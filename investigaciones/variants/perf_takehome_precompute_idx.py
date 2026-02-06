"""
Optimization: Precompute partial idx update before hash.

Current flow for idx update:
1. [After hash] bit = val & 1
2. [After hash] partial = idx * 2 + (1-fp)  [multiply_add]
3. [After hash] idx = partial + bit

Optimized flow:
1. [Before hash] partial = idx * 2 + (1-fp)  [can run in parallel with hash]
2. [After hash] bit = val & 1
3. [After hash] idx = partial + bit

This moves 1 VALU op from the post-hash critical path to before the hash,
allowing better overlap with hash computation.

Savings: For depths where we precompute (2-9, excluding 0,1,10):
- 8 depths in first cycle (rounds 2-9) + 2 depths in second (rounds 13-14) = 10 rounds
- Wait, let me recalculate...

Rounds with idx update: 1,2,3,4,5,6,7,8,9, 12,13,14,15 = 13 rounds
But for round 1 (depth 1), idx is SET during the round, not read.
So we can precompute for rounds 2-9 and 12-15 = 12 rounds.

Actually, depth 1 sets idx = fp + 1 + bit, then updates to idx = 2*idx + ...
So we CAN precompute for depth 1 too, but we need idx to be valid first.

Let me think more carefully:
- Round 0 (depth 0): idx not used, no update
- Round 1 (depth 1): idx set to fp+1+bit, then updated -> can't precompute (idx just set)
- Round 2-9 (depth 2-9): idx read for load/vselect, then updated -> CAN precompute
- Round 10 (depth 10): idx read, no update
- Round 11 (depth 0): idx=0, no update
- Round 12 (depth 1): idx set, then updated -> can't precompute
- Round 13-15 (depth 2-4): idx read, then updated -> CAN precompute

Rounds where we can precompute: 2,3,4,5,6,7,8,9,13,14,15 = 11 rounds
Vectors: 32
Savings: 11 * 32 = 352 VALU ops that can overlap with hash

At current 40% of bundles having VALU=4, filling those with precomputed ops could help.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}
        self.aggressive_schedule = False

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots, vliw=False):
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        def _addrs_range(base, length):
            return range(base, base + length)

        def _rw_sets(engine, slot):
            reads = set()
            writes = set()
            use_mem = not self.aggressive_schedule
            MEM = ("MEM",)

            if engine == "alu":
                _op, dest, a1, a2 = slot
                writes.add(dest)
                reads.add(a1)
                reads.add(a2)
            elif engine == "valu":
                op = slot[0]
                if op == "vbroadcast":
                    _, dest, src = slot
                    writes.update(_addrs_range(dest, VLEN))
                    reads.add(src)
                    if use_mem:
                        writes.add(MEM)
                elif op == "multiply_add":
                    _, dest, a, b, c = slot
                    writes.update(_addrs_range(dest, VLEN))
                    reads.update(_addrs_range(a, VLEN))
                    reads.update(_addrs_range(b, VLEN))
                    reads.update(_addrs_range(c, VLEN))
                else:
                    _, dest, a1, a2 = slot
                    writes.update(_addrs_range(dest, VLEN))
                    reads.update(_addrs_range(a1, VLEN))
                    reads.update(_addrs_range(a2, VLEN))
            elif engine == "load":
                op = slot[0]
                if op == "load":
                    _, dest, addr = slot
                    writes.add(dest)
                    reads.add(addr)
                    if use_mem:
                        reads.add(MEM)
                elif op == "load_offset":
                    _, dest, addr, offset = slot
                    writes.add(dest + offset)
                    reads.add(addr + offset)
                    if use_mem:
                        reads.add(MEM)
                elif op == "vload":
                    _, dest, addr = slot
                    writes.update(_addrs_range(dest, VLEN))
                    reads.add(addr)
                    if use_mem:
                        reads.add(MEM)
                elif op == "const":
                    _, dest, _val = slot
                    writes.add(dest)
            elif engine == "store":
                op = slot[0]
                if op == "store":
                    _, addr, src = slot
                    reads.add(addr)
                    reads.add(src)
                elif op == "vstore":
                    _, addr, src = slot
                    reads.add(addr)
                    reads.update(_addrs_range(src, VLEN))
                    if use_mem:
                        writes.add(MEM)
            elif engine == "flow":
                op = slot[0]
                if op == "select":
                    _, dest, cond, a, b = slot
                    writes.add(dest)
                    reads.add(cond)
                    reads.add(a)
                    reads.add(b)
                elif op == "addimm":
                    _, dest, a, _imm = slot
                    writes.add(dest)
                    reads.add(a)
                elif op == "coreid":
                    _, dest = slot
                    writes.add(dest)
                elif op == "vselect":
                    _, dest, cond, a, b = slot
                    writes.update(_addrs_range(dest, VLEN))
                    reads.update(_addrs_range(cond, VLEN))
                    reads.update(_addrs_range(a, VLEN))
                    reads.update(_addrs_range(b, VLEN))
            return reads, writes

        instrs = []

        def can_pack(engine, reads, writes, cur_counts, cur_writes):
            if cur_counts.get(engine, 0) + 1 > SLOT_LIMITS.get(engine, 0):
                return False
            if (reads & cur_writes) or (writes & cur_writes):
                return False
            return True

        cur = {}
        cur_counts = {k: 0 for k in SLOT_LIMITS.keys()}
        cur_writes = set()

        def flush():
            nonlocal cur, cur_counts, cur_writes
            if cur:
                instrs.append(cur)
            cur = {}
            cur_counts = {k: 0 for k in SLOT_LIMITS.keys()}
            cur_writes = set()

        pending = list(slots)
        window_size = 1024

        while pending:
            if pending[0][0] == "debug":
                flush()
                engine, slot = pending.pop(0)
                instrs.append({engine: [slot]})
                continue

            while True:
                first_debug = None
                for i in range(min(window_size, len(pending))):
                    if pending[i][0] == "debug":
                        first_debug = i
                        break
                scan_limit = first_debug if first_debug is not None else min(
                    window_size, len(pending)
                )
                if scan_limit == 0:
                    break

                prefix_reads = set()
                prefix_writes = set()
                chosen_i = None
                chosen_rw = None
                for i in range(scan_limit):
                    engine, slot = pending[i]
                    reads, writes = _rw_sets(engine, slot)
                    if (writes & prefix_reads) or (writes & prefix_writes) or (
                        reads & prefix_writes
                    ):
                        prefix_reads |= reads
                        prefix_writes |= writes
                        continue
                    if can_pack(engine, reads, writes, cur_counts, cur_writes):
                        chosen_i = i
                        chosen_rw = (reads, writes)
                        break
                    prefix_reads |= reads
                    prefix_writes |= writes

                if chosen_i is None:
                    break

                engine, slot = pending.pop(chosen_i)
                reads, writes = chosen_rw
                cur.setdefault(engine, []).append(slot)
                cur_counts[engine] = cur_counts.get(engine, 0) + 1
                cur_writes |= writes

            if cur:
                flush()
                continue

            engine, slot = pending.pop(0)
            reads, writes = _rw_sets(engine, slot)
            cur.setdefault(engine, []).append(slot)
            cur_counts[engine] = cur_counts.get(engine, 0) + 1
            cur_writes |= writes
            flush()

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space at {self.scratch_ptr}"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def vector_const(self, val, name=None):
        if val not in self.vec_const_map:
            addr = self.alloc_scratch(name, length=VLEN)
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.vec_const_map[val] = addr
        return self.vec_const_map[val]

    def build_hash_vec_multi(self, val_addrs, tmp1_addrs, tmp2_addrs, round, i_base, emit_debug):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mult = 1 + (1 << val3)
                for u in range(len(val_addrs)):
                    slots.append(("valu", ("multiply_add", val_addrs[u], val_addrs[u],
                                           self.vector_const(mult), self.vector_const(val1))))
            else:
                for u in range(len(val_addrs)):
                    slots.append(("valu", (op1, tmp1_addrs[u], val_addrs[u], self.vector_const(val1))))
                for u in range(len(val_addrs)):
                    slots.append(("valu", (op3, tmp2_addrs[u], val_addrs[u], self.vector_const(val3))))
                for u in range(len(val_addrs)):
                    slots.append(("valu", (op2, val_addrs[u], tmp1_addrs[u], tmp2_addrs[u])))
        return slots

    def build_hash_vec_multi_stages(self, val_addrs, tmp1_addrs, tmp2_addrs, round, i_base, emit_debug):
        stages = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            stage_slots = []
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mult = 1 + (1 << val3)
                for u in range(len(val_addrs)):
                    stage_slots.append(("valu", ("multiply_add", val_addrs[u], val_addrs[u],
                                                  self.vector_const(mult), self.vector_const(val1))))
            else:
                for u in range(len(val_addrs)):
                    stage_slots.append(("valu", (op1, tmp1_addrs[u], val_addrs[u], self.vector_const(val1))))
                for u in range(len(val_addrs)):
                    stage_slots.append(("valu", (op3, tmp2_addrs[u], val_addrs[u], self.vector_const(val3))))
                for u in range(len(val_addrs)):
                    stage_slots.append(("valu", (op2, val_addrs[u], tmp1_addrs[u], tmp2_addrs[u])))
            stages.append(stage_slots)
        return stages

    def build_hash_pipeline_addr(self, v_idx, v_val, v_tmp1, v_tmp2, round, i_base,
                                  emit_debug, vec_count, hash_group=3):
        slots = []
        group_size = min(hash_group, vec_count)
        num_groups = (vec_count + group_size - 1) // group_size
        load_progress = [0] * num_groups

        def emit_load_for_group(g, group_start, group_vecs):
            idx = load_progress[g]
            u = group_start + idx // VLEN
            lane = idx % VLEN
            slots.append(("load", ("load_offset", v_tmp1[u], v_idx[u], lane)))
            load_progress[g] = idx + 1

        for g in range(num_groups):
            group_start = g * group_size
            group_vecs = min(group_size, vec_count - group_start)
            total_loads = group_vecs * VLEN
            while load_progress[g] < total_loads:
                emit_load_for_group(g, group_start, group_vecs)

            for u in range(group_start, group_start + group_vecs):
                slots.append(("valu", ("^", v_val[u], v_val[u], v_tmp1[u])))

            stages = self.build_hash_vec_multi_stages(
                v_val[group_start:group_start + group_vecs],
                v_tmp1[group_start:group_start + group_vecs],
                v_tmp2[group_start:group_start + group_vecs],
                round, i_base + group_start * VLEN, emit_debug
            )

            next_g = g + 1
            if next_g < num_groups:
                next_start = next_g * group_size
                next_vecs = min(group_size, vec_count - next_start)
                next_total = next_vecs * VLEN
                remaining = next_total - load_progress[next_g]
                loads_per_stage = (remaining + (len(stages) * 2) - 1) // (len(stages) * 2) if remaining > 0 else 0
            else:
                next_start = next_vecs = next_total = loads_per_stage = 0

            for stage_slots in stages:
                if next_g < num_groups and loads_per_stage:
                    for _ in range(loads_per_stage):
                        if load_progress[next_g] >= next_total:
                            break
                        emit_load_for_group(next_g, next_start, next_vecs)
                slots.extend(stage_slots)
                if next_g < num_groups and loads_per_stage:
                    for _ in range(loads_per_stage):
                        if load_progress[next_g] >= next_total:
                            break
                        emit_load_for_group(next_g, next_start, next_vecs)

            if next_g < num_groups:
                while load_progress[next_g] < next_total:
                    emit_load_for_group(next_g, next_start, next_vecs)

        return slots

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        if batch_size == 256 and rounds == 16 and forest_height == 10:
            self.aggressive_schedule = True
            emit_debug = False
            store_indices = True

            tmp1 = self.alloc_scratch("tmp1")
            tmp2 = self.alloc_scratch("tmp2")

            forest_values_p = self.alloc_scratch("forest_values_p", 1)
            inp_indices_p = self.alloc_scratch("inp_indices_p", 1)
            inp_values_p = self.alloc_scratch("inp_values_p", 1)
            forest_values_p_val = 7
            inp_indices_p_val = forest_values_p_val + n_nodes
            inp_values_p_val = forest_values_p_val + n_nodes + batch_size

            zero_const = self.alloc_scratch("zero_const")
            one_const = self.alloc_scratch("one_const")
            two_const = self.alloc_scratch("two_const")
            vlen_const = self.alloc_scratch("vlen_const")
            const_3 = self.alloc_scratch("const_3")
            const_4 = self.alloc_scratch("const_4")
            const_5 = self.alloc_scratch("const_5")
            const_6 = self.alloc_scratch("const_6")

            root_val = self.alloc_scratch("root_val")
            level1_left = self.alloc_scratch("level1_left")
            level1_right = self.alloc_scratch("level1_right")
            level2_vals = [self.alloc_scratch(f"level2_val_{i}") for i in range(4)]

            v_one = self.alloc_scratch("v_one", length=VLEN)
            v_two = self.alloc_scratch("v_two", length=VLEN)
            v_root_val = self.alloc_scratch("v_root_val", length=VLEN)
            v_level1_left = self.alloc_scratch("v_level1_left", length=VLEN)
            v_level1_right = self.alloc_scratch("v_level1_right", length=VLEN)
            v_level2 = [self.alloc_scratch(f"v_level2_{i}", length=VLEN) for i in range(4)]
            v_forest_values_p = self.alloc_scratch("v_forest_values_p", length=VLEN)
            v_base_plus1 = self.alloc_scratch("v_base_plus1", length=VLEN)
            v_base_minus1 = self.alloc_scratch("v_base_minus1", length=VLEN)
            v_neg_forest = self.alloc_scratch("v_neg_forest", length=VLEN)

            init = []

            init.append(("load", ("const", forest_values_p, forest_values_p_val)))
            init.append(("load", ("const", inp_indices_p, inp_indices_p_val)))
            init.append(("load", ("const", inp_values_p, inp_values_p_val)))
            init.append(("load", ("const", zero_const, 0)))
            init.append(("load", ("const", one_const, 1)))
            init.append(("load", ("const", two_const, 2)))
            init.append(("load", ("const", vlen_const, VLEN)))
            init.append(("load", ("const", const_3, 3)))
            init.append(("load", ("const", const_4, 4)))
            init.append(("load", ("const", const_5, 5)))
            init.append(("load", ("const", const_6, 6)))

            init.append(("valu", ("vbroadcast", v_one, one_const)))
            init.append(("valu", ("vbroadcast", v_two, two_const)))

            init.append(("load", ("load", root_val, forest_values_p)))
            init.append(("alu", ("+", tmp1, forest_values_p, one_const)))
            init.append(("alu", ("+", tmp2, forest_values_p, two_const)))
            init.append(("load", ("load", level1_left, tmp1)))
            init.append(("load", ("load", level1_right, tmp2)))

            init.append(("alu", ("+", tmp1, forest_values_p, const_3)))
            init.append(("load", ("load", level2_vals[0], tmp1)))
            init.append(("alu", ("+", tmp1, forest_values_p, const_4)))
            init.append(("load", ("load", level2_vals[1], tmp1)))
            init.append(("alu", ("+", tmp1, forest_values_p, const_5)))
            init.append(("load", ("load", level2_vals[2], tmp1)))
            init.append(("alu", ("+", tmp1, forest_values_p, const_6)))
            init.append(("load", ("load", level2_vals[3], tmp1)))

            init.append(("valu", ("vbroadcast", v_root_val, root_val)))
            init.append(("valu", ("vbroadcast", v_level1_left, level1_left)))
            init.append(("valu", ("vbroadcast", v_level1_right, level1_right)))
            init.append(("valu", ("vbroadcast", v_level2[0], level2_vals[0])))
            init.append(("valu", ("vbroadcast", v_level2[1], level2_vals[1])))
            init.append(("valu", ("vbroadcast", v_level2[2], level2_vals[2])))
            init.append(("valu", ("vbroadcast", v_level2[3], level2_vals[3])))
            init.append(("valu", ("vbroadcast", v_forest_values_p, forest_values_p)))

            init.append(("valu", ("+", v_base_plus1, v_forest_values_p, v_one)))
            init.append(("valu", ("-", v_base_minus1, v_one, v_forest_values_p)))
            init.append(("valu", ("-", v_neg_forest, v_base_minus1, v_one)))

            hash_consts = [9, 16, 19, 33, 4097, 374761393, 2127912214,
                          3042594569, 3345072700, 3550635116, 4251993797]
            hash_scalar_addrs = {}
            hash_vector_addrs = {}
            for hc in hash_consts:
                scalar_addr = self.alloc_scratch(f"hc_s_{hc}")
                vector_addr = self.alloc_scratch(f"hc_v_{hc}", length=VLEN)
                hash_scalar_addrs[hc] = scalar_addr
                hash_vector_addrs[hc] = vector_addr
                init.append(("load", ("const", scalar_addr, hc)))
            for hc in hash_consts:
                init.append(("valu", ("vbroadcast", hash_vector_addrs[hc], hash_scalar_addrs[hc])))

            init.append(("flow", ("pause",)))

            init_instrs = self.build(init, vliw=True)
            self.instrs.extend(init_instrs)

            self.const_map[0] = zero_const
            self.const_map[1] = one_const
            self.const_map[2] = two_const
            self.const_map[VLEN] = vlen_const
            self.vec_const_map[1] = v_one
            self.vec_const_map[2] = v_two
            for hc in hash_consts:
                self.const_map[hc] = hash_scalar_addrs[hc]
                self.vec_const_map[hc] = hash_vector_addrs[hc]

            body = []

            UNROLL_MAIN = 32

            v_idx = [self.alloc_scratch(f"v_idx{u}", length=VLEN) for u in range(UNROLL_MAIN)]
            v_val = [self.alloc_scratch(f"v_val{u}", length=VLEN) for u in range(UNROLL_MAIN)]
            v_tmp1 = [self.alloc_scratch(f"v_tmp1_{u}", length=VLEN) for u in range(UNROLL_MAIN)]
            v_tmp2 = [self.alloc_scratch(f"v_tmp2_{u}", length=VLEN) for u in range(UNROLL_MAIN)]
            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)

            # NEW: Allocate partial_idx vectors for precomputation
            v_partial_idx = [self.alloc_scratch(f"v_partial_idx{u}", length=VLEN) for u in range(UNROLL_MAIN)]

            tmp_val_addr_u = [self.alloc_scratch(f"tmp_val_addr{u}") for u in range(UNROLL_MAIN)]
            tmp_idx_addr_u = [self.alloc_scratch(f"tmp_idx_addr{u}") for u in range(UNROLL_MAIN)] if store_indices else []

            print(f"Scratch usage after allocation: {self.scratch_ptr} / {SCRATCH_SIZE}")

            def emit_precompute_partial_idx(start, count):
                """Precompute partial_idx = idx * 2 + (1-fp) before hash."""
                for u in range(start, start + count):
                    body.append(("valu", ("multiply_add", v_partial_idx[u], v_idx[u], v_two, v_base_minus1)))

            def emit_finalize_idx_update(start, count):
                """Finalize idx update after hash: idx = partial + bit."""
                for u in range(start, start + count):
                    body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))  # bit
                    body.append(("valu", ("+", v_idx[u], v_partial_idx[u], v_tmp1[u])))  # idx = partial + bit

            def emit_hash_only_range(round_idx, depth, start, count, precompute_next_idx=False):
                v_idx_l = v_idx[start:start + count]
                v_val_l = v_val[start:start + count]
                v_tmp1_l = v_tmp1[start:start + count]
                v_tmp2_l = v_tmp2[start:start + count]

                if depth == 0:
                    for u in range(count):
                        body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_root_val)))
                    # Precompute partial idx for NEXT round (depth 1) if needed
                    # But depth 1 SETS idx, not reads it, so no precompute needed
                    body.extend(self.build_hash_vec_multi(v_val_l, v_tmp1_l, v_tmp2_l,
                                                          round_idx, start * VLEN, emit_debug))
                elif depth == 1:
                    for u in range(count):
                        body.append(("valu", ("&", v_tmp1_l[u], v_val_l[u], v_one)))
                        body.append(("valu", ("+", v_idx_l[u], v_base_plus1, v_tmp1_l[u])))
                        body.append(("flow", ("vselect", v_tmp1_l[u], v_tmp1_l[u],
                                              v_level1_right, v_level1_left)))
                        body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_tmp1_l[u])))
                    # For depth 1, idx was just SET, so precompute for depth 2
                    if precompute_next_idx:
                        emit_precompute_partial_idx(start, count)
                    body.extend(self.build_hash_vec_multi(v_val_l, v_tmp1_l, v_tmp2_l,
                                                          round_idx, start * VLEN, emit_debug))
                elif depth == 2:
                    # Precompute partial_idx BEFORE hash (idx was set in previous round)
                    if precompute_next_idx:
                        emit_precompute_partial_idx(start, count)
                    for u in range(count):
                        body.append(("valu", ("&", v_tmp1_l[u], v_idx_l[u], v_one)))
                        body.append(("valu", ("&", v_tmp2_l[u], v_idx_l[u], v_two)))
                        body.append(("flow", ("vselect", v_tmp3_shared, v_tmp1_l[u],
                                              v_level2[3], v_level2[2])))
                        body.append(("flow", ("vselect", v_tmp1_l[u], v_tmp1_l[u],
                                              v_level2[1], v_level2[0])))
                        body.append(("flow", ("vselect", v_tmp1_l[u], v_tmp2_l[u],
                                              v_tmp1_l[u], v_tmp3_shared)))
                        body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_tmp1_l[u])))
                    body.extend(self.build_hash_vec_multi(v_val_l, v_tmp1_l, v_tmp2_l,
                                                          round_idx, start * VLEN, emit_debug))
                else:
                    # Depths 3-10: Precompute partial_idx BEFORE loads/hash
                    if precompute_next_idx and depth != forest_height:
                        emit_precompute_partial_idx(start, count)
                    body.extend(self.build_hash_pipeline_addr(v_idx_l, v_val_l, v_tmp1_l, v_tmp2_l,
                                                               round_idx, start * VLEN, emit_debug,
                                                               count, hash_group=3))

            def emit_group(vec_count, base_offset_const, base_is_zero=False):
                if base_is_zero:
                    base_val_addr = self.scratch["inp_values_p"]
                    if store_indices:
                        base_idx_addr = self.scratch["inp_indices_p"]
                else:
                    body.append(("alu", ("+", tmp_val_addr_u[0], self.scratch["inp_values_p"], base_offset_const)))
                    base_val_addr = tmp_val_addr_u[0]
                    if store_indices:
                        body.append(("alu", ("+", tmp_idx_addr_u[0], self.scratch["inp_indices_p"], base_offset_const)))
                        base_idx_addr = tmp_idx_addr_u[0]
                for u in range(1, vec_count):
                    if u == 1:
                        body.append(("alu", ("+", tmp_val_addr_u[u], base_val_addr, vlen_const)))
                        if store_indices:
                            body.append(("alu", ("+", tmp_idx_addr_u[u], base_idx_addr, vlen_const)))
                    else:
                        body.append(("alu", ("+", tmp_val_addr_u[u], tmp_val_addr_u[u - 1], vlen_const)))
                        if store_indices:
                            body.append(("alu", ("+", tmp_idx_addr_u[u], tmp_idx_addr_u[u - 1], vlen_const)))

                for u in range(vec_count):
                    if u == 0:
                        body.append(("load", ("vload", v_val[u], base_val_addr)))
                    else:
                        body.append(("load", ("vload", v_val[u], tmp_val_addr_u[u])))

                round_depths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]
                for round_idx in range(rounds):
                    depth = round_depths[round_idx]
                    next_depth = round_depths[(round_idx + 1) % rounds] if round_idx < rounds - 1 else -1

                    # Determine if we should precompute partial_idx
                    # We can precompute if:
                    # - Next round needs idx update (next_depth != 0 and next_depth != forest_height)
                    # - AND idx is already valid (not being SET this round)
                    can_precompute = (depth >= 1 and depth != forest_height and
                                     next_depth != 0 and next_depth != forest_height and
                                     next_depth != -1)

                    chunk = 1
                    if depth > 2:
                        starts = (0, 16, 4, 20, 8, 24, 12, 28, 2, 18, 6, 22, 10, 26, 14, 30,
                                  1, 17, 5, 21, 9, 25, 13, 29, 3, 19, 7, 23, 11, 27, 15, 31)
                    else:
                        starts = (0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                                  1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31)

                    for start in starts:
                        count = min(chunk, vec_count - start)
                        emit_hash_only_range(round_idx, depth, start, count, precompute_next_idx=can_precompute)

                        # Finalize idx update using precomputed partial (or do full update)
                        if depth != 0 and depth != forest_height:
                            # Check if we precomputed in the PREVIOUS round
                            prev_depth = round_depths[round_idx - 1] if round_idx > 0 else -1
                            prev_precomputed = (prev_depth >= 1 and prev_depth != forest_height and
                                               depth != 0 and depth != forest_height)

                            if prev_precomputed and round_idx > 0:
                                # Use precomputed partial
                                emit_finalize_idx_update(start, count)
                            else:
                                # Full idx update (no precomputation available)
                                for u in range(start, start + count):
                                    body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                    body.append(("valu", ("multiply_add", v_idx[u], v_idx[u], v_two, v_base_minus1)))
                                    body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))

                if store_indices:
                    for u in range(vec_count):
                        body.append(("valu", ("+", v_tmp1[u], v_idx[u], v_neg_forest)))
                    for u in range(vec_count):
                        if u == 0:
                            body.append(("store", ("vstore", base_idx_addr, v_tmp1[u])))
                        else:
                            body.append(("store", ("vstore", tmp_idx_addr_u[u], v_tmp1[u])))
                        if u == 0:
                            body.append(("store", ("vstore", base_val_addr, v_val[u])))
                        else:
                            body.append(("store", ("vstore", tmp_val_addr_u[u], v_val[u])))
                else:
                    for u in range(vec_count):
                        if u == 0:
                            body.append(("store", ("vstore", base_val_addr, v_val[u])))
                        else:
                            body.append(("store", ("vstore", tmp_val_addr_u[u], v_val[u])))

            emit_group(UNROLL_MAIN, zero_const, base_is_zero=True)

            body_instrs = self.build(body, vliw=True)
            self.instrs.extend(body_instrs)
            self.instrs.append({"flow": [("pause",)]})
            self.aggressive_schedule = False
            return

        # Fallback for non-standard parameters (copy from original)
        raise NotImplementedError("Only batch_size=256, rounds=16, forest_height=10 supported")


BASELINE = 147734

def do_kernel_test(forest_height, rounds, batch_size, seed=123, trace=False, prints=False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES,
                      value_trace=value_trace, trace=trace)
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        assert (machine.mem[inp_values_p:inp_values_p + len(inp.values)] ==
                ref_mem[inp_values_p:inp_values_p + len(inp.values)]), f"Incorrect values on round {i}"
        inp_indices_p = ref_mem[5]
        assert (machine.mem[inp_indices_p:inp_indices_p + len(inp.indices)] ==
                ref_mem[inp_indices_p:inp_indices_p + len(inp.indices)]), f"Incorrect indices on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


if __name__ == "__main__":
    do_kernel_test(10, 16, 256)
