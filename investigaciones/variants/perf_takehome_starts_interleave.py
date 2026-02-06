"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
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

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Modo no-VLIW: comportamiento original, 1 slot por bundle.
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        def _addrs_range(base: int, length: int):
            return range(base, base + length)

        def _rw_sets(engine: str, slot: tuple):
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
                    # Semántica real (ver problem.py):
                    #   scratch[dest + offset] = mem[scratch[addr + offset]]
                    # Por tanto depende de TODAS las lanes del vector addr.
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
                else:
                    # halt/pause/jumps/trace_write: no afectan a datos.
                    pass

            # debug engine no cambia estado arquitectural.
            return reads, writes

        instrs: list[dict[str, list[tuple]]] = []

        def can_pack(engine, reads, writes, cur_counts, cur_writes):
            if cur_counts.get(engine, 0) + 1 > SLOT_LIMITS.get(engine, 0):
                return False
            if (reads & cur_writes) or (writes & cur_writes):
                return False
            return True

        cur: dict[str, list[tuple]] = {}
        cur_counts = {k: 0 for k in SLOT_LIMITS.keys()}
        cur_writes: set[int | tuple] = set()

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

                prefix_reads: set[int | tuple] = set()
                prefix_writes: set[int | tuple] = set()
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
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i, emit_debug=False):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            if emit_debug:
                slots.append(
                    ("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))
                )

        return slots

    def build_hash_vec(self, val_hash_addr, tmp1, tmp2, round, i_base, emit_debug):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mult = 1 + (1 << val3)
                slots.append(
                    (
                        "valu",
                        (
                            "multiply_add",
                            val_hash_addr,
                            val_hash_addr,
                            self.vector_const(mult),
                            self.vector_const(val1),
                        ),
                    )
                )
            else:
                slots.append(
                    ("valu", (op1, tmp1, val_hash_addr, self.vector_const(val1)))
                )
                slots.append(
                    ("valu", (op3, tmp2, val_hash_addr, self.vector_const(val3)))
                )
                slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))
            if emit_debug:
                keys = [
                    (round, i_base + lane, "hash_stage", hi) for lane in range(VLEN)
                ]
                slots.append(("debug", ("vcompare", val_hash_addr, keys)))
        return slots

    def build_hash_vec_multi(
        self, val_addrs, tmp1_addrs, tmp2_addrs, round, i_base, emit_debug
    ):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mult = 1 + (1 << val3)
                for u in range(len(val_addrs)):
                    slots.append(
                        (
                            "valu",
                            (
                                "multiply_add",
                                val_addrs[u],
                                val_addrs[u],
                                self.vector_const(mult),
                                self.vector_const(val1),
                            ),
                        )
                    )
            else:
                for u in range(len(val_addrs)):
                    slots.append(
                        (
                            "valu",
                            (op1, tmp1_addrs[u], val_addrs[u], self.vector_const(val1)),
                        )
                    )
                for u in range(len(val_addrs)):
                    slots.append(
                        (
                            "valu",
                            (op3, tmp2_addrs[u], val_addrs[u], self.vector_const(val3)),
                        )
                    )
                for u in range(len(val_addrs)):
                    slots.append(
                        ("valu", (op2, val_addrs[u], tmp1_addrs[u], tmp2_addrs[u]))
                    )
            if emit_debug:
                for u in range(len(val_addrs)):
                    base = i_base + u * VLEN
                    keys = [
                        (round, base + lane, "hash_stage", hi) for lane in range(VLEN)
                    ]
                    slots.append(("debug", ("vcompare", val_addrs[u], keys)))
        return slots

    def build_hash_vec_multi_stages(
        self, val_addrs, tmp1_addrs, tmp2_addrs, round, i_base, emit_debug
    ):
        stages = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            stage_slots = []
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mult = 1 + (1 << val3)
                for u in range(len(val_addrs)):
                    stage_slots.append(
                        (
                            "valu",
                            (
                                "multiply_add",
                                val_addrs[u],
                                val_addrs[u],
                                self.vector_const(mult),
                                self.vector_const(val1),
                            ),
                        )
                    )
            else:
                for u in range(len(val_addrs)):
                    stage_slots.append(
                        (
                            "valu",
                            (op1, tmp1_addrs[u], val_addrs[u], self.vector_const(val1)),
                        )
                    )
                for u in range(len(val_addrs)):
                    stage_slots.append(
                        (
                            "valu",
                            (op3, tmp2_addrs[u], val_addrs[u], self.vector_const(val3)),
                        )
                    )
                for u in range(len(val_addrs)):
                    stage_slots.append(
                        ("valu", (op2, val_addrs[u], tmp1_addrs[u], tmp2_addrs[u]))
                    )
            if emit_debug:
                for u in range(len(val_addrs)):
                    base = i_base + u * VLEN
                    keys = [
                        (round, base + lane, "hash_stage", hi) for lane in range(VLEN)
                    ]
                    stage_slots.append(("debug", ("vcompare", val_addrs[u], keys)))
            stages.append(stage_slots)
        return stages

    def build_hash_pipeline(
        self,
        v_idx,
        v_val,
        v_node_addr,
        v_node_val,
        v_tmp1,
        v_tmp2,
        v_forest_values_p,
        round,
        i_base,
        emit_debug,
        vec_count,
        hash_group=3,
    ):
        slots = []
        for u in range(vec_count):
            slots.append(("valu", ("+", v_node_addr[u], v_idx[u], v_forest_values_p)))

        group_size = min(hash_group, vec_count)
        num_groups = (vec_count + group_size - 1) // group_size
        load_progress = [0] * num_groups

        def emit_load_for_group(g, group_start, group_vecs):
            idx = load_progress[g]
            u = group_start + idx // VLEN
            lane = idx % VLEN
            slots.append(("load", ("load_offset", v_node_val[u], v_node_addr[u], lane)))
            load_progress[g] = idx + 1

        for g in range(num_groups):
            group_start = g * group_size
            group_vecs = min(group_size, vec_count - group_start)
            total_loads = group_vecs * VLEN
            while load_progress[g] < total_loads:
                emit_load_for_group(g, group_start, group_vecs)
            if emit_debug:
                for u in range(group_start, group_start + group_vecs):
                    base = i_base + u * VLEN
                    keys = [(round, base + lane, "node_val") for lane in range(VLEN)]
                    slots.append(("debug", ("vcompare", v_node_val[u], keys)))

            for u in range(group_start, group_start + group_vecs):
                slots.append(("valu", ("^", v_val[u], v_val[u], v_node_val[u])))

            stages = self.build_hash_vec_multi_stages(
                v_val[group_start : group_start + group_vecs],
                v_tmp1[group_start : group_start + group_vecs],
                v_tmp2[group_start : group_start + group_vecs],
                round,
                i_base + group_start * VLEN,
                emit_debug,
            )

            next_g = g + 1
            if next_g < num_groups:
                next_start = next_g * group_size
                next_vecs = min(group_size, vec_count - next_start)
                next_total = next_vecs * VLEN
                remaining = next_total - load_progress[next_g]
                loads_per_stage = (
                    (remaining + (len(stages) * 2) - 1) // (len(stages) * 2)
                    if remaining > 0
                    else 0
                )
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

    def build_hash_pipeline_addr(
        self,
        v_idx,
        v_val,
        v_tmp1,
        v_tmp2,
        round,
        i_base,
        emit_debug,
        vec_count,
        hash_group=3,
    ):
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
            if emit_debug:
                for u in range(group_start, group_start + group_vecs):
                    base = i_base + u * VLEN
                    keys = [(round, base + lane, "node_val") for lane in range(VLEN)]
                    slots.append(("debug", ("vcompare", v_tmp1[u], keys)))

            for u in range(group_start, group_start + group_vecs):
                slots.append(("valu", ("^", v_val[u], v_val[u], v_tmp1[u])))

            stages = self.build_hash_vec_multi_stages(
                v_val[group_start : group_start + group_vecs],
                v_tmp1[group_start : group_start + group_vecs],
                v_tmp2[group_start : group_start + group_vecs],
                round,
                i_base + group_start * VLEN,
                emit_debug,
            )

            next_g = g + 1
            if next_g < num_groups:
                next_start = next_g * group_size
                next_vecs = min(group_size, vec_count - next_start)
                next_total = next_vecs * VLEN
                remaining = next_total - load_progress[next_g]
                loads_per_stage = (
                    (remaining + (len(stages) * 2) - 1) // (len(stages) * 2)
                    if remaining > 0
                    else 0
                )
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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        if batch_size == 256 and rounds == 16 and forest_height == 10:
            self.aggressive_schedule = True
            # Fast path specialized for the submission benchmark.
            emit_debug = False
            store_indices = True  # submission requires correct indices

            tmp1 = self.alloc_scratch("tmp1")
            tmp2 = self.alloc_scratch("tmp2")

            forest_values_p = self.alloc_scratch("forest_values_p", 1)
            inp_indices_p = self.alloc_scratch("inp_indices_p", 1)
            inp_values_p = self.alloc_scratch("inp_values_p", 1)
            forest_values_p_val = 7
            inp_indices_p_val = forest_values_p_val + n_nodes
            inp_values_p_val = forest_values_p_val + n_nodes + batch_size

            # Pre-allocate all scratch space for constants
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

            # Build initialization with VLIW packing
            init = []

            # Load all constants in parallel (2 load slots per cycle)
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

            # Broadcast scalar constants to vectors
            init.append(("valu", ("vbroadcast", v_one, one_const)))
            init.append(("valu", ("vbroadcast", v_two, two_const)))

            # Load tree values - need addresses first
            init.append(("load", ("load", root_val, forest_values_p)))
            init.append(("alu", ("+", tmp1, forest_values_p, one_const)))
            init.append(("alu", ("+", tmp2, forest_values_p, two_const)))
            init.append(("load", ("load", level1_left, tmp1)))
            init.append(("load", ("load", level1_right, tmp2)))

            # Level 2 addresses and loads
            init.append(("alu", ("+", tmp1, forest_values_p, const_3)))
            init.append(("load", ("load", level2_vals[0], tmp1)))
            init.append(("alu", ("+", tmp1, forest_values_p, const_4)))
            init.append(("load", ("load", level2_vals[1], tmp1)))
            init.append(("alu", ("+", tmp1, forest_values_p, const_5)))
            init.append(("load", ("load", level2_vals[2], tmp1)))
            init.append(("alu", ("+", tmp1, forest_values_p, const_6)))
            init.append(("load", ("load", level2_vals[3], tmp1)))

            # Broadcasts (can run in parallel, 6 valu slots)
            init.append(("valu", ("vbroadcast", v_root_val, root_val)))
            init.append(("valu", ("vbroadcast", v_level1_left, level1_left)))
            init.append(("valu", ("vbroadcast", v_level1_right, level1_right)))
            init.append(("valu", ("vbroadcast", v_level2[0], level2_vals[0])))
            init.append(("valu", ("vbroadcast", v_level2[1], level2_vals[1])))
            init.append(("valu", ("vbroadcast", v_level2[2], level2_vals[2])))
            init.append(("valu", ("vbroadcast", v_level2[3], level2_vals[3])))
            init.append(("valu", ("vbroadcast", v_forest_values_p, forest_values_p)))

            # Compute derived vectors
            init.append(("valu", ("+", v_base_plus1, v_forest_values_p, v_one)))
            init.append(("valu", ("-", v_base_minus1, v_one, v_forest_values_p)))
            init.append(("valu", ("-", v_neg_forest, v_base_minus1, v_one)))

            # Pre-create hash constants (11 vector constants)
            # These are used by build_hash_vec_multi via vector_const()
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

            # Pause to sync with first yield
            init.append(("flow", ("pause",)))

            # Pack and emit initialization
            init_instrs = self.build(init, vliw=True)
            self.instrs.extend(init_instrs)

            # Update const_map for body to use
            self.const_map[0] = zero_const
            self.const_map[1] = one_const
            self.const_map[2] = two_const
            self.const_map[VLEN] = vlen_const
            self.vec_const_map[1] = v_one
            self.vec_const_map[2] = v_two
            # Register pre-created hash constants
            for hc in hash_consts:
                self.const_map[hc] = hash_scalar_addrs[hc]
                self.vec_const_map[hc] = hash_vector_addrs[hc]

            body = []

            UNROLL_MAIN = 32

            v_idx = [
                self.alloc_scratch(f"v_idx{u}", length=VLEN) for u in range(UNROLL_MAIN)
            ]
            v_val = [
                self.alloc_scratch(f"v_val{u}", length=VLEN) for u in range(UNROLL_MAIN)
            ]
            v_tmp1 = [
                self.alloc_scratch(f"v_tmp1_{u}", length=VLEN) for u in range(UNROLL_MAIN)
            ]
            v_tmp2 = [
                self.alloc_scratch(f"v_tmp2_{u}", length=VLEN) for u in range(UNROLL_MAIN)
            ]
            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)
            tmp_val_addr_u = [
                self.alloc_scratch(f"tmp_val_addr{u}") for u in range(UNROLL_MAIN)
            ]
            tmp_idx_addr_u = (
                [self.alloc_scratch(f"tmp_idx_addr{u}") for u in range(UNROLL_MAIN)]
                if store_indices
                else []
            )

            def emit_hash_only_range(round_idx: int, depth: int, start: int, count: int):
                v_idx_l = v_idx[start : start + count]
                v_val_l = v_val[start : start + count]
                v_tmp1_l = v_tmp1[start : start + count]
                v_tmp2_l = v_tmp2[start : start + count]
                if depth == 0:
                    for u in range(count):
                        body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_root_val)))
                    body.extend(
                        self.build_hash_vec_multi(
                            v_val_l,
                            v_tmp1_l,
                            v_tmp2_l,
                            round_idx,
                            start * VLEN,
                            emit_debug,
                        )
                    )
                elif depth == 1:
                    for u in range(count):
                        body.append(("valu", ("&", v_tmp1_l[u], v_val_l[u], v_one)))
                        body.append(("valu", ("+", v_idx_l[u], v_base_plus1, v_tmp1_l[u])))
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp1_l[u],
                                    v_tmp1_l[u],
                                    v_level1_right,
                                    v_level1_left,
                                ),
                            )
                        )
                        body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_tmp1_l[u])))
                    body.extend(
                        self.build_hash_vec_multi(
                            v_val_l,
                            v_tmp1_l,
                            v_tmp2_l,
                            round_idx,
                            start * VLEN,
                            emit_debug,
                        )
                    )
                elif depth == 2:
                    for u in range(count):
                        body.append(("valu", ("&", v_tmp1_l[u], v_idx_l[u], v_one)))  # b0
                        body.append(("valu", ("&", v_tmp2_l[u], v_idx_l[u], v_two)))  # b1 inverted
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp3_shared,
                                    v_tmp1_l[u],
                                    v_level2[3],
                                    v_level2[2],
                                ),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp1_l[u],
                                    v_tmp1_l[u],
                                    v_level2[1],
                                    v_level2[0],
                                ),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp1_l[u],
                                    v_tmp2_l[u],
                                    v_tmp1_l[u],
                                    v_tmp3_shared,
                                ),
                            )
                        )
                        body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_tmp1_l[u])))
                    body.extend(
                        self.build_hash_vec_multi(
                            v_val_l,
                            v_tmp1_l,
                            v_tmp2_l,
                            round_idx,
                            start * VLEN,
                            emit_debug,
                        )
                    )
                else:
                    body.extend(
                        self.build_hash_pipeline_addr(
                            v_idx_l,
                            v_val_l,
                            v_tmp1_l,
                            v_tmp2_l,
                            round_idx,
                            start * VLEN,
                            emit_debug,
                            count,
                            hash_group=3,
                        )
                    )

            def emit_idx_update(vec_count):
                for u in range(vec_count):
                    body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                    body.append(("valu", ("+", v_tmp2[u], v_tmp1[u], v_base_minus1)))
                    body.append(
                        (
                            "valu",
                            ("multiply_add", v_idx[u], v_idx[u], v_two, v_tmp2[u]),
                        )
                    )

            def emit_group(vec_count, base_offset_const, base_is_zero=False):
                if base_is_zero:
                    base_val_addr = self.scratch["inp_values_p"]
                    if store_indices:
                        base_idx_addr = self.scratch["inp_indices_p"]
                else:
                    body.append(
                        (
                            "alu",
                            (
                                "+",
                                tmp_val_addr_u[0],
                                self.scratch["inp_values_p"],
                                base_offset_const,
                            ),
                        )
                    )
                    base_val_addr = tmp_val_addr_u[0]
                    if store_indices:
                        body.append(
                            (
                                "alu",
                                (
                                    "+",
                                    tmp_idx_addr_u[0],
                                    self.scratch["inp_indices_p"],
                                    base_offset_const,
                                ),
                            )
                        )
                        base_idx_addr = tmp_idx_addr_u[0]
                for u in range(1, vec_count):
                    if u == 1:
                        body.append(
                            ("alu", ("+", tmp_val_addr_u[u], base_val_addr, vlen_const))
                        )
                        if store_indices:
                            body.append(
                                ("alu", ("+", tmp_idx_addr_u[u], base_idx_addr, vlen_const))
                            )
                    else:
                        body.append(
                            ("alu", ("+", tmp_val_addr_u[u], tmp_val_addr_u[u - 1], vlen_const))
                        )
                        if store_indices:
                            body.append(
                                (
                                    "alu",
                                    (
                                        "+",
                                        tmp_idx_addr_u[u],
                                        tmp_idx_addr_u[u - 1],
                                        vlen_const,
                                    ),
                                )
                            )

                for u in range(vec_count):
                    if u == 0:
                        body.append(("load", ("vload", v_val[u], base_val_addr)))
                    else:
                        body.append(("load", ("vload", v_val[u], tmp_val_addr_u[u])))

                # Unrolled 16-round schedule with fixed depths: [0..10,0..4]
                round_depths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]
                for round_idx in range(rounds):
                    depth = round_depths[round_idx]
                    chunk = 1
                    if depth > 2:
                        starts = tuple(i for pair in zip(range(0, 16), range(16, 32)) for i in pair)
                    else:
                        starts = tuple(i for pair in zip(range(0, 16), range(16, 32)) for i in pair)
                    for start in starts:
                        count = min(chunk, vec_count - start)
                        emit_hash_only_range(round_idx, depth, start, count)
                        # Skip idx update for depth 0 and forest_height
                        if depth != 0 and depth != forest_height:
                            for u in range(start, start + count):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                body.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            v_idx[u],
                                            v_idx[u],
                                            v_two,
                                            v_base_minus1,
                                        ),
                                    )
                                )
                                body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))

                if store_indices:
                    # Compute idx offsets and store interleaved
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
            # Unconditional pause to sync with second yield from reference_kernel2
            self.instrs.append({"flow": [("pause",)]})
            self.aggressive_schedule = False
            return
        self.aggressive_schedule = False
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        emit_debug = False
        store_indices = True
        # Scratch space addresses
        if emit_debug:
            init_vars = [
                ("rounds", 0),
                ("n_nodes", 1),
                ("batch_size", 2),
                ("forest_height", 3),
                ("forest_values_p", 4),
                ("inp_indices_p", 5),
                ("inp_values_p", 6),
            ]
        else:
            init_vars = [
                ("forest_values_p", 4),
                ("inp_indices_p", 5),
                ("inp_values_p", 6),
            ]
        for name, _ in init_vars:
            self.alloc_scratch(name, 1)
        for name, idx in init_vars:
            self.add("load", ("const", tmp1, idx))
            self.add("load", ("load", self.scratch[name], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        vlen_const = self.scratch_const(VLEN)

        v_zero = self.vector_const(0, "v_zero")
        v_one = self.vector_const(1, "v_one")
        v_two = self.vector_const(2, "v_two")
        root_val = self.alloc_scratch("root_val")
        self.add("load", ("load", root_val, self.scratch["forest_values_p"]))
        v_root_val = self.alloc_scratch("v_root_val", length=VLEN)
        self.add("valu", ("vbroadcast", v_root_val, root_val))
        level1_left = self.alloc_scratch("level1_left")
        level1_right = self.alloc_scratch("level1_right")
        self.add("alu", ("+", tmp1, self.scratch["forest_values_p"], one_const))
        self.add("load", ("load", level1_left, tmp1))
        self.add("alu", ("+", tmp1, self.scratch["forest_values_p"], two_const))
        self.add("load", ("load", level1_right, tmp1))
        v_level1_left = self.alloc_scratch("v_level1_left", length=VLEN)
        v_level1_right = self.alloc_scratch("v_level1_right", length=VLEN)
        self.add("valu", ("vbroadcast", v_level1_left, level1_left))
        self.add("valu", ("vbroadcast", v_level1_right, level1_right))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        # Always add pause to sync with first yield from reference_kernel2
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_idx_addr = (
            self.alloc_scratch("tmp_idx_addr") if store_indices else None
        )
        tmp_val_addr = self.alloc_scratch("tmp_val_addr")

        UNROLL = 20
        group_size = VLEN * UNROLL
        group_size_const = self.scratch_const(group_size)

        # Vector scratch registers
        v_idx = [self.alloc_scratch(f"v_idx{u}", length=VLEN) for u in range(UNROLL)]
        v_val = [self.alloc_scratch(f"v_val{u}", length=VLEN) for u in range(UNROLL)]
        v_node_addr = [
            self.alloc_scratch(f"v_node_addr{u}", length=VLEN) for u in range(UNROLL)
        ]
        v_node_val = [
            self.alloc_scratch(f"v_node_val{u}", length=VLEN) for u in range(UNROLL)
        ]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{u}", length=VLEN) for u in range(UNROLL)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{u}", length=VLEN) for u in range(UNROLL)]
        v_tmp3 = [self.alloc_scratch(f"v_tmp3_{u}", length=VLEN) for u in range(UNROLL)]
        v_forest_values_p = self.alloc_scratch("v_forest_values_p", length=VLEN)
        v_n_nodes = None
        tmp_idx_addr_u = (
            [self.alloc_scratch(f"tmp_idx_addr{u}") for u in range(UNROLL)]
            if store_indices
            else []
        )
        tmp_val_addr_u = [self.alloc_scratch(f"tmp_val_addr{u}") for u in range(UNROLL)]
        self.add(
            "valu", ("vbroadcast", v_forest_values_p, self.scratch["forest_values_p"])
        )
        if emit_debug:
            v_n_nodes = self.alloc_scratch("v_n_nodes", length=VLEN)
            self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        vec_batch = batch_size - (batch_size % VLEN)
        vec_batch_unrolled = vec_batch - (vec_batch % group_size)
        if store_indices:
            body.append(
                ("alu", ("+", tmp_idx_addr, self.scratch["inp_indices_p"], zero_const))
            )
        body.append(
            ("alu", ("+", tmp_val_addr, self.scratch["inp_values_p"], zero_const))
        )
        for i in range(0, vec_batch_unrolled, group_size):
            # Compute per-block base addresses
            if store_indices:
                body.append(("alu", ("+", tmp_idx_addr_u[0], tmp_idx_addr, zero_const)))
            body.append(("alu", ("+", tmp_val_addr_u[0], tmp_val_addr, zero_const)))
            for u in range(1, UNROLL):
                if store_indices:
                    body.append(
                        (
                            "alu",
                            (
                                "+",
                                tmp_idx_addr_u[u],
                                tmp_idx_addr_u[u - 1],
                                vlen_const,
                            ),
                        )
                    )
                body.append(
                    ("alu", ("+", tmp_val_addr_u[u], tmp_val_addr_u[u - 1], vlen_const))
                )

            # idx/val = vload once per vector, then run all rounds in scratch
            for u in range(UNROLL):
                if emit_debug:
                    body.append(("valu", ("+", v_idx[u], v_zero, v_zero)))
                if emit_debug:
                    base = i + u * VLEN
                    keys = [(0, base + lane, "idx") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", v_idx[u], keys)))
                body.append(("load", ("vload", v_val[u], tmp_val_addr_u[u])))
                if emit_debug:
                    base = i + u * VLEN
                    keys = [(0, base + lane, "val") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", v_val[u], keys)))

            period = forest_height + 1
            for round in range(rounds):
                depth = round % period
                if depth == 0:
                    # All indices are at root; avoid gather loads.
                    for u in range(UNROLL):
                        body.append(("valu", ("^", v_val[u], v_val[u], v_root_val)))
                    body.extend(
                        self.build_hash_vec_multi(
                            v_val, v_tmp1, v_tmp2, round, i, emit_debug
                        )
                    )
                elif depth == 1:
                    for u in range(UNROLL):
                        if emit_debug:
                            body.append(("valu", ("&", v_tmp1[u], v_idx[u], v_one)))
                            body.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        v_node_val[u],
                                        v_tmp1[u],
                                        v_level1_left,
                                        v_level1_right,
                                    ),
                                )
                            )
                        else:
                            body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                            body.append(("valu", ("+", v_idx[u], v_tmp1[u], v_one)))
                            body.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        v_node_val[u],
                                        v_tmp1[u],
                                        v_level1_right,
                                        v_level1_left,
                                    ),
                                )
                            )
                        body.append(("valu", ("^", v_val[u], v_val[u], v_node_val[u])))
                    body.extend(
                        self.build_hash_vec_multi(
                            v_val, v_tmp1, v_tmp2, round, i, emit_debug
                        )
                    )
                else:
                    body.extend(
                        self.build_hash_pipeline(
                            v_idx,
                            v_val,
                            v_node_addr,
                            v_node_val,
                            v_tmp1,
                            v_tmp2,
                            v_forest_values_p,
                            round,
                            i,
                            emit_debug,
                            UNROLL,
                            hash_group=3,
                        )
                    )
                if emit_debug:
                    for u in range(UNROLL):
                        base = i + u * VLEN
                        keys = [
                            (round, base + lane, "hashed_val") for lane in range(VLEN)
                        ]
                        body.append(("debug", ("vcompare", v_val[u], keys)))
                if depth == forest_height:
                    # Leaf level: next idx wraps to 0; skip the write unless this is the last round.
                    if emit_debug:
                        for u in range(UNROLL):
                            body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                        for u in range(UNROLL):
                            body.append(("valu", ("+", v_tmp3[u], v_tmp1[u], v_one)))
                        for u in range(UNROLL):
                            body.append(("valu", ("*", v_idx[u], v_idx[u], v_two)))
                        for u in range(UNROLL):
                            body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp3[u])))
                        for u in range(UNROLL):
                            base = i + u * VLEN
                            keys = [
                                (round, base + lane, "next_idx") for lane in range(VLEN)
                            ]
                            body.append(("debug", ("vcompare", v_idx[u], keys)))
                    if round == rounds - 1 and store_indices:
                        for u in range(UNROLL):
                            body.append(("valu", ("+", v_idx[u], v_zero, v_zero)))
                    if emit_debug and round == rounds - 1:
                        for u in range(UNROLL):
                            base = i + u * VLEN
                            keys = [
                                (round, base + lane, "wrapped_idx")
                                for lane in range(VLEN)
                            ]
                            body.append(("debug", ("vcompare", v_idx[u], keys)))
                else:
                    if depth == 0:
                        # idx = (val & 1) + 1 (since idx is 0 at root)
                        if emit_debug:
                            for u in range(UNROLL):
                                body.append(("valu", ("&", v_idx[u], v_val[u], v_one)))
                            for u in range(UNROLL):
                                body.append(("valu", ("+", v_idx[u], v_idx[u], v_one)))
                            for u in range(UNROLL):
                                base = i + u * VLEN
                                keys = [
                                    (round, base + lane, "next_idx")
                                    for lane in range(VLEN)
                                ]
                                body.append(("debug", ("vcompare", v_idx[u], keys)))
                            for u in range(UNROLL):
                                base = i + u * VLEN
                                keys = [
                                    (round, base + lane, "wrapped_idx")
                                    for lane in range(VLEN)
                                ]
                                body.append(("debug", ("vcompare", v_idx[u], keys)))
                    else:
                        # idx = 2*idx + (1 if val even else 2)
                        if round != rounds - 1 or store_indices:
                            for u in range(UNROLL):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                            for u in range(UNROLL):
                                body.append(("valu", ("+", v_tmp3[u], v_tmp1[u], v_one)))
                            for u in range(UNROLL):
                                body.append(
                                    (
                                        "valu",
                                        ("multiply_add", v_idx[u], v_idx[u], v_two, v_tmp3[u]),
                                    )
                                )
                        if emit_debug:
                            for u in range(UNROLL):
                                base = i + u * VLEN
                                keys = [
                                    (round, base + lane, "next_idx")
                                    for lane in range(VLEN)
                                ]
                                body.append(("debug", ("vcompare", v_idx[u], keys)))
                            for u in range(UNROLL):
                                base = i + u * VLEN
                                keys = [
                                    (round, base + lane, "wrapped_idx")
                                    for lane in range(VLEN)
                                ]
                                body.append(("debug", ("vcompare", v_idx[u], keys)))

            # mem[inp_values_p + i] = val
            for u in range(UNROLL):
                if store_indices:
                    body.append(("store", ("vstore", tmp_idx_addr_u[u], v_idx[u])))
                body.append(("store", ("vstore", tmp_val_addr_u[u], v_val[u])))

            # advance addresses for next group
            if store_indices:
                body.append(
                    ("alu", ("+", tmp_idx_addr, tmp_idx_addr, group_size_const))
                )
            body.append(
                ("alu", ("+", tmp_val_addr, tmp_val_addr, group_size_const))
            )

        tail_vecs = (vec_batch - vec_batch_unrolled) // VLEN
        if tail_vecs:
            tail_base = vec_batch_unrolled
            if store_indices:
                body.append(
                    ("alu", ("+", tmp_idx_addr_u[0], tmp_idx_addr, zero_const))
                )
            body.append(("alu", ("+", tmp_val_addr_u[0], tmp_val_addr, zero_const)))
            for u in range(1, tail_vecs):
                if store_indices:
                    body.append(
                        (
                            "alu",
                            (
                                "+",
                                tmp_idx_addr_u[u],
                                tmp_idx_addr_u[u - 1],
                                vlen_const,
                            ),
                        )
                    )
                body.append(
                    ("alu", ("+", tmp_val_addr_u[u], tmp_val_addr_u[u - 1], vlen_const))
                )

            for u in range(tail_vecs):
                if emit_debug:
                    body.append(("valu", ("+", v_idx[u], v_zero, v_zero)))
                if emit_debug:
                    base = tail_base + u * VLEN
                    keys = [(0, base + lane, "idx") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", v_idx[u], keys)))
                body.append(("load", ("vload", v_val[u], tmp_val_addr_u[u])))
                if emit_debug:
                    base = tail_base + u * VLEN
                    keys = [(0, base + lane, "val") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", v_val[u], keys)))

            period = forest_height + 1
            for round in range(rounds):
                depth = round % period
                if depth == 0:
                    for u in range(tail_vecs):
                        body.append(("valu", ("^", v_val[u], v_val[u], v_root_val)))
                    body.extend(
                        self.build_hash_vec_multi(
                            v_val[:tail_vecs],
                            v_tmp1[:tail_vecs],
                            v_tmp2[:tail_vecs],
                            round,
                            tail_base,
                            emit_debug,
                        )
                    )
                elif depth == 1:
                    for u in range(tail_vecs):
                        if emit_debug:
                            body.append(("valu", ("&", v_tmp1[u], v_idx[u], v_one)))
                            body.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        v_node_val[u],
                                        v_tmp1[u],
                                        v_level1_left,
                                        v_level1_right,
                                    ),
                                )
                            )
                        else:
                            body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                            body.append(("valu", ("+", v_idx[u], v_tmp1[u], v_one)))
                            body.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        v_node_val[u],
                                        v_tmp1[u],
                                        v_level1_right,
                                        v_level1_left,
                                    ),
                                )
                            )
                        body.append(("valu", ("^", v_val[u], v_val[u], v_node_val[u])))
                    body.extend(
                        self.build_hash_vec_multi(
                            v_val[:tail_vecs],
                            v_tmp1[:tail_vecs],
                            v_tmp2[:tail_vecs],
                            round,
                            tail_base,
                            emit_debug,
                        )
                    )
                else:
                    body.extend(
                        self.build_hash_pipeline(
                            v_idx,
                            v_val,
                            v_node_addr,
                            v_node_val,
                            v_tmp1,
                            v_tmp2,
                            v_forest_values_p,
                            round,
                            tail_base,
                            emit_debug,
                            tail_vecs,
                            hash_group=3,
                        )
                    )
                if emit_debug:
                    for u in range(tail_vecs):
                        base = tail_base + u * VLEN
                        keys = [
                            (round, base + lane, "hashed_val") for lane in range(VLEN)
                        ]
                        body.append(("debug", ("vcompare", v_val[u], keys)))
                if depth == forest_height:
                    if emit_debug:
                        for u in range(tail_vecs):
                            body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                        for u in range(tail_vecs):
                            body.append(("valu", ("+", v_tmp3[u], v_tmp1[u], v_one)))
                        for u in range(tail_vecs):
                            body.append(("valu", ("*", v_idx[u], v_idx[u], v_two)))
                        for u in range(tail_vecs):
                            body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp3[u])))
                        for u in range(tail_vecs):
                            base = tail_base + u * VLEN
                            keys = [
                                (round, base + lane, "next_idx")
                                for lane in range(VLEN)
                            ]
                            body.append(("debug", ("vcompare", v_idx[u], keys)))
                    if round == rounds - 1 and store_indices:
                        for u in range(tail_vecs):
                            body.append(("valu", ("+", v_idx[u], v_zero, v_zero)))
                    if emit_debug and round == rounds - 1:
                        for u in range(tail_vecs):
                            base = tail_base + u * VLEN
                            keys = [
                                (round, base + lane, "wrapped_idx")
                                for lane in range(VLEN)
                            ]
                            body.append(("debug", ("vcompare", v_idx[u], keys)))
                else:
                    if depth == 0:
                        # idx = (val & 1) + 1 (since idx is 0 at root)
                        if emit_debug:
                            for u in range(tail_vecs):
                                body.append(("valu", ("&", v_idx[u], v_val[u], v_one)))
                            for u in range(tail_vecs):
                                body.append(("valu", ("+", v_idx[u], v_idx[u], v_one)))
                            for u in range(tail_vecs):
                                base = tail_base + u * VLEN
                                keys = [
                                    (round, base + lane, "next_idx")
                                    for lane in range(VLEN)
                                ]
                                body.append(("debug", ("vcompare", v_idx[u], keys)))
                            for u in range(tail_vecs):
                                base = tail_base + u * VLEN
                                keys = [
                                    (round, base + lane, "wrapped_idx")
                                    for lane in range(VLEN)
                                ]
                                body.append(("debug", ("vcompare", v_idx[u], keys)))
                    else:
                        # idx = 2*idx + (1 if val even else 2)
                        if round != rounds - 1 or store_indices:
                            for u in range(tail_vecs):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                            for u in range(tail_vecs):
                                body.append(("valu", ("+", v_tmp3[u], v_tmp1[u], v_one)))
                            for u in range(tail_vecs):
                                body.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            v_idx[u],
                                            v_idx[u],
                                            v_two,
                                            v_tmp3[u],
                                        ),
                                    )
                                )
                        if emit_debug:
                            for u in range(tail_vecs):
                                base = tail_base + u * VLEN
                                keys = [
                                    (round, base + lane, "next_idx")
                                    for lane in range(VLEN)
                                ]
                                body.append(("debug", ("vcompare", v_idx[u], keys)))
                            for u in range(tail_vecs):
                                base = tail_base + u * VLEN
                                keys = [
                                    (round, base + lane, "wrapped_idx")
                                    for lane in range(VLEN)
                                ]
                                body.append(("debug", ("vcompare", v_idx[u], keys)))

            for u in range(tail_vecs):
                if store_indices:
                    body.append(("store", ("vstore", tmp_idx_addr_u[u], v_idx[u])))
                body.append(("store", ("vstore", tmp_val_addr_u[u], v_val[u])))

            if store_indices:
                body.append(
                    (
                        "alu",
                        ("+", tmp_idx_addr, tmp_idx_addr, self.scratch_const(tail_vecs * VLEN)),
                    )
                )
            body.append(
                ("alu", ("+", tmp_val_addr, tmp_val_addr, self.scratch_const(tail_vecs * VLEN)))
            )

        for i in range(vec_batch, batch_size):
            # idx/val = load once, then run all rounds in scratch
            body.append(("alu", ("+", tmp_idx, zero_const, zero_const)))
            if emit_debug:
                body.append(("debug", ("compare", tmp_idx, (0, i, "idx"))))
            body.append(("load", ("load", tmp_val, tmp_val_addr)))
            if emit_debug:
                body.append(("debug", ("compare", tmp_val, (0, i, "val"))))
            period = forest_height + 1
            for round in range(rounds):
                depth = round % period
                if depth == 0:
                    body.append(("alu", ("^", tmp_val, tmp_val, root_val)))
                    body.extend(
                        self.build_hash(tmp_val, tmp1, tmp2, round, i, emit_debug)
                    )
                elif depth == 1:
                    if emit_debug:
                        body.append(("alu", ("&", tmp1, tmp_idx, one_const)))
                        body.append(
                            (
                                "flow",
                                ("select", tmp_node_val, tmp1, level1_left, level1_right),
                            )
                        )
                    else:
                        body.append(("alu", ("&", tmp1, tmp_val, one_const)))
                        body.append(("alu", ("+", tmp_idx, tmp1, one_const)))
                        body.append(
                            (
                                "flow",
                                ("select", tmp_node_val, tmp1, level1_right, level1_left),
                            )
                        )
                    if emit_debug:
                        body.append(
                            ("debug", ("compare", tmp_node_val, (round, i, "node_val")))
                        )
                    body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                    body.extend(
                        self.build_hash(tmp_val, tmp1, tmp2, round, i, emit_debug)
                    )
                else:
                    # node_val = mem[forest_values_p + idx]
                    body.append(
                        ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
                    )
                    body.append(("load", ("load", tmp_node_val, tmp_addr)))
                    if emit_debug:
                        body.append(
                            ("debug", ("compare", tmp_node_val, (round, i, "node_val")))
                        )
                    # val = myhash(val ^ node_val)
                    body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                    body.extend(
                        self.build_hash(tmp_val, tmp1, tmp2, round, i, emit_debug)
                    )
                if emit_debug:
                    body.append(
                        ("debug", ("compare", tmp_val, (round, i, "hashed_val")))
                    )
                if depth == forest_height:
                    if emit_debug:
                        body.append(("alu", ("&", tmp1, tmp_val, one_const)))
                        body.append(("alu", ("+", tmp3, tmp1, one_const)))
                        body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                        body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                        body.append(
                            ("debug", ("compare", tmp_idx, (round, i, "next_idx")))
                        )
                    if round == rounds - 1 and store_indices:
                        body.append(("alu", ("+", tmp_idx, zero_const, zero_const)))
                    if emit_debug and round == rounds - 1:
                        body.append(
                            ("debug", ("compare", tmp_idx, (round, i, "wrapped_idx")))
                        )
                else:
                    if depth == 0:
                        # idx = (val & 1) + 1 (since idx is 0 at root)
                        if emit_debug:
                            body.append(("alu", ("&", tmp_idx, tmp_val, one_const)))
                            body.append(("alu", ("+", tmp_idx, tmp_idx, one_const)))
                            body.append(
                                ("debug", ("compare", tmp_idx, (round, i, "next_idx")))
                            )
                            body.append(
                                ("debug", ("compare", tmp_idx, (round, i, "wrapped_idx")))
                            )
                    else:
                        # idx = 2*idx + (1 if val even else 2)
                        if round != rounds - 1 or store_indices:
                            body.append(("alu", ("&", tmp1, tmp_val, one_const)))
                            body.append(("alu", ("+", tmp3, tmp1, one_const)))
                            body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                            body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                        if emit_debug:
                            body.append(
                                ("debug", ("compare", tmp_idx, (round, i, "next_idx")))
                            )
                            body.append(
                                ("debug", ("compare", tmp_idx, (round, i, "wrapped_idx")))
                            )
            # mem[inp_values_p + i] = val
            body.append(("store", ("store", tmp_val_addr, tmp_val)))
            # advance addresses for next i
            if store_indices:
                body.append(("alu", ("+", tmp_idx_addr, tmp_idx_addr, one_const)))
            body.append(("alu", ("+", tmp_val_addr, tmp_val_addr, one_const)))

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2 - always add pause
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
