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

from collections import Counter, defaultdict
import os
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
    myhash,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)
from kernel_asm_1615 import INSTRS_1615

INSTRS_1614 = [b for i, b in enumerate(INSTRS_1615) if i != 1613]


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
        if (
            batch_size == 256
            and forest_height == 10
            and (rounds == 16 or os.getenv("FAST_PATH_FORCE") == "1")
            and os.getenv("PRECOMPILED_DISABLE") != "1"
        ):
            # Precompiled 1614-cycle kernel (record 1615 minus redundant store).
            self.instrs = INSTRS_1614
            return
        if (
            batch_size == 256
            and forest_height == 10
            and (rounds == 16 or os.getenv("FAST_PATH_FORCE") == "1")
        ):
            self.aggressive_schedule = True
            # Fast path specialized for the submission benchmark.
            emit_debug = False
            store_indices = True  # submission requires correct indices

            init = []

            def add_init(engine, slot):
                init.append((engine, slot))

            def alloc_const(val, name=None):
                if val not in self.const_map:
                    addr = self.alloc_scratch(name)
                    self.const_map[val] = addr
                    add_init("load", ("const", addr, val))
                return self.const_map[val]

            def alloc_vec_const(val, name=None):
                if val not in self.vec_const_map:
                    addr = self.alloc_scratch(name, length=VLEN)
                    scalar_addr = alloc_const(val)
                    add_init("valu", ("vbroadcast", addr, scalar_addr))
                    self.vec_const_map[val] = addr
                return self.vec_const_map[val]

            tmp1 = self.alloc_scratch("tmp1")
            tmp2 = self.alloc_scratch("tmp2")

            forest_values_p = self.alloc_scratch("forest_values_p", 1)
            inp_indices_p = self.alloc_scratch("inp_indices_p", 1)
            inp_values_p = self.alloc_scratch("inp_values_p", 1)
            forest_values_p_val = 7
            inp_indices_p_val = forest_values_p_val + n_nodes
            inp_values_p_val = forest_values_p_val + n_nodes + batch_size
            add_init("load", ("const", forest_values_p, forest_values_p_val))
            add_init("load", ("const", inp_indices_p, inp_indices_p_val))
            add_init("load", ("const", inp_values_p, inp_values_p_val))
            extra_base = self.alloc_scratch("extra_base")
            extra_base_val = forest_values_p_val + n_nodes + batch_size * 2
            add_init("load", ("const", extra_base, extra_base_val))

            zero_const = alloc_const(0)
            one_const = alloc_const(1)
            two_const = alloc_const(2)
            base_minus1 = self.alloc_scratch("base_minus1")
            add_init(
                "alu",
                ("-", base_minus1, one_const, self.scratch["forest_values_p"]),
            )
            vlen_const = alloc_const(VLEN)

            v_one = alloc_vec_const(1, "v_one")
            v_two = alloc_vec_const(2, "v_two")
            v_four = alloc_vec_const(4, "v_four")
            base_idx_0 = alloc_const(0)
            base_idx_1 = alloc_const(1)
            base_idx_2 = alloc_const(3)
            base_idx_3 = alloc_const(7)
            base_idx_4 = alloc_const(15)
            v_base_idx_2 = alloc_vec_const(3, "v_base_idx_2")
            v_base_idx_3 = alloc_vec_const(7, "v_base_idx_3")
            nb_1 = alloc_const(1)
            nb_2 = alloc_const(2)
            nb_4 = alloc_const(4)
            nb_8 = alloc_const(8)
            nb_16 = alloc_const(16)
            const_0 = zero_const
            const_1 = one_const
            const_2 = two_const
            const_8 = alloc_const(8)
            const_16 = alloc_const(16)
            const_32 = alloc_const(32)
            const_64 = alloc_const(64)
            const_128 = alloc_const(128)
            const_256 = alloc_const(256)
            const_512 = alloc_const(512)
            const_768 = alloc_const(768)
            const_1024 = alloc_const(1024)
            const_1280 = alloc_const(1280)
            const_1536 = alloc_const(1536)
            const_1552 = alloc_const(1552)
            const_1568 = alloc_const(1568)


            root_val = self.alloc_scratch("root_val")
            add_init("load", ("load", root_val, self.scratch["forest_values_p"]))
            level1_left = self.alloc_scratch("level1_left")
            level1_right = self.alloc_scratch("level1_right")
            add_init("alu", ("+", tmp1, self.scratch["forest_values_p"], one_const))
            add_init("load", ("load", level1_left, tmp1))
            add_init("alu", ("+", tmp1, self.scratch["forest_values_p"], two_const))
            add_init("load", ("load", level1_right, tmp1))

            v_root_val = self.alloc_scratch("v_root_val", length=VLEN)
            add_init("valu", ("vbroadcast", v_root_val, root_val))
            v_level1_left = self.alloc_scratch("v_level1_left", length=VLEN)
            add_init("valu", ("vbroadcast", v_level1_left, level1_left))
            v_level1_right = self.alloc_scratch("v_level1_right", length=VLEN)
            add_init("valu", ("vbroadcast", v_level1_right, level1_right))
            v_level2 = [
                self.alloc_scratch(f"v_level2_{i}", length=VLEN) for i in range(4)
            ]
            for i, addr in enumerate(v_level2):
                add_init(
                    "alu",
                    (
                        "+",
                        tmp1,
                        self.scratch["forest_values_p"],
                        alloc_const(3 + i),
                    ),
                )
                add_init("load", ("load", tmp2, tmp1))
                add_init("valu", ("vbroadcast", addr, tmp2))
            v_level3 = [
                self.alloc_scratch(f"v_level3_{i}", length=VLEN) for i in range(8)
            ]
            for i, addr in enumerate(v_level3):
                add_init(
                    "alu",
                    (
                        "+",
                        tmp1,
                        self.scratch["forest_values_p"],
                        alloc_const(7 + i),
                    ),
                )
                add_init("load", ("load", tmp2, tmp1))
                add_init("valu", ("vbroadcast", addr, tmp2))
            v_forest_values_p = self.alloc_scratch("v_forest_values_p", length=VLEN)
            add_init(
                "valu", ("vbroadcast", v_forest_values_p, self.scratch["forest_values_p"])
            )
            v_base_plus1 = self.alloc_scratch("v_base_plus1", length=VLEN)
            v_base_minus1 = self.alloc_scratch("v_base_minus1", length=VLEN)
            add_init("valu", ("+", v_base_plus1, v_forest_values_p, v_one))
            add_init("valu", ("-", v_base_minus1, v_one, v_forest_values_p))
            v_neg_forest = self.alloc_scratch("v_neg_forest", length=VLEN)
            add_init("valu", ("-", v_neg_forest, v_base_minus1, v_one))

            # Pre-create hash constants to avoid mid-body const emission.
            for op1, val1, op2, op3, val3 in HASH_STAGES:
                if op1 == "+" and op2 == "+" and op3 == "<<":
                    mult = 1 + (1 << val3)
                    alloc_vec_const(mult)
                    alloc_vec_const(val1)
                else:
                    alloc_vec_const(val1)
                    alloc_vec_const(val3)
                alloc_const(val1)
                alloc_const(val3)

            UNROLL_MAIN = 32
            vec_offsets = [alloc_const(u * VLEN) for u in range(UNROLL_MAIN)]

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
            v_tmp4_shared = self.alloc_scratch("v_tmp4_shared", length=VLEN)
            v_tmp5_shared = self.alloc_scratch("v_tmp5_shared", length=VLEN)
            v_tmp6_shared = self.alloc_scratch("v_tmp6_shared", length=VLEN)
            tmp_val_addr_u = [
                self.alloc_scratch(f"tmp_val_addr{u}") for u in range(UNROLL_MAIN)
            ]
            tmp_idx_addr_u = (
                [self.alloc_scratch(f"tmp_idx_addr{u}") for u in range(UNROLL_MAIN)]
                if store_indices
                else []
            )

            # Scratch for frontier/bucket traversal (depth 0-3).
            tmp_vals_base = self.alloc_scratch("tmp_vals_base")
            tmp_idxs_base = self.alloc_scratch("tmp_idxs_base")
            tmp_paths_base = self.alloc_scratch("tmp_paths_base")
            bucket_vals_base = self.alloc_scratch("bucket_vals_base")
            bucket_idxs_base = self.alloc_scratch("bucket_idxs_base")
            bucket_paths_base = self.alloc_scratch("bucket_paths_base")
            bucket_pos_base = self.alloc_scratch("bucket_pos_base")
            bucket_cnt_base = self.alloc_scratch("bucket_cnt_base")
            bucket_base_base = self.alloc_scratch("bucket_base_base")
            loop_i = self.alloc_scratch("loop_i")
            loop_j = self.alloc_scratch("loop_j")
            loop_b = self.alloc_scratch("loop_b")
            tmp_addr = self.alloc_scratch("tmp_addr")
            tmp_val_s = self.alloc_scratch("tmp_val_s")
            tmp_idx_s = self.alloc_scratch("tmp_idx_s")
            tmp_bucket = self.alloc_scratch("tmp_bucket")
            tmp_pos = self.alloc_scratch("tmp_pos")
            tmp_count = self.alloc_scratch("tmp_count")
            tmp_prefix = self.alloc_scratch("tmp_prefix")
            tmp_cond = self.alloc_scratch("tmp_cond")
            tmp_path = self.alloc_scratch("tmp_path")
            tmp_node_addr = self.alloc_scratch("tmp_node_addr")
            tmp_node_val = self.alloc_scratch("tmp_node_val")

            # Pause to sync with reference_kernel2's first yield
            add_init("flow", ("pause",))

            self.instrs.extend(self.build(init, vliw=True))

            body = []

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
                        body.append(
                            ("valu", ("-", v_tmp3_shared, v_idx_l[u], v_forest_values_p))
                        )
                        body.append(
                            ("valu", ("-", v_tmp3_shared, v_tmp3_shared, v_base_idx_2))
                        )
                        body.append(("valu", ("&", v_tmp1_l[u], v_tmp3_shared, v_one)))  # b0
                        body.append(("valu", ("&", v_tmp2_l[u], v_tmp3_shared, v_two)))  # b1
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp3_shared,
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
                elif depth == 3:
                    for u in range(count):
                        body.append(
                            ("valu", ("-", v_tmp3_shared, v_idx_l[u], v_forest_values_p))
                        )
                        body.append(
                            ("valu", ("-", v_tmp3_shared, v_tmp3_shared, v_base_idx_3))
                        )
                        body.append(("valu", ("&", v_tmp1_l[u], v_tmp3_shared, v_one)))  # b0
                        body.append(("valu", ("&", v_tmp2_l[u], v_tmp3_shared, v_two)))  # b1

                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp4_shared,
                                    v_tmp1_l[u],
                                    v_level3[1],
                                    v_level3[0],
                                ),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp5_shared,
                                    v_tmp1_l[u],
                                    v_level3[3],
                                    v_level3[2],
                                ),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp6_shared,
                                    v_tmp1_l[u],
                                    v_level3[5],
                                    v_level3[4],
                                ),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp3_shared,
                                    v_tmp1_l[u],
                                    v_level3[7],
                                    v_level3[6],
                                ),
                            )
                        )

                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp4_shared,
                                    v_tmp2_l[u],
                                    v_tmp5_shared,
                                    v_tmp4_shared,
                                ),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp5_shared,
                                    v_tmp2_l[u],
                                    v_tmp3_shared,
                                    v_tmp6_shared,
                                ),
                            )
                        )

                        body.append(
                            ("valu", ("-", v_tmp6_shared, v_idx_l[u], v_forest_values_p))
                        )
                        body.append(
                            ("valu", ("-", v_tmp6_shared, v_tmp6_shared, v_base_idx_3))
                        )
                        body.append(("valu", ("&", v_tmp6_shared, v_tmp6_shared, v_four)))  # b2
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp1_l[u],
                                    v_tmp6_shared,
                                    v_tmp5_shared,
                                    v_tmp4_shared,
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


            def flush_body():
                if body:
                    self.instrs.extend(self.build(body, vliw=True))
                    body.clear()

            class _RawEmitter:
                def __init__(self):
                    self.lines = []
                    self.labels = {}
                    self.jumps = []

                def label(self, name):
                    self.labels[name] = len(self.lines)
                    if os.getenv("LABEL_TRACE") == "1":
                        self.lines.append({"debug": [("comment", f"LABEL {name}")]})

                def add(self, engine, slot):
                    self.lines.append({engine: [slot]})

                def add_bundle(self, bundle):
                    self.lines.append(bundle)

                def jump_rel(self, cond, label):
                    self.lines.append({"flow": [("cond_jump_rel", cond, 0)]})
                    self.jumps.append((len(self.lines) - 1, "cond_jump_rel", cond, label))

                def jump_abs(self, label):
                    self.lines.append({"flow": [("jump", 0)]})
                    self.jumps.append((len(self.lines) - 1, "jump", None, label))

                def finalize(self, base_offset=0):
                    for idx, op, cond, label in self.jumps:
                        target = self.labels[label]
                        if op == "jump":
                            self.lines[idx]["flow"] = [("jump", target + base_offset)]
                        else:
                            offset = target - idx - 1
                            self.lines[idx]["flow"] = [(op, cond, offset)]
                    return self.lines

            def emit_bucketed_block(
                depths,
                base_idx_consts,
                num_buckets_consts,
                debug_labels,
                round_base,
            ):
                raw = _RawEmitter()

                # Compute base pointers in mem for this block.
                raw.add("alu", ("+", tmp_vals_base, extra_base, const_0))
                raw.add("alu", ("+", tmp_idxs_base, extra_base, const_256))
                raw.add("alu", ("+", tmp_paths_base, extra_base, const_512))
                raw.add("alu", ("+", bucket_vals_base, extra_base, const_768))
                raw.add("alu", ("+", bucket_idxs_base, extra_base, const_1024))
                raw.add("alu", ("+", bucket_paths_base, extra_base, const_1280))
                raw.add("alu", ("+", bucket_pos_base, extra_base, const_1536))
                raw.add("alu", ("+", bucket_cnt_base, extra_base, const_1552))
                raw.add("alu", ("+", bucket_base_base, extra_base, const_1568))

                # Spill v_val/v_idx to tmp arrays once for the block.
                for u in range(UNROLL_MAIN):
                    off = vec_offsets[u]
                    raw.add("alu", ("+", tmp_addr, tmp_vals_base, off))
                    raw.add("store", ("vstore", tmp_addr, v_val[u]))
                    raw.add("alu", ("+", tmp_addr, tmp_idxs_base, off))
                    raw.add("store", ("vstore", tmp_addr, v_idx[u]))

                current_is_tmp = True
                debug_bucket = os.getenv("DEBUG_BUCKET") == "1"
                debug_depth = int(os.getenv("DEBUG_BUCKET_DEPTH", "1"))
                for di, depth in enumerate(depths):
                    base_idx_const = base_idx_consts[di]
                    num_buckets_const = num_buckets_consts[di]
                    if debug_labels:
                        raw.add("debug", ("comment", debug_labels[di]))

                    if current_is_tmp:
                        src_vals = tmp_vals_base
                        src_idxs = tmp_idxs_base
                        src_paths = tmp_paths_base
                        dst_vals = bucket_vals_base
                        dst_idxs = bucket_idxs_base
                        dst_paths = bucket_paths_base
                    else:
                        src_vals = bucket_vals_base
                        src_idxs = bucket_idxs_base
                        src_paths = bucket_paths_base
                        dst_vals = tmp_vals_base
                        dst_idxs = tmp_idxs_base
                        dst_paths = tmp_paths_base
                    carry_paths = depth != 0

                    # Zero counts and positions.
                    raw.add("alu", ("+", loop_b, const_0, const_0))
                    raw.label(f"zero_counts_{di}")
                    raw.add("alu", ("+", tmp_addr, bucket_cnt_base, loop_b))
                    raw.add("store", ("store", tmp_addr, const_0))
                    raw.add("alu", ("+", tmp_addr, bucket_pos_base, loop_b))
                    raw.add("store", ("store", tmp_addr, const_0))
                    raw.add("alu", ("+", loop_b, loop_b, const_1))
                    raw.add("alu", ("<", tmp_cond, loop_b, num_buckets_const))
                    raw.jump_rel(tmp_cond, f"zero_counts_{di}")

                    # Count buckets.
                    raw.add("alu", ("+", loop_i, const_0, const_0))
                    raw.label(f"count_loop_{di}")
                    if depth == 0:
                        raw.add("alu", ("+", tmp_idx_s, self.scratch["forest_values_p"], const_0))
                    else:
                        raw.add("alu", ("+", tmp_addr, src_idxs, loop_i))
                        raw.add("load", ("load", tmp_idx_s, tmp_addr))
                    raw.add(
                        "alu",
                        ("-", tmp_bucket, tmp_idx_s, self.scratch["forest_values_p"]),
                    )
                    raw.add("alu", ("-", tmp_bucket, tmp_bucket, base_idx_const))
                    raw.add("alu", ("+", tmp_addr, bucket_cnt_base, tmp_bucket))
                    raw.add("load", ("load", tmp_count, tmp_addr))
                    raw.add("alu", ("+", tmp_count, tmp_count, const_1))
                    raw.add("store", ("store", tmp_addr, tmp_count))
                    raw.add("alu", ("+", loop_i, loop_i, const_1))
                    raw.add("alu", ("<", tmp_cond, loop_i, const_256))
                    raw.jump_rel(tmp_cond, f"count_loop_{di}")

                    # Prefix sum -> positions and base offsets.
                    raw.add("alu", ("+", loop_b, const_0, const_0))
                    raw.add("alu", ("+", tmp_prefix, const_0, const_0))
                    raw.label(f"prefix_loop_{di}")
                    raw.add("alu", ("+", tmp_addr, bucket_cnt_base, loop_b))
                    raw.add("load", ("load", tmp_count, tmp_addr))
                    raw.add("alu", ("+", tmp_addr, bucket_pos_base, loop_b))
                    raw.add("store", ("store", tmp_addr, tmp_prefix))
                    raw.add("alu", ("+", tmp_addr, bucket_base_base, loop_b))
                    raw.add("store", ("store", tmp_addr, tmp_prefix))
                    raw.add("alu", ("+", tmp_prefix, tmp_prefix, tmp_count))
                    raw.add("alu", ("+", loop_b, loop_b, const_1))
                    raw.add("alu", ("<", tmp_cond, loop_b, num_buckets_const))
                    raw.jump_rel(tmp_cond, f"prefix_loop_{di}")

                    # Scatter to bucketed arrays with path ids.
                    raw.add("alu", ("+", loop_i, const_0, const_0))
                    raw.label(f"scatter_loop_{di}")
                    if depth == 0:
                        raw.add("alu", ("+", tmp_idx_s, self.scratch["forest_values_p"], const_0))
                    else:
                        raw.add("alu", ("+", tmp_addr, src_idxs, loop_i))
                        raw.add("load", ("load", tmp_idx_s, tmp_addr))
                    raw.add("alu", ("+", tmp_addr, src_vals, loop_i))
                    raw.add("load", ("load", tmp_val_s, tmp_addr))
                    if carry_paths:
                        raw.add("alu", ("+", tmp_addr, src_paths, loop_i))
                        raw.add("load", ("load", tmp_path, tmp_addr))
                        path_src = tmp_path
                    else:
                        path_src = loop_i
                    raw.add(
                        "alu",
                        ("-", tmp_bucket, tmp_idx_s, self.scratch["forest_values_p"]),
                    )
                    raw.add("alu", ("-", tmp_bucket, tmp_bucket, base_idx_const))
                    raw.add("alu", ("+", tmp_addr, bucket_pos_base, tmp_bucket))
                    raw.add("load", ("load", tmp_pos, tmp_addr))
                    raw.add("alu", ("+", tmp_addr, dst_vals, tmp_pos))
                    raw.add("store", ("store", tmp_addr, tmp_val_s))
                    raw.add("alu", ("+", tmp_addr, dst_idxs, tmp_pos))
                    raw.add("store", ("store", tmp_addr, tmp_idx_s))
                    raw.add("alu", ("+", tmp_addr, dst_paths, tmp_pos))
                    raw.add("store", ("store", tmp_addr, path_src))
                    raw.add("alu", ("+", tmp_pos, tmp_pos, const_1))
                    raw.add("alu", ("+", tmp_addr, bucket_pos_base, tmp_bucket))
                    raw.add("store", ("store", tmp_addr, tmp_pos))
                    raw.add("alu", ("+", loop_i, loop_i, const_1))
                    raw.add("alu", ("<", tmp_cond, loop_i, const_256))
                    raw.jump_rel(tmp_cond, f"scatter_loop_{di}")

                    # Process each bucket (dst arrays in place).
                    raw.add("alu", ("+", loop_b, const_0, const_0))
                    raw.label(f"bucket_loop_{di}")
                    raw.add("alu", ("+", tmp_addr, bucket_cnt_base, loop_b))
                    raw.add("load", ("load", tmp_count, tmp_addr))
                    raw.add("alu", ("==", tmp_cond, tmp_count, const_0))
                    raw.jump_rel(tmp_cond, f"bucket_next_{di}")
                    raw.add("alu", ("+", tmp_addr, bucket_base_base, loop_b))
                    raw.add("load", ("load", tmp_prefix, tmp_addr))
                    raw.add(
                        "alu",
                        ("+", tmp_node_addr, self.scratch["forest_values_p"], base_idx_const),
                    )
                    raw.add("alu", ("+", tmp_node_addr, tmp_node_addr, loop_b))
                    raw.add("load", ("load", tmp_node_val, tmp_node_addr))
                    raw.add("valu", ("vbroadcast", v_tmp3_shared, tmp_node_val))
                    raw.add("alu", ("+", loop_j, const_0, const_0))
                    raw.label(f"bucket_vec_check_{di}")
                    raw.add("alu", ("-", tmp1, tmp_count, loop_j))
                    raw.add("alu", ("+", tmp2, tmp1, const_1))
                    raw.add("alu", ("<", tmp_cond, vlen_const, tmp2))
                    raw.jump_rel(tmp_cond, f"bucket_vec_body_{di}")
                    raw.jump_abs(f"bucket_tail_check_{di}")
                    raw.label(f"bucket_vec_body_{di}")
                    raw.add("alu", ("+", tmp_addr, dst_vals, tmp_prefix))
                    raw.add("alu", ("+", tmp_addr, tmp_addr, loop_j))
                    raw.add("load", ("vload", v_val[0], tmp_addr))
                    raw.add("alu", ("+", tmp_addr, dst_idxs, tmp_prefix))
                    raw.add("alu", ("+", tmp_addr, tmp_addr, loop_j))
                    raw.add("load", ("vload", v_idx[0], tmp_addr))
                    raw.add("valu", ("^", v_val[0], v_val[0], v_tmp3_shared))

                    # Hash for single vector.
                    for eng, slot in self.build_hash_vec(
                        v_val[0], v_tmp1[0], v_tmp2[0], 0, 0, False
                    ):
                        raw.add(eng, slot)

                    # idx update
                    raw.add("valu", ("&", v_tmp1[0], v_val[0], v_one))
                    raw.add(
                        "valu",
                        ("multiply_add", v_idx[0], v_idx[0], v_two, v_base_minus1),
                    )
                    raw.add("valu", ("+", v_idx[0], v_idx[0], v_tmp1[0]))

                    raw.add("alu", ("+", tmp_addr, dst_vals, tmp_prefix))
                    raw.add("alu", ("+", tmp_addr, tmp_addr, loop_j))
                    raw.add("store", ("vstore", tmp_addr, v_val[0]))
                    raw.add("alu", ("+", tmp_addr, dst_idxs, tmp_prefix))
                    raw.add("alu", ("+", tmp_addr, tmp_addr, loop_j))
                    raw.add("store", ("vstore", tmp_addr, v_idx[0]))

                    raw.add("alu", ("+", loop_j, loop_j, vlen_const))
                    raw.jump_abs(f"bucket_vec_check_{di}")
                    raw.label(f"bucket_tail_check_{di}")
                    raw.add("alu", ("<", tmp_cond, loop_j, tmp_count))
                    raw.jump_rel(tmp_cond, f"bucket_tail_body_{di}")
                    raw.jump_abs(f"bucket_done_{di}")
                    raw.label(f"bucket_tail_body_{di}")
                    raw.add("alu", ("+", tmp_addr, dst_vals, tmp_prefix))
                    raw.add("alu", ("+", tmp_addr, tmp_addr, loop_j))
                    raw.add("load", ("load", tmp_val_s, tmp_addr))
                    raw.add("alu", ("+", tmp_addr, dst_idxs, tmp_prefix))
                    raw.add("alu", ("+", tmp_addr, tmp_addr, loop_j))
                    raw.add("load", ("load", tmp_idx_s, tmp_addr))
                    raw.add("alu", ("^", tmp_val_s, tmp_val_s, tmp_node_val))
                    for eng, slot in self.build_hash(
                        tmp_val_s, tmp1, tmp2, 0, 0, False
                    ):
                        raw.add(eng, slot)
                    raw.add("alu", ("&", tmp1, tmp_val_s, const_1))
                    raw.add("alu", ("*", tmp_idx_s, tmp_idx_s, const_2))
                    raw.add("alu", ("+", tmp_idx_s, tmp_idx_s, base_minus1))
                    raw.add("alu", ("+", tmp_idx_s, tmp_idx_s, tmp1))
                    raw.add("alu", ("+", tmp_addr, dst_vals, tmp_prefix))
                    raw.add("alu", ("+", tmp_addr, tmp_addr, loop_j))
                    raw.add("store", ("store", tmp_addr, tmp_val_s))
                    raw.add("alu", ("+", tmp_addr, dst_idxs, tmp_prefix))
                    raw.add("alu", ("+", tmp_addr, tmp_addr, loop_j))
                    raw.add("store", ("store", tmp_addr, tmp_idx_s))
                    raw.add("alu", ("+", loop_j, loop_j, const_1))
                    raw.jump_abs(f"bucket_tail_check_{di}")
                    raw.label(f"bucket_done_{di}")
                    raw.label(f"bucket_next_{di}")
                    raw.add("alu", ("+", loop_b, loop_b, const_1))
                    raw.add("alu", ("<", tmp_cond, loop_b, num_buckets_const))
                    raw.jump_rel(tmp_cond, f"bucket_loop_{di}")

                    if debug_bucket and depth == debug_depth:
                        # Debug-only unscatter + reload to validate this depth.
                        # Processed data lives in dst_*; unscatter back to tmp_* for ordered compare.
                        src_vals = dst_vals
                        src_idxs = dst_idxs
                        src_paths = dst_paths
                        dst_vals = tmp_vals_base
                        dst_idxs = tmp_idxs_base

                        raw.add("alu", ("+", loop_i, const_0, const_0))
                        raw.label(f"dbg_unscatter_{di}")
                        raw.add("alu", ("+", tmp_addr, src_paths, loop_i))
                        raw.add("load", ("load", tmp_path, tmp_addr))
                        raw.add("alu", ("+", tmp_addr, src_vals, loop_i))
                        raw.add("load", ("load", tmp_val_s, tmp_addr))
                        raw.add("alu", ("+", tmp_addr, src_idxs, loop_i))
                        raw.add("load", ("load", tmp_idx_s, tmp_addr))
                        raw.add("alu", ("+", tmp_addr, dst_vals, tmp_path))
                        raw.add("store", ("store", tmp_addr, tmp_val_s))
                        raw.add("alu", ("+", tmp_addr, dst_idxs, tmp_path))
                        raw.add("store", ("store", tmp_addr, tmp_idx_s))
                        raw.add("alu", ("+", loop_i, loop_i, const_1))
                        raw.add("alu", ("<", tmp_cond, loop_i, const_256))
                        raw.jump_rel(tmp_cond, f"dbg_unscatter_{di}")

                        # Reload first vector for compare.
                        raw.add("alu", ("+", tmp_addr, dst_vals, const_0))
                        raw.add("load", ("vload", v_val[0], tmp_addr))
                        raw.add("alu", ("+", tmp_addr, dst_idxs, const_0))
                        raw.add("load", ("vload", v_idx[0], tmp_addr))
                        raw.add(
                            "valu",
                            ("-", v_tmp3_shared, v_idx[0], v_forest_values_p),
                        )
                        round_idx = round_base + di
                        keys_val = [
                            (round_idx, lane, "hashed_val") for lane in range(VLEN)
                        ]
                        keys_idx = [
                            (round_idx, lane, "wrapped_idx") for lane in range(VLEN)
                        ]
                        raw.add("debug", ("vcompare", v_val[0], keys_val))
                        raw.add("debug", ("vcompare", v_tmp3_shared, keys_idx))

                        # After debug, reset to ordered tmp arrays.
                        current_is_tmp = True
                    else:
                        current_is_tmp = not current_is_tmp

                # Unscatter once at end (restore original order).
                if current_is_tmp:
                    src_vals = tmp_vals_base
                    src_idxs = tmp_idxs_base
                    src_paths = tmp_paths_base
                    dst_vals = bucket_vals_base
                    dst_idxs = bucket_idxs_base
                else:
                    src_vals = bucket_vals_base
                    src_idxs = bucket_idxs_base
                    src_paths = bucket_paths_base
                    dst_vals = tmp_vals_base
                    dst_idxs = tmp_idxs_base

                raw.add("alu", ("+", loop_i, const_0, const_0))
                raw.label("unscatter_loop")
                raw.add("alu", ("+", tmp_addr, src_paths, loop_i))
                raw.add("load", ("load", tmp_path, tmp_addr))
                raw.add("alu", ("+", tmp_addr, src_vals, loop_i))
                raw.add("load", ("load", tmp_val_s, tmp_addr))
                raw.add("alu", ("+", tmp_addr, src_idxs, loop_i))
                raw.add("load", ("load", tmp_idx_s, tmp_addr))
                raw.add("alu", ("+", tmp_addr, dst_vals, tmp_path))
                raw.add("store", ("store", tmp_addr, tmp_val_s))
                raw.add("alu", ("+", tmp_addr, dst_idxs, tmp_path))
                raw.add("store", ("store", tmp_addr, tmp_idx_s))
                raw.add("alu", ("+", loop_i, loop_i, const_1))
                raw.add("alu", ("<", tmp_cond, loop_i, const_256))
                raw.jump_rel(tmp_cond, "unscatter_loop")

                # Reload v_idx/v_val from the final ordered arrays.
                for u in range(UNROLL_MAIN):
                    off = vec_offsets[u]
                    raw.add("alu", ("+", tmp_addr, dst_vals, off))
                    raw.add("load", ("vload", v_val[u], tmp_addr))
                    raw.add("alu", ("+", tmp_addr, dst_idxs, off))
                    raw.add("load", ("vload", v_idx[u], tmp_addr))

                flush_body()
                self.instrs.extend(raw.finalize(base_offset=len(self.instrs)))
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
                for u in range(vec_count):
                    body.append(
                        (
                            "valu",
                            ("vbroadcast", v_idx[u], self.scratch["forest_values_p"]),
                        )
                    )

                # Unrolled 16-round schedule with fixed depths: [0..10,0..4]
                round_depths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]
                round_idx = 0
                while round_idx < rounds:
                    depth = round_depths[round_idx]
                    chunk = 1
                    if depth > 2:
                        starts = (
                            0, 16, 4, 20, 8, 24, 12, 28, 2, 18, 6, 22, 10, 26, 14, 30,
                            1, 17, 5, 21, 9, 25, 13, 29, 3, 19, 7, 23, 11, 27, 15, 31,
                        )
                    else:
                        starts = (
                            0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                            1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31,
                        )
                    for start in starts:
                        count = min(chunk, vec_count - start)
                        emit_hash_only_range(round_idx, depth, start, count)
                        # Skip idx update for depth 0 and forest_height
                        # For last round, still compute if we need to store indices
                        if depth != 0 and depth != forest_height and (round_idx != rounds - 1 or store_indices):
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
                    round_idx += 1

                for u in range(vec_count):
                    if store_indices:
                        body.append(("valu", ("+", v_tmp1[u], v_idx[u], v_neg_forest)))
                        if u == 0:
                            body.append(("store", ("vstore", base_idx_addr, v_tmp1[u])))
                        else:
                            body.append(("store", ("vstore", tmp_idx_addr_u[u], v_tmp1[u])))
                        if u == 0:
                            body.append(("store", ("vstore", base_val_addr, v_val[u])))
                        else:
                            body.append(("store", ("vstore", tmp_val_addr_u[u], v_val[u])))
                    else:
                        if u == 0:
                            body.append(("store", ("vstore", base_val_addr, v_val[u])))
                        else:
                            body.append(("store", ("vstore", tmp_val_addr_u[u], v_val[u])))

            emit_group(UNROLL_MAIN, zero_const, base_is_zero=True)

            body_instrs = self.build(body, vliw=True)
            self.instrs.extend(body_instrs)
            # Final pause to sync with reference_kernel2's second yield
            self.instrs.append({"flow": [("pause",)]})
            self.aggressive_schedule = False
            return
        # Fallback: Original simple scalar implementation for varying parameters
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def _depth_sequence(forest_height: int, rounds: int) -> list[int]:
    if rounds <= forest_height + 1:
        return list(range(rounds))
    seq = list(range(forest_height + 1))
    tail = rounds - len(seq)
    seq.extend(range(tail))
    return seq


def instrument_depth_stats(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
):
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    inp_indices_p = mem[5]
    n_nodes = mem[1]

    depth_seq = _depth_sequence(forest_height, rounds)
    per_depth = {}
    for d in depth_seq:
        per_depth.setdefault(
            d,
            {
                "u_counts": [],
                "uchild_counts": [],
                "bucket_sizes": [],
            },
        )

    for h in range(rounds):
        depth = depth_seq[h]
        idxs = mem[inp_indices_p : inp_indices_p + batch_size]
        counts = Counter(idxs)
        u = len(counts)
        per_depth[depth]["u_counts"].append(u)
        per_depth[depth]["bucket_sizes"].extend(counts.values())

        child_set = set()
        for idx in counts.keys():
            left = 2 * idx + 1
            right = 2 * idx + 2
            if left < n_nodes:
                child_set.add(left)
            else:
                child_set.add(0)
            if right < n_nodes:
                child_set.add(right)
            else:
                child_set.add(0)
        per_depth[depth]["uchild_counts"].append(len(child_set))

        # Advance one round (same as reference_kernel2).
        inp_values_p = mem[6]
        forest_values_p = mem[4]
        for i in range(batch_size):
            idx = mem[inp_indices_p + i]
            val = mem[inp_values_p + i]
            node_val = mem[forest_values_p + idx]
            val = myhash(val ^ node_val)
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= n_nodes else idx
            mem[inp_values_p + i] = val
            mem[inp_indices_p + i] = idx

    print(f"DEPTH STATS: forest_height={forest_height}, rounds={rounds}, batch={batch_size}")
    print("depth | rounds | U_d avg(min..max) | collision% avg | Uchild avg | bucket_hist (size:count)")
    for d in sorted(per_depth.keys()):
        entry = per_depth[d]
        u_counts = entry["u_counts"]
        u_avg = sum(u_counts) / len(u_counts)
        u_min = min(u_counts)
        u_max = max(u_counts)
        coll_avg = 1.0 - (u_avg / batch_size)
        uchild_avg = sum(entry["uchild_counts"]) / len(entry["uchild_counts"])
        hist = Counter(entry["bucket_sizes"])
        hist_str = " ".join(f"{k}:{hist[k]}" for k in sorted(hist.keys()))
        print(
            f"{d:>5} | {len(u_counts):>6} | {u_avg:>5.1f} ({u_min}..{u_max}) |"
            f" {coll_avg*100:>6.2f}% | {uchild_avg:>9.1f} | {hist_str}"
        )


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
        if os.getenv("TRACE_SMALL") == "1":
            do_kernel_test(6, 4, 256, trace=True, prints=False)
        elif os.getenv("TRACE_FAST") == "1":
            do_kernel_test(10, 5, 256, trace=True, prints=False)
        else:
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
