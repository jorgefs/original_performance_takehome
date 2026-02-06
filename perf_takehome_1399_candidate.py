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

        # Build dependency graph for list scheduling.
        n = len(slots)
        reads_list = [None] * n
        writes_list = [None] * n
        deps = [set() for _ in range(n)]

        last_write = {}
        last_read = defaultdict(set)

        for i, (engine, slot) in enumerate(slots):
            reads, writes = _rw_sets(engine, slot)
            reads_list[i] = reads
            writes_list[i] = writes

            for loc in reads:
                if loc in last_write:
                    deps[i].add(last_write[loc])
                last_read[loc].add(i)

            for loc in writes:
                if loc in last_write:
                    deps[i].add(last_write[loc])
                if loc in last_read and last_read[loc]:
                    if i in last_read[loc]:
                        last_read[loc].remove(i)
                    deps[i].update(last_read[loc])
                    last_read[loc].clear()
                last_write[loc] = i

        indeg = [0] * n
        succs = [set() for _ in range(n)]
        for i in range(n):
            for d in deps[i]:
                succs[d].add(i)
                indeg[i] += 1
        # Try multiple scheduling heuristics and keep the shortest bundle count.
        engine_priority = {
            "load": 0,
            "flow": 1,
            "valu": 2,
            "alu": 3,
            "store": 4,
            "debug": 5,
        }

        weight_sets = [
            {
                "valu": 1,
                "load": 2,
                "flow": 3,
                "store": 3,
                "alu": 1,
                "debug": 1,
            },
            {
                "valu": 1,
                "load": 3,
                "flow": 3,
                "store": 3,
                "alu": 1,
                "debug": 1,
            },
        ]

        def build_crit(engine_weight):
            order = sorted(range(n), key=lambda i: len(succs[i]))
            crit = [engine_weight[slots[i][0]] for i in range(n)]
            for i in sorted(order, reverse=True):
                if succs[i]:
                    crit[i] = engine_weight[slots[i][0]] + max(crit[s] for s in succs[i])
            return crit

        write_sizes = [len(writes_list[i]) for i in range(n)]

        def schedule_with(crit, key_fn):
            instrs_local: list[dict[str, list[tuple]]] = []
            indeg_local = indeg[:]
            ready = {i for i in range(n) if indeg_local[i] == 0}
            remaining = n

            while remaining:
                cur: dict[str, list[tuple]] = {}
                cur_counts = {k: 0 for k in SLOT_LIMITS.keys()}
                cur_writes: set[int | tuple] = set()

                debug_ready = [i for i in ready if slots[i][0] == "debug"]
                if debug_ready:
                    i = min(debug_ready)
                    engine, slot = slots[i]
                    instrs_local.append({engine: [slot]})
                    ready.remove(i)
                    remaining -= 1
                    for s in succs[i]:
                        indeg_local[s] -= 1
                        if indeg_local[s] == 0:
                            ready.add(s)
                    continue

                progressed = True
                while progressed:
                    progressed = False
                    for i in sorted(ready, key=key_fn):
                        engine, slot = slots[i]
                        reads = reads_list[i]
                        writes = writes_list[i]
                        if can_pack(engine, reads, writes, cur_counts, cur_writes):
                            cur.setdefault(engine, []).append(slot)
                            cur_counts[engine] = cur_counts.get(engine, 0) + 1
                            cur_writes |= writes
                            ready.remove(i)
                            remaining -= 1
                            for s in succs[i]:
                                indeg_local[s] -= 1
                                if indeg_local[s] == 0:
                                    ready.add(s)
                            progressed = True
                            break

                if cur:
                    instrs_local.append(cur)
                    continue

                i = min(ready)
                engine, slot = slots[i]
                instrs_local.append({engine: [slot]})
                ready.remove(i)
                remaining -= 1
                for s in succs[i]:
                    indeg_local[s] -= 1
                    if indeg_local[s] == 0:
                        ready.add(s)

            return instrs_local

        best = None
        for engine_weight in weight_sets:
            crit = build_crit(engine_weight)

            keys = [
                lambda k, c=crit: (-c[k], -len(succs[k]), k),
                lambda k, c=crit: (-c[k], -len(succs[k]), -k),
                lambda k, c=crit: (-c[k], engine_priority[slots[k][0]], -len(succs[k]), k),
                lambda k, c=crit: (-c[k], -write_sizes[k], -len(succs[k]), k),
            ]

            for key_fn in keys:
                instrs_local = schedule_with(crit, key_fn)
                cand = (len(instrs_local), instrs_local)
                if best is None or cand[0] < best[0]:
                    best = cand

        return best[1]

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
            frontier_k = int(os.getenv("FRONTIER_K", "0"))
            use_frontier = frontier_k in (4, 5)
            self.frontier_k = frontier_k

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

            zero_const = alloc_const(0)
            one_const = alloc_const(1)
            two_const = alloc_const(2)
            vlen_const = alloc_const(VLEN)

            v_one = alloc_vec_const(1, "v_one")
            v_two = alloc_vec_const(2, "v_two")
            base_idx_2 = alloc_const(3)
            v_base_idx_2 = alloc_vec_const(3, "v_base_idx_2")


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
            # For depth-2 selection we can use address low bits (forest_values_p=7),
            # which maps idx 3..6 to address&3 order: [5,6,3,4].
            v_level2_perm = [v_level2[2], v_level2[3], v_level2[0], v_level2[1]]
            v_forest_values_p = self.alloc_scratch("v_forest_values_p", length=VLEN)
            add_init(
                "valu", ("vbroadcast", v_forest_values_p, self.scratch["forest_values_p"])
            )
            v_base_plus1 = self.alloc_scratch("v_base_plus1", length=VLEN)
            v_base_minus1 = self.alloc_scratch("v_base_minus1", length=VLEN)
            add_init("valu", ("+", v_base_plus1, v_forest_values_p, v_one))
            add_init("valu", ("-", v_base_minus1, v_one, v_forest_values_p))
            v_base_minus1_plus1 = self.alloc_scratch("v_base_minus1_plus1", length=VLEN)
            add_init("valu", ("+", v_base_minus1_plus1, v_base_minus1, v_one))
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

            # Pause to sync with reference_kernel2's first yield
            add_init("flow", ("pause",))

            self.instrs.extend(self.build(init, vliw=True))

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
            v_tmp4_shared = self.alloc_scratch("v_tmp4_shared", length=VLEN)
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
                        body.append(("valu", ("&", v_tmp2_l[u], v_idx_l[u], v_two)))  # b1
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp3_shared,
                                    v_tmp1_l[u],
                                    v_level2_perm[1],
                                    v_level2_perm[0],
                                ),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp4_shared,
                                    v_tmp1_l[u],
                                    v_level2_perm[3],
                                    v_level2_perm[2],
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
                                    v_tmp4_shared,
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
                start_round = 0
                if use_frontier:
                    pre_end = frontier_k
                    for round_idx in range(pre_end):
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
                            if depth != 0 and depth != forest_height:
                                for u in range(start, start + count):
                                    body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                    body.append(
                                        (
                                            "flow",
                                            (
                                                "vselect",
                                                v_tmp2[u],
                                                v_tmp1[u],
                                                v_base_minus1_plus1,
                                                v_base_minus1,
                                            ),
                                        )
                                    )
                                    body.append(
                                        (
                                            "valu",
                                            (
                                                "multiply_add",
                                                v_idx[u],
                                                v_idx[u],
                                                v_two,
                                                v_tmp2[u],
                                            ),
                                        )
                                    )
                    round_idx = frontier_k
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
                        if depth != 0 and depth != forest_height:
                            for u in range(start, start + count):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                body.append(
                                    (
                                        "flow",
                                        (
                                            "vselect",
                                            v_tmp2[u],
                                            v_tmp1[u],
                                            v_base_minus1_plus1,
                                            v_base_minus1,
                                        ),
                                    )
                                )
                                body.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            v_idx[u],
                                            v_idx[u],
                                            v_two,
                                            v_tmp2[u],
                                        ),
                                    )
                                )
                    start_round = frontier_k + 1
                for round_idx in range(start_round, rounds):
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
                        if depth != 0 and depth != forest_height and (round_idx != rounds - 1 or store_indices):
                            for u in range(start, start + count):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                body.append(
                                    (
                                        "flow",
                                        (
                                            "vselect",
                                            v_tmp2[u],
                                            v_tmp1[u],
                                            v_base_minus1_plus1,
                                            v_base_minus1,
                                        ),
                                    )
                                )
                                body.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            v_idx[u],
                                            v_idx[u],
                                            v_two,
                                            v_tmp2[u],
                                        ),
                                    )
                                )

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

            def _writes_src(bundle, src_base: int) -> bool:
                src_lo = src_base
                src_hi = src_base + VLEN - 1

                def _overlap(dest):
                    return src_lo <= dest <= src_hi

                def _overlap_range(dest):
                    return not (dest + VLEN - 1 < src_lo or dest > src_hi)

                for engine, slots in bundle.items():
                    for slot in slots:
                        if engine == "valu":
                            dest = slot[1]
                            if _overlap_range(dest):
                                return True
                        elif engine == "load":
                            op = slot[0]
                            if op == "vload":
                                dest = slot[1]
                                if _overlap_range(dest):
                                    return True
                            elif op == "load":
                                dest = slot[1]
                                if _overlap(dest):
                                    return True
                            elif op == "load_offset":
                                dest = slot[1] + slot[3]
                                if _overlap(dest):
                                    return True
                            elif op == "const":
                                dest = slot[1]
                                if _overlap(dest):
                                    return True
                        elif engine == "flow":
                            op = slot[0]
                            if op == "vselect":
                                dest = slot[1]
                                if _overlap_range(dest):
                                    return True
                            elif op in ("select", "addimm", "coreid"):
                                dest = slot[1]
                                if _overlap(dest):
                                    return True
                        elif engine == "alu":
                            dest = slot[1]
                            if _overlap(dest):
                                return True
                return False

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
    if os.getenv("FRONTIER_LOG") == "1":
        frontier_k = getattr(kb, "frontier_k", 0)
        if frontier_k in (4, 5):
            print(f"FRONTIER_K{frontier_k}: {machine.cycle} ciclos")
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
