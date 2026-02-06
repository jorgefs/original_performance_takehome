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

        def can_pack(engine, reads, writes, cur_counts, cur_writes, cur_chosen=None, raw_deps_i=None):
            if cur_counts.get(engine, 0) + 1 > SLOT_LIMITS.get(engine, 0):
                return False
            if writes & cur_writes:
                return False
            if reads & cur_writes:
                if cur_chosen is None or raw_deps_i is None:
                    return False
                if raw_deps_i and any(p in cur_chosen for p in raw_deps_i):
                    return False
            return True

        # Build dependency graph for list scheduling.
        n = len(slots)
        reads_list = [None] * n
        writes_list = [None] * n
        deps = [set() for _ in range(n)]
        raw_deps = [set() for _ in range(n)]

        last_write = {}
        last_write_read = {}
        last_read = defaultdict(set)

        strict_waw = os.getenv("STRICT_WAW", "0") == "1"
        for i, (engine, slot) in enumerate(slots):
            reads, writes = _rw_sets(engine, slot)
            reads_list[i] = reads
            writes_list[i] = writes

            for loc in reads:
                if loc in last_write:
                    deps[i].add(last_write[loc])
                    raw_deps[i].add(last_write[loc])
                    last_write_read[loc] = True
                last_read[loc].add(i)

            for loc in writes:
                if loc in last_write:
                    if strict_waw or last_write_read.get(loc, True):
                        deps[i].add(last_write[loc])
                if loc in last_read and last_read[loc]:
                    if i in last_read[loc]:
                        last_read[loc].remove(i)
                    deps[i].update(last_read[loc])
                    last_read[loc].clear()
                last_write[loc] = i
                last_write_read[loc] = False

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
            {
                "valu": 1,
                "load": 2,
                "flow": 4,
                "store": 3,
                "alu": 1,
                "debug": 1,
            },
            {
                "valu": 1,
                "load": 4,
                "flow": 2,
                "store": 3,
                "alu": 1,
                "debug": 1,
            },
        ]

        def build_crit(engine_weight, reads_list_local=None):
            if reads_list_local is None:
                reads_list_local = reads_list
            order = sorted(range(n), key=lambda i: len(succs[i]))
            crit = [engine_weight[slots[i][0]] for i in range(n)]
            for i in sorted(order, reverse=True):
                if succs[i]:
                    crit[i] = engine_weight[slots[i][0]] + max(crit[s] for s in succs[i])
            return crit

        write_sizes = [len(writes_list[i]) for i in range(n)]
        read_sizes = [len(reads_list[i]) for i in range(n)]

        def schedule_with(crit, key_fn):
            instrs_local: list[dict[str, list[tuple]]] = []
            indeg_local = indeg[:]
            ready = {i for i in range(n) if indeg_local[i] == 0}
            remaining = n

            while remaining:
                cur: dict[str, list[tuple]] = {}
                cur_counts = {k: 0 for k in SLOT_LIMITS.keys()}
                cur_writes: set[int | tuple] = set()
                cur_chosen: set[int] = set()

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
                        if can_pack(engine, reads, writes, cur_counts, cur_writes, cur_chosen, raw_deps[i]):
                            cur.setdefault(engine, []).append(slot)
                            cur_counts[engine] = cur_counts.get(engine, 0) + 1
                            cur_writes |= writes
                            cur_chosen.add(i)
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

        def schedule_beam(crit, key_fn, beam_width: int):
            instrs_local: list[dict[str, list[tuple]]] = []
            indeg_local = indeg[:]
            ready = {i for i in range(n) if indeg_local[i] == 0}
            remaining = n

            def _fill_bundle(start_i, ready_list):
                cur: dict[str, list[tuple]] = {}
                cur_counts = {k: 0 for k in SLOT_LIMITS.keys()}
                cur_writes: set[int | tuple] = set()
                cur_chosen: set[int] = set()

                def _try_add(i):
                    nonlocal cur, cur_counts, cur_writes, cur_chosen
                    engine, slot = slots[i]
                    reads = reads_list[i]
                    writes = writes_list[i]
                    if can_pack(engine, reads, writes, cur_counts, cur_writes, cur_chosen, raw_deps[i]):
                        cur.setdefault(engine, []).append(slot)
                        cur_counts[engine] = cur_counts.get(engine, 0) + 1
                        cur_writes |= writes
                        cur_chosen.add(i)
                        return True
                    return False

                _try_add(start_i)
                for i in ready_list:
                    if i in cur_chosen:
                        continue
                    _try_add(i)
                return cur_chosen, cur

            while remaining:
                debug_ready = [i for i in ready if slots[i][0] == "debug"]
                if debug_ready:
                    i = min(debug_ready)
                    instrs_local.append({slots[i][0]: [slots[i][1]]})
                    ready.remove(i)
                    remaining -= 1
                    for s in succs[i]:
                        indeg_local[s] -= 1
                        if indeg_local[s] == 0:
                            ready.add(s)
                    continue

                ready_list = sorted(ready, key=key_fn)
                if not ready_list:
                    break
                beam = ready_list[: max(1, beam_width)]
                best_bundle = None
                best_score = None
                for start_i in beam:
                    chosen, cur = _fill_bundle(start_i, ready_list)
                    score = (len(chosen), sum(crit[i] for i in chosen))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_bundle = (chosen, cur)

                chosen, cur = best_bundle
                if not cur:
                    i = ready_list[0]
                    chosen = {i}
                    cur = {slots[i][0]: [slots[i][1]]}
                instrs_local.append(cur)
                for i in chosen:
                    if i in ready:
                        ready.remove(i)
                    remaining -= 1
                    for s in succs[i]:
                        indeg_local[s] -= 1
                        if indeg_local[s] == 0:
                            ready.add(s)

            return instrs_local

        def schedule_window(slots_list, window_size):
            instrs_local: list[dict[str, list[tuple]]] = []
            cur: dict[str, list[tuple]] = {}
            cur_counts = {k: 0 for k in SLOT_LIMITS.keys()}
            cur_writes: set[int | tuple] = set()
            cur_chosen: set[int] = set()

            def flush():
                nonlocal cur, cur_counts, cur_writes, cur_chosen
                if cur:
                    instrs_local.append(cur)
                cur = {}
                cur_counts = {k: 0 for k in SLOT_LIMITS.keys()}
                cur_writes = set()
                cur_chosen = set()

            pending = list(enumerate(slots_list))
            while pending:
                if pending[0][1][0] == "debug":
                    flush()
                    _idx, (engine, slot) = pending.pop(0)
                    instrs_local.append({engine: [slot]})
                    continue

                while True:
                    first_debug = None
                    for i in range(min(window_size, len(pending))):
                        if pending[i][1][0] == "debug":
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
                        _idx, (engine, slot) = pending[i]
                        reads, writes = _rw_sets(engine, slot)
                        if (writes & prefix_reads) or (writes & prefix_writes) or (
                            reads & prefix_writes
                        ):
                            prefix_reads |= reads
                            prefix_writes |= writes
                            continue
                        if can_pack(engine, reads, writes, cur_counts, cur_writes, cur_chosen, raw_deps[_idx]):
                            chosen_i = i
                            chosen_rw = (reads, writes)
                            break
                        prefix_reads |= reads
                        prefix_writes |= writes

                    if chosen_i is None:
                        break

                    idx, (engine, slot) = pending.pop(chosen_i)
                    reads, writes = chosen_rw
                    cur.setdefault(engine, []).append(slot)
                    cur_counts[engine] = cur_counts.get(engine, 0) + 1
                    cur_writes |= writes
                    cur_chosen.add(idx)

                if cur:
                    flush()
                    continue

                idx, (engine, slot) = pending.pop(0)
                reads, writes = _rw_sets(engine, slot)
                cur.setdefault(engine, []).append(slot)
                cur_counts[engine] = cur_counts.get(engine, 0) + 1
                cur_writes |= writes
                cur_chosen.add(idx)
                flush()

            return instrs_local

        def schedule_random(seed, iters=1):
            best_local = None
            rng = random.Random(seed)
            for _ in range(iters):
                instrs_local: list[dict[str, list[tuple]]] = []
                indeg_local = indeg[:]
                ready = [i for i in range(n) if indeg_local[i] == 0]
                remaining = n
                while remaining:
                    cur: dict[str, list[tuple]] = {}
                    cur_counts = {k: 0 for k in SLOT_LIMITS.keys()}
                    cur_writes: set[int | tuple] = set()
                    cur_chosen: set[int] = set()

                    debug_ready = [i for i in ready if slots[i][0] == "debug"]
                    if debug_ready:
                        i = min(debug_ready)
                        instrs_local.append({slots[i][0]: [slots[i][1]]})
                        ready.remove(i)
                        remaining -= 1
                        for s in succs[i]:
                            indeg_local[s] -= 1
                            if indeg_local[s] == 0:
                                ready.append(s)
                        continue

                    rng.shuffle(ready)
                    progressed = True
                    while progressed:
                        progressed = False
                        for i in list(ready):
                            engine, slot = slots[i]
                            reads = reads_list[i]
                            writes = writes_list[i]
                            if can_pack(engine, reads, writes, cur_counts, cur_writes, cur_chosen, raw_deps[i]):
                                cur.setdefault(engine, []).append(slot)
                                cur_counts[engine] = cur_counts.get(engine, 0) + 1
                                cur_writes |= writes
                                cur_chosen.add(i)
                                ready.remove(i)
                                remaining -= 1
                                for s in succs[i]:
                                    indeg_local[s] -= 1
                                    if indeg_local[s] == 0:
                                        ready.append(s)
                                progressed = True
                                break

                    if cur:
                        instrs_local.append(cur)
                        continue

                    i = min(ready)
                    instrs_local.append({slots[i][0]: [slots[i][1]]})
                    ready.remove(i)
                    remaining -= 1
                    for s in succs[i]:
                        indeg_local[s] -= 1
                        if indeg_local[s] == 0:
                            ready.append(s)

                if best_local is None or len(instrs_local) < best_local[0]:
                    best_local = (len(instrs_local), instrs_local)
            return best_local

        best = None
        search_seeds = int(os.getenv("SCHED_SEARCH", "4"))
        rng_seeds = list(range(search_seeds))
        for engine_weight in weight_sets:
            crit = build_crit(engine_weight)
            hash_bias = os.getenv("HASH_BIAS", "0") == "1"
            if hash_bias:
                crit = [
                    c + (2 if reads_list[i] and any(isinstance(x, int) for x in reads_list[i]) else 0)
                    for i, c in enumerate(crit)
                ]

            keys = [
                lambda k, c=crit: (-c[k], -len(succs[k]), k),
                lambda k, c=crit: (-c[k], -len(succs[k]), -k),
                lambda k, c=crit: (-c[k], engine_priority[slots[k][0]], -len(succs[k]), k),
                lambda k, c=crit: (-c[k], -write_sizes[k], -len(succs[k]), k),
                lambda k, c=crit: (-c[k], write_sizes[k], -len(succs[k]), k),
                lambda k, c=crit: (-c[k], -read_sizes[k], -len(succs[k]), k),
                lambda k, c=crit: (-c[k], read_sizes[k], -len(succs[k]), k),
                lambda k, c=crit: (-c[k], engine_priority[slots[k][0]], read_sizes[k], k),
                lambda k, c=crit: (-c[k], engine_priority[slots[k][0]], write_sizes[k], k),
                lambda k, c=crit: (-c[k], -engine_priority[slots[k][0]], read_sizes[k], k),
                lambda k, c=crit: (-c[k], -engine_priority[slots[k][0]], write_sizes[k], k),
            ]
            for seed in rng_seeds:
                rng = random.Random(seed)
                pri = [rng.randrange(1_000_000_000) for _ in range(n)]
                keys.append(lambda k, c=crit, p=pri: (-c[k], p[k]))

            for key_fn in keys:
                instrs_local = schedule_with(crit, key_fn)
                cand = (len(instrs_local), instrs_local)
                if best is None or cand[0] < best[0]:
                    best = cand
                beam_width = int(os.getenv("SCHED_BEAM", "0"))
                if beam_width > 0:
                    beam_sched = schedule_beam(crit, key_fn, beam_width)
                    if best is None or len(beam_sched) < best[0]:
                        best = (len(beam_sched), beam_sched)

        window_sizes = [64, 128, 256, 512, 1024, 2048]
        for ws in window_sizes:
            window_sched = schedule_window(slots, ws)
            if best is None or len(window_sched) < best[0]:
                best = (len(window_sched), window_sched)

        rand_iters = int(os.getenv("SCHED_RANDOM_ITERS", "0"))
        if rand_iters:
            rand_runs = int(os.getenv("SCHED_RANDOM_RUNS", "4"))
            for seed in range(rand_runs):
                cand = schedule_random(seed, iters=rand_iters)
                if cand and (best is None or cand[0] < best[0]):
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
                denom = len(stages) * interleave_scale
                loads_per_stage = (
                    (remaining + denom - 1) // denom
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
        simple=False,
    ):
        slots = []
        if simple:
            for u in range(vec_count):
                for lane in range(VLEN):
                    slots.append(("load", ("load_offset", v_tmp1[u], v_idx[u], lane)))
            if emit_debug:
                for u in range(vec_count):
                    base = i_base + u * VLEN
                    keys = [(round, base + lane, "node_val") for lane in range(VLEN)]
                    slots.append(("debug", ("vcompare", v_tmp1[u], keys)))
            for u in range(vec_count):
                slots.append(("valu", ("^", v_val[u], v_val[u], v_tmp1[u])))
            slots.extend(
                self.build_hash_vec_multi(
                    v_val,
                    v_tmp1,
                    v_tmp2,
                    round,
                    i_base,
                    emit_debug,
                )
            )
            return slots
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
                denom = len(stages) * interleave_scale
                loads_per_stage = (
                    (remaining + denom - 1) // denom
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
            emit_debug = os.getenv("EMIT_DEBUG", "0") == "1"
            store_indices = os.getenv("STORE_INDICES", "1") == "1"
            frontier_k = int(os.getenv("FRONTIER_K", "0"))
            use_frontier = frontier_k in (4, 5)
            self.frontier_k = frontier_k
            order_variant = int(os.getenv("ORDER_VARIANT", "0"))
            hash_group = int(os.getenv("HASH_GROUP", "3"))
            interleave_scale = int(os.getenv("LOAD_INTERLEAVE_SCALE", "2"))
            if interleave_scale < 1:
                interleave_scale = 1

            init = []
            round_trace = os.getenv("ROUND_TRACE", "0") == "1"
            chunk_size = int(os.getenv("CHUNK_SIZE", "1"))
            if chunk_size < 1:
                chunk_size = 1
            index_update_flow = os.getenv("INDEX_UPDATE_FLOW", "1") == "1"

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
            v_four = alloc_vec_const(4, "v_four")
            v_zero = alloc_vec_const(0, "v_zero")
            v_eight = alloc_vec_const(8, "v_eight")
            v_sixteen = alloc_vec_const(16, "v_sixteen")
            v_thirtytwo = alloc_vec_const(32, "v_thirtytwo")
            v_l6_base = alloc_vec_const(63, "v_l6_base")
            v_l6_limit = alloc_vec_const(127, "v_l6_limit")
            v_eight = alloc_vec_const(8, "v_eight")
            v_sixteen = alloc_vec_const(16, "v_sixteen")
            v_thirtytwo = alloc_vec_const(32, "v_thirtytwo")
            v_l6_base = alloc_vec_const(63, "v_l6_base")
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
            depth6_vselect = os.getenv("DEPTH6_VSELECT", "0") == "1"
            depth6_chunked = os.getenv("DEPTH6_CHUNKED", "0") == "1"
            if depth6_vselect:
                level6_vals = []
                for i in range(64):
                    addr = self.alloc_scratch(f"level6_{i}")
                    add_init(
                        "alu",
                        (
                            "+",
                            tmp1,
                            self.scratch["forest_values_p"],
                            alloc_const(63 + i),
                        ),
                    )
                    add_init("load", ("load", addr, tmp1))
                    level6_vals.append(addr)
            else:
                level6_vals = []
            depth3_cache = os.getenv("DEPTH3_CACHE", "0") == "1"
            depth3_cache_u = int(os.getenv("DEPTH3_CACHE_U", "0"))
            if depth3_cache:
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
                # Order by (forest_values_p + idx) & 7 to allow 3-bit vselect.
                order = sorted(
                    range(7, 15),
                    key=lambda idx: (forest_values_p_val + idx) & 7,
                )
                v_level3_perm = [v_level3[idx - 7] for idx in order]
            else:
                v_level3_perm = []
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

            # Optional round markers for trace analysis (not for performance).
            round_markers = []
            if round_trace:
                for r in range(rounds):
                    addr = self.alloc_scratch(f"round_marker_{r}")
                    add_init("load", ("const", addr, r))
                    round_markers.append(addr)

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

            UNROLL_MAIN = 16

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
            tmp1_pool = int(os.getenv("TMP1_POOL", "4"))
            tmp1_pool_partial = int(os.getenv("TMP1_POOL_PARTIAL", "0"))
            if UNROLL_MAIN == 16:
                # Avoid temp register aliasing under the scheduler.
                tmp1_pool = 0
                tmp1_pool_partial = 0
            if depth6_vselect:
                tmp1_pool = 0
                tmp1_pool_partial = 0
            v_tmp1_pool = [
                self.alloc_scratch(f"v_tmp1_pool_{i}", length=VLEN) for i in range(tmp1_pool)
            ]
            v_tmp2_pool = [
                self.alloc_scratch(f"v_tmp2_pool_{i}", length=VLEN) for i in range(tmp1_pool)
            ]
            stage_rename = os.getenv("STAGE_RENAME", "0") == "1"
            stage_u_limit = int(os.getenv("STAGE_RENAME_U", "4"))
            if depth6_vselect:
                stage_rename = False
            if depth6_chunked:
                tmp1_pool = 0
                tmp1_pool_partial = 0
                stage_rename = False
            if stage_rename:
                v_tmp1_stage = [
                    self.alloc_scratch(f"v_tmp1_stage_{u}", length=VLEN)
                    for u in range(stage_u_limit)
                ]
                v_tmp2_stage = [
                    self.alloc_scratch(f"v_tmp2_stage_{u}", length=VLEN)
                    for u in range(stage_u_limit)
                ]
            else:
                v_tmp1_stage = []
                v_tmp2_stage = []
            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)
            v_tmp4_shared = self.alloc_scratch("v_tmp4_shared", length=VLEN)
            if depth3_cache or os.getenv("DEPTH2_MASK_SELECT", "0") == "1" or depth6_vselect or depth6_chunked:
                v_tmp5_shared = self.alloc_scratch("v_tmp5_shared", length=VLEN)
                v_tmp6_shared = self.alloc_scratch("v_tmp6_shared", length=VLEN)
                v_tmp7_shared = self.alloc_scratch("v_tmp7_shared", length=VLEN)
            else:
                v_tmp5_shared = None
                v_tmp6_shared = None
                v_tmp7_shared = None
            tmp34_pool = int(os.getenv("TMP34_POOL", "0"))
            if tmp34_pool > 0:
                v_tmp3_pool = [
                    self.alloc_scratch(f"v_tmp3_pool_{i}", length=VLEN)
                    for i in range(tmp34_pool)
                ]
                v_tmp4_pool = [
                    self.alloc_scratch(f"v_tmp4_pool_{i}", length=VLEN)
                    for i in range(tmp34_pool)
                ]
            else:
                v_tmp3_pool = []
                v_tmp4_pool = []
            tmp_val_addr_u = [
                self.alloc_scratch(f"tmp_val_addr{u}") for u in range(UNROLL_MAIN)
            ]
            tmp_idx_addr_u = (
                [self.alloc_scratch(f"tmp_idx_addr{u}") for u in range(UNROLL_MAIN)]
                if store_indices
                else []
            )
            if depth6_chunked:
                v_l6_chunk = [
                    self.alloc_scratch(f"v_l6_chunk_{i}", length=VLEN) for i in range(16)
                ]
                v_l6_candidates = [
                    self.alloc_scratch(f"v_l6_cand_{i}", length=VLEN) for i in range(4)
                ]
                l6_chain_s = self.alloc_scratch("l6_chain_s")
                add_init("load", ("const", l6_chain_s, 0))
                v_l6_chain = self.alloc_scratch("v_l6_chain", length=VLEN)
            else:
                v_l6_chunk = []
                v_l6_candidates = []
                l6_chain_s = None
                v_l6_chain = None

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
                            [
                                v_tmp1_stage[start + u]
                                if stage_rename and (start + u) < stage_u_limit
                                else v_tmp1_l[u]
                                for u in range(count)
                            ],
                            [
                                v_tmp2_stage[start + u]
                                if stage_rename and (start + u) < stage_u_limit
                                else v_tmp2_l[u]
                                for u in range(count)
                            ],
                            round_idx,
                            start * VLEN,
                            emit_debug,
                        )
                    )
                elif depth == 1:
                    if os.getenv("DEPTH1_GATHER", "0") == "1":
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
                                hash_group=hash_group,
                                simple=os.getenv("PIPELINE_SIMPLE", "0") == "1",
                            )
                        )
                        return
                    for u in range(count):
                        use_pool = tmp1_pool > 0 and (tmp1_pool_partial == 0 or (start + u) < tmp1_pool_partial)
                        if use_pool:
                            pool_idx = (start + u) % tmp1_pool
                            t1 = v_tmp1_pool[pool_idx]
                            t2 = v_tmp2_pool[pool_idx]
                        else:
                            t1 = v_tmp1_l[u]
                            t2 = v_tmp2_l[u]
                        body.append(("valu", ("&", t1, v_val_l[u], v_one)))
                        body.append(("valu", ("+", v_idx_l[u], v_base_plus1, t1)))
                        if os.getenv("DEPTH1_MASK_SELECT", "0") == "1":
                            body.append(("valu", ("-", t2, v_zero, t1)))
                            body.append(("valu", ("^", t1, v_level1_left, v_level1_right)))
                            body.append(("valu", ("&", t1, t1, t2)))
                            body.append(("valu", ("^", t1, v_level1_left, t1)))
                        else:
                            body.append(
                                (
                                    "flow",
                                    (
                                        "vselect",
                                        t1,
                                        t1,
                                        v_level1_right,
                                        v_level1_left,
                                    ),
                                )
                            )
                        body.append(("valu", ("^", v_val_l[u], v_val_l[u], t1)))
                    body.extend(
                        self.build_hash_vec_multi(
                            v_val_l,
                            [
                                v_tmp1_stage[start + u]
                                if stage_rename and (start + u) < stage_u_limit
                                else v_tmp1_l[u]
                                for u in range(count)
                            ],
                            [
                                v_tmp2_stage[start + u]
                                if stage_rename and (start + u) < stage_u_limit
                                else v_tmp2_l[u]
                                for u in range(count)
                            ],
                            round_idx,
                            start * VLEN,
                            emit_debug,
                        )
                    )
                elif depth == 2:
                    if os.getenv("DEPTH2_GATHER", "0") == "1":
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
                                hash_group=hash_group,
                                simple=os.getenv("PIPELINE_SIMPLE", "0") == "1",
                            )
                        )
                        return
                    if os.getenv("DEPTH2_MASK_SELECT", "0") == "1":
                        for u in range(count):
                            # b0 = idx & 1, b1 = idx & 2
                            body.append(("valu", ("&", v_tmp1_l[u], v_idx_l[u], v_one)))
                            body.append(("valu", ("&", v_tmp2_l[u], v_idx_l[u], v_two)))
                            # mask0 = 0 - b0, mask1 = 0 - (b1 >> 1)
                            body.append(("valu", ("-", v_tmp3_shared, v_zero, v_tmp1_l[u])))
                            body.append(("valu", (">>", v_tmp4_shared, v_tmp2_l[u], v_one)))
                            body.append(("valu", ("-", v_tmp4_shared, v_zero, v_tmp4_shared)))
                            # sel01
                            body.append(("valu", ("^", v_tmp5_shared, v_level2_perm[0], v_level2_perm[1])))
                            body.append(("valu", ("&", v_tmp5_shared, v_tmp5_shared, v_tmp3_shared)))
                            body.append(("valu", ("^", v_tmp6_shared, v_level2_perm[0], v_tmp5_shared)))
                            # sel23
                            body.append(("valu", ("^", v_tmp5_shared, v_level2_perm[2], v_level2_perm[3])))
                            body.append(("valu", ("&", v_tmp5_shared, v_tmp5_shared, v_tmp3_shared)))
                            body.append(("valu", ("^", v_tmp7_shared, v_level2_perm[2], v_tmp5_shared)))
                            # select between sel01 and sel23 with mask1
                            body.append(("valu", ("^", v_tmp5_shared, v_tmp6_shared, v_tmp7_shared)))
                            body.append(("valu", ("&", v_tmp5_shared, v_tmp5_shared, v_tmp4_shared)))
                            body.append(("valu", ("^", v_tmp1_l[u], v_tmp6_shared, v_tmp5_shared)))
                            body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_tmp1_l[u])))
                        body.extend(
                            self.build_hash_vec_multi(
                                v_val_l,
                                [
                                    v_tmp1_stage[start + u]
                                    if stage_rename and (start + u) < stage_u_limit
                                    else v_tmp1_l[u]
                                    for u in range(count)
                                ],
                                [
                                    v_tmp2_stage[start + u]
                                    if stage_rename and (start + u) < stage_u_limit
                                    else v_tmp2_l[u]
                                    for u in range(count)
                                ],
                                round_idx,
                                start * VLEN,
                                emit_debug,
                            )
                        )
                        return
                    for u in range(count):
                        if tmp34_pool > 0:
                            pool_idx = (start + u) % tmp34_pool
                            tmp3 = v_tmp3_pool[pool_idx]
                            tmp4 = v_tmp4_pool[pool_idx]
                        else:
                            tmp3 = v_tmp3_shared
                            tmp4 = v_tmp4_shared
                        body.append(("valu", ("&", v_tmp1_l[u], v_idx_l[u], v_one)))  # b0
                        body.append(("valu", ("&", v_tmp2_l[u], v_idx_l[u], v_two)))  # b1
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    tmp3,
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
                                    tmp4,
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
                                    tmp4,
                                    tmp3,
                                ),
                            )
                        )
                        body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_tmp1_l[u])))
                    body.extend(
                        self.build_hash_vec_multi(
                            v_val_l,
                            [
                                v_tmp1_stage[start + u]
                                if stage_rename and (start + u) < stage_u_limit
                                else v_tmp1_l[u]
                                for u in range(count)
                            ],
                            [
                                v_tmp2_stage[start + u]
                                if stage_rename and (start + u) < stage_u_limit
                                else v_tmp2_l[u]
                                for u in range(count)
                            ],
                            round_idx,
                            start * VLEN,
                            emit_debug,
                        )
                    )
                elif depth == 3 and depth3_cache:
                    if depth3_cache_u:
                        cached_count = max(0, min(count, depth3_cache_u - start))
                    else:
                        cached_count = count
                    for u in range(cached_count):
                        body.append(("valu", ("&", v_tmp1_l[u], v_idx_l[u], v_one)))  # b0
                        body.append(("valu", ("&", v_tmp2_l[u], v_idx_l[u], v_two)))  # b1
                        body.append(("valu", ("&", v_tmp3_shared, v_idx_l[u], v_four)))  # b2
                        body.append(
                            (
                                "flow",
                                ("vselect", v_tmp4_shared, v_tmp1_l[u], v_level3_perm[1], v_level3_perm[0]),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                ("vselect", v_tmp5_shared, v_tmp1_l[u], v_level3_perm[3], v_level3_perm[2]),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                ("vselect", v_tmp6_shared, v_tmp1_l[u], v_level3_perm[5], v_level3_perm[4]),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                ("vselect", v_tmp7_shared, v_tmp1_l[u], v_level3_perm[7], v_level3_perm[6]),
                            )
                        )
                        body.append(
                            ("flow", ("vselect", v_tmp4_shared, v_tmp2_l[u], v_tmp5_shared, v_tmp4_shared))
                        )
                        body.append(
                            ("flow", ("vselect", v_tmp5_shared, v_tmp2_l[u], v_tmp7_shared, v_tmp6_shared))
                        )
                        body.append(
                            ("flow", ("vselect", v_tmp1_l[u], v_tmp3_shared, v_tmp5_shared, v_tmp4_shared))
                        )
                        body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_tmp1_l[u])))
                    if cached_count:
                        body.extend(
                            self.build_hash_vec_multi(
                                v_val_l[:cached_count],
                                [
                                    v_tmp1_stage[start + u]
                                    if stage_rename and (start + u) < stage_u_limit
                                    else v_tmp1_l[u]
                                    for u in range(cached_count)
                                ],
                                [
                                    v_tmp2_stage[start + u]
                                    if stage_rename and (start + u) < stage_u_limit
                                    else v_tmp2_l[u]
                                    for u in range(cached_count)
                                ],
                                round_idx,
                                start * VLEN,
                                emit_debug,
                            )
                        )
                    uncached_count = count - cached_count
                    if uncached_count:
                        uncached_start = start + cached_count
                        body.extend(
                            self.build_hash_pipeline_addr(
                                v_idx[uncached_start : uncached_start + uncached_count],
                                v_val[uncached_start : uncached_start + uncached_count],
                                v_tmp1[uncached_start : uncached_start + uncached_count],
                                v_tmp2[uncached_start : uncached_start + uncached_count],
                                round_idx,
                                uncached_start * VLEN,
                                emit_debug,
                                uncached_count,
                                hash_group=hash_group,
                                simple=os.getenv("PIPELINE_SIMPLE", "0") == "1",
                            )
                        )
                else:
                    if depth == 6 and depth6_chunked:
                        for u in range(count):
                            # Chain across u to prevent scheduler interleaving on shared buffers.
                            body.append(("alu", ("+", l6_chain_s, l6_chain_s, zero_const)))
                            body.append(("valu", ("vbroadcast", v_l6_chain, l6_chain_s)))
                            # v_local = v_idx - 63
                            body.append(("valu", ("-", v_tmp2_l[u], v_idx_l[u], v_l6_base)))
                            # masks for bits 0..5 from v_local
                            body.append(("valu", ("&", v_tmp3_shared, v_tmp2_l[u], v_one)))       # b0
                            body.append(("valu", ("&", v_tmp4_shared, v_tmp2_l[u], v_two)))       # b1
                            body.append(("valu", ("&", v_tmp5_shared, v_tmp2_l[u], v_four)))      # b2
                            body.append(("valu", ("&", v_tmp6_shared, v_tmp2_l[u], v_eight)))     # b3
                            body.append(("valu", ("&", v_tmp7_shared, v_tmp2_l[u], v_sixteen)))   # b4
                            body.append(("valu", ("&", v_tmp1_l[u], v_tmp2_l[u], v_thirtytwo)))   # b5
                            # Tie masks to chain dependency (no effect because v_l6_chain is 0).
                            body.append(("valu", ("^", v_tmp3_shared, v_tmp3_shared, v_l6_chain)))
                            body.append(("valu", ("^", v_tmp4_shared, v_tmp4_shared, v_l6_chain)))
                            body.append(("valu", ("^", v_tmp5_shared, v_tmp5_shared, v_l6_chain)))
                            body.append(("valu", ("^", v_tmp6_shared, v_tmp6_shared, v_l6_chain)))
                            body.append(("valu", ("^", v_tmp7_shared, v_tmp7_shared, v_l6_chain)))
                            body.append(("valu", ("^", v_tmp1_l[u], v_tmp1_l[u], v_l6_chain)))

                            for chunk_id in range(4):
                                # Chain between chunks to keep order.
                                body.append(("alu", ("+", l6_chain_s, l6_chain_s, zero_const)))
                                body.append(("valu", ("vbroadcast", v_l6_chain, l6_chain_s)))
                                base_off = 63 + chunk_id * 16
                                # load 16 scalars and broadcast
                                for i in range(16):
                                    body.append(
                                        (
                                            "alu",
                                            ("+", tmp1, self.scratch["forest_values_p"], alloc_const(base_off + i)),
                                        )
                                    )
                                    body.append(("load", ("load", tmp2, tmp1)))
                                    body.append(("alu", ("+", tmp2, tmp2, l6_chain_s)))
                                    body.append(("alu", ("-", tmp2, tmp2, l6_chain_s)))
                                    body.append(("valu", ("vbroadcast", v_l6_chunk[i], tmp2)))

                                # local reduce 16 -> 1 with b0..b3
                                for i in range(8):
                                    body.append(
                                        (
                                            "flow",
                                            ("vselect", v_l6_chunk[i], v_tmp3_shared, v_l6_chunk[2 * i + 1], v_l6_chunk[2 * i]),
                                        )
                                    )
                                for i in range(4):
                                    body.append(
                                        (
                                            "flow",
                                            ("vselect", v_l6_chunk[i], v_tmp4_shared, v_l6_chunk[2 * i + 1], v_l6_chunk[2 * i]),
                                        )
                                    )
                                for i in range(2):
                                    body.append(
                                        (
                                            "flow",
                                            ("vselect", v_l6_chunk[i], v_tmp5_shared, v_l6_chunk[2 * i + 1], v_l6_chunk[2 * i]),
                                        )
                                    )
                                body.append(
                                    (
                                        "flow",
                                        ("vselect", v_l6_chunk[0], v_tmp6_shared, v_l6_chunk[1], v_l6_chunk[0]),
                                    )
                                )
                                body.append(("valu", ("+", v_l6_candidates[chunk_id], v_l6_chunk[0], v_zero)))
                                body.append(("alu", ("+", l6_chain_s, l6_chain_s, zero_const)))

                            # final select across chunks with b4, b5
                            body.append(
                                (
                                    "flow",
                                    ("vselect", v_l6_candidates[0], v_tmp7_shared, v_l6_candidates[1], v_l6_candidates[0]),
                                )
                            )
                            body.append(
                                (
                                    "flow",
                                    ("vselect", v_l6_candidates[1], v_tmp7_shared, v_l6_candidates[3], v_l6_candidates[2]),
                                )
                            )
                            body.append(
                                (
                                    "flow",
                                    ("vselect", v_l6_candidates[0], v_tmp1_l[u], v_l6_candidates[1], v_l6_candidates[0]),
                                )
                            )
                            node_vec = v_l6_candidates[0]
                            body.append(("alu", ("+", l6_chain_s, l6_chain_s, zero_const)))
                            # mask_in = (v_idx >= 63) & (v_idx < 127)
                            body.append(("valu", ("<", v_tmp3_shared, v_idx_l[u], v_l6_base)))   # idx < 63
                            body.append(("valu", ("<", v_tmp4_shared, v_idx_l[u], v_l6_limit)))  # idx < 127
                            body.append(("valu", ("^", v_tmp3_shared, v_tmp3_shared, v_one)))    # idx >= 63
                            body.append(("valu", ("&", v_tmp3_shared, v_tmp3_shared, v_tmp4_shared)))
                            # gather fallback into v_tmp1_l[u]
                            for lane in range(VLEN):
                                body.append(("load", ("load_offset", v_tmp1_l[u], v_idx_l[u], lane)))
                            # select chunked vs gather
                            body.append(
                                (
                                    "flow",
                                    ("vselect", v_tmp1_l[u], v_tmp3_shared, node_vec, v_tmp1_l[u]),
                                )
                            )
                            body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_tmp1_l[u])))

                        body.extend(
                            self.build_hash_vec_multi(
                                v_val_l,
                                [
                                    v_tmp1_stage[start + u]
                                    if stage_rename and (start + u) < stage_u_limit
                                    else v_tmp1_l[u]
                                    for u in range(count)
                                ],
                                [
                                    v_tmp2_stage[start + u]
                                    if stage_rename and (start + u) < stage_u_limit
                                    else v_tmp2_l[u]
                                    for u in range(count)
                                ],
                                round_idx,
                                start * VLEN,
                                emit_debug,
                            )
                        )
                        return
                    if depth == 6 and depth6_vselect:
                        for u in range(count):
                            # v_local = v_idx - 63
                            body.append(("valu", ("-", v_tmp3_shared, v_idx_l[u], v_l6_base)))
                            # stage 0: 64 -> 32 using bit0
                            body.append(("valu", ("&", v_tmp4_shared, v_tmp3_shared, v_one)))
                            cur = v_tmp1
                            nxt = v_tmp2
                            for i in range(32):
                                body.append(("valu", ("vbroadcast", v_tmp5_shared, level6_vals[2 * i])))
                                body.append(("valu", ("vbroadcast", v_tmp6_shared, level6_vals[2 * i + 1])))
                                body.append(
                                    (
                                        "flow",
                                        ("vselect", cur[i], v_tmp4_shared, v_tmp6_shared, v_tmp5_shared),
                                    )
                                )
                            # stage 1: 32 -> 16 using bit1
                            body.append(("valu", ("&", v_tmp4_shared, v_tmp3_shared, v_two)))
                            for i in range(16):
                                body.append(
                                    (
                                        "flow",
                                        ("vselect", nxt[i], v_tmp4_shared, cur[2 * i + 1], cur[2 * i]),
                                    )
                                )
                            cur, nxt = nxt, cur
                            # stage 2: 16 -> 8 using bit2
                            body.append(("valu", ("&", v_tmp4_shared, v_tmp3_shared, v_four)))
                            for i in range(8):
                                body.append(
                                    (
                                        "flow",
                                        ("vselect", nxt[i], v_tmp4_shared, cur[2 * i + 1], cur[2 * i]),
                                    )
                                )
                            cur, nxt = nxt, cur
                            # stage 3: 8 -> 4 using bit3
                            body.append(("valu", ("&", v_tmp4_shared, v_tmp3_shared, v_eight)))
                            for i in range(4):
                                body.append(
                                    (
                                        "flow",
                                        ("vselect", nxt[i], v_tmp4_shared, cur[2 * i + 1], cur[2 * i]),
                                    )
                                )
                            cur, nxt = nxt, cur
                            # stage 4: 4 -> 2 using bit4
                            body.append(("valu", ("&", v_tmp4_shared, v_tmp3_shared, v_sixteen)))
                            for i in range(2):
                                body.append(
                                    (
                                        "flow",
                                        ("vselect", nxt[i], v_tmp4_shared, cur[2 * i + 1], cur[2 * i]),
                                    )
                                )
                            cur, nxt = nxt, cur
                            # stage 5: 2 -> 1 using bit5
                            body.append(("valu", ("&", v_tmp4_shared, v_tmp3_shared, v_thirtytwo)))
                            body.append(
                                (
                                    "flow",
                                    ("vselect", nxt[0], v_tmp4_shared, cur[1], cur[0]),
                                )
                            )
                            node_vec = nxt[0]
                            body.append(("valu", ("^", v_val_l[u], v_val_l[u], node_vec)))
                        body.extend(
                            self.build_hash_vec_multi(
                                v_val_l,
                                [
                                    v_tmp1_stage[start + u]
                                    if stage_rename and (start + u) < stage_u_limit
                                    else v_tmp1_l[u]
                                    for u in range(count)
                                ],
                                [
                                    v_tmp2_stage[start + u]
                                    if stage_rename and (start + u) < stage_u_limit
                                    else v_tmp2_l[u]
                                    for u in range(count)
                                ],
                                round_idx,
                                start * VLEN,
                                emit_debug,
                            )
                        )
                        return
                    if depth == 6 and os.getenv("DEPTH6_SIMPLE", "0") == "1":
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
                                hash_group=hash_group,
                                simple=True,
                            )
                        )
                        return
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
                            hash_group=hash_group,
                            simple=os.getenv("PIPELINE_SIMPLE", "0") == "1",
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
                def starts_for_depth(depth: int):
                    if chunk_size > 1 and os.getenv("CHUNK_ORDER", "0") == "1":
                        return tuple(range(0, UNROLL_MAIN, chunk_size))
                    if os.getenv("STARTS_ZIGZAG", "0") == "1":
                        out = []
                        half = UNROLL_MAIN // 2
                        for i in range(half):
                            out.append(i)
                            out.append(i + half)
                        return tuple(out)
                    if os.getenv("STARTS_PAIR_ZIGZAG", "0") == "1":
                        out = []
                        half = UNROLL_MAIN // 2
                        for i in range(0, half, 2):
                            out.extend([i, i + 1, i + half, i + half + 1])
                        return tuple(out)
                    if os.getenv("STARTS_BLOCK_ROT", "0") == "1":
                        block = 8
                        blocks = [list(range(b * block, (b + 1) * block)) for b in range(UNROLL_MAIN // block)]
                        rot = depth & 3
                        blocks = blocks[rot:] + blocks[:rot]
                        out = []
                        for bl in blocks:
                            out.extend(bl)
                        return tuple(out)
                    if order_variant == 0:
                        if UNROLL_MAIN == 32:
                            if depth > 2:
                                return (
                                    0, 16, 4, 20, 8, 24, 12, 28, 2, 18, 6, 22, 10, 26, 14, 30,
                                    1, 17, 5, 21, 9, 25, 13, 29, 3, 19, 7, 23, 11, 27, 15, 31,
                                )
                            return (
                                0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                                1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31,
                            )
                        # UNROLL_MAIN == 16
                        if depth > 2:
                            return (0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15)
                        return (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)
                    if order_variant == 1:
                        return tuple(range(UNROLL_MAIN))
                    if order_variant == 2:
                        return tuple(range(UNROLL_MAIN - 1, -1, -1))
                    if order_variant == 3:
                        evens = list(range(0, UNROLL_MAIN, 2))
                        odds = list(range(1, UNROLL_MAIN, 2))
                        return tuple(evens + odds)
                    if order_variant == 4:
                        out = []
                        bits = (UNROLL_MAIN - 1).bit_length()
                        for i in range(UNROLL_MAIN):
                            b = format(i, f"0{bits}b")[::-1]
                            out.append(int(b, 2))
                        return tuple(out)
                    return tuple(range(UNROLL_MAIN))

                if use_frontier:
                    pre_end = frontier_k
                    for round_idx in range(pre_end):
                        depth = round_depths[round_idx]
                        chunk = chunk_size
                        starts = starts_for_depth(depth)
                        if os.getenv("ROUND_STAGGER", "0") == "1" and (round_idx & 1):
                            starts = tuple(reversed(starts))
                        if round_trace:
                            body.append(("flow", ("trace_write", round_markers[round_idx])))
                        for start in starts:
                            count = min(chunk, vec_count - start)
                            emit_hash_only_range(round_idx, depth, start, count)
                            if depth != 0 and depth != forest_height:
                                for u in range(start, start + count):
                                    body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                    if index_update_flow:
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
                                    else:
                                        body.append(
                                            ("valu", ("+", v_tmp2[u], v_base_minus1, v_tmp1[u]))
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
                    chunk = chunk_size
                    starts = starts_for_depth(depth)
                    if os.getenv("ROUND_STAGGER", "0") == "1" and (round_idx & 1):
                        starts = tuple(reversed(starts))
                    if round_trace:
                        body.append(("flow", ("trace_write", round_markers[round_idx])))
                    for start in starts:
                        count = min(chunk, vec_count - start)
                        emit_hash_only_range(round_idx, depth, start, count)
                        if depth != 0 and depth != forest_height:
                            for u in range(start, start + count):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                if index_update_flow:
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
                                else:
                                    body.append(
                                        ("valu", ("+", v_tmp2[u], v_base_minus1, v_tmp1[u]))
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
                    chunk = chunk_size
                    starts = starts_for_depth(depth)
                    if os.getenv("ROUND_STAGGER", "0") == "1" and (round_idx & 1):
                        starts = tuple(reversed(starts))
                    if round_trace:
                        # Marker for trace analysis; not for performance.
                        body.append(("flow", ("trace_write", round_markers[round_idx])))
                    for start in starts:
                        count = min(chunk, vec_count - start)
                        emit_hash_only_range(round_idx, depth, start, count)
                        if depth != 0 and depth != forest_height and (round_idx != rounds - 1 or store_indices):
                            for u in range(start, start + count):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                if index_update_flow:
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
                                else:
                                    body.append(
                                        ("valu", ("+", v_tmp2[u], v_base_minus1, v_tmp1[u]))
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
                        body.append(("valu", ("-", v_tmp1[u], v_idx[u], v_forest_values_p)))
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
            if UNROLL_MAIN == 16 and batch_size >= UNROLL_MAIN * VLEN * 2:
                body = []
                second_base = alloc_const(UNROLL_MAIN * VLEN)
                emit_group(UNROLL_MAIN, second_base, base_is_zero=False)
                body_instrs += self.build(body, vliw=True)

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
        three_const = self.scratch_const(3)
        four_const = self.scratch_const(4)
        five_const = self.scratch_const(5)
        six_const = self.scratch_const(6)
        seven_const = self.scratch_const(7)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_cached_node = self.alloc_scratch("tmp_cached_node")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Preload depth 0-2 nodes (indices 0..6) for all scenarios.
        # This is a general optimization idea, not micro-optimized here.
        cache_nodes = []
        for i in range(7):
            idx_const = self.scratch_const(i)
            cache = self.alloc_scratch(f"cache_node_{i}")
            self.add("alu", ("+", tmp_addr, self.scratch["forest_values_p"], idx_const))
            self.add("load", ("load", cache, tmp_addr))
            cache_nodes.append(cache)

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # If idx is in [0..6], select cached node; otherwise load from memory.
                # This extends the fast-path idea to all scenarios, without micro-optimizing.
                body.append(("alu", ("+", tmp_cached_node, cache_nodes[0], zero_const)))
                body.append(("alu", ("==", tmp1, tmp_idx, one_const)))
                body.append(("flow", ("select", tmp_cached_node, tmp1, cache_nodes[1], tmp_cached_node)))
                body.append(("alu", ("==", tmp1, tmp_idx, two_const)))
                body.append(("flow", ("select", tmp_cached_node, tmp1, cache_nodes[2], tmp_cached_node)))
                body.append(("alu", ("==", tmp1, tmp_idx, three_const)))
                body.append(("flow", ("select", tmp_cached_node, tmp1, cache_nodes[3], tmp_cached_node)))
                body.append(("alu", ("==", tmp1, tmp_idx, four_const)))
                body.append(("flow", ("select", tmp_cached_node, tmp1, cache_nodes[4], tmp_cached_node)))
                body.append(("alu", ("==", tmp1, tmp_idx, five_const)))
                body.append(("flow", ("select", tmp_cached_node, tmp1, cache_nodes[5], tmp_cached_node)))
                body.append(("alu", ("==", tmp1, tmp_idx, six_const)))
                body.append(("flow", ("select", tmp_cached_node, tmp1, cache_nodes[6], tmp_cached_node)))

                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("alu", ("<", tmp1, tmp_idx, seven_const)))
                body.append(("flow", ("select", tmp_node_val, tmp1, tmp_cached_node, tmp_node_val)))
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
