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

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        def slot_rw(engine, slot):
            reads = set()
            writes = set()
            is_load = False
            is_store = False
            if engine == "alu":
                _, dest, a1, a2 = slot
                reads.update([a1, a2])
                writes.add(dest)
            elif engine == "valu":
                op = slot[0]
                if op == "vbroadcast":
                    _, dest, src = slot
                    reads.add(src)
                    writes.update(range(dest, dest + VLEN))
                elif op == "multiply_add":
                    _, dest, a, b, c = slot
                    reads.update(range(a, a + VLEN))
                    reads.update(range(b, b + VLEN))
                    reads.update(range(c, c + VLEN))
                    writes.update(range(dest, dest + VLEN))
                else:
                    _, dest, a1, a2 = slot
                    reads.update(range(a1, a1 + VLEN))
                    reads.update(range(a2, a2 + VLEN))
                    writes.update(range(dest, dest + VLEN))
            elif engine == "load":
                op = slot[0]
                if op == "load":
                    _, dest, addr = slot
                    reads.add(addr)
                    writes.add(dest)
                    is_load = True
                elif op == "load_offset":
                    _, dest, addr, offset = slot
                    reads.add(addr + offset)
                    writes.add(dest + offset)
                    is_load = True
                elif op == "vload":
                    _, dest, addr = slot
                    reads.add(addr)
                    writes.update(range(dest, dest + VLEN))
                    is_load = True
                elif op == "const":
                    _, dest, _val = slot
                    writes.add(dest)
            elif engine == "store":
                op = slot[0]
                if op == "store":
                    _, addr, src = slot
                    reads.add(addr)
                    reads.add(src)
                    is_store = True
                elif op == "vstore":
                    _, addr, src = slot
                    reads.add(addr)
                    reads.update(range(src, src + VLEN))
                    is_store = True
            elif engine == "flow":
                op = slot[0]
                if op == "select":
                    _, dest, cond, a, b = slot
                    reads.update([cond, a, b])
                    writes.add(dest)
                elif op == "add_imm":
                    _, dest, a, _imm = slot
                    reads.add(a)
                    writes.add(dest)
                elif op == "vselect":
                    _, dest, cond, a, b = slot
                    reads.update(range(cond, cond + VLEN))
                    reads.update(range(a, a + VLEN))
                    reads.update(range(b, b + VLEN))
                    writes.update(range(dest, dest + VLEN))
                elif op == "trace_write":
                    _, val = slot
                    reads.add(val)
                elif op == "cond_jump":
                    _, cond, addr = slot
                    reads.update([cond, addr])
                elif op == "cond_jump_rel":
                    _, cond, _offset = slot
                    reads.add(cond)
                elif op == "jump":
                    pass
                elif op == "jump_indirect":
                    _, addr = slot
                    reads.add(addr)
                elif op == "coreid":
                    _, dest = slot
                    writes.add(dest)
            elif engine == "debug":
                op = slot[0]
                if op == "compare":
                    _, loc, _key = slot
                    reads.add(loc)
                elif op == "vcompare":
                    _, loc, _keys = slot
                    reads.update(range(loc, loc + VLEN))
            return reads, writes, is_load, is_store

        instrs = []
        bundle = {}
        bundle_writes = set()
        bundle_has_store = False

        def flush_bundle():
            nonlocal bundle, bundle_writes, bundle_has_store
            if bundle:
                instrs.append(bundle)
                bundle = {}
                bundle_writes = set()
                bundle_has_store = False

        for engine, slot in slots:
            is_pause = engine == "flow" and slot[0] == "pause"
            reads, writes, is_load, is_store = slot_rw(engine, slot)
            if is_pause:
                flush_bundle()
                instrs.append({engine: [slot]})
                continue
            if bundle.get(engine) and len(bundle[engine]) >= SLOT_LIMITS[engine]:
                flush_bundle()
            if reads & bundle_writes:
                flush_bundle()
            if is_load and bundle_has_store:
                flush_bundle()
            if bundle.get(engine) and len(bundle[engine]) >= SLOT_LIMITS[engine]:
                flush_bundle()

            bundle.setdefault(engine, []).append(slot)
            bundle_writes.update(writes)
            if is_store:
                bundle_has_store = True

        flush_bundle()
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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i, emit_debug=True):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mult = 1 + (1 << val3)
                slots.append(
                    ("alu", ("*", tmp1, val_hash_addr, self.scratch_const(mult)))
                )
                slots.append(
                    ("alu", ("+", val_hash_addr, tmp1, self.scratch_const(val1)))
                )
            else:
                slots.append(
                    ("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1)))
                )
                slots.append(
                    ("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3)))
                )
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
                keys = [(round, i_base + lane, "hash_stage", hi) for lane in range(VLEN)]
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
                    keys = [(round, base + lane, "hash_stage", hi) for lane in range(VLEN)]
                    slots.append(("debug", ("vcompare", val_addrs[u], keys)))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        emit_debug = False
        # Scratch space addresses
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
        vlen_const = self.scratch_const(VLEN)

        v_zero = self.vector_const(0, "v_zero")
        v_one = self.vector_const(1, "v_one")
        v_two = self.vector_const(2, "v_two")

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_idx_addr = self.alloc_scratch("tmp_idx_addr")
        tmp_val_addr = self.alloc_scratch("tmp_val_addr")

        UNROLL = 6
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
        v_n_nodes = self.alloc_scratch("v_n_nodes", length=VLEN)
        tmp_idx_addr_u = [self.alloc_scratch(f"tmp_idx_addr{u}") for u in range(UNROLL)]
        tmp_val_addr_u = [self.alloc_scratch(f"tmp_val_addr{u}") for u in range(UNROLL)]
        self.add("valu", ("vbroadcast", v_forest_values_p, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        vec_batch = batch_size - (batch_size % VLEN)
        vec_batch_unrolled = vec_batch - (vec_batch % group_size)
        body.append(
            ("alu", ("+", tmp_idx_addr, self.scratch["inp_indices_p"], zero_const))
        )
        body.append(
            ("alu", ("+", tmp_val_addr, self.scratch["inp_values_p"], zero_const))
        )
        for i in range(0, vec_batch_unrolled, group_size):
            # Compute per-block base addresses
            body.append(("alu", ("+", tmp_idx_addr_u[0], tmp_idx_addr, zero_const)))
            body.append(("alu", ("+", tmp_val_addr_u[0], tmp_val_addr, zero_const)))
            for u in range(1, UNROLL):
                body.append(
                    ("alu", ("+", tmp_idx_addr_u[u], tmp_idx_addr_u[u - 1], vlen_const))
                )
                body.append(
                    ("alu", ("+", tmp_val_addr_u[u], tmp_val_addr_u[u - 1], vlen_const))
                )

            # idx/val = vload once per vector, then run all rounds in scratch
            for u in range(UNROLL):
                body.append(("load", ("vload", v_idx[u], tmp_idx_addr_u[u])))
                if emit_debug:
                    base = i + u * VLEN
                    keys = [(0, base + lane, "idx") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", v_idx[u], keys)))
                body.append(("load", ("vload", v_val[u], tmp_val_addr_u[u])))
                if emit_debug:
                    base = i + u * VLEN
                    keys = [(0, base + lane, "val") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", v_val[u], keys)))

            for round in range(rounds):
                for u in range(UNROLL):
                    # node_val = gather(forest_values_p + idx)
                    body.append(
                        ("valu", ("+", v_node_addr[u], v_idx[u], v_forest_values_p))
                    )
                    for lane in range(VLEN):
                        body.append(
                            (
                                "load",
                                ("load_offset", v_node_val[u], v_node_addr[u], lane),
                            )
                        )
                    if emit_debug:
                        base = i + u * VLEN
                        keys = [
                            (round, base + lane, "node_val") for lane in range(VLEN)
                        ]
                        body.append(("debug", ("vcompare", v_node_val[u], keys)))
                # val = myhash(val ^ node_val)
                for u in range(UNROLL):
                    body.append(("valu", ("^", v_val[u], v_val[u], v_node_val[u])))
                body.extend(
                    self.build_hash_vec_multi(
                        v_val, v_tmp1, v_tmp2, round, i, emit_debug
                    )
                )
                if emit_debug:
                    for u in range(UNROLL):
                        base = i + u * VLEN
                        keys = [
                            (round, base + lane, "hashed_val") for lane in range(VLEN)
                        ]
                        body.append(("debug", ("vcompare", v_val[u], keys)))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                for u in range(UNROLL):
                    body.append(("valu", ("%", v_tmp1[u], v_val[u], v_two)))
                for u in range(UNROLL):
                    body.append(("valu", ("==", v_tmp1[u], v_tmp1[u], v_zero)))
                for u in range(UNROLL):
                    body.append(("valu", ("-", v_tmp3[u], v_two, v_tmp1[u])))
                for u in range(UNROLL):
                    body.append(("valu", ("*", v_idx[u], v_idx[u], v_two)))
                for u in range(UNROLL):
                    body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp3[u])))
                if emit_debug:
                    for u in range(UNROLL):
                        base = i + u * VLEN
                        keys = [
                            (round, base + lane, "next_idx") for lane in range(VLEN)
                        ]
                        body.append(("debug", ("vcompare", v_idx[u], keys)))
                # idx = 0 if idx >= n_nodes else idx
                for u in range(UNROLL):
                    body.append(("valu", ("<", v_tmp1[u], v_idx[u], v_n_nodes)))
                for u in range(UNROLL):
                    body.append(("valu", ("*", v_idx[u], v_idx[u], v_tmp1[u])))
                if emit_debug:
                    for u in range(UNROLL):
                        base = i + u * VLEN
                        keys = [
                            (round, base + lane, "wrapped_idx") for lane in range(VLEN)
                        ]
                        body.append(("debug", ("vcompare", v_idx[u], keys)))

            # mem[inp_indices_p + i] = idx, mem[inp_values_p + i] = val
            for u in range(UNROLL):
                body.append(("store", ("vstore", tmp_idx_addr_u[u], v_idx[u])))
                body.append(("store", ("vstore", tmp_val_addr_u[u], v_val[u])))

            # advance addresses for next group
            body.append(
                ("alu", ("+", tmp_idx_addr, tmp_idx_addr, group_size_const))
            )
            body.append(
                ("alu", ("+", tmp_val_addr, tmp_val_addr, group_size_const))
            )

        for i in range(vec_batch_unrolled, vec_batch, VLEN):
            # idx/val = vload once, then run all rounds in scratch
            body.append(("load", ("vload", v_idx[0], tmp_idx_addr)))
            if emit_debug:
                keys = [(0, i + lane, "idx") for lane in range(VLEN)]
                body.append(("debug", ("vcompare", v_idx[0], keys)))
            body.append(("load", ("vload", v_val[0], tmp_val_addr)))
            if emit_debug:
                keys = [(0, i + lane, "val") for lane in range(VLEN)]
                body.append(("debug", ("vcompare", v_val[0], keys)))
            for round in range(rounds):
                # node_val = gather(forest_values_p + idx)
                body.append(("valu", ("+", v_node_addr[0], v_idx[0], v_forest_values_p)))
                for lane in range(VLEN):
                    body.append(
                        ("load", ("load_offset", v_node_val[0], v_node_addr[0], lane))
                    )
                if emit_debug:
                    keys = [(round, i + lane, "node_val") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", v_node_val[0], keys)))
                # val = myhash(val ^ node_val)
                body.append(("valu", ("^", v_val[0], v_val[0], v_node_val[0])))
                body.extend(
                    self.build_hash_vec(v_val[0], v_tmp1[0], v_tmp2[0], round, i, emit_debug)
                )
                if emit_debug:
                    keys = [(round, i + lane, "hashed_val") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", v_val[0], keys)))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("valu", ("%", v_tmp1[0], v_val[0], v_two)))
                body.append(("valu", ("==", v_tmp1[0], v_tmp1[0], v_zero)))
                body.append(("valu", ("-", v_tmp3[0], v_two, v_tmp1[0])))
                body.append(("valu", ("*", v_idx[0], v_idx[0], v_two)))
                body.append(("valu", ("+", v_idx[0], v_idx[0], v_tmp3[0])))
                if emit_debug:
                    keys = [(round, i + lane, "next_idx") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", v_idx[0], keys)))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", v_tmp1[0], v_idx[0], v_n_nodes)))
                body.append(("valu", ("*", v_idx[0], v_idx[0], v_tmp1[0])))
                if emit_debug:
                    keys = [(round, i + lane, "wrapped_idx") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", v_idx[0], keys)))
            # mem[inp_indices_p + i] = idx
            body.append(("store", ("vstore", tmp_idx_addr, v_idx[0])))
            # mem[inp_values_p + i] = val
            body.append(("store", ("vstore", tmp_val_addr, v_val[0])))
            # advance addresses for next block
            body.append(("alu", ("+", tmp_idx_addr, tmp_idx_addr, vlen_const)))
            body.append(("alu", ("+", tmp_val_addr, tmp_val_addr, vlen_const)))

        for i in range(vec_batch, batch_size):
            # idx/val = load once, then run all rounds in scratch
            body.append(("load", ("load", tmp_idx, tmp_idx_addr)))
            if emit_debug:
                body.append(("debug", ("compare", tmp_idx, (0, i, "idx"))))
            body.append(("load", ("load", tmp_val, tmp_val_addr)))
            if emit_debug:
                body.append(("debug", ("compare", tmp_val, (0, i, "val"))))
            for round in range(rounds):
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
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i, emit_debug))
                if emit_debug:
                    body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("alu", ("-", tmp3, two_const, tmp1)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                if emit_debug:
                    body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("alu", ("*", tmp_idx, tmp_idx, tmp1)))
                if emit_debug:
                    body.append(
                        ("debug", ("compare", tmp_idx, (round, i, "wrapped_idx")))
                    )
            # mem[inp_indices_p + i] = idx
            body.append(("store", ("store", tmp_idx_addr, tmp_idx)))
            # mem[inp_values_p + i] = val
            body.append(("store", ("store", tmp_val_addr, tmp_val)))
            # advance addresses for next i
            body.append(("alu", ("+", tmp_idx_addr, tmp_idx_addr, one_const)))
            body.append(("alu", ("+", tmp_val_addr, tmp_val_addr, one_const)))

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
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
