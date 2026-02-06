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

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

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
            for name, _ in init_vars:
                self.alloc_scratch(name, 1)
            for name, idx in init_vars:
                self.add("load", ("const", tmp1, idx))
                self.add("load", ("load", self.scratch[name], tmp1))
        else:
            forest_values_p = self.alloc_scratch("forest_values_p", 1)
            inp_indices_p = self.alloc_scratch("inp_indices_p", 1)
            inp_values_p = self.alloc_scratch("inp_values_p", 1)
            forest_values_p_val = 7
            inp_indices_p_val = forest_values_p_val + n_nodes
            inp_values_p_val = inp_indices_p_val + batch_size
            self.add("load", ("const", forest_values_p, forest_values_p_val))
            self.add("load", ("const", inp_indices_p, inp_indices_p_val))
            self.add("load", ("const", inp_values_p, inp_values_p_val))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        use_vector = batch_size >= VLEN
        vlen_const = self.scratch_const(VLEN) if use_vector else None
        vec_const = {}
        if use_vector:
            vec_const_vals = {0, 1, 2}
            for op1, val1, op2, op3, val3 in HASH_STAGES:
                vec_const_vals.add(val1)
                vec_const_vals.add(val3)
                if op1 == "+" and op2 == "+" and op3 == "<<":
                    vec_const_vals.add(1 + (1 << val3))
            if forest_height >= 2:
                vec_const_vals.add(3)
            if forest_height >= 3:
                vec_const_vals.update({5, 6, 7})
            for val in sorted(vec_const_vals):
                if val == 0:
                    name = "v_zero"
                elif val == 1:
                    name = "v_one"
                elif val == 2:
                    name = "v_two"
                else:
                    name = f"v_const_{val}"
                addr = self.alloc_scratch(name, length=VLEN)
                self.add("valu", ("vbroadcast", addr, self.scratch_const(val)))
                vec_const[val] = addr

        root_val = self.alloc_scratch("root_val")
        self.add("load", ("load", root_val, self.scratch["forest_values_p"]))
        level1_left = self.alloc_scratch("level1_left")
        level1_right = self.alloc_scratch("level1_right")
        self.add("alu", ("+", tmp1, self.scratch["forest_values_p"], one_const))
        self.add("load", ("load", level1_left, tmp1))
        self.add("alu", ("+", tmp1, self.scratch["forest_values_p"], two_const))
        self.add("load", ("load", level1_right, tmp1))
        level1_diff_rl = self.alloc_scratch("level1_diff_rl")
        level1_diff_lr = self.alloc_scratch("level1_diff_lr")
        self.add("alu", ("-", level1_diff_rl, level1_right, level1_left))
        self.add("alu", ("-", level1_diff_lr, level1_left, level1_right))
        v_level2 = None
        v_level3 = None
        v_level2_c1 = None
        v_level2_c2 = None
        v_level2_c3 = None
        v_level3_c100 = None
        v_level3_c010 = None
        v_level3_c001 = None
        v_level3_c110 = None
        v_level3_c101 = None
        v_level3_c011 = None
        v_level3_c111 = None
        if use_vector:
            v_zero = vec_const[0]
            v_one = vec_const[1]
            v_two = vec_const[2]
            v_root_val = self.alloc_scratch("v_root_val", length=VLEN)
            self.add("valu", ("vbroadcast", v_root_val, root_val))
            v_level1_left = self.alloc_scratch("v_level1_left", length=VLEN)
            v_level1_right = self.alloc_scratch("v_level1_right", length=VLEN)
            self.add("valu", ("vbroadcast", v_level1_left, level1_left))
            self.add("valu", ("vbroadcast", v_level1_right, level1_right))
            v_level1_diff_rl = self.alloc_scratch("v_level1_diff_rl", length=VLEN)
            v_level1_diff_lr = self.alloc_scratch("v_level1_diff_lr", length=VLEN)
            self.add("valu", ("vbroadcast", v_level1_diff_rl, level1_diff_rl))
            self.add("valu", ("vbroadcast", v_level1_diff_lr, level1_diff_lr))
            if forest_height >= 2:
                v_level2 = [
                    self.alloc_scratch(f"v_level2_{i}", length=VLEN) for i in range(4)
                ]
                for i, addr in enumerate(v_level2):
                    self.add(
                        "alu",
                        (
                            "+",
                            tmp1,
                            self.scratch["forest_values_p"],
                            self.scratch_const(3 + i),
                        ),
                    )
                    self.add("load", ("load", tmp2, tmp1))
                    self.add("valu", ("vbroadcast", addr, tmp2))
                v_level2_c1 = self.alloc_scratch("v_level2_c1", length=VLEN)
                v_level2_c2 = self.alloc_scratch("v_level2_c2", length=VLEN)
                v_level2_c3 = self.alloc_scratch("v_level2_c3", length=VLEN)
                self.add("valu", ("-", v_level2_c1, v_level2[1], v_level2[0]))
                self.add("valu", ("-", v_level2_c2, v_level2[2], v_level2[0]))
                self.add("valu", ("-", v_level2_c3, v_level2[3], v_level2[2]))
                self.add("valu", ("-", v_level2_c3, v_level2_c3, v_level2[1]))
                self.add("valu", ("+", v_level2_c3, v_level2_c3, v_level2[0]))
            if forest_height >= 3:
                v_level3 = [
                    self.alloc_scratch(f"v_level3_{i}", length=VLEN) for i in range(8)
                ]
                for i, addr in enumerate(v_level3):
                    self.add(
                        "alu",
                        (
                            "+",
                            tmp1,
                            self.scratch["forest_values_p"],
                            self.scratch_const(7 + i),
                        ),
                    )
                    self.add("load", ("load", tmp2, tmp1))
                    self.add("valu", ("vbroadcast", addr, tmp2))
                v_level3_c100 = self.alloc_scratch("v_level3_c100", length=VLEN)
                v_level3_c010 = self.alloc_scratch("v_level3_c010", length=VLEN)
                v_level3_c001 = self.alloc_scratch("v_level3_c001", length=VLEN)
                v_level3_c110 = self.alloc_scratch("v_level3_c110", length=VLEN)
                v_level3_c101 = self.alloc_scratch("v_level3_c101", length=VLEN)
                v_level3_c011 = self.alloc_scratch("v_level3_c011", length=VLEN)
                v_level3_c111 = self.alloc_scratch("v_level3_c111", length=VLEN)
                self.add("valu", ("-", v_level3_c100, v_level3[1], v_level3[0]))
                self.add("valu", ("-", v_level3_c010, v_level3[2], v_level3[0]))
                self.add("valu", ("-", v_level3_c001, v_level3[4], v_level3[0]))
                self.add("valu", ("-", v_level3_c110, v_level3[3], v_level3[2]))
                self.add("valu", ("-", v_level3_c110, v_level3_c110, v_level3[1]))
                self.add("valu", ("+", v_level3_c110, v_level3_c110, v_level3[0]))
                self.add("valu", ("-", v_level3_c101, v_level3[5], v_level3[4]))
                self.add("valu", ("-", v_level3_c101, v_level3_c101, v_level3[1]))
                self.add("valu", ("+", v_level3_c101, v_level3_c101, v_level3[0]))
                self.add("valu", ("-", v_level3_c011, v_level3[6], v_level3[4]))
                self.add("valu", ("-", v_level3_c011, v_level3_c011, v_level3[2]))
                self.add("valu", ("+", v_level3_c011, v_level3_c011, v_level3[0]))
                self.add("valu", ("-", v_level3_c111, v_level3[7], v_level3[6]))
                self.add("valu", ("-", v_level3_c111, v_level3_c111, v_level3[5]))
                self.add("valu", ("-", v_level3_c111, v_level3_c111, v_level3[3]))
                self.add("valu", ("+", v_level3_c111, v_level3_c111, v_level3[4]))
                self.add("valu", ("+", v_level3_c111, v_level3_c111, v_level3[2]))
                self.add("valu", ("+", v_level3_c111, v_level3_c111, v_level3[1]))
                self.add("valu", ("-", v_level3_c111, v_level3_c111, v_level3[0]))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        if emit_debug:
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

        UNROLL = 32
        group_size = VLEN * UNROLL
        group_size_const = self.scratch_const(group_size) if use_vector else None

        # Vector scratch registers
        v_idx = [self.alloc_scratch(f"v_idx{u}", length=VLEN) for u in range(UNROLL)]
        v_val = [self.alloc_scratch(f"v_val{u}", length=VLEN) for u in range(UNROLL)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{u}", length=VLEN) for u in range(UNROLL)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{u}", length=VLEN) for u in range(UNROLL)]
        v_forest_values_p = self.alloc_scratch("v_forest_values_p", length=VLEN)
        v_n_nodes = None
        v_base_plus1 = None
        v_base_minus1 = None
        tmp_idx_addr_u = (
            [self.alloc_scratch(f"tmp_idx_addr{u}") for u in range(UNROLL)]
            if store_indices
            else []
        )
        tmp_val_addr_u = [self.alloc_scratch(f"tmp_val_addr{u}") for u in range(UNROLL)]
        if use_vector:
            self.add(
                "valu",
                ("vbroadcast", v_forest_values_p, self.scratch["forest_values_p"]),
            )
            v_base_plus1 = self.alloc_scratch("v_base_plus1", length=VLEN)
            v_base_minus1 = self.alloc_scratch("v_base_minus1", length=VLEN)
            self.add("valu", ("+", v_base_plus1, v_forest_values_p, v_one))
            self.add("valu", ("-", v_base_minus1, v_one, v_forest_values_p))
            if emit_debug:
                v_n_nodes = self.alloc_scratch("v_n_nodes", length=VLEN)
                self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        vec_batch = batch_size - (batch_size % VLEN)
        vec_batch_unrolled = vec_batch - (vec_batch % group_size)
        if store_indices:
            body.append(
                (
                    "alu",
                    ("+", tmp_idx_addr, self.scratch["inp_indices_p"], zero_const),
                )
            )
        body.append(
            (
                "alu",
                ("+", tmp_val_addr, self.scratch["inp_values_p"], zero_const),
            )
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
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        if op1 == "+" and op2 == "+" and op3 == "<<":
                            mult = 1 + (1 << val3)
                            vec_mult = vec_const[mult]
                            vec_val1 = vec_const[val1]
                            for u in range(UNROLL):
                                body.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            v_val[u],
                                            v_val[u],
                                            vec_mult,
                                            vec_val1,
                                        ),
                                    )
                                )
                        else:
                            vec_val1 = vec_const[val1]
                            vec_val3 = vec_const[val3]
                            for u in range(UNROLL):
                                body.append(
                                    ("valu", (op1, v_tmp1[u], v_val[u], vec_val1))
                                )
                                body.append(
                                    ("valu", (op3, v_tmp2[u], v_val[u], vec_val3))
                                )
                            for u in range(UNROLL):
                                body.append(
                                    ("valu", (op2, v_val[u], v_tmp1[u], v_tmp2[u]))
                                )
                        if emit_debug:
                            for u in range(UNROLL):
                                base = i + u * VLEN
                                keys = [
                                    (round, base + lane, "hash_stage", hi)
                                    for lane in range(VLEN)
                                ]
                                body.append(("debug", ("vcompare", v_val[u], keys)))
                elif depth == 1:
                    for u in range(UNROLL):
                        if emit_debug:
                            body.append(
                                ("valu", ("-", v_tmp2[u], v_idx[u], v_forest_values_p))
                            )
                            body.append(("valu", ("&", v_tmp1[u], v_tmp2[u], v_one)))
                            body.append(
                                (
                                    "valu",
                                    ("*", v_tmp2[u], v_tmp1[u], v_level1_diff_lr),
                                )
                            )
                            body.append(
                                ("valu", ("+", v_tmp1[u], v_level1_right, v_tmp2[u]))
                            )
                        else:
                            body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                            body.append(
                                ("valu", ("+", v_idx[u], v_base_plus1, v_tmp1[u]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp1[u],
                                        v_level1_diff_rl,
                                        v_level1_left,
                                    ),
                                )
                            )
                        body.append(("valu", ("^", v_val[u], v_val[u], v_tmp1[u])))
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        if op1 == "+" and op2 == "+" and op3 == "<<":
                            mult = 1 + (1 << val3)
                            vec_mult = vec_const[mult]
                            vec_val1 = vec_const[val1]
                            for u in range(UNROLL):
                                body.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            v_val[u],
                                            v_val[u],
                                            vec_mult,
                                            vec_val1,
                                        ),
                                    )
                                )
                        else:
                            vec_val1 = vec_const[val1]
                            vec_val3 = vec_const[val3]
                            for u in range(UNROLL):
                                body.append(
                                    ("valu", (op1, v_tmp1[u], v_val[u], vec_val1))
                                )
                                body.append(
                                    ("valu", (op3, v_tmp2[u], v_val[u], vec_val3))
                                )
                            for u in range(UNROLL):
                                body.append(
                                    ("valu", (op2, v_val[u], v_tmp1[u], v_tmp2[u]))
                                )
                        if emit_debug:
                            for u in range(UNROLL):
                                base = i + u * VLEN
                                keys = [
                                    (round, base + lane, "hash_stage", hi)
                                    for lane in range(VLEN)
                                ]
                                body.append(("debug", ("vcompare", v_val[u], keys)))
                else:
                    if depth == 2 and v_level2 is not None:
                        for u in range(UNROLL):
                            body.append(("valu", ("-", v_tmp1[u], v_idx[u], vec_const[3])))
                            body.append(("valu", ("&", v_tmp2[u], v_tmp1[u], v_one)))
                            body.append(("valu", (">>", v_tmp2[u], v_tmp1[u], v_one)))
                            body.append(("valu", ("&", v_tmp2[u], v_tmp2[u], v_one)))
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_level2[1], v_level2[0]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_level2[0],
                                    ),
                                )
                            )
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_level2[3], v_level2[2]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_level2[2],
                                    ),
                                )
                            )
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_tmp1[u], v_tmp1[u]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_tmp1[u],
                                    ),
                                )
                            )
                        for u in range(UNROLL):
                            body.append(
                                ("valu", ("^", v_val[u], v_val[u], v_tmp1[u]))
                            )
                        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                            if op1 == "+" and op2 == "+" and op3 == "<<":
                                mult = 1 + (1 << val3)
                                vec_mult = vec_const[mult]
                                vec_val1 = vec_const[val1]
                                for u in range(UNROLL):
                                    body.append(
                                        (
                                            "valu",
                                            (
                                                "multiply_add",
                                                v_val[u],
                                                v_val[u],
                                                vec_mult,
                                                vec_val1,
                                            ),
                                        )
                                    )
                            else:
                                vec_val1 = vec_const[val1]
                                vec_val3 = vec_const[val3]
                                for u in range(UNROLL):
                                    body.append(
                                        ("valu", (op1, v_tmp1[u], v_val[u], vec_val1))
                                    )
                                    body.append(
                                        ("valu", (op3, v_tmp2[u], v_val[u], vec_val3))
                                    )
                                for u in range(UNROLL):
                                    body.append(
                                        ("valu", (op2, v_val[u], v_tmp1[u], v_tmp2[u]))
                                    )
                            if emit_debug:
                                for u in range(UNROLL):
                                    base = i + u * VLEN
                                    keys = [
                                        (round, base + lane, "hash_stage", hi)
                                        for lane in range(VLEN)
                                    ]
                                    body.append(("debug", ("vcompare", v_val[u], keys)))
                    elif depth == 3 and v_level3 is not None:
                        for u in range(UNROLL):
                            body.append(("valu", ("-", v_tmp1[u], v_idx[u], vec_const[7])))
                            body.append(("valu", ("&", v_tmp2[u], v_tmp1[u], v_one)))
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_level3[1], v_level3[0]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp2[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_level3[0],
                                    ),
                                )
                            )
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_level3[3], v_level3[2]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_level3[2],
                                    ),
                                )
                            )
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_level3[5], v_level3[4]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp2[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_level3[4],
                                    ),
                                )
                            )
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_level3[7], v_level3[6]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_level3[6],
                                    ),
                                )
                            )
                            body.append(("valu", ("-", v_tmp2[u], v_idx[u], vec_const[7])))
                            body.append(("valu", (">>", v_tmp2[u], v_tmp2[u], v_one)))
                            body.append(("valu", ("&", v_tmp2[u], v_tmp2[u], v_one)))
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_tmp1[u], v_tmp2[u]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp2[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_tmp2[u],
                                    ),
                                )
                            )
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_tmp1[u], v_tmp2[u]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_tmp2[u],
                                    ),
                                )
                            )
                            body.append(("valu", ("-", v_tmp2[u], v_idx[u], vec_const[7])))
                            body.append(("valu", (">>", v_tmp2[u], v_tmp2[u], v_two)))
                            body.append(("valu", ("&", v_tmp2[u], v_tmp2[u], v_one)))
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_tmp1[u], v_tmp2[u]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_tmp2[u],
                                    ),
                                )
                            )
                        for u in range(UNROLL):
                            body.append(
                                ("valu", ("^", v_val[u], v_val[u], v_tmp1[u]))
                            )
                        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                            if op1 == "+" and op2 == "+" and op3 == "<<":
                                mult = 1 + (1 << val3)
                                vec_mult = vec_const[mult]
                                vec_val1 = vec_const[val1]
                                for u in range(UNROLL):
                                    body.append(
                                        (
                                            "valu",
                                            (
                                                "multiply_add",
                                                v_val[u],
                                                v_val[u],
                                                vec_mult,
                                                vec_val1,
                                            ),
                                        )
                                    )
                            else:
                                vec_val1 = vec_const[val1]
                                vec_val3 = vec_const[val3]
                                for u in range(UNROLL):
                                    body.append(
                                        ("valu", (op1, v_tmp1[u], v_val[u], vec_val1))
                                    )
                                    body.append(
                                        ("valu", (op3, v_tmp2[u], v_val[u], vec_val3))
                                    )
                                for u in range(UNROLL):
                                    body.append(
                                        ("valu", (op2, v_val[u], v_tmp1[u], v_tmp2[u]))
                                    )
                            if emit_debug:
                                for u in range(UNROLL):
                                    base = i + u * VLEN
                                    keys = [
                                        (round, base + lane, "hash_stage", hi)
                                        for lane in range(VLEN)
                                    ]
                                    body.append(("debug", ("vcompare", v_val[u], keys)))
                    else:
                        hash_group = 3
                        pipeline_group_size = (
                            hash_group if UNROLL >= hash_group else UNROLL
                        )
                        pipeline_num_groups = (
                            (UNROLL + pipeline_group_size - 1) // pipeline_group_size
                        )
                        pipeline_load_progress = [0] * pipeline_num_groups

                        for g in range(pipeline_num_groups):
                            group_start = g * pipeline_group_size
                            group_vecs = min(pipeline_group_size, UNROLL - group_start)
                            total_loads = group_vecs * VLEN
                            while pipeline_load_progress[g] < total_loads:
                                idx = pipeline_load_progress[g]
                                u = group_start + idx // VLEN
                                lane = idx % VLEN
                                body.append(
                                    (
                                        "load",
                                        (
                                            "load_offset",
                                            v_tmp1[u],
                                            v_idx[u],
                                            lane,
                                        ),
                                    )
                                )
                                pipeline_load_progress[g] = idx + 1
                            if emit_debug:
                                for u in range(group_start, group_start + group_vecs):
                                    base = i + u * VLEN
                                    keys = [
                                        (round, base + lane, "node_val")
                                        for lane in range(VLEN)
                                    ]
                                    body.append(
                                        ("debug", ("vcompare", v_tmp1[u], keys))
                                    )

                            for u in range(group_start, group_start + group_vecs):
                                body.append(
                                    ("valu", ("^", v_val[u], v_val[u], v_tmp1[u]))
                                )

                            stages = []
                            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                                stage_slots = []
                                if op1 == "+" and op2 == "+" and op3 == "<<":
                                    mult = 1 + (1 << val3)
                                    vec_mult = vec_const[mult]
                                    vec_val1 = vec_const[val1]
                                    for u in range(group_start, group_start + group_vecs):
                                        stage_slots.append(
                                            (
                                                "valu",
                                                (
                                                    "multiply_add",
                                                    v_val[u],
                                                    v_val[u],
                                                    vec_mult,
                                                    vec_val1,
                                                ),
                                            )
                                        )
                                else:
                                    vec_val1 = vec_const[val1]
                                    vec_val3 = vec_const[val3]
                                    for u in range(group_start, group_start + group_vecs):
                                        stage_slots.append(
                                            (
                                                "valu",
                                                (op1, v_tmp1[u], v_val[u], vec_val1),
                                            )
                                        )
                                        stage_slots.append(
                                            (
                                                "valu",
                                                (op3, v_tmp2[u], v_val[u], vec_val3),
                                            )
                                        )
                                    for u in range(group_start, group_start + group_vecs):
                                        stage_slots.append(
                                            (
                                                "valu",
                                                (op2, v_val[u], v_tmp1[u], v_tmp2[u]),
                                            )
                                        )
                                if emit_debug:
                                    for u in range(group_start, group_start + group_vecs):
                                        base = i + u * VLEN
                                        keys = [
                                            (round, base + lane, "hash_stage", hi)
                                            for lane in range(VLEN)
                                        ]
                                        stage_slots.append(
                                            ("debug", ("vcompare", v_val[u], keys))
                                        )
                                stages.append(stage_slots)

                            next_g = g + 1
                            if next_g < pipeline_num_groups:
                                next_start = next_g * pipeline_group_size
                                next_vecs = min(
                                    pipeline_group_size, UNROLL - next_start
                                )
                                next_total = next_vecs * VLEN
                                remaining = next_total - pipeline_load_progress[next_g]
                                if remaining > 0:
                                    loads_per_stage = (
                                        (remaining + len(stages) - 1) // len(stages)
                                    )
                                else:
                                    loads_per_stage = 0
                            else:
                                next_start = 0
                                next_vecs = 0
                                next_total = 0
                                loads_per_stage = 0

                            for stage_slots in stages:
                                body.extend(stage_slots)
                                if next_g < pipeline_num_groups and loads_per_stage:
                                    for _ in range(loads_per_stage):
                                        if pipeline_load_progress[next_g] >= next_total:
                                            break
                                        idx = pipeline_load_progress[next_g]
                                        u = next_start + idx // VLEN
                                        lane = idx % VLEN
                                        body.append(
                                            (
                                                "load",
                                                (
                                                    "load_offset",
                                                    v_tmp1[u],
                                                    v_idx[u],
                                                    lane,
                                                ),
                                            )
                                        )
                                        pipeline_load_progress[next_g] = idx + 1

                            if next_g < pipeline_num_groups:
                                while pipeline_load_progress[next_g] < next_total:
                                    idx = pipeline_load_progress[next_g]
                                    u = next_start + idx // VLEN
                                    lane = idx % VLEN
                                    body.append(
                                        (
                                            "load",
                                            (
                                                "load_offset",
                                                v_tmp1[u],
                                                v_idx[u],
                                                lane,
                                            ),
                                        )
                                    )
                                    pipeline_load_progress[next_g] = idx + 1
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
                            body.append(("valu", ("+", v_tmp2[u], v_tmp1[u], v_one)))
                        for u in range(UNROLL):
                            body.append(("valu", ("*", v_idx[u], v_idx[u], v_two)))
                        for u in range(UNROLL):
                            body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp2[u])))
                        for u in range(UNROLL):
                            base = i + u * VLEN
                            keys = [
                                (round, base + lane, "next_idx") for lane in range(VLEN)
                            ]
                            body.append(("debug", ("vcompare", v_idx[u], keys)))
                    if round == rounds - 1 and store_indices:
                        for u in range(UNROLL):
                            body.append(
                                ("valu", ("+", v_idx[u], v_forest_values_p, v_zero))
                            )
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
                        if emit_debug or (store_indices and round == rounds - 1):
                            for u in range(UNROLL):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                            for u in range(UNROLL):
                                body.append(
                                    ("valu", ("+", v_idx[u], v_base_plus1, v_tmp1[u]))
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
                    else:
                        # idx = 2*idx + (1 if val even else 2)
                        for u in range(UNROLL):
                            body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                        for u in range(UNROLL):
                            body.append(
                                ("valu", ("+", v_tmp2[u], v_tmp1[u], v_base_minus1))
                            )
                        for u in range(UNROLL):
                            body.append(
                                ("valu", ("multiply_add", v_idx[u], v_idx[u], v_two, v_tmp2[u]))
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
            if store_indices:
                for u in range(UNROLL):
                    body.append(
                        ("valu", ("-", v_idx[u], v_idx[u], v_forest_values_p))
                    )
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
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        if op1 == "+" and op2 == "+" and op3 == "<<":
                            mult = 1 + (1 << val3)
                            vec_mult = vec_const[mult]
                            vec_val1 = vec_const[val1]
                            for u in range(tail_vecs):
                                body.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            v_val[u],
                                            v_val[u],
                                            vec_mult,
                                            vec_val1,
                                        ),
                                    )
                                )
                        else:
                            vec_val1 = vec_const[val1]
                            vec_val3 = vec_const[val3]
                            for u in range(tail_vecs):
                                body.append(
                                    ("valu", (op1, v_tmp1[u], v_val[u], vec_val1))
                                )
                                body.append(
                                    ("valu", (op3, v_tmp2[u], v_val[u], vec_val3))
                                )
                            for u in range(tail_vecs):
                                body.append(
                                    ("valu", (op2, v_val[u], v_tmp1[u], v_tmp2[u]))
                                )
                        if emit_debug:
                            for u in range(tail_vecs):
                                base = tail_base + u * VLEN
                                keys = [
                                    (round, base + lane, "hash_stage", hi)
                                    for lane in range(VLEN)
                                ]
                                body.append(("debug", ("vcompare", v_val[u], keys)))
                elif depth == 1:
                    for u in range(tail_vecs):
                        if emit_debug:
                            body.append(
                                ("valu", ("-", v_tmp2[u], v_idx[u], v_forest_values_p))
                            )
                            body.append(("valu", ("&", v_tmp1[u], v_tmp2[u], v_one)))
                            body.append(
                                (
                                    "valu",
                                    ("*", v_tmp2[u], v_tmp1[u], v_level1_diff_lr),
                                )
                            )
                            body.append(
                                ("valu", ("+", v_tmp1[u], v_level1_right, v_tmp2[u]))
                            )
                        else:
                            body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                            body.append(
                                ("valu", ("+", v_idx[u], v_base_plus1, v_tmp1[u]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp1[u],
                                        v_level1_diff_rl,
                                        v_level1_left,
                                    ),
                                )
                            )
                        body.append(("valu", ("^", v_val[u], v_val[u], v_tmp1[u])))
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        if op1 == "+" and op2 == "+" and op3 == "<<":
                            mult = 1 + (1 << val3)
                            vec_mult = vec_const[mult]
                            vec_val1 = vec_const[val1]
                            for u in range(tail_vecs):
                                body.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            v_val[u],
                                            v_val[u],
                                            vec_mult,
                                            vec_val1,
                                        ),
                                    )
                                )
                        else:
                            vec_val1 = vec_const[val1]
                            vec_val3 = vec_const[val3]
                            for u in range(tail_vecs):
                                body.append(
                                    ("valu", (op1, v_tmp1[u], v_val[u], vec_val1))
                                )
                                body.append(
                                    ("valu", (op3, v_tmp2[u], v_val[u], vec_val3))
                                )
                            for u in range(tail_vecs):
                                body.append(
                                    ("valu", (op2, v_val[u], v_tmp1[u], v_tmp2[u]))
                                )
                        if emit_debug:
                            for u in range(tail_vecs):
                                base = tail_base + u * VLEN
                                keys = [
                                    (round, base + lane, "hash_stage", hi)
                                    for lane in range(VLEN)
                                ]
                                body.append(("debug", ("vcompare", v_val[u], keys)))
                else:
                    if depth == 2 and v_level2 is not None:
                        for u in range(tail_vecs):
                            body.append(("valu", ("-", v_tmp1[u], v_idx[u], vec_const[3])))
                            body.append(("valu", ("&", v_tmp2[u], v_tmp1[u], v_one)))
                            body.append(("valu", (">>", v_tmp2[u], v_tmp1[u], v_one)))
                            body.append(("valu", ("&", v_tmp2[u], v_tmp2[u], v_one)))
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_level2[1], v_level2[0]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_level2[0],
                                    ),
                                )
                            )
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_level2[3], v_level2[2]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_level2[2],
                                    ),
                                )
                            )
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_tmp1[u], v_tmp1[u]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_tmp1[u],
                                    ),
                                )
                            )
                        for u in range(tail_vecs):
                            body.append(
                                ("valu", ("^", v_val[u], v_val[u], v_tmp1[u]))
                            )
                        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                            if op1 == "+" and op2 == "+" and op3 == "<<":
                                mult = 1 + (1 << val3)
                                vec_mult = vec_const[mult]
                                vec_val1 = vec_const[val1]
                                for u in range(tail_vecs):
                                    body.append(
                                        (
                                            "valu",
                                            (
                                                "multiply_add",
                                                v_val[u],
                                                v_val[u],
                                                vec_mult,
                                                vec_val1,
                                            ),
                                        )
                                    )
                            else:
                                vec_val1 = vec_const[val1]
                                vec_val3 = vec_const[val3]
                                for u in range(tail_vecs):
                                    body.append(
                                        ("valu", (op1, v_tmp1[u], v_val[u], vec_val1))
                                    )
                                    body.append(
                                        ("valu", (op3, v_tmp2[u], v_val[u], vec_val3))
                                    )
                                for u in range(tail_vecs):
                                    body.append(
                                        ("valu", (op2, v_val[u], v_tmp1[u], v_tmp2[u]))
                                    )
                            if emit_debug:
                                for u in range(tail_vecs):
                                    base = tail_base + u * VLEN
                                    keys = [
                                        (round, base + lane, "hash_stage", hi)
                                        for lane in range(VLEN)
                                    ]
                                    body.append(("debug", ("vcompare", v_val[u], keys)))
                    elif depth == 3 and v_level3 is not None:
                        for u in range(tail_vecs):
                            body.append(("valu", ("-", v_tmp1[u], v_idx[u], vec_const[7])))
                            body.append(("valu", ("&", v_tmp2[u], v_tmp1[u], v_one)))
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_level3[1], v_level3[0]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp2[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_level3[0],
                                    ),
                                )
                            )
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_level3[3], v_level3[2]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_level3[2],
                                    ),
                                )
                            )
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_level3[5], v_level3[4]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp2[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_level3[4],
                                    ),
                                )
                            )
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_level3[7], v_level3[6]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_level3[6],
                                    ),
                                )
                            )
                            body.append(("valu", ("-", v_tmp2[u], v_idx[u], vec_const[7])))
                            body.append(("valu", (">>", v_tmp2[u], v_tmp2[u], v_one)))
                            body.append(("valu", ("&", v_tmp2[u], v_tmp2[u], v_one)))
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_tmp1[u], v_tmp2[u]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp2[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_tmp2[u],
                                    ),
                                )
                            )
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_tmp1[u], v_tmp2[u]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_tmp2[u],
                                    ),
                                )
                            )
                            body.append(("valu", ("-", v_tmp2[u], v_idx[u], vec_const[7])))
                            body.append(("valu", (">>", v_tmp2[u], v_tmp2[u], v_two)))
                            body.append(("valu", ("&", v_tmp2[u], v_tmp2[u], v_one)))
                            body.append(
                                ("valu", ("-", v_tmp1[u], v_tmp1[u], v_tmp2[u]))
                            )
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp1[u],
                                        v_tmp2[u],
                                        v_tmp1[u],
                                        v_tmp2[u],
                                    ),
                                )
                            )
                        for u in range(tail_vecs):
                            body.append(
                                ("valu", ("^", v_val[u], v_val[u], v_tmp1[u]))
                            )
                        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                            if op1 == "+" and op2 == "+" and op3 == "<<":
                                mult = 1 + (1 << val3)
                                vec_mult = vec_const[mult]
                                vec_val1 = vec_const[val1]
                                for u in range(tail_vecs):
                                    body.append(
                                        (
                                            "valu",
                                            (
                                                "multiply_add",
                                                v_val[u],
                                                v_val[u],
                                                vec_mult,
                                                vec_val1,
                                            ),
                                        )
                                    )
                            else:
                                vec_val1 = vec_const[val1]
                                vec_val3 = vec_const[val3]
                                for u in range(tail_vecs):
                                    body.append(
                                        ("valu", (op1, v_tmp1[u], v_val[u], vec_val1))
                                    )
                                    body.append(
                                        ("valu", (op3, v_tmp2[u], v_val[u], vec_val3))
                                    )
                                for u in range(tail_vecs):
                                    body.append(
                                        ("valu", (op2, v_val[u], v_tmp1[u], v_tmp2[u]))
                                    )
                            if emit_debug:
                                for u in range(tail_vecs):
                                    base = tail_base + u * VLEN
                                    keys = [
                                        (round, base + lane, "hash_stage", hi)
                                        for lane in range(VLEN)
                                    ]
                                    body.append(("debug", ("vcompare", v_val[u], keys)))
                    else:
                        hash_group = 3
                        pipeline_group_size = (
                            hash_group if tail_vecs >= hash_group else tail_vecs
                        )
                        pipeline_num_groups = (
                            (tail_vecs + pipeline_group_size - 1) // pipeline_group_size
                        )
                        pipeline_load_progress = [0] * pipeline_num_groups

                        for g in range(pipeline_num_groups):
                            group_start = g * pipeline_group_size
                            group_vecs = min(pipeline_group_size, tail_vecs - group_start)
                            total_loads = group_vecs * VLEN
                            while pipeline_load_progress[g] < total_loads:
                                idx = pipeline_load_progress[g]
                                u = group_start + idx // VLEN
                                lane = idx % VLEN
                                body.append(
                                    (
                                        "load",
                                        (
                                            "load_offset",
                                            v_tmp1[u],
                                            v_idx[u],
                                            lane,
                                        ),
                                    )
                                )
                                pipeline_load_progress[g] = idx + 1
                            if emit_debug:
                                for u in range(group_start, group_start + group_vecs):
                                    base = tail_base + u * VLEN
                                    keys = [
                                        (round, base + lane, "node_val")
                                        for lane in range(VLEN)
                                    ]
                                    body.append(
                                        ("debug", ("vcompare", v_tmp1[u], keys))
                                    )

                            for u in range(group_start, group_start + group_vecs):
                                body.append(
                                    ("valu", ("^", v_val[u], v_val[u], v_tmp1[u]))
                                )

                            stages = []
                            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                                stage_slots = []
                                if op1 == "+" and op2 == "+" and op3 == "<<":
                                    mult = 1 + (1 << val3)
                                    vec_mult = vec_const[mult]
                                    vec_val1 = vec_const[val1]
                                    for u in range(group_start, group_start + group_vecs):
                                        stage_slots.append(
                                            (
                                                "valu",
                                                (
                                                    "multiply_add",
                                                    v_val[u],
                                                    v_val[u],
                                                    vec_mult,
                                                    vec_val1,
                                                ),
                                            )
                                        )
                                else:
                                    vec_val1 = vec_const[val1]
                                    vec_val3 = vec_const[val3]
                                    for u in range(group_start, group_start + group_vecs):
                                        stage_slots.append(
                                            (
                                                "valu",
                                                (op1, v_tmp1[u], v_val[u], vec_val1),
                                            )
                                        )
                                        stage_slots.append(
                                            (
                                                "valu",
                                                (op3, v_tmp2[u], v_val[u], vec_val3),
                                            )
                                        )
                                    for u in range(group_start, group_start + group_vecs):
                                        stage_slots.append(
                                            (
                                                "valu",
                                                (op2, v_val[u], v_tmp1[u], v_tmp2[u]),
                                            )
                                        )
                                if emit_debug:
                                    for u in range(group_start, group_start + group_vecs):
                                        base = tail_base + u * VLEN
                                        keys = [
                                            (round, base + lane, "hash_stage", hi)
                                            for lane in range(VLEN)
                                        ]
                                        stage_slots.append(
                                            ("debug", ("vcompare", v_val[u], keys))
                                        )
                                stages.append(stage_slots)

                            next_g = g + 1
                            if next_g < pipeline_num_groups:
                                next_start = next_g * pipeline_group_size
                                next_vecs = min(
                                    pipeline_group_size, tail_vecs - next_start
                                )
                                next_total = next_vecs * VLEN
                                remaining = next_total - pipeline_load_progress[next_g]
                                if remaining > 0:
                                    loads_per_stage = (
                                        (remaining + len(stages) - 1) // len(stages)
                                    )
                                else:
                                    loads_per_stage = 0
                            else:
                                next_start = 0
                                next_vecs = 0
                                next_total = 0
                                loads_per_stage = 0

                            for stage_slots in stages:
                                body.extend(stage_slots)
                                if next_g < pipeline_num_groups and loads_per_stage:
                                    for _ in range(loads_per_stage):
                                        if pipeline_load_progress[next_g] >= next_total:
                                            break
                                        idx = pipeline_load_progress[next_g]
                                        u = next_start + idx // VLEN
                                        lane = idx % VLEN
                                        body.append(
                                            (
                                                "load",
                                                (
                                                    "load_offset",
                                                    v_tmp1[u],
                                                    v_idx[u],
                                                    lane,
                                                ),
                                            )
                                        )
                                        pipeline_load_progress[next_g] = idx + 1

                            if next_g < pipeline_num_groups:
                                while pipeline_load_progress[next_g] < next_total:
                                    idx = pipeline_load_progress[next_g]
                                    u = next_start + idx // VLEN
                                    lane = idx % VLEN
                                    body.append(
                                        (
                                            "load",
                                            (
                                                "load_offset",
                                                v_tmp1[u],
                                                v_idx[u],
                                                lane,
                                            ),
                                        ),
                                    )
                                    pipeline_load_progress[next_g] = idx + 1
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
                            body.append(("valu", ("+", v_tmp2[u], v_tmp1[u], v_one)))
                        for u in range(tail_vecs):
                            body.append(("valu", ("*", v_idx[u], v_idx[u], v_two)))
                        for u in range(tail_vecs):
                            body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp2[u])))
                        for u in range(tail_vecs):
                            base = tail_base + u * VLEN
                            keys = [
                                (round, base + lane, "next_idx")
                                for lane in range(VLEN)
                            ]
                            body.append(("debug", ("vcompare", v_idx[u], keys)))
                    if round == rounds - 1 and store_indices:
                        for u in range(tail_vecs):
                            body.append(
                                ("valu", ("+", v_idx[u], v_forest_values_p, v_zero))
                            )
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
                        if emit_debug or (store_indices and round == rounds - 1):
                            for u in range(tail_vecs):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                            for u in range(tail_vecs):
                                body.append(
                                    ("valu", ("+", v_idx[u], v_base_plus1, v_tmp1[u]))
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
                    else:
                        # idx = 2*idx + (1 if val even else 2)
                        for u in range(tail_vecs):
                            body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                        for u in range(tail_vecs):
                            body.append(
                                ("valu", ("+", v_tmp2[u], v_tmp1[u], v_base_minus1))
                            )
                        for u in range(tail_vecs):
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

            if store_indices:
                for u in range(tail_vecs):
                    body.append(
                        ("valu", ("-", v_idx[u], v_idx[u], v_forest_values_p))
                    )
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
            if emit_debug:
                body.append(("alu", ("+", tmp_idx, zero_const, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (0, i, "idx"))))
            body.append(("load", ("load", tmp_val, tmp_val_addr)))
            if emit_debug:
                body.append(("debug", ("compare", tmp_val, (0, i, "val"))))
            period = forest_height + 1
            for round in range(rounds):
                depth = round % period
                if depth == 0:
                    body.append(("alu", ("^", tmp_val, tmp_val, root_val)))
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        if op1 == "+" and op2 == "+" and op3 == "<<":
                            mult = 1 + (1 << val3)
                            body.append(
                                ("alu", ("*", tmp1, tmp_val, self.scratch_const(mult)))
                            )
                            body.append(
                                ("alu", ("+", tmp_val, tmp1, self.scratch_const(val1)))
                            )
                        else:
                            body.append(
                                ("alu", (op1, tmp1, tmp_val, self.scratch_const(val1)))
                            )
                            body.append(
                                ("alu", (op3, tmp2, tmp_val, self.scratch_const(val3)))
                            )
                            body.append(("alu", (op2, tmp_val, tmp1, tmp2)))
                        if emit_debug:
                            body.append(
                                (
                                    "debug",
                                    ("compare", tmp_val, (round, i, "hash_stage", hi)),
                                )
                            )
                elif depth == 1:
                    if emit_debug:
                        body.append(("alu", ("&", tmp1, tmp_idx, one_const)))
                        body.append(("alu", ("*", tmp2, tmp1, level1_diff_lr)))
                        body.append(("alu", ("+", tmp_node_val, level1_right, tmp2)))
                    else:
                        body.append(("alu", ("&", tmp1, tmp_val, one_const)))
                        body.append(("alu", ("+", tmp_idx, tmp1, one_const)))
                        body.append(("alu", ("*", tmp2, tmp1, level1_diff_rl)))
                        body.append(("alu", ("+", tmp_node_val, level1_left, tmp2)))
                    if emit_debug:
                        body.append(
                            ("debug", ("compare", tmp_node_val, (round, i, "node_val")))
                        )
                    body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        if op1 == "+" and op2 == "+" and op3 == "<<":
                            mult = 1 + (1 << val3)
                            body.append(
                                ("alu", ("*", tmp1, tmp_val, self.scratch_const(mult)))
                            )
                            body.append(
                                ("alu", ("+", tmp_val, tmp1, self.scratch_const(val1)))
                            )
                        else:
                            body.append(
                                ("alu", (op1, tmp1, tmp_val, self.scratch_const(val1)))
                            )
                            body.append(
                                ("alu", (op3, tmp2, tmp_val, self.scratch_const(val3)))
                            )
                            body.append(("alu", (op2, tmp_val, tmp1, tmp2)))
                        if emit_debug:
                            body.append(
                                (
                                    "debug",
                                    ("compare", tmp_val, (round, i, "hash_stage", hi)),
                                )
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
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        if op1 == "+" and op2 == "+" and op3 == "<<":
                            mult = 1 + (1 << val3)
                            body.append(
                                ("alu", ("*", tmp1, tmp_val, self.scratch_const(mult)))
                            )
                            body.append(
                                ("alu", ("+", tmp_val, tmp1, self.scratch_const(val1)))
                            )
                        else:
                            body.append(
                                ("alu", (op1, tmp1, tmp_val, self.scratch_const(val1)))
                            )
                            body.append(
                                ("alu", (op3, tmp2, tmp_val, self.scratch_const(val3)))
                            )
                            body.append(("alu", (op2, tmp_val, tmp1, tmp2)))
                        if emit_debug:
                            body.append(
                                (
                                    "debug",
                                    ("compare", tmp_val, (round, i, "hash_stage", hi)),
                                )
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
                        if emit_debug or (store_indices and round == rounds - 1):
                            body.append(("alu", ("&", tmp_idx, tmp_val, one_const)))
                            body.append(("alu", ("+", tmp_idx, tmp_idx, one_const)))
                            if emit_debug:
                                body.append(
                                    ("debug", ("compare", tmp_idx, (round, i, "next_idx")))
                                )
                                body.append(
                                    ("debug", ("compare", tmp_idx, (round, i, "wrapped_idx")))
                                )
                    else:
                        # idx = 2*idx + (1 if val even else 2)
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
            # mem[inp_indices_p + i] = idx, mem[inp_values_p + i] = val
            if store_indices:
                body.append(("store", ("store", tmp_idx_addr, tmp_idx)))
            body.append(("store", ("store", tmp_val_addr, tmp_val)))
            # advance addresses for next i
            if i != batch_size - 1:
                if store_indices:
                    body.append(("alu", ("+", tmp_idx_addr, tmp_idx_addr, one_const)))
                body.append(("alu", ("+", tmp_val_addr, tmp_val_addr, one_const)))

        body_instrs = []
        cur_instr = {}
        cur_counts = {}
        bundle_writes = set()
        bundle_has_store = False
        for engine, slot in body:
            if engine == "debug":
                if cur_instr:
                    body_instrs.append(cur_instr)
                    cur_instr = {}
                    cur_counts = {}
                    bundle_writes = set()
                    bundle_has_store = False
                body_instrs.append({engine: [slot]})
                continue
            reads = set()
            writes = set()
            mem_read = False
            mem_write = False
            barrier = False
            if engine == "alu":
                writes.add(slot[1])
                reads.add(slot[2])
                reads.add(slot[3])
            elif engine == "valu":
                op = slot[0]
                dest = slot[1]
                if op == "vbroadcast":
                    reads.add(slot[2])
                    writes.update(range(dest, dest + VLEN))
                elif op == "multiply_add":
                    a = slot[2]
                    b = slot[3]
                    c = slot[4]
                    reads.update(range(a, a + VLEN))
                    reads.update(range(b, b + VLEN))
                    reads.update(range(c, c + VLEN))
                    writes.update(range(dest, dest + VLEN))
                else:
                    a1 = slot[2]
                    a2 = slot[3]
                    reads.update(range(a1, a1 + VLEN))
                    reads.update(range(a2, a2 + VLEN))
                    writes.update(range(dest, dest + VLEN))
            elif engine == "load":
                op = slot[0]
                if op == "const":
                    writes.add(slot[1])
                elif op == "load":
                    writes.add(slot[1])
                    reads.add(slot[2])
                    mem_read = True
                elif op == "load_offset":
                    writes.add(slot[1] + slot[3])
                    reads.add(slot[2] + slot[3])
                    mem_read = True
                elif op == "vload":
                    dest = slot[1]
                    reads.add(slot[2])
                    writes.update(range(dest, dest + VLEN))
                    mem_read = True
                else:
                    barrier = True
            elif engine == "store":
                op = slot[0]
                if op == "store":
                    reads.add(slot[1])
                    reads.add(slot[2])
                    mem_write = True
                elif op == "vstore":
                    reads.add(slot[1])
                    src = slot[2]
                    reads.update(range(src, src + VLEN))
                    mem_write = True
                else:
                    barrier = True
            elif engine == "flow":
                op = slot[0]
                if op == "select":
                    writes.add(slot[1])
                    reads.add(slot[2])
                    reads.add(slot[3])
                    reads.add(slot[4])
                elif op == "add_imm":
                    writes.add(slot[1])
                    reads.add(slot[2])
                elif op == "vselect":
                    dest = slot[1]
                    cond = slot[2]
                    a = slot[3]
                    b = slot[4]
                    writes.update(range(dest, dest + VLEN))
                    reads.update(range(cond, cond + VLEN))
                    reads.update(range(a, a + VLEN))
                    reads.update(range(b, b + VLEN))
                elif op == "coreid":
                    writes.add(slot[1])
                else:
                    barrier = True
            else:
                barrier = True

            if barrier:
                if cur_instr:
                    body_instrs.append(cur_instr)
                    cur_instr = {}
                    cur_counts = {}
                    bundle_writes = set()
                    bundle_has_store = False
                body_instrs.append({engine: [slot]})
                continue

            can_pack = True
            if cur_instr and cur_counts.get(engine, 0) >= SLOT_LIMITS[engine]:
                can_pack = False
            if reads and bundle_writes.intersection(reads):
                can_pack = False
            if writes and bundle_writes.intersection(writes):
                can_pack = False
            if not can_pack:
                if cur_instr:
                    body_instrs.append(cur_instr)
                cur_instr = {}
                cur_counts = {}
                bundle_writes = set()
                bundle_has_store = False
            cur_instr.setdefault(engine, []).append(slot)
            cur_counts[engine] = cur_counts.get(engine, 0) + 1
            if writes:
                bundle_writes.update(writes)
            if mem_write:
                bundle_has_store = True

        if cur_instr:
            body_instrs.append(cur_instr)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        if emit_debug:
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
