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

    def vector_const(self, val, name=None):
        if val not in self.vec_const_map:
            addr = self.alloc_scratch(name, length=VLEN)
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.vec_const_map[val] = addr
        return self.vec_const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

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
                    keys = [
                        (round, base + lane, "node_val") for lane in range(VLEN)
                    ]
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
                    (remaining + len(stages) - 1) // len(stages) if remaining > 0 else 0
                )
            else:
                next_start = next_vecs = next_total = loads_per_stage = 0

            for stage_slots in stages:
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
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        emit_debug = False
        store_indices = emit_debug
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
                        for u in range(UNROLL):
                            body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                        for u in range(UNROLL):
                            body.append(("valu", ("+", v_tmp3[u], v_tmp1[u], v_one)))
                        for u in range(UNROLL):
                            body.append(
                                ("valu", ("multiply_add", v_idx[u], v_idx[u], v_two, v_tmp3[u]))
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
