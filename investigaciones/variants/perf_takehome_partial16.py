"""
Optimization: Use 16 v_partial vectors instead of 32.
Process in two halves, precomputing partial_idx for each half separately.

Scratch savings: 32 - 16 = 16 vectors = 128 words
This should fit within the 217 free words we have.
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

import sys
sys.path.insert(0, r'C:\Users\OEM\proyectos_gito\test2\original_performance_takehome')
from perf_takehome import KernelBuilder as OriginalKernelBuilder, BASELINE


class KernelBuilder(OriginalKernelBuilder):
    """KernelBuilder with half-batch precomputation."""

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
            PARTIAL_SIZE = 16  # Only 16 partial vectors, reused for both halves

            v_idx = [self.alloc_scratch(f"v_idx{u}", length=VLEN) for u in range(UNROLL_MAIN)]
            v_val = [self.alloc_scratch(f"v_val{u}", length=VLEN) for u in range(UNROLL_MAIN)]
            v_tmp1 = [self.alloc_scratch(f"v_tmp1_{u}", length=VLEN) for u in range(UNROLL_MAIN)]
            v_tmp2 = [self.alloc_scratch(f"v_tmp2_{u}", length=VLEN) for u in range(UNROLL_MAIN)]
            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)

            # Only 16 partial vectors - will be reused
            v_partial = [self.alloc_scratch(f"v_partial{u}", length=VLEN) for u in range(PARTIAL_SIZE)]

            tmp_val_addr_u = [self.alloc_scratch(f"tmp_val_addr{u}") for u in range(UNROLL_MAIN)]
            tmp_idx_addr_u = [self.alloc_scratch(f"tmp_idx_addr{u}") for u in range(UNROLL_MAIN)] if store_indices else []

            print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

            vec_count = UNROLL_MAIN
            base_val_addr = self.scratch["inp_values_p"]
            base_idx_addr = self.scratch["inp_indices_p"] if store_indices else None

            # Setup addresses
            for u in range(1, vec_count):
                if u == 1:
                    body.append(("alu", ("+", tmp_val_addr_u[u], base_val_addr, vlen_const)))
                    if store_indices:
                        body.append(("alu", ("+", tmp_idx_addr_u[u], base_idx_addr, vlen_const)))
                else:
                    body.append(("alu", ("+", tmp_val_addr_u[u], tmp_val_addr_u[u - 1], vlen_const)))
                    if store_indices:
                        body.append(("alu", ("+", tmp_idx_addr_u[u], tmp_idx_addr_u[u - 1], vlen_const)))

            # Load all values
            for u in range(vec_count):
                if u == 0:
                    body.append(("load", ("vload", v_val[u], base_val_addr)))
                else:
                    body.append(("load", ("vload", v_val[u], tmp_val_addr_u[u])))

            round_depths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]

            for round_idx in range(rounds):
                depth = round_depths[round_idx]

                # Process first half (vectors 0-15)
                half = 0
                start_u = 0
                end_u = PARTIAL_SIZE

                # Emit hash for first half
                if depth == 0:
                    for u in range(start_u, end_u):
                        body.append(("valu", ("^", v_val[u], v_val[u], v_root_val)))
                    body.extend(self.build_hash_vec_multi(v_val[start_u:end_u], v_tmp1[start_u:end_u],
                                                          v_tmp2[start_u:end_u], round_idx, start_u * VLEN, emit_debug))
                elif depth == 1:
                    for u in range(start_u, end_u):
                        body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                        body.append(("valu", ("+", v_idx[u], v_base_plus1, v_tmp1[u])))
                        body.append(("flow", ("vselect", v_tmp1[u], v_tmp1[u], v_level1_right, v_level1_left)))
                        body.append(("valu", ("^", v_val[u], v_val[u], v_tmp1[u])))
                    body.extend(self.build_hash_vec_multi(v_val[start_u:end_u], v_tmp1[start_u:end_u],
                                                          v_tmp2[start_u:end_u], round_idx, start_u * VLEN, emit_debug))
                elif depth == 2:
                    for u in range(start_u, end_u):
                        body.append(("valu", ("&", v_tmp1[u], v_idx[u], v_one)))
                        body.append(("valu", ("&", v_tmp2[u], v_idx[u], v_two)))
                        body.append(("flow", ("vselect", v_tmp3_shared, v_tmp1[u], v_level2[3], v_level2[2])))
                        body.append(("flow", ("vselect", v_tmp1[u], v_tmp1[u], v_level2[1], v_level2[0])))
                        body.append(("flow", ("vselect", v_tmp1[u], v_tmp2[u], v_tmp1[u], v_tmp3_shared)))
                        body.append(("valu", ("^", v_val[u], v_val[u], v_tmp1[u])))
                    body.extend(self.build_hash_vec_multi(v_val[start_u:end_u], v_tmp1[start_u:end_u],
                                                          v_tmp2[start_u:end_u], round_idx, start_u * VLEN, emit_debug))
                else:
                    body.extend(self.build_hash_pipeline_addr(v_idx[start_u:end_u], v_val[start_u:end_u],
                                                               v_tmp1[start_u:end_u], v_tmp2[start_u:end_u],
                                                               round_idx, start_u * VLEN, emit_debug,
                                                               end_u - start_u, hash_group=3))

                # Idx update for first half
                if depth != 0 and depth != forest_height:
                    for u in range(start_u, end_u):
                        body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                        body.append(("valu", ("multiply_add", v_idx[u], v_idx[u], v_two, v_base_minus1)))
                        body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))

                # Process second half (vectors 16-31)
                half = 1
                start_u = PARTIAL_SIZE
                end_u = UNROLL_MAIN

                # Emit hash for second half
                if depth == 0:
                    for u in range(start_u, end_u):
                        body.append(("valu", ("^", v_val[u], v_val[u], v_root_val)))
                    body.extend(self.build_hash_vec_multi(v_val[start_u:end_u], v_tmp1[start_u:end_u],
                                                          v_tmp2[start_u:end_u], round_idx, start_u * VLEN, emit_debug))
                elif depth == 1:
                    for u in range(start_u, end_u):
                        body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                        body.append(("valu", ("+", v_idx[u], v_base_plus1, v_tmp1[u])))
                        body.append(("flow", ("vselect", v_tmp1[u], v_tmp1[u], v_level1_right, v_level1_left)))
                        body.append(("valu", ("^", v_val[u], v_val[u], v_tmp1[u])))
                    body.extend(self.build_hash_vec_multi(v_val[start_u:end_u], v_tmp1[start_u:end_u],
                                                          v_tmp2[start_u:end_u], round_idx, start_u * VLEN, emit_debug))
                elif depth == 2:
                    for u in range(start_u, end_u):
                        body.append(("valu", ("&", v_tmp1[u], v_idx[u], v_one)))
                        body.append(("valu", ("&", v_tmp2[u], v_idx[u], v_two)))
                        body.append(("flow", ("vselect", v_tmp3_shared, v_tmp1[u], v_level2[3], v_level2[2])))
                        body.append(("flow", ("vselect", v_tmp1[u], v_tmp1[u], v_level2[1], v_level2[0])))
                        body.append(("flow", ("vselect", v_tmp1[u], v_tmp2[u], v_tmp1[u], v_tmp3_shared)))
                        body.append(("valu", ("^", v_val[u], v_val[u], v_tmp1[u])))
                    body.extend(self.build_hash_vec_multi(v_val[start_u:end_u], v_tmp1[start_u:end_u],
                                                          v_tmp2[start_u:end_u], round_idx, start_u * VLEN, emit_debug))
                else:
                    body.extend(self.build_hash_pipeline_addr(v_idx[start_u:end_u], v_val[start_u:end_u],
                                                               v_tmp1[start_u:end_u], v_tmp2[start_u:end_u],
                                                               round_idx, start_u * VLEN, emit_debug,
                                                               end_u - start_u, hash_group=3))

                # Idx update for second half
                if depth != 0 and depth != forest_height:
                    for u in range(start_u, end_u):
                        body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                        body.append(("valu", ("multiply_add", v_idx[u], v_idx[u], v_two, v_base_minus1)))
                        body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))

            # Store results
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

            body_instrs = self.build(body, vliw=True)
            self.instrs.extend(body_instrs)
            self.instrs.append({"flow": [("pause",)]})
            self.aggressive_schedule = False
            return

        return super().build_kernel(forest_height, n_nodes, batch_size, rounds)


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

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


if __name__ == "__main__":
    do_kernel_test(10, 16, 256)
