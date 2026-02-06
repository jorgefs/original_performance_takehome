"""
Implement depth 3 bucketing optimization (v2 - fixed addressing):
v_idx contains forest_values_p + tree_idx, so we need to adjust.

At depth 3, tree_idx is in [7, 14], so bucket = tree_idx - 7 = (v_idx - forest_values_p) - 7
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add scratch allocations for depth 3 buckets (early allocation before init)
old_allocs = '''            v_neg_forest = self.alloc_scratch("v_neg_forest", length=VLEN)'''

new_allocs = '''            v_neg_forest = self.alloc_scratch("v_neg_forest", length=VLEN)

            # Depth 3 bucketing scratch (allocated early so init can use d3_offset)
            d3_node_val_s = [self.alloc_scratch(f"d3_node_s_{i}") for i in range(8)]  # 8 scalars
            d3_node_val_v = [self.alloc_scratch(f"d3_node_v_{i}", length=VLEN) for i in range(8)]  # 8 vectors
            d3_sel01 = self.alloc_scratch("d3_sel01", length=VLEN)
            d3_sel23 = self.alloc_scratch("d3_sel23", length=VLEN)
            d3_sel45 = self.alloc_scratch("d3_sel45", length=VLEN)
            d3_sel67 = self.alloc_scratch("d3_sel67", length=VLEN)
            d3_sel03 = self.alloc_scratch("d3_sel03", length=VLEN)
            d3_sel47 = self.alloc_scratch("d3_sel47", length=VLEN)
            d3_offset = self.alloc_scratch("d3_offset", length=VLEN)  # forest_values_p + 7'''

# Also need to add v_four later
old_allocs2 = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)'''
new_allocs2 = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)
            v_four = self.vector_const(4)'''

if old_allocs in content:
    content = content.replace(old_allocs, new_allocs)
    print("Added depth 3 early scratch allocations")
else:
    print("ERROR: Could not find early scratch allocation point")

if old_allocs2 in content:
    content = content.replace(old_allocs2, new_allocs2)
    print("Added v_four allocation")
else:
    print("ERROR: Could not find v_tmp3_shared allocation point")

# Add initialization for d3_offset
old_init_end = '''            init.append(("valu", ("-", v_neg_forest, v_base_minus1, v_one)))'''

new_init_end = '''            init.append(("valu", ("-", v_neg_forest, v_base_minus1, v_one)))
            # d3_offset = forest_values_p + 7 for depth 3 bucketing
            init.append(("valu", ("+", d3_offset, v_forest_values_p, self.vector_const(7))))'''

if old_init_end in content:
    content = content.replace(old_init_end, new_init_end)
    print("Added depth 3 offset initialization")
else:
    print("ERROR: Could not find init end point")

# Modify the else clause (depth > 2) to add depth 3 special case
old_else_clause = '''                else:
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
                    )'''

# The depth 3 bucketing logic
new_else_clause = '''                elif depth == 3:
                    # Depth 3 bucketing: 8 scalar loads + vselect instead of gather
                    # At depth 3, tree_idx in [7, 14], bucket = tree_idx - 7
                    # v_idx = forest_values_p + tree_idx, so bucket = v_idx - (forest_values_p + 7) = v_idx - d3_offset

                    # Load all 8 node values once at start of round
                    if start == 0:
                        for node_idx in range(8):
                            # Load forest_values[7 + node_idx]
                            actual_addr_const = self.scratch_const(forest_values_p_val + 7 + node_idx)
                            body.append(("load", ("load", d3_node_val_s[node_idx], actual_addr_const)))
                        # Broadcast to vectors
                        for node_idx in range(8):
                            body.append(("valu", ("vbroadcast", d3_node_val_v[node_idx], d3_node_val_s[node_idx])))

                    # For each vector, select correct value using bucket bits
                    for u in range(count):
                        # bucket = v_idx - d3_offset = v_idx - (forest_values_p + 7)
                        body.append(("valu", ("-", v_tmp1_l[u], v_idx_l[u], d3_offset)))

                        # Extract bit 0
                        body.append(("valu", ("&", v_tmp2_l[u], v_tmp1_l[u], v_one)))

                        # First level vselect: pick pairs based on bit 0
                        body.append(("flow", ("vselect", d3_sel01, v_tmp2_l[u], d3_node_val_v[1], d3_node_val_v[0])))
                        body.append(("flow", ("vselect", d3_sel23, v_tmp2_l[u], d3_node_val_v[3], d3_node_val_v[2])))
                        body.append(("flow", ("vselect", d3_sel45, v_tmp2_l[u], d3_node_val_v[5], d3_node_val_v[4])))
                        body.append(("flow", ("vselect", d3_sel67, v_tmp2_l[u], d3_node_val_v[7], d3_node_val_v[6])))

                        # Extract bit 1
                        body.append(("valu", ("&", v_tmp2_l[u], v_tmp1_l[u], v_two)))

                        # Second level vselect: pick quads based on bit 1
                        body.append(("flow", ("vselect", d3_sel03, v_tmp2_l[u], d3_sel23, d3_sel01)))
                        body.append(("flow", ("vselect", d3_sel47, v_tmp2_l[u], d3_sel67, d3_sel45)))

                        # Extract bit 2
                        body.append(("valu", ("&", v_tmp2_l[u], v_tmp1_l[u], v_four)))

                        # Third level vselect: pick final based on bit 2
                        body.append(("flow", ("vselect", v_tmp1_l[u], v_tmp2_l[u], d3_sel47, d3_sel03)))

                        # XOR with val
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
                    )'''

if old_else_clause in content:
    content = content.replace(old_else_clause, new_else_clause)
    print("Added depth 3 bucketing logic")
else:
    print("ERROR: Could not find else clause for depth > 2")

with open('perf_takehome_depth3_bucket_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Created: perf_takehome_depth3_bucket_v2.py")
