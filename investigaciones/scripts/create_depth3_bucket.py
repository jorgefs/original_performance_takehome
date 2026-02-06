"""
Implement depth 3 bucketing optimization:
Instead of 256 load_offset per round at depth 3, use:
- 8 scalar loads (one per possible node at depth 3: idx 7-14)
- 8 vbroadcasts to vector
- 3 vselects per vector to pick correct value based on idx bits

Expected savings: ~52 cycles (from 256 to 204 cycles for depth 3 rounds)
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the emit_hash_only_range function and modify the else clause (depth > 2)
# We need to add special handling for depth == 3

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

new_else_clause = '''                elif depth == 3:
                    # Depth 3 bucketing: use 8 scalar loads + vselect instead of load_offset
                    # At depth 3, idx values are in range [7, 14] (8 possible values)
                    # idx bits 0,1,2 determine which of the 8 nodes: idx - 7 = bits 0-2

                    # Load all 8 possible node values once per chunk
                    if start == 0:  # Only load once per round
                        for node_idx in range(8):  # nodes 7-14
                            actual_idx = 7 + node_idx
                            # Load forest_values[actual_idx] into a temp scalar
                            tmp_addr = self.alloc_scratch(f"d3_tmp_addr_{node_idx}")
                            body.append(("alu", ("+", tmp_addr, forest_values_p, self.scratch_const(actual_idx))))
                            body.append(("load", ("load", d3_node_val_s[node_idx], tmp_addr)))
                        # Broadcast to vectors
                        for node_idx in range(8):
                            body.append(("valu", ("vbroadcast", d3_node_val_v[node_idx], d3_node_val_s[node_idx])))

                    # For each vector, select the correct value using idx bits
                    for u in range(count):
                        # Extract bits 0, 1, 2 from (idx - 7)
                        body.append(("valu", ("-", v_tmp1_l[u], v_idx_l[u], v_seven)))  # idx - 7
                        body.append(("valu", ("&", v_tmp2_l[u], v_tmp1_l[u], v_one)))   # bit 0
                        # Use nested vselect to pick from 8 options based on 3 bits
                        # bit2=0: pick from nodes 0-3, bit2=1: pick from nodes 4-7
                        # bit1=0: pick from nodes 0,1 or 4,5, bit1=1: pick from nodes 2,3 or 6,7
                        # bit0=0: pick node 0,2,4,6, bit0=1: pick node 1,3,5,7

                        # First level: pick based on bit 0 (pairs)
                        body.append(("flow", ("vselect", d3_sel01, v_tmp2_l[u], d3_node_val_v[1], d3_node_val_v[0])))
                        body.append(("flow", ("vselect", d3_sel23, v_tmp2_l[u], d3_node_val_v[3], d3_node_val_v[2])))
                        body.append(("flow", ("vselect", d3_sel45, v_tmp2_l[u], d3_node_val_v[5], d3_node_val_v[4])))
                        body.append(("flow", ("vselect", d3_sel67, v_tmp2_l[u], d3_node_val_v[7], d3_node_val_v[6])))

                        # Second level: pick based on bit 1 (quads)
                        body.append(("valu", ("&", v_tmp2_l[u], v_tmp1_l[u], v_two)))   # bit 1
                        body.append(("flow", ("vselect", d3_sel03, v_tmp2_l[u], d3_sel23, d3_sel01)))
                        body.append(("flow", ("vselect", d3_sel47, v_tmp2_l[u], d3_sel67, d3_sel45)))

                        # Third level: pick based on bit 2 (final)
                        body.append(("valu", ("&", v_tmp2_l[u], v_tmp1_l[u], v_four)))  # bit 2
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

# We also need to add the scratch allocations for depth 3 buckets
old_allocs = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)'''

new_allocs = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)

            # Depth 3 bucketing scratch
            d3_node_val_s = [self.alloc_scratch(f"d3_node_s_{i}") for i in range(8)]  # 8 scalars
            d3_node_val_v = [self.alloc_scratch(f"d3_node_v_{i}", length=VLEN) for i in range(8)]  # 8 vectors
            d3_sel01 = self.alloc_scratch("d3_sel01", length=VLEN)
            d3_sel23 = self.alloc_scratch("d3_sel23", length=VLEN)
            d3_sel45 = self.alloc_scratch("d3_sel45", length=VLEN)
            d3_sel67 = self.alloc_scratch("d3_sel67", length=VLEN)
            d3_sel03 = self.alloc_scratch("d3_sel03", length=VLEN)
            d3_sel47 = self.alloc_scratch("d3_sel47", length=VLEN)
            v_seven = self.vector_const(7)
            v_four = self.vector_const(4)'''

if old_allocs in content:
    content = content.replace(old_allocs, new_allocs)
    print("Added depth 3 scratch allocations")
else:
    print("ERROR: Could not find scratch allocation point")

if old_else_clause in content:
    content = content.replace(old_else_clause, new_else_clause)
    print("Added depth 3 bucketing logic")
else:
    print("ERROR: Could not find else clause for depth > 2")

with open('perf_takehome_depth3_bucket.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Created: perf_takehome_depth3_bucket.py")
