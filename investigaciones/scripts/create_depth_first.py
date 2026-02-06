"""
Depth-first processing: Instead of processing all 256 elements through round 0,
then all through round 1, etc., we process batches of elements through ALL rounds.

Current (breadth-first):
  Round 0: elements 0-255
  Round 1: elements 0-255
  ...
  Round 15: elements 0-255

New (depth-first with batch=8 vectors = 64 elements):
  Batch A (elements 0-63): rounds 0-15
  Batch B (elements 64-127): rounds 0-15
  Batch C (elements 128-191): rounds 0-15
  Batch D (elements 192-255): rounds 0-15

This creates more independent work because while batch A is in round 5,
we can start batch B in round 0 (or overlap idx calculations).
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the main processing loop and restructure it
# The key change is to process smaller batches through all rounds

old_emit_group_call = '''            emit_group(UNROLL_MAIN, zero_const, base_is_zero=True)'''

# For depth-first, we'll process in smaller chunks
# But we need to be careful about scratch space for intermediate values

new_emit_group_call = '''            # Depth-first: process in batches of 8 vectors through all rounds
            BATCH_SIZE = 8  # 8 vectors = 64 elements per batch
            num_batches = UNROLL_MAIN // BATCH_SIZE  # 32 / 8 = 4 batches

            for batch in range(num_batches):
                batch_start = batch * BATCH_SIZE
                batch_end = batch_start + BATCH_SIZE

                # Load values for this batch
                for u in range(batch_start, batch_end):
                    if u == 0:
                        body.append(("load", ("vload", v_val[u], self.scratch["inp_values_p"])))
                    else:
                        body.append(("load", ("vload", v_val[u], tmp_val_addr_u[u])))

                # Initialize idx for this batch
                for u in range(batch_start, batch_end):
                    body.append(("valu", ("+", v_idx[u], zero_vec, zero_vec)))

                # Process all rounds for this batch
                for round_idx in range(rounds):
                    depth = round_depths[round_idx]
                    chunk = 1

                    # Adjust starts for this batch
                    batch_vec_count = BATCH_SIZE
                    if depth > 2:
                        # Use first BATCH_SIZE elements of the starts tuple
                        batch_starts = tuple(s for s in (
                            0, 4, 2, 6, 1, 5, 3, 7
                        ) if s < BATCH_SIZE)
                    else:
                        batch_starts = tuple(range(BATCH_SIZE))

                    for local_start in batch_starts:
                        actual_start = batch_start + local_start
                        count = min(chunk, batch_end - actual_start)
                        if count <= 0:
                            continue

                        v_idx_l = v_idx[actual_start : actual_start + count]
                        v_val_l = v_val[actual_start : actual_start + count]
                        v_tmp1_l = v_tmp1[actual_start : actual_start + count]
                        v_tmp2_l = v_tmp2[actual_start : actual_start + count]

                        if depth == 0:
                            for u in range(count):
                                body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_root_val)))
                            body.extend(
                                self.build_hash_vec_multi(
                                    v_val_l, v_tmp1_l, v_tmp2_l,
                                    round_idx, actual_start * VLEN, emit_debug
                                )
                            )
                        elif depth == 1:
                            for u in range(count):
                                body.append(("valu", ("&", v_tmp1_l[u], v_val_l[u], v_one)))
                                body.append(("valu", ("+", v_idx_l[u], v_base_plus1, v_tmp1_l[u])))
                                body.append((
                                    "flow",
                                    ("vselect", v_tmp1_l[u], v_tmp1_l[u], v_level1_right, v_level1_left)
                                ))
                                body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_tmp1_l[u])))
                            body.extend(
                                self.build_hash_vec_multi(
                                    v_val_l, v_tmp1_l, v_tmp2_l,
                                    round_idx, actual_start * VLEN, emit_debug
                                )
                            )
                        elif depth == 2:
                            for u in range(count):
                                body.append(("valu", ("&", v_tmp1_l[u], v_idx_l[u], v_one)))
                                body.append(("valu", ("&", v_tmp2_l[u], v_idx_l[u], v_two)))
                                body.append((
                                    "flow",
                                    ("vselect", v_tmp3_shared, v_tmp1_l[u], v_level2[3], v_level2[2])
                                ))
                                body.append((
                                    "flow",
                                    ("vselect", v_tmp1_l[u], v_tmp1_l[u], v_level2[1], v_level2[0])
                                ))
                                body.append((
                                    "flow",
                                    ("vselect", v_tmp1_l[u], v_tmp2_l[u], v_tmp1_l[u], v_tmp3_shared)
                                ))
                                body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_tmp1_l[u])))
                            body.extend(
                                self.build_hash_vec_multi(
                                    v_val_l, v_tmp1_l, v_tmp2_l,
                                    round_idx, actual_start * VLEN, emit_debug
                                )
                            )
                        else:
                            body.extend(
                                self.build_hash_pipeline_addr(
                                    v_idx_l, v_val_l, v_tmp1_l, v_tmp2_l,
                                    round_idx, actual_start * VLEN, emit_debug,
                                    count, hash_group=3
                                )
                            )

                        # idx update
                        if depth != 0 and depth != forest_height:
                            for u in range(actual_start, actual_start + count):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                body.append((
                                    "valu",
                                    ("multiply_add", v_idx[u], v_idx[u], v_two, v_base_minus1)
                                ))
                                body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))

                # Store results for this batch
                if store_indices:
                    for u in range(batch_start, batch_end):
                        body.append(("valu", ("+", v_tmp1[u], v_idx[u], v_neg_forest)))
                    for u in range(batch_start, batch_end):
                        if batch == 0 and u == 0:
                            body.append(("store", ("vstore", self.scratch["inp_indices_p"], v_tmp1[u])))
                        else:
                            body.append(("store", ("vstore", tmp_idx_addr_u[u], v_tmp1[u])))
                for u in range(batch_start, batch_end):
                    if batch == 0 and u == 0:
                        body.append(("store", ("vstore", self.scratch["inp_values_p"], v_val[u])))
                    else:
                        body.append(("store", ("vstore", tmp_val_addr_u[u], v_val[u])))'''

if old_emit_group_call in content:
    content = content.replace(old_emit_group_call, new_emit_group_call)
    print("Replaced with depth-first processing")
else:
    print("ERROR: Could not find emit_group call")

with open('perf_takehome_depth_first.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_depth_first.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"Depth-first: {match.group(1)} cycles (baseline: 1615)")
elif 'OK' in output:
    print("Test passed but no cycle count")
else:
    print("Test failed:")
    print(output[-1000:])
