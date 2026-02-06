"""
Bulk emit approach: Instead of processing vectors one at a time through starts,
emit ALL loads for all vectors, then ALL XORs, then ALL hash operations.

This maximizes the amount of independent work available at each step.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# For depth > 2, replace the starts-based loop with bulk emission
old_round_loop = '''                for round_idx in range(rounds):
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
                        if depth != 0 and depth != forest_height:
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
                                body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))'''

new_round_loop = '''                for round_idx in range(rounds):
                    depth = round_depths[round_idx]
                    chunk = 1

                    # For depth > 2, use bulk emission
                    if depth > 2:
                        # Step 1: Emit ALL loads for all 32 vectors
                        for u in range(vec_count):
                            for lane in range(VLEN):
                                body.append(("load", ("load_offset", v_tmp1[u], v_idx[u], lane)))

                        # Step 2: Emit ALL XORs
                        for u in range(vec_count):
                            body.append(("valu", ("^", v_val[u], v_val[u], v_tmp1[u])))

                        # Step 3: Emit ALL hash operations (interleaved by stage)
                        body.extend(
                            self.build_hash_vec_multi(
                                v_val[:vec_count],
                                v_tmp1[:vec_count],
                                v_tmp2[:vec_count],
                                round_idx,
                                0,
                                emit_debug,
                            )
                        )

                        # Step 4: Emit ALL idx updates (except for forest_height)
                        if depth != forest_height:
                            # Emit bit extraction for all
                            for u in range(vec_count):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                            # Emit multiply_add for all
                            for u in range(vec_count):
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
                            # Emit final add for all
                            for u in range(vec_count):
                                body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))
                    else:
                        # For depth 0, 1, 2: use original starts-based approach
                        starts = (
                            0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                            1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31,
                        )
                        for start in starts:
                            count = min(chunk, vec_count - start)
                            emit_hash_only_range(round_idx, depth, start, count)
                            # Skip idx update for depth 0 and forest_height
                            if depth != 0 and depth != forest_height:
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
                                    body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))'''

if old_round_loop in content:
    content = content.replace(old_round_loop, new_round_loop)
    print("Replaced with bulk emit version")
else:
    print("ERROR: Could not find round loop")

with open('perf_takehome_bulk.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_bulk.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"Bulk emit: {match.group(1)} cycles (baseline: 1615)")
elif 'OK' in output:
    print("Test passed")
else:
    print("Test result:")
    if 'AssertionError' in output:
        print("Correctness FAILED!")
    print(output[-1000:])
