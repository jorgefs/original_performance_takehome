"""
Wave pipeline: Process multiple "waves" of vectors simultaneously.
While wave A does hash stage 3, wave B does hash stage 0, wave C does loads.

This creates truly independent work by having different vectors at different
points in the computation.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Modify the round loop to interleave work from different starting points
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

# New version: emit operations in a way that creates more independent work
# by interleaving idx updates with hash computations
new_round_loop = '''                for round_idx in range(rounds):
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

                    # For depth > 2, try to overlap idx update with hash
                    # The idx update only depends on the val AFTER XOR, not after hash
                    # So we can compute idx_update(u) while hash(u-1) is running

                    if depth > 2:
                        # Emit in pairs: do hash for u, then idx_update for u-1
                        # This creates overlap between idx_update and hash
                        prev_start = None
                        for start in starts:
                            count = min(chunk, vec_count - start)
                            emit_hash_only_range(round_idx, depth, start, count)

                            # Emit idx update for PREVIOUS vector (from previous iteration)
                            if prev_start is not None and depth != forest_height:
                                for u in range(prev_start, prev_start + 1):
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
                                    body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))
                            prev_start = start

                        # Emit final idx update
                        if prev_start is not None and depth != 0 and depth != forest_height:
                            for u in range(prev_start, prev_start + 1):
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
                                body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))
                    else:
                        # For shallow depths, use original ordering
                        for start in starts:
                            count = min(chunk, vec_count - start)
                            emit_hash_only_range(round_idx, depth, start, count)
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
    print("Replaced with wave pipeline")
else:
    print("ERROR: Could not find round loop")

with open('perf_takehome_wave.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_wave.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"Wave pipeline: {match.group(1)} cycles (baseline: 1615)")
elif 'OK' in output:
    print("Test passed")
else:
    print("Test failed or no cycles found")
    if 'AssertionError' in output:
        print("Correctness failed!")
    print(output[-500:])
