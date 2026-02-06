"""
Pipelined idx calculation: Move the multiply_add (which is independent of hash)
to BEFORE the hash, so it can run in parallel with the hash stages.

Current flow:
  load -> XOR -> hash (6 stages) -> bit_extract -> multiply_add -> add

New flow:
  load -> XOR -> [multiply_add runs in parallel with hash] -> bit_extract -> add

The multiply_add (idx*2 + base_minus1) doesn't depend on the hash result,
only on the current idx which is known at the start of the round.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Modify the round loop to emit multiply_add BEFORE the hash for depth > 2
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

# New version: For depth > 2, emit multiply_add BEFORE hash in build_hash_pipeline_addr
# Then after hash, only emit bit extraction and final add
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

                    # For depth > 2 with pipelining: emit multiply_add before hash
                    # The multiply_add (idx*2 + base_minus1) is independent of hash result
                    if depth > 2 and depth != forest_height:
                        # Pre-emit all multiply_adds (they don't depend on hash)
                        for u in range(vec_count):
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        v_tmp2[u],  # Store in tmp2, not idx yet
                                        v_idx[u],
                                        v_two,
                                        v_base_minus1,
                                    ),
                                )
                            )

                    for start in starts:
                        count = min(chunk, vec_count - start)
                        emit_hash_only_range(round_idx, depth, start, count)
                        # Skip idx update for depth 0 and forest_height
                        if depth != 0 and depth != forest_height:
                            if depth > 2:
                                # For depth > 2: only emit bit extract and final add
                                # (multiply_add was done earlier)
                                for u in range(start, start + count):
                                    body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                    body.append(("valu", ("+", v_idx[u], v_tmp2[u], v_tmp1[u])))
                            else:
                                # For depth 1, 2: use original pattern
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
    print("Replaced with pipelined idx version")
else:
    print("ERROR: Could not find round loop pattern")

with open('perf_takehome_pipelined_idx.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_pipelined_idx.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"Pipelined idx: {match.group(1)} cycles (baseline: 1615)")
elif 'OK' in output:
    print("Test passed")
else:
    print("Test result:")
    if 'AssertionError' in output:
        print("Correctness FAILED!")
    print(output[-700:])
