"""
Try emitting idx update operations BEFORE hash operations.

Currently: load -> XOR -> hash -> idx_update
New:       load -> XOR -> idx_update -> hash

The idx_update only depends on val (from XOR), not on the hash result.
Doing idx_update first gives the scheduler more independent work during hash.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the pattern where hash is followed by idx update and swap them
old_pattern = '''                    for start in starts:
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

new_pattern = '''                    for start in starts:
                        count = min(chunk, vec_count - start)
                        # Emit idx update BEFORE hash (except for depth 0 and forest_height)
                        # This works because idx_update depends on val after XOR, before hash
                        if depth != 0 and depth != forest_height and depth > 2:
                            # For depth > 2, we do XOR inside emit_hash_only_range
                            # So we need a different approach - just emit normally
                            emit_hash_only_range(round_idx, depth, start, count)
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
                                body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))
                        else:
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

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    print("Pattern replaced")
else:
    print("ERROR: Pattern not found")

with open('perf_takehome_idx_before.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_idx_before.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=120
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"Result: {match.group(1)} cycles (baseline: 1615)")
elif 'OK' in output:
    print("Test passed but no cycle count")
else:
    print("Test failed:")
    print(output[-800:])
