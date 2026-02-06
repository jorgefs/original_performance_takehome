"""
Speculative load: While hash is computing for depth N, preload both
possible nodes for depth N+1 (left and right children).

This hides load latency by overlapping it with hash computation.
When hash completes, we already have both possible next nodes loaded,
and just need to select the correct one.

Trade-off: 2x loads but removes load from critical path.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add scratch for speculative node values
old_alloc = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)'''
new_alloc = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)

            # Speculative loads: left and right child node values
            v_spec_left = self.alloc_scratch("v_spec_left", length=VLEN)
            v_spec_right = self.alloc_scratch("v_spec_right", length=VLEN)
            v_spec_idx_left = self.alloc_scratch("v_spec_idx_left", length=VLEN)
            v_spec_idx_right = self.alloc_scratch("v_spec_idx_right", length=VLEN)'''

if old_alloc in content:
    content = content.replace(old_alloc, new_alloc)
    print("Added speculative load scratch")
else:
    print("ERROR: Could not find scratch allocation")

# The key change: for depth > 2, we modify build_hash_pipeline_addr
# to interleave speculative loads for the NEXT iteration

# For now, let's try a simpler approach: move the multiply_add (which doesn't
# depend on hash) to BEFORE the hash, so it can run in parallel

old_idx_pattern = '''                        # Skip idx update for depth 0 and forest_height
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

# New pattern: separate the multiply_add (no hash dependency) from the rest
# Emit multiply_add BEFORE hash if possible, and the rest AFTER
new_idx_pattern = '''                        # Skip idx update for depth 0 and forest_height
                        if depth != 0 and depth != forest_height:
                            # First pass: emit multiply_add (doesn't depend on hash result)
                            # This computes idx*2 + base_minus1, which can run during hash
                            for u in range(start, start + count):
                                body.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            v_tmp2[u],  # Use tmp2 to store intermediate
                                            v_idx[u],
                                            v_two,
                                            v_base_minus1,
                                        ),
                                    )
                                )
                            # Second pass: emit bit extraction and final add (depends on hash)
                            for u in range(start, start + count):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                            for u in range(start, start + count):
                                body.append(("valu", ("+", v_idx[u], v_tmp2[u], v_tmp1[u])))'''

if old_idx_pattern in content:
    content = content.replace(old_idx_pattern, new_idx_pattern)
    print("Replaced idx update with split version")
else:
    print("ERROR: Could not find idx pattern")

with open('perf_takehome_spec_load.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_spec_load.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"Split idx update: {match.group(1)} cycles (baseline: 1615)")
elif 'OK' in output:
    print("Test passed")
else:
    print("Test result:")
    if 'AssertionError' in output:
        print("Correctness FAILED!")
    print(output[-700:])
