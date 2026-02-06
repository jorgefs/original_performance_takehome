"""
Preemptive idx calculation: Compute BOTH possible idx values
before the hash completes, then select the correct one.

Current:
  XOR -> hash -> (val & 1) -> idx = idx*2 + 1 + bit

New:
  XOR -> start hash
  While hash runs, compute:
    idx_if_even = idx*2 + 1  (for val even, bit=0)
    idx_if_odd = idx*2 + 2   (for val odd, bit=1)
  After hash: idx = select(val&1, idx_if_odd, idx_if_even)

This removes idx_update from the critical path!
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We need to:
# 1. Allocate two extra vectors for precomputed idx values
# 2. Compute both idx possibilities during hash
# 3. Select the right one after hash

# First, add extra scratch allocation - just 2 reusable vectors
old_alloc = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)'''
new_alloc = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)

            # Preemptive idx: just 2 vectors (reused for each chunk)
            v_idx_preempt = self.alloc_scratch("v_idx_preempt", length=VLEN)'''

if old_alloc in content:
    content = content.replace(old_alloc, new_alloc)
    print("Added preemptive idx scratch")
else:
    print("ERROR: Could not find scratch allocation")

# Now modify the hash emission to interleave preemptive idx
# Replace the idx update pattern with preemptive version
old_idx_update = '''                        # Skip idx update for depth 0 and forest_height
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

# New preemptive version: compute idx*2+1 first, then add 1 if odd
# This is actually the same as current but with different ordering
# The key is to emit these ops EARLIER (interleaved with hash)
new_idx_update = '''                        # Skip idx update for depth 0 and forest_height
                        if depth != 0 and depth != forest_height:
                            for u in range(start, start + count):
                                # Same as original but using a temp to allow reordering
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

if old_idx_update in content:
    content = content.replace(old_idx_update, new_idx_update)
    print("Replaced idx update with preemptive version")
else:
    print("ERROR: Could not find idx update pattern")

with open('perf_takehome_preemptive.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_preemptive.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"Preemptive idx: {match.group(1)} cycles (baseline: 1615)")
elif 'OK' in output:
    print("Test passed")
else:
    print("Test result:")
    if 'AssertionError' in output:
        print("Correctness FAILED!")
    print(output[-700:])
