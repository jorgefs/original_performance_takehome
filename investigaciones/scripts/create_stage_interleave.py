"""
Stage interleaving: Instead of completing all hash stages for vector A before
starting vector B, interleave the stages:
- A stage 0, B stage 0
- A stage 1, B stage 1
- ...

This maximizes independent work at each step because:
- A stage 1 depends on A stage 0, but NOT on B stage 0
- So A stage 1 and B stage 0 can run in parallel if emitted in same window

The key change: modify build_hash_vec_multi_stages to emit stages for
multiple vectors interleaved by stage, not by vector.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The current build_hash_vec_multi already interleaves by operation within each stage
# But operations for all vectors in a stage are emitted together
# Let's try emitting operations for pairs of vectors alternately

# Actually, looking at build_hash_vec_multi_stages more carefully:
# For each stage, it emits all ops for all vectors in that stage
# The scheduler then packs them

# The issue might be that we're not giving the scheduler enough independent work
# Let me try a different approach: use chunk=2 but with the optimized starts

old_chunk = '''                    chunk = 1
                    if depth > 2:'''

new_chunk = '''                    chunk = 2  # Process 2 vectors at a time for better interleaving
                    if depth > 2:'''

if old_chunk in content:
    content = content.replace(old_chunk, new_chunk)
    print("Changed chunk from 1 to 2")
else:
    print("ERROR: Could not find chunk pattern")

# Also need to adjust starts for chunk=2 (16 starts instead of 32)
old_deep_starts = '''                        starts = (
                            0, 16, 4, 20, 8, 24, 12, 28, 2, 18, 6, 22, 10, 26, 14, 30,
                            1, 17, 5, 21, 9, 25, 13, 29, 3, 19, 7, 23, 11, 27, 15, 31,
                        )'''

new_deep_starts = '''                        starts = (
                            0, 16, 4, 20, 8, 24, 12, 28, 2, 18, 6, 22, 10, 26, 14, 30,
                        )'''

if old_deep_starts in content:
    content = content.replace(old_deep_starts, new_deep_starts)
    print("Adjusted deep starts for chunk=2")

old_shallow_starts = '''                        starts = (
                            0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                            1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31,
                        )'''

new_shallow_starts = '''                        starts = (
                            0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                        )'''

if old_shallow_starts in content:
    content = content.replace(old_shallow_starts, new_shallow_starts)
    print("Adjusted shallow starts for chunk=2")

with open('perf_takehome_chunk2_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_chunk2_fixed.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"Chunk=2 with fixed starts: {match.group(1)} cycles (baseline: 1615)")
elif 'OK' in output:
    print("Test passed")
else:
    print("Test result:")
    if 'AssertionError' in output:
        print("Correctness FAILED!")
    print(output[-1000:])
