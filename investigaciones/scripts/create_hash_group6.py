"""
Try increasing hash_group from 3 to 6.

With hash_group=6, we process 6 vectors through hash stages together.
This provides more independent work within each stage:
- Stage 0: 6 multiply_add ops = 1 cycle (fills 6 VALU slots)
- Stage 1: 6 op1 + 6 op3 = 12 ops = 2 cycles, then 6 op2 = 1 cycle

More vectors in flight means more opportunities for the scheduler to
fill empty VALU slots.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Change hash_group from 3 to 6 in build_hash_pipeline_addr calls
content = content.replace('hash_group=3,', 'hash_group=6,')
content = content.replace('hash_group=3)', 'hash_group=6)')

print("Changed hash_group from 3 to 6")

with open('perf_takehome_hashgroup6.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_hashgroup6.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"hash_group=6: {match.group(1)} cycles (baseline: 1615)")
elif 'OK' in output:
    print("Test passed")
else:
    print("Test result:")
    if 'AssertionError' in output:
        print("Correctness FAILED!")
    print(output[-1000:])
