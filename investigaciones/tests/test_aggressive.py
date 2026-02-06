"""Test with aggressive_schedule = False to see if it helps."""
import subprocess
import re

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Try with aggressive_schedule = False
modified = content.replace('self.aggressive_schedule = True', 'self.aggressive_schedule = False')

with open('perf_takehome_noaggr.py', 'w', encoding='utf-8') as f:
    f.write(modified)

result = subprocess.run(
    ['python', 'perf_takehome_noaggr.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=120
)

output = result.stdout + result.stderr
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    cycles = int(match.group(1))
    print(f"aggressive_schedule=False: {cycles} cycles (baseline: 1615)")
else:
    print("Test failed or no cycle count found")
    print(output[-500:])
