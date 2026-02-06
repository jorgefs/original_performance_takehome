"""Test different scheduler window sizes."""
import subprocess
import re

def test_window(size):
    with open('perf_takehome.py', 'r', encoding='utf-8') as f:
        content = f.read()

    modified = content.replace('window_size = 1024', f'window_size = {size}')

    filename = f'perf_takehome_win{size}.py'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(modified)

    result = subprocess.run(
        ['python', filename, 'Tests.test_kernel_cycles'],
        capture_output=True, text=True, timeout=300
    )

    output = result.stdout + result.stderr
    match = re.search(r'CYCLES:\s*(\d+)', output)
    if match:
        return int(match.group(1))
    return None

print("Testing scheduler window sizes:")
print(f"  {'baseline (1024)':20s}: 1615 cycles")
for size in [512, 2048, 4096, 8192]:
    try:
        cycles = test_window(size)
        diff = cycles - 1615 if cycles else "N/A"
        print(f"  window={size:<14d}: {cycles} cycles ({'+' if isinstance(diff, int) and diff > 0 else ''}{diff})")
    except Exception as e:
        print(f"  window={size:<14d}: ERROR - {str(e)[:100]}")
