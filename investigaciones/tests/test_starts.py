"""Test different starts orderings."""
import subprocess
import re

def test_starts(name, deep_starts, shallow_starts):
    with open('perf_takehome.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace starts patterns
    old_starts = '''                    if depth > 2:
                        starts = (
                            0, 16, 4, 20, 8, 24, 12, 28, 2, 18, 6, 22, 10, 26, 14, 30,
                            1, 17, 5, 21, 9, 25, 13, 29, 3, 19, 7, 23, 11, 27, 15, 31,
                        )
                    else:
                        starts = (
                            0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                            1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31,
                        )'''

    new_starts = f'''                    if depth > 2:
                        starts = {deep_starts}
                    else:
                        starts = {shallow_starts}'''

    modified = content.replace(old_starts, new_starts)

    filename = f'perf_takehome_starts_{name}.py'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(modified)

    result = subprocess.run(
        ['python', filename, 'Tests.test_kernel_cycles'],
        capture_output=True, text=True, timeout=120
    )

    output = result.stdout + result.stderr
    match = re.search(r'CYCLES:\s*(\d+)', output)
    if match:
        return int(match.group(1))
    else:
        return None

# Test different patterns
patterns = [
    ("sequential", "tuple(range(32))", "tuple(range(32))"),
    ("reverse", "tuple(range(31, -1, -1))", "tuple(range(31, -1, -1))"),
    ("stride2", "tuple(range(0, 32, 2)) + tuple(range(1, 32, 2))", "tuple(range(0, 32, 2)) + tuple(range(1, 32, 2))"),
    ("stride4", "tuple(range(0, 32, 4)) + tuple(range(1, 32, 4)) + tuple(range(2, 32, 4)) + tuple(range(3, 32, 4))",
               "tuple(range(0, 32, 4)) + tuple(range(1, 32, 4)) + tuple(range(2, 32, 4)) + tuple(range(3, 32, 4))"),
    ("halfhalf", "tuple(range(0, 16)) + tuple(range(16, 32))", "tuple(range(0, 16)) + tuple(range(16, 32))"),
    ("interleave", "tuple(i for pair in zip(range(0, 16), range(16, 32)) for i in pair)",
                  "tuple(i for pair in zip(range(0, 16), range(16, 32)) for i in pair)"),
    ("bitrev", "tuple(sorted(range(32), key=lambda x: int(f'{x:05b}'[::-1], 2)))",
               "tuple(sorted(range(32), key=lambda x: int(f'{x:05b}'[::-1], 2)))"),
]

print("Testing different starts patterns:")
print(f"  {'baseline':20s}: 1615 cycles")
for name, deep, shallow in patterns:
    try:
        cycles = test_starts(name, deep, shallow)
        diff = cycles - 1615 if cycles else "N/A"
        print(f"  {name:20s}: {cycles} cycles ({'+' if isinstance(diff, int) and diff > 0 else ''}{diff})")
    except Exception as e:
        print(f"  {name:20s}: ERROR - {str(e)[:100]}")
