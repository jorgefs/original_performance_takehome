"""Test different hash_group values to find optimal scheduling."""
import subprocess
import re

def test_hash_group(hg):
    # Create a modified kernel with the specified hash_group
    with open('perf_takehome.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace hash_group=3 with the test value
    modified = content.replace('hash_group=3', f'hash_group={hg}')

    # Write to temp file
    filename = f'perf_takehome_hg{hg}.py'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(modified)

    # Run the test
    result = subprocess.run(
        ['python', filename, 'Tests.test_kernel_cycles'],
        capture_output=True, text=True, timeout=120
    )

    # Extract cycle count from output
    output = result.stdout + result.stderr
    match = re.search(r'CYCLES:\s*(\d+)', output)
    if match:
        return int(match.group(1))
    elif 'OK' in output:
        # Try to find cycles differently
        return None
    else:
        raise Exception(output[-500:] if len(output) > 500 else output)

# Test different hash_group values
print("Testing hash_group values:")
for hg in [1, 2, 3, 4, 5, 6, 8, 16, 32]:
    try:
        cycles = test_hash_group(hg)
        print(f"  hash_group={hg}: {cycles} cycles")
    except Exception as e:
        print(f"  hash_group={hg}: ERROR - {str(e)[:200]}")
