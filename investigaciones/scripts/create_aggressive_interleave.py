"""
Aggressive interleaving: emit operations in a more fine-grained order
to give the scheduler more independent work at any time.

Current approach in build_hash_pipeline_addr:
1. Emit all address computations
2. For each group:
   a. Emit all load_offset for the group
   b. Emit XOR and hash for the group
   c. Interleave next group's loads during hash

New approach:
- Process in smaller chunks (1 vector at a time instead of hash_group)
- Emit each vector's work interleaved with loads for next vectors
- This creates more overlap between vectors
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace hash_group=3 with hash_group=1 for maximum interleaving
# This processes one vector at a time, allowing more overlap
old_hash_group = 'hash_group=3'
new_hash_group = 'hash_group=1'

content = content.replace(old_hash_group, new_hash_group)

with open('perf_takehome_hg1.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test it
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_hg1.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=120, cwd='.'
)

print("hash_group=1 test:")
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
