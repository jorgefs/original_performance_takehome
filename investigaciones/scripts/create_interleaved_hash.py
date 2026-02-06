"""
Interleaved hash emission: instead of emitting all stage 0 for all vectors,
then all stage 1, etc., we interleave stages across vectors.

This creates more opportunities for the scheduler to find independent work.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace build_hash_vec_multi to use interleaved emission
old_build_hash = '''    def build_hash_vec_multi(
        self, val_addrs, tmp1_addrs, tmp2_addrs, round, i_base, emit_debug
    ):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mult = 1 + (1 << val3)
                for u in range(len(val_addrs)):
                    slots.append(
                        (
                            "valu",
                            (
                                "multiply_add",
                                val_addrs[u],
                                val_addrs[u],
                                self.vector_const(mult),
                                self.vector_const(val1),
                            ),
                        )
                    )
            else:
                for u in range(len(val_addrs)):
                    slots.append(
                        (
                            "valu",
                            (op1, tmp1_addrs[u], val_addrs[u], self.vector_const(val1)),
                        )
                    )
                for u in range(len(val_addrs)):
                    slots.append(
                        (
                            "valu",
                            (op3, tmp2_addrs[u], val_addrs[u], self.vector_const(val3)),
                        )
                    )
                for u in range(len(val_addrs)):
                    slots.append(
                        ("valu", (op2, val_addrs[u], tmp1_addrs[u], tmp2_addrs[u]))
                    )
            if emit_debug:
                for u in range(len(val_addrs)):
                    base = i_base + u * VLEN
                    keys = [
                        (round, base + lane, "hash_stage", hi) for lane in range(VLEN)
                    ]
                    slots.append(("debug", ("vcompare", val_addrs[u], keys)))
        return slots'''

# New version: interleave operations more aggressively
new_build_hash = '''    def build_hash_vec_multi(
        self, val_addrs, tmp1_addrs, tmp2_addrs, round, i_base, emit_debug
    ):
        slots = []
        n = len(val_addrs)

        # Precompute stage info
        stage_info = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            is_mad = op1 == "+" and op2 == "+" and op3 == "<<"
            if is_mad:
                mult = 1 + (1 << val3)
                stage_info.append(('mad', mult, val1))
            else:
                stage_info.append(('triple', op1, val1, op2, op3, val3))

        # Interleaved emission: process 2 vectors at a time through all stages
        # This creates more overlap between different vectors' dependencies
        for u_start in range(0, n, 2):
            u_end = min(u_start + 2, n)
            for hi, info in enumerate(stage_info):
                if info[0] == 'mad':
                    _, mult, val1 = info
                    for u in range(u_start, u_end):
                        slots.append(
                            (
                                "valu",
                                (
                                    "multiply_add",
                                    val_addrs[u],
                                    val_addrs[u],
                                    self.vector_const(mult),
                                    self.vector_const(val1),
                                ),
                            )
                        )
                else:
                    _, op1, val1, op2, op3, val3 = info
                    for u in range(u_start, u_end):
                        slots.append(
                            (
                                "valu",
                                (op1, tmp1_addrs[u], val_addrs[u], self.vector_const(val1)),
                            )
                        )
                    for u in range(u_start, u_end):
                        slots.append(
                            (
                                "valu",
                                (op3, tmp2_addrs[u], val_addrs[u], self.vector_const(val3)),
                            )
                        )
                    for u in range(u_start, u_end):
                        slots.append(
                            ("valu", (op2, val_addrs[u], tmp1_addrs[u], tmp2_addrs[u]))
                        )
                if emit_debug:
                    for u in range(u_start, u_end):
                        base = i_base + u * VLEN
                        keys = [
                            (round, base + lane, "hash_stage", hi) for lane in range(VLEN)
                        ]
                        slots.append(("debug", ("vcompare", val_addrs[u], keys)))
        return slots'''

if old_build_hash in content:
    content = content.replace(old_build_hash, new_build_hash)
    print("Replaced build_hash_vec_multi with interleaved version")
else:
    print("ERROR: Could not find build_hash_vec_multi")

with open('perf_takehome_interleaved_hash.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test it
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_interleaved_hash.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=120, cwd='.'
)

print("\nInterleaved hash test:")
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
