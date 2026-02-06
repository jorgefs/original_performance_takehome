"""Analyze the init phase cycle usage."""
import sys
sys.path.insert(0, '.')
from perf_takehome import KernelBuilder, VLEN

def analyze_init():
    forest_height = 10
    rounds = 16
    batch_size = 256
    n_nodes = (1 << (forest_height + 1)) - 1

    builder = KernelBuilder()
    builder.build_kernel(forest_height, n_nodes, batch_size, rounds)
    instrs = builder.instrs

    # Find the first pause (end of init)
    init_end = 0
    for i, instr in enumerate(instrs):
        if 'flow' in instr and any(slot[0] == 'pause' for slot in instr.get('flow', [])):
            init_end = i
            break

    print(f"=== Init Phase ===")
    print(f"Init instructions: {init_end + 1} bundles (cycles)")

    # Count ops in init
    init_ops = {'alu': 0, 'valu': 0, 'load': 0, 'store': 0, 'flow': 0}
    for i in range(init_end + 1):
        for engine, slots in instrs[i].items():
            if engine in init_ops:
                init_ops[engine] += len(slots)

    for engine, count in sorted(init_ops.items()):
        print(f"  {engine}: {count}")

    # Find second pause (end of main loop)
    body_end = init_end + 1
    for i, instr in enumerate(instrs[init_end + 1:], init_end + 1):
        if 'flow' in instr and any(slot[0] == 'pause' for slot in instr.get('flow', [])):
            body_end = i
            break

    print(f"\n=== Body Phase ===")
    print(f"Body instructions: {body_end - init_end - 1} bundles (cycles)")

    # Count ops in body
    body_ops = {'alu': 0, 'valu': 0, 'load': 0, 'store': 0, 'flow': 0}
    for i in range(init_end + 1, body_end):
        for engine, slots in instrs[i].items():
            if engine in body_ops:
                body_ops[engine] += len(slots)

    for engine, count in sorted(body_ops.items()):
        print(f"  {engine}: {count}")

    print(f"\n=== Total ===")
    print(f"Total cycles: {len(instrs)}")
    print(f"  Init: {init_end + 1} ({100*(init_end+1)/len(instrs):.1f}%)")
    print(f"  Body: {body_end - init_end - 1} ({100*(body_end-init_end-1)/len(instrs):.1f}%)")
    print(f"  Remaining: {len(instrs) - body_end} ({100*(len(instrs)-body_end)/len(instrs):.1f}%)")

if __name__ == "__main__":
    analyze_init()
