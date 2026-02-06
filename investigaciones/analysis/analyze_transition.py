"""Analyze the transition between 6-VALU and 4-VALU cycles."""
import sys
sys.path.insert(0, '.')
from perf_takehome import KernelBuilder, VLEN

def analyze():
    forest_height = 10
    rounds = 16
    batch_size = 256
    n_nodes = (1 << (forest_height + 1)) - 1

    builder = KernelBuilder()
    builder.build_kernel(forest_height, n_nodes, batch_size, rounds)
    instrs = builder.instrs

    # Find where we transition from 6-VALU to 4-VALU
    print("=== Transition Points ===")
    prev_valu = 0
    transitions = []
    for i, instr in enumerate(instrs):
        valu_count = len(instr.get('valu', []))
        load_count = len(instr.get('load', []))
        has_load_offset = any(s[0] == 'load_offset' for s in instr.get('load', []))

        if prev_valu != valu_count:
            transitions.append((i, prev_valu, valu_count, has_load_offset))
        prev_valu = valu_count

    # Show transitions around cycle 446
    print("\nTransitions around the big 4-VALU stretch (cycle 446):")
    for t in transitions:
        if 400 < t[0] < 500:
            print(f"  Cycle {t[0]}: {t[1]} -> {t[2]} VALU (load_offset={t[3]})")

    # Analyze cycles just before and after the 4-VALU stretch
    print("\n=== Cycles Around Transition ===")
    print("\nBefore (cycles 440-445):")
    for i in range(440, 446):
        instr = instrs[i]
        valu = instr.get('valu', [])
        load = instr.get('load', [])
        print(f"  {i}: VALU={len(valu)} {[s[0] for s in valu]}")
        print(f"       LOAD={len(load)} {[s[0] for s in load]}")

    print("\nStart of 4-VALU stretch (cycles 446-449):")
    for i in range(446, 450):
        instr = instrs[i]
        valu = instr.get('valu', [])
        load = instr.get('load', [])
        print(f"  {i}: VALU={len(valu)} {[s[0] for s in valu]}")
        print(f"       LOAD={len(load)} {[s[0] for s in load]}")

    print("\nEnd of 4-VALU stretch (cycles 1089-1092):")
    for i in range(1089, 1093):
        instr = instrs[i]
        valu = instr.get('valu', [])
        load = instr.get('load', [])
        print(f"  {i}: VALU={len(valu)} {[s[0] for s in valu]}")
        print(f"       LOAD={len(load)} {[s[0] for s in load]}")

    # Count how many different VALU operations are needed per cycle in 4-VALU stretch
    print("\n=== Why Only 4 VALU? Dependency Analysis ===")
    # The scheduler can't add more ops if:
    # 1. No more ops fit (dependencies)
    # 2. Slot limit reached (already 6)
    # Since we have 4, something is blocking the other 2 slots

    # Let's check what VALUs are available but not scheduled
    # This would require tracing the scheduler, but we can infer:
    # During depth>2, the main work is:
    # - load_offset (2/cycle)
    # - XOR node_val with val (1 per vector)
    # - 6 hash stages per vector
    # - idx update (2 ops per vector)

    # With 32 vectors and 2 load_offset per cycle:
    # - 256 load_offset total -> 128 cycles minimum
    # - For each load_offset, soon after we need XOR and hash

    # The scheduler might be limited by:
    # - Read-after-write dependencies (hash stage i depends on i-1)
    # - Only having 4 independent ops available at any moment

if __name__ == "__main__":
    analyze()
