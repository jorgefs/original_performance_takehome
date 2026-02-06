"""Analyze the cycles with < 4 VALU operations - potential for easy savings."""
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

    print("=== Cycles with < 4 VALU Operations ===")
    print(f"Total instructions: {len(instrs)}")
    print()

    light_cycles = []
    for i, instr in enumerate(instrs):
        valu_count = len(instr.get('valu', []))
        if valu_count < 4:
            light_cycles.append((i, valu_count, instr))

    print(f"Found {len(light_cycles)} cycles with < 4 VALU")
    print()

    # Group by VALU count
    by_count = {}
    for i, count, instr in light_cycles:
        by_count.setdefault(count, []).append((i, instr))

    for count in sorted(by_count.keys()):
        cycles = by_count[count]
        print(f"=== {count} VALU cycles ({len(cycles)} total) ===")
        for i, instr in cycles[:5]:  # Show first 5 of each type
            engines = []
            for e, slots in instr.items():
                if slots:
                    ops = [s[0] if isinstance(s, tuple) and s else s for s in slots]
                    engines.append(f"{e}={ops}")
            print(f"  Cycle {i}: {', '.join(engines)}")
        if len(cycles) > 5:
            print(f"  ... and {len(cycles) - 5} more")
        print()

    # Potential savings
    total_wasted = sum(4 - count for i, count, _ in light_cycles)
    potential_saved = total_wasted / 6
    print(f"=== Potential Savings ===")
    print(f"Total 'wasted' VALU slots in light cycles: {total_wasted}")
    print(f"If we could fill to 4 VALU each: {potential_saved:.1f} cycles saved")

    # Check what's AFTER these light cycles
    print()
    print("=== Context Around Light Cycles ===")
    for i, count, instr in light_cycles[:3]:
        print(f"\nCycle {i} (VALU={count}):")
        if i > 0:
            prev = instrs[i-1]
            prev_valu = len(prev.get('valu', []))
            print(f"  Before (cycle {i-1}): VALU={prev_valu}")
        print(f"  Current: {dict((k, len(v)) for k, v in instr.items())}")
        if i < len(instrs) - 1:
            next_instr = instrs[i+1]
            next_valu = len(next_instr.get('valu', []))
            print(f"  After (cycle {i+1}): VALU={next_valu}")

if __name__ == "__main__":
    analyze()
