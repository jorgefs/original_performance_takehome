"""Analyze per-cycle slot utilization."""
import sys
sys.path.insert(0, '.')
from perf_takehome import KernelBuilder, VLEN
from collections import Counter

def analyze_utilization():
    forest_height = 10
    rounds = 16
    batch_size = 256
    n_nodes = (1 << (forest_height + 1)) - 1

    builder = KernelBuilder()
    builder.build_kernel(forest_height, n_nodes, batch_size, rounds)
    instrs = builder.instrs

    # Slot limits per engine
    limits = {'alu': 12, 'valu': 6, 'load': 2, 'store': 2, 'flow': 1}

    # Count utilization histograms
    valu_utilization = Counter()
    load_utilization = Counter()
    flow_utilization = Counter()

    # Track cycles with low utilization
    low_valu_cycles = []

    for i, instr in enumerate(instrs):
        valu_count = len(instr.get('valu', []))
        load_count = len(instr.get('load', []))
        flow_count = len(instr.get('flow', []))

        valu_utilization[valu_count] += 1
        load_utilization[load_count] += 1
        flow_utilization[flow_count] += 1

        # Track cycles where VALU is not full but could have been
        if valu_count < 6 and valu_count > 0:
            low_valu_cycles.append((i, valu_count, instr))

    print("=== VALU Utilization Histogram (6 slots max) ===")
    for count in range(7):
        n = valu_utilization[count]
        bar = '#' * (n // 20)
        print(f"  {count} slots: {n:4d} cycles ({100*n/len(instrs):5.1f}%) {bar}")

    print("\n=== Load Utilization Histogram (2 slots max) ===")
    for count in range(3):
        n = load_utilization[count]
        bar = '#' * (n // 20)
        print(f"  {count} slots: {n:4d} cycles ({100*n/len(instrs):5.1f}%) {bar}")

    print("\n=== Flow Utilization Histogram (1 slot max) ===")
    for count in range(2):
        n = flow_utilization[count]
        bar = '#' * (n // 20)
        print(f"  {count} slots: {n:4d} cycles ({100*n/len(instrs):5.1f}%) {bar}")

    # Calculate theoretical if VALU was always 6
    wasted_valu_slots = sum((6 - count) * valu_utilization[count] for count in range(6))
    wasted_cycles = wasted_valu_slots / 6
    print(f"\n=== Efficiency Analysis ===")
    print(f"Total VALU ops: 8216")
    print(f"Cycles with VALU ops: {sum(valu_utilization[c] for c in range(1,7))}")
    print(f"Wasted VALU slots: {wasted_valu_slots}")
    print(f"Potential cycles saved: {wasted_cycles:.0f}")

    # Show some example low-utilization cycles
    print(f"\n=== Sample Low-VALU Cycles (showing first 10) ===")
    for i, (cycle, valu_count, instr) in enumerate(low_valu_cycles[:10]):
        engines = ', '.join(f"{e}={len(s)}" for e, s in instr.items() if s)
        print(f"  Cycle {cycle}: VALU={valu_count}, {engines}")

if __name__ == "__main__":
    analyze_utilization()
