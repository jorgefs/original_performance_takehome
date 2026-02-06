"""Analyze the relationship between load_offset and VALU counts."""
import sys
sys.path.insert(0, '.')
from perf_takehome import KernelBuilder, VLEN
from collections import Counter

def analyze():
    forest_height = 10
    rounds = 16
    batch_size = 256
    n_nodes = (1 << (forest_height + 1)) - 1

    builder = KernelBuilder()
    builder.build_kernel(forest_height, n_nodes, batch_size, rounds)
    instrs = builder.instrs

    # Count load_offset by VALU count
    load_offset_by_valu = Counter()
    cycles_by_valu_and_load = Counter()

    total_load_offset = 0
    for i, instr in enumerate(instrs):
        valu_count = len(instr.get('valu', []))
        load_slots = instr.get('load', [])
        load_offset_count = sum(1 for s in load_slots if s[0] == 'load_offset')

        total_load_offset += load_offset_count
        load_offset_by_valu[valu_count] += load_offset_count
        cycles_by_valu_and_load[(valu_count, load_offset_count)] += 1

    print("=== Load_offset Distribution by VALU Count ===")
    for valu in sorted(load_offset_by_valu.keys()):
        count = load_offset_by_valu[valu]
        print(f"  VALU={valu}: {count} load_offset ops ({100*count/total_load_offset:.1f}%)")

    print()
    print("=== Cycle Distribution (VALU, load_offset) ===")
    for (valu, loads), count in sorted(cycles_by_valu_and_load.items()):
        print(f"  VALU={valu}, load_offset={loads}: {count} cycles")

    # Check cycles 446-1091 specifically
    print()
    print("=== Stretch 446-1091 Analysis ===")
    valu_ops_in_stretch = []
    for i in range(446, 1092):
        instr = instrs[i]
        valu_count = len(instr.get('valu', []))
        valu_ops_in_stretch.extend([s[0] for s in instr.get('valu', [])])

    print(f"Total VALU ops in stretch: {len(valu_ops_in_stretch)}")
    print("VALU op distribution:")
    op_counts = Counter(valu_ops_in_stretch)
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")

if __name__ == "__main__":
    analyze()
