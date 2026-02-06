"""Analyze where 4-VALU cycles occur."""
import sys
sys.path.insert(0, '.')
from perf_takehome import KernelBuilder, VLEN

def analyze_4valu():
    forest_height = 10
    rounds = 16
    batch_size = 256
    n_nodes = (1 << (forest_height + 1)) - 1

    builder = KernelBuilder()
    builder.build_kernel(forest_height, n_nodes, batch_size, rounds)
    instrs = builder.instrs

    # Find stretches of 4-VALU cycles
    stretches = []
    current_stretch = None

    for i, instr in enumerate(instrs):
        valu_count = len(instr.get('valu', []))
        load_count = len(instr.get('load', []))

        if valu_count == 4:
            if current_stretch is None:
                current_stretch = {'start': i, 'count': 1, 'loads': load_count}
            else:
                current_stretch['count'] += 1
                current_stretch['loads'] += load_count
        else:
            if current_stretch is not None:
                current_stretch['end'] = i - 1
                stretches.append(current_stretch)
                current_stretch = None

    if current_stretch:
        current_stretch['end'] = len(instrs) - 1
        stretches.append(current_stretch)

    print(f"=== 4-VALU Cycle Stretches ===")
    print(f"Total stretches: {len(stretches)}")
    print(f"Total 4-VALU cycles: {sum(s['count'] for s in stretches)}")
    print()

    # Group by stretch length
    by_length = {}
    for s in stretches:
        length = s['count']
        by_length.setdefault(length, []).append(s)

    print("Distribution by stretch length:")
    for length in sorted(by_length.keys(), reverse=True)[:10]:
        count = len(by_length[length])
        total_cycles = length * count
        print(f"  {length:3d} cycles: {count:3d} stretches ({total_cycles} total cycles)")

    # Show the longest stretches
    print()
    print("=== Longest 4-VALU Stretches ===")
    longest = sorted(stretches, key=lambda x: -x['count'])[:10]
    for s in longest:
        avg_loads = s['loads'] / s['count'] if s['count'] > 0 else 0
        print(f"  Cycles {s['start']}-{s['end']}: {s['count']} cycles, avg loads={avg_loads:.1f}")

    # Analyze what operations are in these cycles
    print()
    print("=== Sample 4-VALU Cycles (from longest stretch) ===")
    if longest:
        start = longest[0]['start']
        for i in range(start, min(start + 5, longest[0]['end'] + 1)):
            instr = instrs[i]
            valu_ops = [slot[0] for slot in instr.get('valu', [])]
            load_ops = [slot[0] for slot in instr.get('load', [])]
            print(f"  Cycle {i}: VALU={valu_ops}, LOAD={load_ops}")

if __name__ == "__main__":
    analyze_4valu()
