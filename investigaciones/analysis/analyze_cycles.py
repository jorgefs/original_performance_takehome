"""Analyze cycle breakdown and instruction mix for the current kernel."""
import sys
sys.path.insert(0, '.')
from perf_takehome import KernelBuilder, VLEN
from problem import Machine, Core

def analyze_kernel():
    forest_height = 10
    rounds = 16
    batch_size = 256
    n_nodes = (1 << (forest_height + 1)) - 1

    builder = KernelBuilder()
    builder.build_kernel(forest_height, n_nodes, batch_size, rounds)
    instrs = builder.instrs

    # Count instruction types
    counts = {
        'alu': 0,
        'valu': 0,
        'load': 0,
        'store': 0,
        'flow': 0,
        'debug': 0,
    }

    valu_ops = {}
    load_ops = {}
    flow_ops = {}

    for instr in instrs:
        for engine, slots in instr.items():
            if engine in counts:
                counts[engine] += len(slots)
                if engine == 'valu':
                    for slot in slots:
                        op = slot[0]
                        valu_ops[op] = valu_ops.get(op, 0) + 1
                elif engine == 'load':
                    for slot in slots:
                        op = slot[0]
                        load_ops[op] = load_ops.get(op, 0) + 1
                elif engine == 'flow':
                    for slot in slots:
                        op = slot[0]
                        flow_ops[op] = flow_ops.get(op, 0) + 1

    print("=== Instruction Mix ===")
    print(f"Total bundles (cycles): {len(instrs)}")
    print()
    for engine, count in sorted(counts.items()):
        print(f"{engine:8s}: {count:6d}")
    print()

    print("=== VALU Operations ===")
    for op, count in sorted(valu_ops.items(), key=lambda x: -x[1]):
        print(f"  {op:20s}: {count:6d}")
    print()

    print("=== Load Operations ===")
    for op, count in sorted(load_ops.items(), key=lambda x: -x[1]):
        print(f"  {op:20s}: {count:6d}")
    print()

    print("=== Flow Operations ===")
    for op, count in sorted(flow_ops.items(), key=lambda x: -x[1]):
        print(f"  {op:20s}: {count:6d}")
    print()

    # Calculate theoretical minimums
    print("=== Theoretical Minimums (ignoring dependencies) ===")
    print(f"VALU limited: {(counts['valu'] + 5) // 6} cycles (6 valu/cycle)")
    print(f"Load limited: {(counts['load'] + 1) // 2} cycles (2 load/cycle)")
    print(f"Store limited: {(counts['store'] + 1) // 2} cycles (2 store/cycle)")
    print(f"Flow limited: {counts['flow']} cycles (1 flow/cycle)")

    # Count load_offset specifically
    load_offset_count = load_ops.get('load_offset', 0)
    print()
    print(f"=== Load Offset Analysis ===")
    print(f"Total load_offset: {load_offset_count}")
    print(f"  → Minimum cycles for load_offset: {(load_offset_count + 1) // 2}")

    # Analyze by depth
    round_depths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]
    loads_per_round = {}
    for r, d in enumerate(round_depths):
        if d > 2:
            loads_per_round[r] = 256  # 32 vectors * 8 elements

    print(f"  Rounds with depth > 2: {list(loads_per_round.keys())}")
    print(f"  Expected load_offset count: {sum(loads_per_round.values())}")

if __name__ == "__main__":
    analyze_kernel()
