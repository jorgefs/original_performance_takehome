#!/usr/bin/env python3
"""
ASM Instrumentation Script for kernel optimization analysis.
Analyzes kernel_asm_1615.txt to identify bottlenecks and optimization opportunities.
"""

import re
from collections import defaultdict, Counter

def parse_asm_file(filename):
    """Parse the ASM file and extract cycles with their operations."""
    cycles = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Match cycle number and operations
            match = re.match(r'\s*(\d+):\s*(.+)', line)
            if match:
                cycle_num = int(match.group(1))
                ops_str = match.group(2)
                # Split by | to get individual operations
                ops = [op.strip() for op in ops_str.split('|')]
                parsed_ops = []
                for op in ops:
                    # Parse engine:operation format
                    if ':' in op:
                        parts = op.split(':', 1)
                        engine = parts[0]
                        operation = parts[1]
                        parsed_ops.append((engine, operation))
                cycles.append((cycle_num, parsed_ops))
    return cycles

def analyze_opcode_counts(cycles):
    """Count opcodes by engine and operation type."""
    engine_counts = Counter()
    op_counts = Counter()
    engine_op_counts = defaultdict(Counter)

    for cycle_num, ops in cycles:
        for engine, op_str in ops:
            engine_counts[engine] += 1
            # Extract operation name
            if op_str.startswith('('):
                # Try to extract the first element of the tuple
                match = re.match(r"\('([^']+)'", op_str)
                if match:
                    op_name = match.group(1)
                else:
                    op_name = op_str[:20]
            else:
                op_name = op_str[:20]
            op_counts[(engine, op_name)] += 1
            engine_op_counts[engine][op_name] += 1

    return engine_counts, op_counts, engine_op_counts

def analyze_load_offset_clusters(cycles):
    """Find load_offset clusters (sequences 0..7 on same base)."""
    clusters = []
    current_cluster = []
    current_base = None

    for cycle_num, ops in cycles:
        for engine, op_str in ops:
            if engine == 'load' and 'load_offset' in op_str:
                # Parse: ('load_offset', dest, base, offset)
                match = re.search(r"'load_offset',\s*(\d+),\s*(\d+),\s*(\d+)", op_str)
                if match:
                    dest = int(match.group(1))
                    base = int(match.group(2))
                    offset = int(match.group(3))

                    if current_base is None or base != current_base:
                        if current_cluster and len(current_cluster) >= 8:
                            clusters.append((current_base, current_cluster))
                        current_cluster = [(cycle_num, dest, offset)]
                        current_base = base
                    else:
                        current_cluster.append((cycle_num, dest, offset))

    # Don't forget the last cluster
    if current_cluster and len(current_cluster) >= 8:
        clusters.append((current_base, current_cluster))

    return clusters

def analyze_vselect_patterns(cycles):
    """Find vselect triplet patterns (3 consecutive vselects for depth 2)."""
    triplets = []
    vselect_sequence = []

    for cycle_num, ops in cycles:
        for engine, op_str in ops:
            if engine == 'flow' and 'vselect' in op_str:
                vselect_sequence.append((cycle_num, op_str))
            else:
                if len(vselect_sequence) >= 3:
                    triplets.append(vselect_sequence[:3])
                vselect_sequence = []

    # Check remaining
    if len(vselect_sequence) >= 3:
        triplets.append(vselect_sequence[:3])

    return triplets

def analyze_cycle_utilization(cycles):
    """Analyze how well cycles are utilized."""
    utilization = Counter()
    underutilized = []

    for cycle_num, ops in cycles:
        num_ops = len(ops)
        utilization[num_ops] += 1
        if num_ops < 4:
            underutilized.append((cycle_num, ops))

    return utilization, underutilized

def segment_phases(cycles):
    """Segment code into init, main loop, and epilogue phases."""
    # Find pause instructions which typically mark phase boundaries
    pause_cycles = []
    for cycle_num, ops in cycles:
        for engine, op_str in ops:
            if engine == 'flow' and 'pause' in op_str:
                pause_cycles.append(cycle_num)

    phases = []
    start = 0
    for pause_cycle in pause_cycles:
        phases.append(('phase', start, pause_cycle))
        start = pause_cycle + 1
    if start < len(cycles):
        phases.append(('phase', start, len(cycles) - 1))

    return phases, pause_cycles

def count_vliw_slot_usage(cycles):
    """Count VLIW slot usage per cycle to find bottlenecks."""
    slot_limits = {'alu': 12, 'valu': 6, 'load': 2, 'store': 2, 'flow': 1, 'debug': 1}
    bottleneck_analysis = defaultdict(int)

    for cycle_num, ops in cycles:
        engine_count = Counter()
        for engine, op_str in ops:
            engine_count[engine] += 1

        # Find which engine is at capacity (bottleneck)
        for engine, count in engine_count.items():
            limit = slot_limits.get(engine, 1)
            if count >= limit:
                bottleneck_analysis[engine] += 1

    return bottleneck_analysis

def analyze_load_offset_opportunities(cycles):
    """Find consecutive load_offset ops that could be replaced with vload."""
    # A vload can load 8 consecutive values in 1 cycle
    # 8 load_offsets with offsets 0-7 on same base could potentially be replaced

    load_offset_sequences = []
    current_seq = []
    current_base = None
    current_dest_base = None

    for cycle_num, ops in cycles:
        for engine, op_str in ops:
            if engine == 'load' and 'load_offset' in op_str:
                match = re.search(r"'load_offset',\s*(\d+),\s*(\d+),\s*(\d+)", op_str)
                if match:
                    dest = int(match.group(1))
                    base = int(match.group(2))
                    offset = int(match.group(3))

                    if (current_base == base and
                        current_dest_base == dest and
                        offset == len(current_seq)):
                        current_seq.append((cycle_num, dest, base, offset))
                    else:
                        if len(current_seq) == 8:
                            load_offset_sequences.append(current_seq)
                        current_seq = [(cycle_num, dest, base, offset)]
                        current_base = base
                        current_dest_base = dest

    if len(current_seq) == 8:
        load_offset_sequences.append(current_seq)

    return load_offset_sequences

def main():
    filename = 'kernel_asm_1615.txt'
    print(f"=== ASM Analysis for {filename} ===\n")

    cycles = parse_asm_file(filename)
    total_cycles = len(cycles)
    print(f"Total VLIW cycles: {total_cycles}")

    # A) Global opcode counts
    print("\n" + "="*60)
    print("A) GLOBAL OPCODE COUNTS")
    print("="*60)

    engine_counts, op_counts, engine_op_counts = analyze_opcode_counts(cycles)

    print("\nBy Engine:")
    for engine, count in sorted(engine_counts.items(), key=lambda x: -x[1]):
        print(f"  {engine}: {count}")

    print("\nTop operations:")
    for (engine, op), count in sorted(op_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {engine}:{op}: {count}")

    # B) Load offset cluster analysis
    print("\n" + "="*60)
    print("B) LOAD_OFFSET CLUSTER ANALYSIS (sequences of 8)")
    print("="*60)

    clusters = analyze_load_offset_clusters(cycles)
    print(f"\nFound {len(clusters)} clusters of 8+ consecutive load_offsets")

    # Count total load_offset operations
    total_load_offset = sum(1 for c, ops in cycles for e, o in ops if 'load_offset' in o)
    print(f"Total load_offset operations: {total_load_offset}")
    print(f"Cycles consumed by load_offset (@ 2/cycle): {total_load_offset / 2:.0f}")

    # Analyze vload-replaceable sequences
    vload_sequences = analyze_load_offset_opportunities(cycles)
    print(f"\nSequences of 8 load_offsets (0..7) on same dest/base: {len(vload_sequences)}")
    if vload_sequences:
        print("  First few sequences:")
        for seq in vload_sequences[:3]:
            start_cycle = seq[0][0]
            end_cycle = seq[-1][0]
            base = seq[0][2]
            dest = seq[0][1]
            print(f"    Cycles {start_cycle}-{end_cycle}: base={base}, dest={dest}")

    # C) vselect analysis
    print("\n" + "="*60)
    print("C) VSELECT ANALYSIS")
    print("="*60)

    total_vselect = sum(1 for c, ops in cycles for e, o in ops if e == 'flow' and 'vselect' in o)
    print(f"Total vselect operations: {total_vselect}")
    print(f"Cycles consumed by vselect (@ 1/cycle): {total_vselect}")

    triplets = analyze_vselect_patterns(cycles)
    print(f"vselect triplet patterns (depth 2 pattern): {len(triplets)}")

    # D) Cycle utilization
    print("\n" + "="*60)
    print("D) CYCLE UTILIZATION")
    print("="*60)

    utilization, underutilized = analyze_cycle_utilization(cycles)

    print("\nOperations per cycle distribution:")
    for num_ops in sorted(utilization.keys()):
        count = utilization[num_ops]
        pct = count / total_cycles * 100
        bar = '#' * int(pct / 2)
        print(f"  {num_ops:2d} ops: {count:4d} cycles ({pct:5.1f}%) {bar}")

    print(f"\nUnderutilized cycles (< 4 ops): {len(underutilized)}")
    if underutilized[:10]:
        print("  First 10:")
        for cycle_num, ops in underutilized[:10]:
            ops_str = ', '.join(f"{e}:{o[:30]}" for e, o in ops)
            print(f"    Cycle {cycle_num}: {ops_str}")

    # E) Phase segmentation
    print("\n" + "="*60)
    print("E) PHASE SEGMENTATION")
    print("="*60)

    phases, pause_cycles = segment_phases(cycles)
    print(f"\nPause instructions at cycles: {pause_cycles}")
    print(f"Phases identified: {len(phases)}")
    for i, (phase_type, start, end) in enumerate(phases):
        cycle_count = end - start + 1
        print(f"  Phase {i}: cycles {start}-{end} ({cycle_count} cycles)")

    # F) Bottleneck analysis
    print("\n" + "="*60)
    print("F) BOTTLENECK ANALYSIS")
    print("="*60)

    bottlenecks = count_vliw_slot_usage(cycles)
    print("\nCycles where engine is at slot limit:")
    for engine, count in sorted(bottlenecks.items(), key=lambda x: -x[1]):
        pct = count / total_cycles * 100
        print(f"  {engine}: {count} cycles ({pct:.1f}%)")

    # G) Theoretical limits
    print("\n" + "="*60)
    print("G) THEORETICAL LIMITS")
    print("="*60)

    # Calculate minimum cycles based on operation counts
    print("\nMinimum cycles required per engine:")
    slot_limits = {'alu': 12, 'valu': 6, 'load': 2, 'store': 2, 'flow': 1}
    for engine, count in sorted(engine_counts.items(), key=lambda x: -x[1]):
        if engine in slot_limits:
            min_cycles = (count + slot_limits[engine] - 1) // slot_limits[engine]
            print(f"  {engine}: {count} ops / {slot_limits[engine]} slots = {min_cycles} min cycles")

    # Identify critical path
    print("\nCritical path (highest min cycles):")
    critical = max(
        ((engine, (count + slot_limits.get(engine, 1) - 1) // slot_limits.get(engine, 1))
         for engine, count in engine_counts.items()
         if engine in slot_limits),
        key=lambda x: x[1])
    print(f"  {critical[0]}: {critical[1]} cycles")

    # H) Optimization opportunities summary
    print("\n" + "="*60)
    print("H) OPTIMIZATION OPPORTUNITIES SUMMARY")
    print("="*60)

    valu_min = (engine_counts.get('valu', 0) + 5) // 6
    load_min = (engine_counts.get('load', 0) + 1) // 2
    flow_min = engine_counts.get('flow', 0)

    print(f"""
Current cycles: {total_cycles}

Key bottlenecks (theoretical minimum cycles):
1. valu: {engine_counts.get('valu', 0)} ops / 6 slots = {valu_min} min cycles  <-- CRITICAL PATH
2. load: {engine_counts.get('load', 0)} ops / 2 slots = {load_min} min cycles
3. flow: {engine_counts.get('flow', 0)} ops / 1 slot  = {flow_min} min cycles (mostly vselect)

Gap analysis:
- Current: {total_cycles} cycles
- valu theoretical min: {valu_min} cycles
- GAP: {total_cycles - valu_min} cycles of potential savings ({(total_cycles - valu_min) / total_cycles * 100:.1f}%)

To reach ~1000 cycles, need to REDUCE valu operations by ~{(valu_min - 1000) * 6} ops!

Optimization opportunities:
1. REDUCE valu ops (critical):
   - multiply_add: {engine_op_counts['valu'].get('multiply_add', 0)} ops (hash computation)
   - ^: {engine_op_counts['valu'].get('^', 0)} ops (XOR in hash)
   - +: {engine_op_counts['valu'].get('+', 0)} ops (additions)
   - >>: {engine_op_counts['valu'].get('>>', 0)} ops (shifts)

2. Replace load_offset with vload:
   - {len(vload_sequences)} sequences of 8 load_offsets could use vload
   - Potential: reduce load ops, freeing cycles for valu

3. Eliminate vselect:
   - {total_vselect} vselect ops for tree traversal
   - Arithmetic alternative: idx = 2*idx + 1 + (val & 1)
""")

if __name__ == '__main__':
    main()
