"""
Analyze the real cause of 659 cycles with VALU=4 and load=2.
Classify them into:
A) Blocked waiting for load_offset result (load latency)
B) Blocked by RAW chain in hash (x depends on prev x)
C) Blocked by flow/vselect or masks
"""
import sys
sys.path.insert(0, '.')
from perf_takehome import KernelBuilder, VLEN, HASH_STAGES

def get_rw_sets(engine, slot):
    """Get read and write sets for an instruction."""
    reads = set()
    writes = set()

    if engine == "alu":
        _op, dest, a1, a2 = slot
        writes.add(dest)
        reads.add(a1)
        reads.add(a2)

    elif engine == "valu":
        op = slot[0]
        if op == "vbroadcast":
            _, dest, src = slot
            for i in range(VLEN):
                writes.add(dest + i)
            reads.add(src)
        elif op == "multiply_add":
            _, dest, a, b, c = slot
            for i in range(VLEN):
                writes.add(dest + i)
                reads.add(a + i)
                reads.add(b + i)
                reads.add(c + i)
        else:
            _, dest, a1, a2 = slot
            for i in range(VLEN):
                writes.add(dest + i)
                reads.add(a1 + i)
                reads.add(a2 + i)

    elif engine == "load":
        op = slot[0]
        if op == "load":
            _, dest, addr = slot
            writes.add(dest)
            reads.add(addr)
        elif op == "vload":
            _, dest, addr = slot
            for i in range(VLEN):
                writes.add(dest + i)
            reads.add(addr)
        elif op == "load_offset":
            _, dest, addr, lane = slot
            writes.add(dest + lane)
            reads.add(addr + lane)
        elif op == "const":
            _, dest, val = slot
            writes.add(dest)

    elif engine == "flow":
        op = slot[0]
        if op == "vselect":
            _, dest, cond, a, b = slot
            for i in range(VLEN):
                writes.add(dest + i)
                reads.add(cond + i)
                reads.add(a + i)
                reads.add(b + i)

    return reads, writes

def analyze():
    forest_height = 10
    rounds = 16
    batch_size = 256
    n_nodes = (1 << (forest_height + 1)) - 1

    builder = KernelBuilder()
    builder.build_kernel(forest_height, n_nodes, batch_size, rounds)
    instrs = builder.instrs

    # Find all VALU=4, load=2 cycles
    target_cycles = []
    for i, instr in enumerate(instrs):
        valu_count = len(instr.get('valu', []))
        load_slots = instr.get('load', [])
        load_offset_count = sum(1 for s in load_slots if s[0] == 'load_offset')
        if valu_count == 4 and load_offset_count == 2:
            target_cycles.append(i)

    print(f"=== Found {len(target_cycles)} cycles with VALU=4 and load_offset=2 ===")
    print()

    # Analyze a sample of these cycles
    sample_size = min(50, len(target_cycles))
    sample_indices = target_cycles[:sample_size]

    # Track what's written in recent cycles (for dependency analysis)
    # We'll look at what was written in the previous N cycles
    lookback = 10

    category_A = 0  # Blocked by load_offset result
    category_B = 0  # Blocked by hash RAW chain
    category_C = 0  # Blocked by flow/vselect

    detailed_analysis = []

    for idx in sample_indices:
        instr = instrs[idx]

        # Get current cycle's writes
        current_writes = set()
        current_reads = set()
        for engine, slots in instr.items():
            for slot in slots:
                r, w = get_rw_sets(engine, slot)
                current_writes |= w
                current_reads |= r

        # Look at what was written in previous cycles
        recent_writes = {}  # addr -> (cycle, engine, op)
        for back in range(1, min(lookback + 1, idx + 1)):
            prev_instr = instrs[idx - back]
            for engine, slots in prev_instr.items():
                for slot in slots:
                    _, w = get_rw_sets(engine, slot)
                    for addr in w:
                        if addr not in recent_writes:
                            recent_writes[addr] = (idx - back, engine, slot[0])

        # Look at what would be next (the operations that couldn't be scheduled)
        # Check the next few instructions in the original emission order
        next_instr_info = None
        if idx + 1 < len(instrs):
            next_instr = instrs[idx + 1]
            next_valu = next_instr.get('valu', [])
            if next_valu:
                # Check what the first VALU op in next cycle reads
                first_valu = next_valu[0]
                r, _ = get_rw_sets('valu', first_valu)
                # Find which of these reads was recently written
                blocking_deps = []
                for addr in r:
                    if addr in recent_writes:
                        cycle, eng, op = recent_writes[addr]
                        if cycle == idx:  # Written this cycle - can't be read this cycle
                            blocking_deps.append((addr, eng, op))
                next_instr_info = (first_valu, blocking_deps)

        # Classify based on what's in this cycle
        load_ops = [s for s in instr.get('load', []) if s[0] == 'load_offset']
        valu_ops = instr.get('valu', [])
        flow_ops = instr.get('flow', [])

        # Heuristic classification:
        # If there are load_offset and the VALU ops read from addresses written by load_offset
        # in recent cycles, then we're load-bound

        load_offset_writes = set()
        for slot in load_ops:
            _, dest, addr, lane = slot
            load_offset_writes.add(dest + lane)

        # Check if current VALU reads from recently loaded addresses
        valu_reads_from_load = False
        for slot in valu_ops:
            r, _ = get_rw_sets('valu', slot)
            for addr in r:
                if addr in recent_writes:
                    cycle, eng, op = recent_writes[addr]
                    if eng == 'load' and op == 'load_offset':
                        valu_reads_from_load = True
                        break

        # Check if VALU ops form a RAW chain (same dest/src pattern)
        hash_raw_chain = False
        valu_addrs = []
        for slot in valu_ops:
            op = slot[0]
            if op in ['^', '+', '>>', '<<', 'multiply_add']:
                if op == 'multiply_add':
                    dest = slot[1]
                    src = slot[2]
                else:
                    dest = slot[1]
                    src = slot[2]
                valu_addrs.append((dest, src))
        # Check for chained operations
        for i in range(len(valu_addrs) - 1):
            if valu_addrs[i][0] == valu_addrs[i+1][1]:  # dest of i is src of i+1
                hash_raw_chain = True
                break

        # Check for flow operations in previous cycles blocking us
        flow_blocking = False
        for back in range(1, min(3, idx + 1)):
            prev_instr = instrs[idx - back]
            if prev_instr.get('flow'):
                for slot in prev_instr['flow']:
                    if slot[0] == 'vselect':
                        # Check if vselect wrote to something we read
                        _, w = get_rw_sets('flow', slot)
                        if w & current_reads:
                            flow_blocking = True

        # Classify
        if valu_reads_from_load:
            category_A += 1
            cat = 'A'
        elif hash_raw_chain:
            category_B += 1
            cat = 'B'
        elif flow_blocking:
            category_C += 1
            cat = 'C'
        else:
            # Default to B (most likely hash chain)
            category_B += 1
            cat = 'B'

        detailed_analysis.append({
            'cycle': idx,
            'category': cat,
            'valu_ops': [s[0] for s in valu_ops],
            'has_load_offset': len(load_ops) > 0,
            'valu_reads_from_load': valu_reads_from_load,
            'hash_raw_chain': hash_raw_chain,
            'flow_blocking': flow_blocking,
        })

    print("=== Classification Results ===")
    print(f"Sample size: {sample_size}")
    print(f"A) Blocked by load_offset latency: {category_A} ({100*category_A/sample_size:.1f}%)")
    print(f"B) Blocked by hash RAW chain:      {category_B} ({100*category_B/sample_size:.1f}%)")
    print(f"C) Blocked by flow/vselect:        {category_C} ({100*category_C/sample_size:.1f}%)")
    print()

    # Show some examples
    print("=== Sample Cycle Details ===")
    for d in detailed_analysis[:10]:
        print(f"Cycle {d['cycle']}: Category {d['category']}")
        print(f"  VALU ops: {d['valu_ops']}")
        print(f"  Reads from load: {d['valu_reads_from_load']}, RAW chain: {d['hash_raw_chain']}, Flow block: {d['flow_blocking']}")
        print()

    # Recommendation
    print("=== Recommendation ===")
    if category_A > category_B and category_A > category_C:
        print("Load latency dominates -> Focus on reducing load_offset (bucketing, caching)")
    elif category_B > category_A and category_B > category_C:
        print("Hash RAW chain dominates -> Need software pipelining between streams")
    else:
        print("Flow/vselect dominates -> Reduce or fuse vselect operations")

if __name__ == "__main__":
    analyze()
