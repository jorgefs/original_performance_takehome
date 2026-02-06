# Kernel Optimization Plan: 1615 → <1500 cycles

## Current State
- **Cycles**: 1615
- **Speedup**: 91.5x over baseline (147,734)
- **Tests Passing**: 5/9
- **ASM File**: `kernel_asm_1615.txt`

## Bottleneck Analysis

### Operation Counts
| Engine | Operations | Slots/Cycle | Min Cycles | Status |
|--------|------------|-------------|------------|--------|
| **valu** | 8216 | 6 | **1370** | **CRITICAL PATH** |
| load | 2621 | 2 | 1311 | 80% saturated |
| flow | 258 | 1 | 258 | vselect bottleneck |
| alu | 68 | 12 | 6 | Negligible |
| store | 64 | 2 | 32 | Negligible |

### valu Breakdown (8216 ops total)
| Operation | Count | Purpose |
|-----------|-------|---------|
| ^ (XOR) | 3072 | Hash stages + node XOR |
| multiply_add | 1952 | 3 hash stages × 512 hashes |
| + | 1025 | Hash stages + index math |
| >> | 1024 | Hash stages |
| << | 512 | Hash stages |
| & | 608 | Index computation (val & 1) |

### Per-Depth Analysis
| Depth | Rounds | valu/round | flow/round | load_offset/round |
|-------|--------|------------|------------|-------------------|
| 0 | 2 | 416 | 0 | 0 |
| 1 | 2 | 576 | 32 | 0 |
| 2 | 2 | 576 | 96 | 0 |
| 3-9,14,15 | 9 | 512 | 0 | 256 |
| 10 | 1 | 416 | 0 | 256 |

## Gap Analysis

```
Current cycles:           1615
valu theoretical min:     1370
Overhead:                 245 cycles (18%)

To reach <1579 (Opus 4.5 2hr):  need to save 36+ cycles
To reach <1548 (Sonnet 4.5):    need to save 67+ cycles
To reach <1487 (Opus 4.5 11hr): need to save 128+ cycles
To reach <1363 (improved):      need to save 252+ cycles
To reach ~1000 (user claims):   need to save 615+ cycles
```

## Optimizations Already Applied

1. **VLIW-packed initialization** (-44 cycles)
   - Collected all init operations and used `self.build(init, vliw=True)`

2. **Pre-created hash constants** (-18 cycles)
   - 11 hash constants pre-allocated and broadcast in init phase

3. **Aggressive scheduling** (enabled)
   - `aggressive_schedule = True` removes MEM dependency tracking

4. **Interleaved vector ordering**
   - Alternating halves: 0,16,2,18,4,20... for memory spread

## Optimizations Attempted (No Improvement)

| Attempt | Result | Reason |
|---------|--------|--------|
| Batched idx updates | +76 cycles | Broke load/hash pipelining |
| hash_group=2,3,4 | 0 change | Scheduler finds same packing |
| Sequential ordering | +5 cycles | Interleaved is better |
| Larger chunks depth 0 | Broken | Incorrect results |

## Proposed Optimizations (Not Yet Implemented)

### A. Reduce idx Update Operations (Potential: 50-100 cycles)
**Current**: 3 valu ops per vector per round
```python
tmp1 = val & 1                    # op 1
idx = multiply_add(idx, 2, base)  # op 2
idx = idx + tmp1                  # op 3
```

**Alternative**: Restructure formula to combine ops
- Requires finding mathematical equivalent with 2 ops
- Challenge: base_minus1 constant can't easily combine with tmp1

### B. Cache Level 3 Tree Values (Potential: 0-64 cycles net)
**Current**: 8 load_offsets per vector at depth 3
**Alternative**: Cache 8 level3 values, use 7 vselects

Analysis:
- Saves: 512 load_offsets → 256 cycles
- Costs: 448 vselects → 448 cycles
- **Net: -192 cycles (worse)**

With 3-vselect approach (like depth 2):
- Saves: 256 cycles
- Costs: 192 vselects → 192 cycles
- **Net: +64 cycles (marginal improvement)**

### C. Eliminate vselect with Arithmetic (Potential: 0 cycles)
Replace `vselect(cond, a, b)` with `b ^ (cond & (a ^ b))`
- Trades 1 flow op for 3 valu ops
- **Makes valu bottleneck worse**

### D. Different Traversal Order (Potential: Unknown)
Batch elements by tree path to:
- Share tree node loads across elements hitting same node
- Reduce redundant hash computations

**Complexity**: High - requires sorting/grouping elements

### E. Hash Stage Fusion (Potential: 500+ cycles if feasible)
Look for mathematical shortcuts in hash computation:
- Stages 1,3,5 use 3 ops each
- Challenge: No obvious way to reduce without changing hash

### F. Software Pipelining Across Rounds (Potential: Unknown)
Instead of: all elements for round R, then round R+1
Try: element 0 for rounds 0-15, then element 1, etc.

**Risk**: May not improve due to different dependency structure

## Theoretical Limits

To reach ~1000 cycles (as claimed achievable):
```
1000 cycles × 6 valu slots = 6000 valu ops max
Current valu ops: 8216
Need to reduce by: 2216 ops (27% reduction)
```

This suggests the ~1000 cycle solutions found a way to:
1. Skip or simplify hash computation for some rounds/depths
2. Batch elements that share tree paths
3. Use a fundamentally different algorithm

## Recommended Next Steps

1. **Profile individual cycles** - Use trace visualization to find cycles with low valu utilization

2. **Analyze hash dependencies** - Look for opportunities to overlap hash stages across different vectors

3. **Investigate tree path batching** - Group elements by their first few tree traversal decisions

4. **Try different interleaving patterns** - The current pattern may not be optimal for all depths

5. **Examine ~1000 cycle solutions** - If examples exist, reverse-engineer their approach

## Files

- `perf_takehome.py` - Main kernel builder (current: 1615 cycles)
- `analyze_asm.py` - ASM instrumentation script
- `kernel_asm_1615.txt` - Current optimized ASM output
- `OPTIMIZATION_ANALYSIS.md` - Detailed analysis document
