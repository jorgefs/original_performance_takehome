# Kernel Optimization Analysis: 1615 → <1500 cycles

## Executive Summary

**Current:** 1615 cycles
**Target:** <1500 cycles (ideally ~1000 as others achieved)

### Critical Finding
The **valu engine is the true bottleneck**, not load:
- valu: 8216 ops / 6 slots = **1370 min cycles** (CRITICAL PATH)
- load: 2621 ops / 2 slots = 1311 min cycles
- flow: 258 ops / 1 slot = 258 min cycles

**To reach ~1000 cycles, must reduce valu ops by ~2200** (from 8216 to ~6000).

---

## Current Operation Breakdown

### By Engine
| Engine | Ops | Min Cycles | Notes |
|--------|-----|------------|-------|
| valu | 8216 | 1370 | **BOTTLENECK** |
| load | 2621 | 1311 | 80% at capacity |
| flow | 258 | 258 | Mostly vselect |
| alu | 68 | 6 | Negligible |
| store | 64 | 32 | Negligible |

### valu Breakdown (8216 total)
| Operation | Count | Purpose |
|-----------|-------|---------|
| ^ (XOR) | 3072 | Hash stages + node XOR |
| multiply_add | 1952 | 3 stages per hash × 512 hashes |
| + | 1025 | Hash stages + index math |
| >> | 1024 | Hash stages |
| << | 512 | Hash stages |
| & | 608 | Index computation (val & 1) |
| vbroadcast | 21 | Init only |
| - | 2 | Init only |

### Hash Function Analysis
6 stages per hash:
- Stages 0,2,4: `multiply_add` (1 valu op each) = 3 ops
- Stages 1,3,5: 3 valu ops each (^, op, shift) = 9 ops
- **Total: 12 valu ops per hash**

With 512 hashes (256 elements ÷ 8 lanes × 16 rounds):
- Hash ops: 512 × 12 = 6144 valu
- XOR with node: ~512 valu
- Index math: ~1500 valu
- **Total: ~8200 valu** ✓ matches

---

## Optimization Proposals (Ranked by ROI)

### A. Reduce Hash Operations (HIGH IMPACT)
**Potential saving: 500-1500 cycles**

The hash dominates valu usage. Options:

1. **Partial hash for early rounds**: Since the hash is deterministic, could we use a reduced-precision approximation for intermediate rounds and only compute full hash on final round?

2. **Instruction-level fusion**: Look for pairs of operations that could be combined. E.g., in stages 1,3,5:
   ```
   tmp1 = val ^ const
   tmp2 = val >> shift
   val = tmp1 ^ tmp2
   ```
   Could this be 2 ops instead of 3 with different sequencing?

3. **Precompute hash of cached tree levels**: For depths 0-2 where tree values are constants, precompute `myhash(cached_val ^ X)` for common X patterns.

### B. Eliminate vselect with Arithmetic (MEDIUM IMPACT)
**Potential saving: 128-256 cycles**

Current depth 1:
```python
vselect(tmp, condition, level1_right, level1_left)  # 1 flow op
```

Alternative with arithmetic (if level1_right = level1_left + delta):
```python
# If level1_right = level1_left + (some_mask * condition)
val = level1_left + (delta & mask_from_condition)  # 2 valu ops
```

This trades 1 flow op for 2 valu ops. Since flow is bottleneck at depth 1, this could help **if done for depth 2 triplets**.

For depth 2 (3 vselects):
- Could use lookup table indexed by (bit0, bit1) = 4 values
- Or arithmetic: `level2[2*b1 + b0]` but needs gather-load

### C. Replace load_offset with vload (MEDIUM IMPACT)
**Potential saving: 100-400 cycles**

Current: 314 sequences of 8 load_offsets (2560 total)
- 8 load_offsets = 4 cycles (2 per cycle)
- 1 vload = 1 cycle

**But:** load_offset does scatter-gather, vload needs contiguous addresses.

**Opportunity:** For elements that happen to have consecutive tree indices, use vload.
- Statistically rare for random data
- Better for structured patterns

### D. Pipeline hash with loads (MEDIUM IMPACT)
**Potential saving: 50-200 cycles**

Current: Hash group of 3 vectors, then load next group.

Better: Interleave loads across groups more aggressively.
- While hashing group G, preload group G+2 (not just G+1)
- Requires more scratch space but hides load latency

### E. Cache Level 3 Tree Values (LOW IMPACT)
**Potential saving: 50-100 cycles**

Current: Levels 0,1,2 are cached (7 values).
Level 3 has 8 values - could cache them too.

Saves:
- 64 load_offset ops (2 rounds × 32 vectors)
- But adds 8 vbroadcast ops in init

Net effect minimal unless combined with other optimizations.

### F. Reduce Index Update Operations (LOW IMPACT)
**Potential saving: 50-100 cycles**

Current index update (depths 1-9):
```python
tmp1 = val & 1           # valu: &
idx = idx * 2 + base_minus1  # valu: multiply_add
idx = idx + tmp1         # valu: +
```
= 3 valu ops

Could combine:
```python
idx = 2*idx + 1 + (val & 1)  # If multiply_add supports this directly
```

Or precompute `v_one_plus_base_minus1`:
```python
tmp1 = val & 1
idx = multiply_add(idx, 2, tmp1 + base_minus1)  # Still 2 ops
```

---

## Recommended Implementation Order

1. **Start with D (Pipeline hash with loads)** - Low risk, moderate gain
2. **Try A.2 (Instruction fusion)** - Analyze if any hash ops can merge
3. **Implement B for depth 2** - Trade vselect for arithmetic
4. **Investigate A.3 (Precompute cached hashes)** - High potential if feasible

---

## To Reach ~1000 Cycles

The gap from 1615 to 1000 is 615 cycles (38% reduction).

This requires reducing valu from 8216 to ~6000 ops (2200 fewer).

**Most likely approach:**
- The ~1000 cycle solutions probably found a way to **skip or simplify hash computation** for certain rounds/depths.
- Or discovered a **mathematical property** of the hash that allows shortcuts.

Would need to analyze what makes rounds/depths mathematically reducible.
