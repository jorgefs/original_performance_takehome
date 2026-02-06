# Session State - Kernel Optimization

## Current Status (Updated 2026-02-02)
- **Cycles**: 1615
- **Target**: < 1579 (next threshold), ideally < 1363 (best known)
- **Speedup**: 91.5x over baseline (147,734)
- **Tests passing**: 5/9

## Submission Thresholds (from submission_tests.py)
```
< 147734  baseline
< 18532   updated starting point
< 2164    opus4 many hours
< 1790    opus45 casual        <- PASSING
< 1579    opus45 2hr           <- NEXT TARGET (need -36 cycles)
< 1548    sonnet45 many hours
< 1487    opus45 11hr
< 1363    opus45 improved      <- BEST KNOWN
```

## VALU Analysis (KEY)
```
Total VALU ops: 8184
Theoretical minimum: 8184 / 6 slots = 1364 cycles
Current: 1615 cycles (84% VALU utilization)
Gap: 251 cycles due to RAW dependencies
```

**CRITICAL**: Target 1363 < theoretical 1364 implies the best solution has FEWER VALU ops, not just better scheduling.

## Bottleneck Analysis
```
valu:  8216 ops / 6 slots = 1370 min cycles (CRITICAL PATH)
load:  2621 ops / 2 slots = 1311 min cycles
flow:   258 ops / 1 slot  =  258 min cycles
alu:     68 ops / 12 slots =   6 min cycles
store:   64 ops / 2 slots  =  32 min cycles
```

## Latest Session Work

### Approaches Tried (ALL FAILED)
| Approach | Result | Notes |
|----------|--------|-------|
| store_indices=False | 1613 | User said already tried before, not useful |
| Skip last round idx_update | 1612 | Minimal improvement |
| v_neg_forest conditional | - | Reverted |

### Fundamental Issue
- RAW dependencies in hash chain prevent full VALU utilization
- Hash: 6 stages, each depends on previous (cannot parallelize within vector)
- idx_update: 3 ops minimum (AND, multiply_add, ADD)
- No obvious way to reduce total VALU ops

## Previous Session Findings

### VALU Utilization Analysis
```
6 slots: 912 cycles (56.5%) - fully utilized
4 slots: 659 cycles (40.8%) - PROBLEM: 2 slots wasted each
Other:    44 cycles (2.7%)
```

### Strategies Tested (ALL FAILED)
| Strategy | Result | Reason |
|----------|--------|--------|
| chunk=2,4 | +6-8 | Slightly worse scheduling |
| depth3 bucketing | +87 | flow engine bottleneck |
| hash_group=1..32 | +0 | Scheduler handles it |
| batched_idx | +76 | Broke load/hash pipelining |
| starts patterns | +4+ | All worse |
| window_size changes | varies | No improvement |

## Key Insight: 4-VALU Stretch
- Cycles 446-1091: 646 consecutive cycles with only 4 VALU
- All have 2 load_offset (fully utilizing load engine)
- Cause: Not enough independent work due to hash dependencies

## What Would Be Needed for 1363
1. Reduce total VALU ops below 8184 (algorithmic change)
2. OR achieve near-perfect scheduling (>98% utilization) - seems impossible

## Possible Future Approaches
1. Mathematical simplification for idx_update formula
2. Cache more tree levels (level 3 = 8 nodes)
3. Find unused simulator instructions
4. Completely different algorithm structure

## Files to Revert
```bash
git checkout -- perf_takehome.py
```

## Command to Resume
```bash
cd "C:\Users\OEM\proyectos_gito\test2\original_performance_takehome"
python tests/submission_tests.py
```
