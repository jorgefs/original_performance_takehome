# Claude Work Summary (Performance Takehome)

This file summarizes the work recorded in the moved notes/logs and the current
state of the optimization effort. It is ASCII-only by design.

## Current status (2026-02-06)
- Default (FAST_SCHED=1): 1397 cycles (perf_takehome.py, fast path, indices not stored)
- Best known in this repo: 1395 cycles (FAST_SCHED=0; full scheduler search)
- Fallback path now includes a generalized (non-optimized) depth0-2 cache idea for presentation.
- Next target remains <1363 cycles (Opus45 improved harness threshold).

## Key conclusions
- Hash RAW chain dominates; it limits independent work for full VALU usage.
- Biggest gains came from avoiding gathers in early depths and from better scheduling.
- Depth6 chunked VSelect (frontier-style) is correct in no-VLIW but breaks under VLIW scheduling due to aliasing/interleaving; not shipping.

## Milestones (selected)
| Cycles | File | Notes |
| ------ | ---- | ----- |
| 1883 | perf_takehome_1883.py | Early baseline in experiment log |
| 1803 | perf_takehome_1803.py | Simplified idx update / scheduling |
| 1782 | perf_takehome_1782.py | Depth2 base_minus3 simplification |
| 1766 | perf_takehome_1766.py | Depth2 b1 from (rel3 & 2) |
| 1752 | perf_takehome_1752.py | Depth2 bits derived from idx |
| 1739 | perf_takehome_1739.py | Depth1 vselect change |
| 1728 | perf_takehome_1728.py | Drop base addr temps in fast path |
| 1723 | perf_takehome_1723.py | Half-interleaved hash scheduling |
| 1708 | perf_takehome_1708.py | Per-round chunking |
| 1694 | perf_takehome_1694.py | Chunk=1 even-odd ordering |
| 1657 | perf_takehome_1657_baseline.py | Later baseline with pipeline |
| 1399 | perf_takehome_1399_candidate_v15.py | Best known 1399 candidate |
| 1397 | perf_takehome.py | Default fast scheduler (FAST_SCHED=1) |
| 1395 | perf_takehome.py | Best with full scheduler search (FAST_SCHED=0) |

Notes:
- perf_takehome_1798.py was reported but reverted due to incorrect output.
- Several depth3 cache and vselect variants regressed or failed correctness.
- Depth6 chunked VSelect experiments live only behind flags in perf_takehome.py and are not correct under VLIW.

## Kept variants (high-signal)
- perf_takehome.py (current best 1399 + generalized fallback)
- perf_takehome_1399_candidate_v15.py (clean fast-path 1399)
- perf_takehome_1399_candidate_v15_general.py (generalized fallback for presentation)
- outline_1399.py (algorithm outline for explanation)
- perf_takehome_1657_baseline.py (later baseline used by many variants)
- perf_takehome_1728.py (fast path addr simplification)
- perf_takehome_1739.py (depth1 vselect change)
- perf_takehome_1752.py (depth2 idx-bit derivation)
- perf_takehome_1782.py (depth2 base_minus3)

## Three key methodological changes that produced big cycle drops
1) Early-depth vector selection (depth 0-2) to remove gathers and exploit contiguity.
2) Hash pipeline interleaving loads/compute to hide memory latency.
3) Reordered 32-vector schedule (order_variant 0) informed by the dependency DAG to maximize bundle utilization.

Everything else was moved under investigaciones/ for archival.
