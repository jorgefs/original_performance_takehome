# Frontier/Bucketing Status (2026-02-02)

## Current state
- Implemented ASM-first frontier/bucketing for depths 0?3 inside `perf_takehome.py`.
- Uses raw flow jumps (cond_jump_rel/jump) and scalar loops to build buckets.
- Depths >=4 still use the classic traversal.
- Kernel length now ~4113 bundles (was 1615).
- `python perf_takehome.py Tests.test_kernel_cycles` timed out at 60s (likely too slow).

## What was added
- Scratch/mem layout for bucket traversal:
  - tmp_vals/tmp_idxs (spill 256 vals/idxs)
  - bucket_vals/bucket_idxs (sorted by bucket)
  - bucket_paths (original path ids)
  - bucket_pos/cnt/base (prefix sums)
- Raw emitter `_RawEmitter` for ASM blocks:
  - zero counts
  - count buckets
  - prefix sum
  - scatter to buckets
  - per-bucket SIMD hash/update
  - unscatter to original order
  - reload v_idx/v_val

## Why cycles exploded
- The bucketing path is **scalar-heavy**:
  - 256-iteration loops for count/scatter/unscatter
  - uses scalar load/store per element (no vectorization)
- Extra flow overhead from cond_jump_rel loops
- Two full data permutations (scatter + unscatter) per depth
- Bucket processing uses only 1 vector at a time (vload/vstore per chunk)
- No VLIW packing inside raw loops (one slot per bundle)

## Interpretation
- The current implementation **reduces gathers** at depth 0?3 but the scalar loop + permutation overhead dominates.
- This can only win if we reduce bookkeeping cost or avoid unscatter for early depths.

## Suggested next steps
1) Restrict bucketing to depth=3 only (depth 0?2 keep classic vselect/cache path).
2) Avoid unscatter for depth 0?2 (ordering is irrelevant if we stay in bucket order through hash/update).
3) Batch per-bucket SIMD processing with multiple vectors per bucket (use vload/vstore of consecutive bucket entries).
4) Add coarse vectorized counting when possible (e.g., bucket index from idx>>bit, reduce scalar loops).

## Files touched
- perf_takehome.py (frontier/bucketing integrated)
- perf_takehome_1615_reconstructed.py (saved 1615 baseline)

