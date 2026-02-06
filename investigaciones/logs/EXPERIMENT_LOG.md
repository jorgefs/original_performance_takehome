# Experiment Log

## 2026-02-06 Update (post-cold-start optimization)

- New local best in active `perf_takehome.py`: **1396 cycles**.
- Change:
  - `IDX_ARITH_MODE=postadd`
  - `IDX_UPDATE_ARITH_ROUNDS=4,15`
- Validation:
  - `python perf_takehome.py Tests.test_kernel_cycles` -> 1396
  - `python tests/submission_tests.py` -> same status as before (fails only `<1363` threshold)

- Minor improvement explored and kept as optional:
  - `STORE_INDICES=0` -> 1393 (small gain, not a large-step path).

- Wide scan of local variants executed and archived in:
  - `investigaciones/logs/scan_variants_results_2026-02-06.json`
  - Result: no pre-existing `perf_takehome*.py` variant under 1399 (outside active file).

- Conclusion:
  - Tuning/knob space is largely exhausted around 1396-1399.
  - Reaching a **>=30 cycle** jump likely requires a structural kernel change (not parameter tuning).

Target benchmark: forest_height=10, rounds=16, batch_size=256 (tests/submission_tests.py)
Baseline after index-store removal: 1883 cycles.

## Pipeline experiments
- Change: build_hash_pipeline_addr interleaves next-group loads before+after each hash stage.
- Result: 1883 cycles (no change).

## Cache experiments
- Level2 cache (4 nodes) with UNROLL=32: out of scratch.
- Level2 cache (UNROLL=24): correctness failed.
- Level2+Level3 cache (UNROLL=16): correctness failed.

## UNROLL / hash_group
- UNROLL=24 with tail: 1900 cycles (worse).
- hash_group=4: 1883 cycles (no change).
- hash_group=2: 1883 cycles (no change).
## Additional experiments
- Level2 cache only (UNROLL=24): 1933 cycles (worse).
- Level3 cache only (UNROLL=16): incorrect output values.
- Pipeline disabled (bulk load then hash): 1883 cycles (no change).
- Level2 cache (UNROLL=28, optimized temps): 1946 cycles (worse).
- Level3 cache (UNROLL=24, recomputed bits): incorrect output values.
- Fast path with aggressive_schedule=False: 1903 cycles (worse).
- UNROLL=16 (tail 16, caches off): 1899 cycles (worse).
- hash_group=6: 1883 cycles (no change).
- hash_group=8: 1883 cycles (no change).
## Iteration (simplify + iterative ideas)
- Bulk load+hash for depth>1 (no pipeline): 1883 cycles (no change), kept for simplicity.
- Level2 cache via vselect: scratch overflow -> reverted.
- Unroll split 16+16: 1899 cycles (worse).
- Pipeline re-enabled: 1883 cycles (no change), reverted to bulk load for simplicity.
- Alternar pipeline/bulk por round (mezcla): 1883 cycles (no change).
- With level2 cache + unroll 8/8, aggressive_schedule=False: 1822 cycles (worse).
- Level2 cache (rel-3 selection), UNROLL=8+8: 1806 cycles (new best).
- Pipeline vs bulk at depth>2 with level2 cache: 1806 cycles (no change).
- hash_group=2/4 with level2 cache: 1814 cycles (no change).
- Level2 cache via multiply_add: 1938 cycles (worse).
- Partial cache (half vectors) depth==2: 1869 cycles (worse).
- Attempted depth==3 cache via vselect: reverted (too complex / no gain).
- UNROLL_MAIN=12 (fixed) broke tail handling for other batch sizes; reverted.
- UNROLL_MAIN=8 for total_vecs<=8 retains 1814 cycles.
- Reordered level2 vselects: 1814 cycles (no change).
- Reworked pipeline scheduling in build_hash_pipeline_addr: 1814 cycles (no change).
- Tail-first scheduling: 1814 cycles (no change).
- pipeline subgroups (hash_group=4 for depth>2): 1814 cycles (no change).
- cache2 only when vec_count==8: 1922 cycles (worse).

## New attempts (2026-01-28)
- Depth==2 uses pipeline_addr (cache removed) + remove level2 preloads: 1905 cycles (worse) -> reverted.
- UNROLL_MAIN=12 (12+4 split): 1814 cycles (no change) -> reverted.
- fast_hash_group=2 for depth>2: 1814 cycles (no change) -> reverted.
- UNROLL_MAIN=16 (no split): out of scratch -> reverted.
- Fast-path depth>2 per-vector load+hash (serial stages): 1814 cycles (no change) -> reverted.
- Depth==3 full cache (8 nodes) with extra temp vector: 1884 cycles (worse) -> reverted.
- Fused depth0+1 rounds in fast path: 1814 cycles (no change) -> reverted.
- Asymmetric unroll 10+6: 1814 cycles (no change) -> reverted.
- fast_hash_group=1: 1814 cycles (no change) -> reverted.
- Tried switching depth>2 to build_hash_pipeline (node_addr), but would need extra v_node_addr/v_node_val scratch; aborted (not run).
- Idea6 (level1 broadcast reuse per-group): 1818 cycles (worse) -> reverted.
- Idea5 (wavefront prefetch lanes): crash IndexError (invalid load_offset) -> reverted.
- Idea1 (partial depth3 hot cache + fallback loads): 1908 cycles (worse) -> reverted.
- Idea2 (depth3 2-bit table layout): incorrect output values -> reverted.
- Idea4 (level2 cache split by vector half): incorrect output values -> reverted.
- Idea3 (predictive b0 cache): incorrect output values -> reverted.
- Idea7 (speculative idx update from v_tmp1): incorrect output values -> reverted.
- Idea8 (reorder round pairs when depth1->depth2): incorrect output values -> reverted.
- Trajectory unroll (TRAJ_UNROLL=2, VLEN=8 actual): 2208 cycles (worse) -> reverted.
- Corrected wavefront prefetch for VLEN=8: 2527 cycles (worse) -> reverted.
- Depth3 full cache with extra temp (corrected selects): 1859 cycles (worse) -> reverted.
- Depth3 cache b2=0 + fallback loads: 1915 cycles (worse) -> reverted.
- Trajectory unroll TRAJ_UNROLL=4 (VLEN=8): 2616 cycles (worse) -> reverted.
- Depth3 2-bit table cache (low/high) attempt: incorrect output values -> reverted.
- fast_hash_group=8 in fast path: 1814 cycles (no change) -> reverted.
- Simplification: hardcode UNROLL_MAIN=16 and remove tail branch (fast path): 1814 cycles (no change).
- Simplification: inline hash_group=3 and remove fast_hash_group variable, drop group_size_main: 1814 cycles (no change).
- Simplification: factor emit_hash_only helper inside fast path emit_group. 1814 cycles (no change).
- Simplification: factor emit_idx_update helper and inline emit_hash_only calls in loop. 1814 cycles (no change).
- Simplification: replace depth % period with manual wrap in fast path loop. 1814 cycles (no change).
- Simplification: remove fast path debug/pause instructions and debug init in emit_group. 1814 cycles (no change).
- Simplification: merge idx update loops (per-u) and last-round idx build. 1803 cycles (new best) -> saved perf_takehome_1803.py.
- Simplification: inline idx_update loops (no helper call) -> still 1803 cycles; keep.
- Simplification: prioritize last-round idx handling (depth==0) before idx_update branch, and isolate leaf idx on last round. 1803 cycles (no change).
- Simplification attempt: reuse depth==1 b0 for idx_update (skip &). Caused out-of-range loads -> reverted to 1803.
- Simplification: reorder depth==2 vselects to favor b1-first. 1803 cycles (no change).
- Simplification: group per-u stores (idx then val) in single branch. 1803 cycles (no change).
- Simplification: depth==2 block compacted (comments + ordering). 1803 cycles (no change).
- Simplification: move last-round index handling to epilogue, remove idx_update on last round. 1798 cycles (new best) -> saved perf_takehome_1798.py.
- Simplification: unrolled round depths via fixed table (0..10,0..4). 1798 cycles (no change).
- Reverted to perf_takehome_1803.py to restore correct indices (skipping last-round idx_update caused incorrect output indices). 1803 cycles.
- Simplification: fixed depth table + always idx_update (no last-round branches). 1803 cycles (no change).
- Simplification attempt: replace + v_base_minus1 with + (v_base_minus1+1). Incorrect output values -> reverted.
- Attempted reuse of depth==2 b0 (rel3) for idx_update: incorrect output values; reverted to 1803.
- Simplification (1): explicit loop for rounds 0..14 + round 15 epilogue (keep idx_update). 1803 cycles (no change).
- Simplification (2): precomputed round_do_idx flags. 1803 cycles (no change).
- Simplification (1+2 combined): explicit last round + round_do_idx flags. 1803 cycles (no change).
- Reordered schedule: hash rounds in pairs then idx_update -> incorrect output values, reverted.
- Reorder attempt: idx_update inside emit_hash_only for depth 1/2 (interleaved per-u) caused out-of-range loads; reverted to 1803.
- Reorder: idx_update split by half (u<mid, u>=mid). 1814 cycles (worse) -> reverted to 1803.
- Depth==2 prefetch load_offset lane0 (speculative) before vselect: 1803 cycles (no change) -> reverted.
- Depth>2 lane-interleaved loads + hash (custom pipeline): 2455 cycles (worse) -> reverted.
- Mixed idx_update schedule: per-u for depths<=2, batched for depths>2. 1803 cycles (no change) -> reverted.
- Hash strategy 1: split hash stages by half (stages interleaved half0->half1). 1803 cycles (no change) -> reverted.
- Hash strategy 2: split hash stages by half (half1->half0). 1803 cycles (no change) -> reverted.
- Hash strategy 3: sequential half hash (build_hash_vec_multi on halves). 1803 cycles (no change) -> reverted.
- Hash strategy 4: hash_group=5 for depth>2 pipeline. 1803 cycles (no change) -> reverted.
- Hash strategy 5: hash_group=7 for depth>2 pipeline. 1803 cycles (no change) -> reverted.
- Hash interleave (stage slices of 4 vectors) caused incorrect output values; reverted.
- Depth==3 precomputed cache via vselect over 8 nodes (v_level3): 1841 cycles (worse) -> saved perf_takehome_1841_depth3cache.py and reverted to 1803.
- hash_group=2 for depth>2 pipeline in fast path: 1803 cycles (no change) -> reverted.
- hash_group=4 for depth>2 pipeline in fast path: 1803 cycles (no change) -> reverted.
- hash_group=6 for depth>2 pipeline in fast path: 1803 cycles (no change) -> reverted.
- Depth2: precompute v_base_minus3 (v_base_minus1 - v_four) and use v_idx + v_base_minus3 (drop extra subtract) -> 1782 cycles (new best) -> saved perf_takehome_1782.py.
- idx_update: replace add with vselect between v_base_minus1/v_base_minus2 (precomputed) -> 1911 cycles (worse) -> saved perf_takehome_1911_vselect_idx.py and reverted to 1782.
- Depth2: compute b1 as (rel3 & 2) directly (use v_two) instead of shift+and -> 1766 cycles (new best) -> saved perf_takehome_1766.py.
- Depth2: derive b0/b1 directly from v_idx bits (b1 inverted), drop rel3 compute -> 1752 cycles (new best) -> saved perf_takehome_1752.py.
- Remove unused v_four vector constant (setup reduction) -> 1751 cycles (new best) -> saved perf_takehome_1751.py.
- Depth1: replace multiply_add (diff/left) with vselect between v_level1_left/right; drop level1_diff_rl -> 1739 cycles (new best) -> saved perf_takehome_1739.py.
- Fast path cleanup: remove unused period var and emit_idx_update helper -> 1739 cycles (no change).
- Depth2: try multiply_add with v_level2 diffs to reduce flow vselects -> 1857 cycles (worse) -> saved perf_takehome_1857_depth2_muladd.py and reverted.
- hash_group=2 with new depth1/depth2 changes -> 1739 cycles (no change) -> reverted.
- v_neg_forest: compute via v_base_minus1 - v_one (remove v_zero broadcast) -> 1738 cycles (new best) -> saved perf_takehome_1738.py.
- hash_group=4 with latest changes -> 1738 cycles (no change) -> reverted.
- Depth1 idx update via multiply_add (idx=2*b0+1) broke indices / load_offset OOB -> saved perf_takehome_idx_muladd_bug.py and reverted.
- idx_update re-ordered: muladd with v_base_minus1 then add b0 -> 1729 cycles (new best) -> saved perf_takehome_1729.py.
- Depth2: replace final vselect with (b1>>1, diff, muladd) to avoid flow slot -> 1859 cycles (worse) -> saved perf_takehome_1859_depth2_muladd_select.py and reverted.
- hash_group=1 with latest changes -> 1729 cycles (no change) -> reverted.
- Fast path: remove tmp_val_addr/tmp_idx_addr base and compute base addrs directly (drop two ALU ops) -> 1728 cycles (new best) -> saved perf_takehome_1728.py.
- Fast path: compute val base from idx base + batch_size const -> 1730 cycles (worse) -> saved perf_takehome_1730_idx_base.py and reverted.
- Depth2: replace one b0 vselect with multiply_add using v_level2_diff01 -> 1840 cycles (worse) -> saved perf_takehome_1840_depth2_one_muladd.py and reverted.
- idx_update: move AND after muladd (same ops) -> 1728 cycles (no change) -> reverted.
- hash_group=16 (single group no interleave) with latest changes -> 1732 cycles (worse) -> reverted.
- hash_group=5 with latest changes -> 1728 cycles (no change) -> reverted.
- hash_group=7 with latest changes -> 1728 cycles (no change) -> reverted.
- Hardcode vector consts for base (v_base_plus1=8, v_base_minus1=-6, v_neg_forest=-7) -> 1729 cycles (worse) -> saved perf_takehome_1729_consts.py and reverted.
- Depth1: use v_idx&1 for node select (skip v_idx recompute) -> load_offset OOB (v_idx uninitialized) -> saved perf_takehome_1728_depth1_idxbit_bug.py and reverted.
- Depth1: init v_idx to base then use v_idx&1 for node select -> load_offset OOB (v_idx drift) -> saved perf_takehome_1728_depth1_init_idx_bug.py and reverted.
- Base vectors from scalar (vbroadcast base±1) instead of v_forest_values_p add/sub -> 1729 cycles (worse) -> saved perf_takehome_1729_base_scalar.py and reverted.
- Fast path: special-case group1 base addresses to avoid base ALU adds for u=0 (direct base addrs) -> 1727 cycles (new best) -> saved perf_takehome_1727.py.
- Precompute base2 addrs in setup (attempt) had builder error; reverted to 1727.
- Precompute base2 addrs (correct) and override base for group2 -> 1729 cycles (worse) -> saved perf_takehome_1729_base2_precompute.py and reverted.
- Base2 addresses as const scratch (no ALU) -> 1729 cycles (worse) -> saved perf_takehome_1729_base2_const.py and reverted.
- Depth2: split b0/b1 and vselect into separate loops (grouped ops) -> 1943 cycles (worse) -> saved perf_takehome_1943_depth2_splitloops.py and reverted.
- Store order: precompute idx outputs then store idx loop, then store val loop -> 1727 cycles (no change).
- Compute v_base_plus1 via v_two - v_base_minus1 (dependency swap) -> 1727 cycles (no change) -> reverted.
- idx_update split into three loops (b0, muladd, add) -> 1763 cycles (worse) -> saved perf_takehome_1763_idxupdate_split.py and reverted.
- Build fast path with aggressive_schedule disabled -> 1771 cycles (worse) -> saved perf_takehome_1771_no_aggr_sched.py and reverted.
- build_hash_pipeline_addr: increase loads_per_stage (factor 1 instead of 2) -> 1727 cycles (no change) -> reverted.
- Extra-room/last-seen cache attempt (per-vector cached idx/val with vselect) in depth>2 pipeline: 1810 cycles (worse) -> saved perf_takehome_1810_cacheA.py and reverted to 1803.
- hash_group=4 in fast-path hash pipeline (current 1727 baseline): 1727 cycles (no change) -> reverted.
- Depth2 attempt: remove v_tmp3 by re-encoding vselects (insufficient temps, incorrect mapping) -> reverted to 1727 before testing.
- Depth>2 bulk load_offset then hash (no pipeline interleave): 1731 cycles (worse) -> reverted to 1727.
- Depth>2 pipeline split into two halves (8+8 vectors): 1727 cycles (no change) -> reverted to 1727.
- UNROLL_MAIN=32 (single group) to process all vectors at once: scratch overflow -> reverted to 1727.
- Hash stages per-vector (compact stage schedule) for depth<=2: 1727 cycles (no change) -> reverted to 1727.
- Depth2 selection via base+diff (vselect b1 + multiply_add b0) with v_level2_diff01/23: incorrect output values -> reverted to 1727.
- Interleave per-round hash pipeline by halves (emit_hash_only_range for 8+8 vectors): 1723 cycles (new best) -> saved perf_takehome_1723.py.
- Per-round chunking 4x4 vectors (hash+idx_update per chunk): 1708 cycles (new best) -> saved perf_takehome_1708.py.
- Per-round chunking 2x8 vectors: 1708 cycles (no change) -> reverted to 1708.
- Chunked depth>2 pipeline hash_group=2: 1708 cycles (no change) -> reverted to 1708.
- Chunked depth>2 pipeline hash_group=1: 1708 cycles (no change) -> reverted to 1708.
- Depth2 base+diff (2x vselect + multiply_add) with chunking: incorrect output values -> reverted to 1708.
- Chunking only for depth>2 (depth<=2 full-width): 1727 cycles (worse) -> reverted to 1708.
- Chunk order permutation (0,8,4,12) with chunk=4: 1708 cycles (no change) -> reverted to 1708.
- Chunked depth>2 pipeline hash_group=4: 1708 cycles (no change) -> reverted to 1708.
- Per-round chunking with tail (chunk=5): 1717 cycles (worse) -> reverted to 1708.
- Per-round chunking with tail (chunk=3): 1699 cycles (new best) -> saved perf_takehome_1699.py.
- Per-round chunking with tail (chunk=1): 1697 cycles (new best) -> saved perf_takehome_1697.py.
- Chunk=1 with even-then-odd order: 1694 cycles (new best) -> saved perf_takehome_1694.py.
- Chunk=1 with interleaved halves order (0,8,1,9,...): 1695 cycles (worse) -> reverted to 1694.
- Chunk=1 with odd-then-even order: 1696 cycles (worse) -> reverted to 1694.
- Depth2 base+diff (corrected mapping) with chunk=1: 1700 cycles (worse) -> reverted to 1694.
- Chunk=1 with stride-4 order (0,4,8,12,...): 1697 cycles (worse) -> reverted to 1694.
- Chunk=1 with bit-reversal order (0,8,4,12,...): 1696 cycles (worse) -> reverted to 1694.
- Chunk=1 with grouped order (0-3,8-11,4-7,12-15): 1695 cycles (worse) -> reverted to 1694.
- Chunk=1 alternating order by round parity: 1696 cycles (worse) -> reverted to 1694.
- Depth>2 prefetch next vector lanes (prefetch=2) with chunk=1: 1696 cycles (worse) -> reverted to 1694.
- Depth>2 prefetch next vector lanes (prefetch=4) with chunk=1: 1696 cycles (worse) -> reverted to 1694.
- Depth>2 lagged idx_update (hash current, idx_update prev): 1696 cycles (worse) -> reverted to 1694.
- Depth>2 reverse order (depth<=2 keep even-odd): 1701 cycles (worse) -> reverted to 1694.
- Depth>2 order odd-then-even (depth<=2 keep even-odd): 1696 cycles (worse) -> reverted to 1694.
- Depth>2 chunk=2 with overlapping order caused load_offset OOB (vector overlap). Reverted to 1694.
- Depth>2 chunk=2 with even starts only: 1696 cycles (worse) -> reverted to 1694.
- Order tweak even then odd reverse: 1702 cycles (worse) -> reverted to 1694.
- UNROLL_MAIN=8 with 4 groups: order list had starts >= vec_count, causing count=0 and division by zero in pipeline; reverted to 1694.
- Order tweak (0,8,2,10,4,12,6,14,1,9,3,11,5,13,7,15): 1695 cycles (worse) -> reverted to 1694.
- Depth2 base+diff (2 vselect + multiply_add) with current chunk=1 order: 1700 cycles (worse) -> reverted to 1694.
- Depth1 multiply_add (diff) instead of vselect: 1706 cycles (worse) -> reverted to 1694.
- Order sequential 0..15: 1697 cycles (worse) -> reverted to 1694.
- Simplification: remove unused emit_idx_update helper (no cycle change, still 1694).
- Chunk=4 with non-overlapping starts (0,4,8,12): 1708 cycles (worse) -> reverted to 1694.
- Simplification: hoist order tuple outside round loop (no cycle change, still 1694).
- Depth>2 order sequential (depth<=2 keep even-odd): 1696 cycles (worse) -> reverted to 1694.
- UNROLL_MAIN=32 with shared temp vectors: 7072 cycles (much worse) -> reverted to 1694.
- Order tweak (evens/odds within halves): 1695 cycles (worse) -> reverted to 1694.
- Round split: hash all vectors then idx_update all vectors: 1723 cycles (worse) -> reverted to 1694.
- Order tweak (0,1,8,9,4,5,12,13,2,3,10,11,6,7,14,15): 1698 cycles (worse) -> reverted to 1694.
- Chunk=3 with non-overlapping starts (0,3,6,9,12,15): 1699 cycles (worse) -> reverted to 1694.
- Simplification: replace count=min(...) with count=1 in fast-path loop (no cycle change, still 1694).
- Depth2 selection reordered (b1inv-first then b0 select): 1694 cycles (no change) -> reverted to 1694.
- Depth<=2 round split (hash then idx_update), depth>2 interleaved: 1723 cycles (worse) -> reverted to 1694.
- Order tweak (0,4,8,12,2,6,10,14,1,5,9,13,3,7,11,15): 1697 cycles (worse) -> reverted to 1694.
- Simplification: remove chunk/min, use count=1 directly (no cycle change, still 1694).
- Order switch at round 8 (even-odd then odd-even): 1694 cycles (no change) -> reverted to 1694.
- Per-group order swap (group1 even-odd, group2 odd-even): 1694 cycles (no change) -> reverted to 1694.
- idx_update reorder (muladd before b0 AND): 1694 cycles (no change) -> reverted to 1694.
- Option1 attempt: keep v_idx relative + build_hash_pipeline (v_node_addr/v_node_val), update idx = 2*idx + 1 + b0, store indices directly: incorrect output values -> reverted to 1694.
- Depth>=8 use hash_group=2 (else 3) in pipeline: 1694 cycles (no change) -> reverted to 1694.
- Simplification: inline count=1 (remove min/loop count) in fast-path round loop: 1694 cycles (no change).
- build_hash_pipeline_addr loads_per_stage denominator=len(stages) (more loads): 1694 cycles (no change) -> reverted.
- Depth>2 order permutation (start at 8): 1698 cycles (worse) -> reverted to 1694.
- Store order swapped (val then idx): 1694 cycles (no change) -> reverted to 1694.
- Depth>2 hash_group=2 on odd rounds: 1694 cycles (no change) -> reverted to 1694.
- Depth==3 bulk loads (no pipeline), else pipeline: 1694 cycles (no change) -> reverted to 1694.
- Hash stages interleaved per-vector (local helper) for depth<=2: 1694 cycles (no change) -> reverted to 1694.
- Group order swap (process second half first): 1695 cycles (worse) -> reverted to 1694.
- Depth==3 hash_group=1 (else 3): 1694 cycles (no change) -> reverted to 1694.
- Order rotate (start at 8): 1695 cycles (worse) -> reverted to 1694.
- Order even then odd reverse: 1702 cycles (worse) -> reverted to 1694.
- Simplification: remove chunk/min, hardcode count=1 in round loop (no change, still 1694).
- Depth<=2 odd-even order (depth>2 even-odd): 1698 cycles (worse) -> reverted to 1694.
- Simplification: hoist order tuple in fast-path round loop (no change, still 1694).
- Depth2 base+diff (2 vselect + multiply_add) with current order: 1700 cycles (worse) -> reverted to 1694.
\n- 2026-01-29: bulk load+hash (no pipeline) for depth>2 with chunk=2, start=even only. Result 1708 cycles (worse). Saved as perf_takehome_1708_bulk2.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: moved idx_update out of per-start loop to after round (emit_idx_update vec_count). Result 1735 cycles (worse). Saved as perf_takehome_1735_idxpost.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: depth==3 cache via vselect with v_level3[0..7] and extra temps (b0/b1/b2 from v_idx). Correctness failed (Incorrect output values). Saved as perf_takehome_depth3_cache_bad.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: depth==3 cache using rel = v_idx - base to derive b0/b1/b2, selection with v_level3 and extra temps. Correctness failed (Incorrect output values). Saved as perf_takehome_depth3_cache_rel_bad.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: chunk=2, even starts only, keep build_hash_pipeline_addr. Result 1708 cycles (worse). Saved as perf_takehome_1708_chunk2.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: hash_group=4 in build_hash_pipeline_addr. Result 1694 cycles (no change). Saved as perf_takehome_1694_hashgroup4.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: depth==3 cache with rel = (v_idx - base - 7) and vselect (extra temps, v_level3, v_four, v_seven). Correctness ok, but 1782 cycles (worse). Saved as perf_takehome_1782_depth3_rel7.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: start order 0..15 sequential (chunk=1). Result 1697 cycles (worse). Saved as perf_takehome_1697_seqorder.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: hash_group=2 in build_hash_pipeline_addr. Result 1694 cycles (no change). Saved as perf_takehome_1694_hashgroup2.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: UNROLL_MAIN=32 with single group. Scratch overflow (assert), no cycles. Saved as perf_takehome_unroll32_scratch_overflow.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: UNROLL_MAIN=24 + UNROLL_TAIL=8 with dynamic start order. Result 1695 cycles (worse). Saved as perf_takehome_1695_unroll24.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: chunk=4 with starts (0,4,8,12). Result 1708 cycles (worse). Saved as perf_takehome_1708_chunk4.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: chunk=4 starts (0,4,8,12) with idx_update after loop for full vec_count. Result 1723 cycles (worse). Saved as perf_takehome_1723_chunk4_idxpost.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: chunk=4 (starts 0,4,8,12) with hash_group=2. Result 1708 cycles (worse). Saved as perf_takehome_1708_chunk4_hg2.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: fast path build with aggressive_schedule disabled before VLIW build. Result 1739 cycles (worse). Saved as perf_takehome_1739_noaggr_fastpath.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: UNROLL_MAIN=8 with 4 groups (offsets 0/64/128/192) and dynamic start list. Result 1767 cycles (worse). Saved as perf_takehome_1767_unroll8_4groups.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: rebuild fast path hash pipeline with hash_group=1 for depth>2. Result 1694 cycles (no change). Reverted to perf_takehome_1694.py.
\n- 2026-01-29: chunk=2 with even starts only, build_hash_pipeline_addr hash_group=3 (default). Result 1708 cycles (worse). Reverted to perf_takehome_1694.py.
\n- 2026-01-29: chunk=2 (even starts), hash_group=1 fast path. Result 1708 cycles (worse). Reverted to perf_takehome_1694.py.
\n- 2026-01-29: build_hash_pipeline_addr with prefetch_factor=1 in fast path. Result 1694 cycles (no change). Reverted to perf_takehome_1694.py.
\n- 2026-01-29: idx_update reorder: v_tmp2 = b0 + v_base_minus1, then muladd with v_tmp2 (remove final add). Result 1700 cycles (worse). Saved as perf_takehome_1700_idxupdate_tmp2.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: build_hash_pipeline_addr prefetch_factor=3 in fast path. Result 1694 cycles (no change). Reverted to perf_takehome_1694.py.
\n- 2026-01-29: simplification: precomputed starts tuple + separate idx/val store loops. Result 1694 cycles (no change). Kept as simplified baseline.
\n- 2026-01-29: simplification: fixed chunk=1 (no min) using starts tuple. Result 1694 cycles (no change). Kept.
\n- 2026-01-29: depth>2 bulk load_offset + build_hash_vec_multi (no pipeline). Result 1694 cycles (no change). Reverted to perf_takehome_1694.py.
\n- 2026-01-29: VLIW scheduler window_size=2048. Result 1694 cycles (no change). Reverted to 1024.
\n- 2026-01-29: depth<=2 use chunk=2 (even starts), depth>2 chunk=1 (even-odd). Result 1706 cycles (worse). Reverted to perf_takehome_1694.py.
\n- 2026-01-29: simplification: removed unused emit_idx_update helper, precomputed starts tuple, removed min(count) for chunk=1. Result 1694 cycles (no change). Kept.
\n- 2026-01-29: depth>=5 chunk=2 (even starts), depth<5 chunk=1 (even-odd). Result 1696 cycles (worse). Reverted to chunk=1.
\n- 2026-01-29: idx_update batched for depth<=2, per-start for depth>2. Result 1723 cycles (worse). Reverted.
\n- 2026-01-29: VLIW scheduler window_size=256. Result 1703 cycles (worse). Reverted to 1024.
\n- 2026-01-29: depth>2 chunk=2 with hash_group=1 (interleave loads across 2 vectors), depth<=2 chunk=1. Result 1696 cycles (worse). Reverted to perf_takehome_1694.py.
\n- 2026-01-29: simplification (kept): remove unused emit_idx_update helper, precompute starts tuple, remove min count for chunk=1. Result 1694 cycles (no change).
\n- 2026-01-29: depth>2 start order interleaved halves (0,8,1,9,...) while depth<=2 even-odd. Result 1698 cycles (worse). Reverted.
\n- 2026-01-29: depth==3 hot4 cache (nodes 7-10) with rel=idx-7, b0/b1 select, b2 fallback to loaded value. Correctness failed (Incorrect output values). Saved as perf_takehome_depth3_hot4_bad.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: depth==3 full cache (8 nodes) with rel = (addr - forest_base - 7), b0/b1/b2 vselect chain. Correct, but 1774 cycles (worse). Saved as perf_takehome_1774_depth3_fullcache.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: depth==3 hot4 cache (nodes 7-10) with rel = addr - forest_base - 7, b0/b1 select, b2 fallback to loaded. Correct but 1864 cycles (worse). Saved as perf_takehome_1864_depth3_hot4.py, reverted to perf_takehome_1694.py.
\n- 2026-01-29: depth>2 wavefront pair (count=2) with interleaved loads while hashing u0; depth>2 chunk=2 starts (0,8,2,10,...). Result 1698 cycles (worse). Reverted to perf_takehome_1694.py.
\n- 2026-01-29: depth>2 wavefront count=3 (u0 hash, u1/u2 loads interleaved), chunk=3 starts (0,3,6,9,12,15). Result 1700 cycles (worse). Reverted to perf_takehome_1694.py.
\n- 2026-01-29: depth>2 wavefront count=2 with even starts (0,2,4,6,8,10,12,14), depth<=2 chunk=1. Result 1696 cycles (worse). Reverted to perf_takehome_1694.py.

## 2026-01-29 Attempt: depth>2 chunk=2 + hash_group=1 (paired starts 0,8,2,10,...)
- Change: For depth>2, used chunk=2 and set hash_group=1 to enable pipeline across two vectors; depth<=2 kept chunk=1.
- Starts: depth>2 starts=(0,8,2,10,4,12,6,14)
- Correctness: passes submission_tests correctness; do_kernel_test(10,16,256) still fails round 0 (baseline behavior due to no pause).
- Cycles: 1698 (worse than 1694 baseline). Reverted.


## 2026-01-29 Attempt: depth>2 chunk=4 + hash_group=1 (starts 0,8,4,12)
- Change: For depth>2, used chunk=4 and hash_group=1 to create 4 groups per call.
- Starts: depth>2 starts=(0,8,4,12)
- Correctness: passes submission_tests correctness.
- Cycles: 1696 (worse than 1694 baseline). Reverted.


## 2026-01-29 Attempt: depth>2 chunk=4 + hash_group=2 (starts 0,8,4,12)
- Change: For depth>2, used chunk=4 and hash_group=2.
- Starts: depth>2 starts=(0,8,4,12)
- Correctness: passes submission_tests correctness.
- Cycles: 1696 (worse than 1694 baseline). Reverted.


## 2026-01-29 Improvement: UNROLL_MAIN=32 + shared v_tmp3
- Change: Increased UNROLL_MAIN to 32 (single group) and replaced per-vector v_tmp3 array with a shared v_tmp3 to fit scratch. Removed unused group_size_const.
- Correctness: passes submission_tests correctness.
- Cycles: 1664 (new best). Snapshot: perf_takehome_1664_unroll32_sharedtmp3.py


## 2026-01-29 Attempt: depth>2 chunk=2 + hash_group=1 with UNROLL_MAIN=32
- Change: For depth>2, chunk=2, starts interleaved across halves; hash_group=1; kept shared v_tmp3.
- Result: submission_tests crashed with IndexError in load_offset (out-of-range memory access). Reverted.


## 2026-01-29 Improvement: interleaved start order with UNROLL_MAIN=32
- Change: Start order interleaved halves: 0,16,1,17,...,15,31 (chunk=1). Shared v_tmp3 retained.
- Correctness: passes submission_tests correctness.
- Cycles: 1661 (new best). Snapshot: perf_takehome_1661_interleaved_starts.py


## 2026-01-29 Attempt: start order stride-8 groups
- Change: Start order 0,8,16,24,1,9,17,25,...,7,15,23,31 (chunk=1).
- Cycles: 1664 (worse than 1661). Reverted.


## 2026-01-29 Improvement: interleaved even/odd within halves
- Change: Start order 0,16,2,18,...,14,30, then 1,17,3,19,...,15,31 (chunk=1). Shared v_tmp3 retained.
- Correctness: passes submission_tests correctness.
- Cycles: 1660 (new best). Snapshot: perf_takehome_1660_interleaved_evenodd_halves.py


## 2026-01-29 Improvement: depth>2 quarter-interleaved start order
- Change: depth>2 starts=0,16,4,20,8,24,12,28,2,18,6,22,10,26,14,30,1,17,5,21,9,25,13,29,3,19,7,23,11,27,15,31; depth<=2 kept even/odd interleaved halves.
- Correctness: passes submission_tests correctness.
- Cycles: 1657 (new best). Snapshot: perf_takehome_1657_depthgt2_quarter_interleave.py


## 2026-01-29 Attempt: depth>2 start order interleave 8/4
- Change: depth>2 starts=0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31.
- Cycles: 1659 (worse than 1657). Reverted.


## 2026-01-29 Attempt: depth<=2 even/odd simple order
- Change: depth<=2 starts=0,2,4,...,30,1,3,...,31 while depth>2 kept quarter interleave.
- Cycles: 1661 (worse than 1657). Reverted.


## 2026-01-29 Attempt: v_tmp3 4-bank instead of shared
- Change: v_tmp3_b0..b3 with banked temp for depth2 selection (u mod 4).
- Cycles: 1657 (no improvement). Reverted to shared v_tmp3.


## 2026-01-29 Attempt: depth>2 chunk=2 hash_group=2
- Change: depth>2 chunk=2, starts=(0,16,4,20,8,24,12,28,2,18,6,22,10,26,14,30), hash_group=2.
- Cycles: 1659 (worse than 1657). Reverted.


## 2026-01-29 Attempt: depth>2 grouped by mod-4 block
- Change: depth>2 starts grouped as [0,16,4,20,8,24,12,28],[1,17,5,21,9,25,13,29],[2,18,6,22,10,26,14,30],[3,19,7,23,11,27,15,31].
- Cycles: 1657 (no improvement). Reverted.


## 2026-01-29 Attempt: move idx update after hash loop per round
- Change: For each round, hashed all vectors first, then ran idx update loop for all vectors.
- Cycles: 1709 (worse). Reverted.


## 2026-01-29 Attempt: depth>2 hash_group=2
- Change: depth>2 uses hash_group=2 (others 3).
- Cycles: 1657 (no change).

## 2026-01-29 Attempt: depth>2 hash_group=4
- Change: depth>2 uses hash_group=4 (others 3).
- Cycles: 1657 (no change). Reverting to hash_group=3.


## 2026-01-29 Attempt: depth3 full cache via vselect
- Change: preloaded v_level3[7..14], added depth==3 vselect chain (rel_minus7 bits).
- Result: correctness failed (Incorrect output values). Reverted.


## 2026-01-29 Attempt: chunk=2 for all depths (even starts only)
- Change: chunk=2, starts=0..30 even (pairs).
- Cycles: 1663 (worse). Reverted.


## 2026-01-29 Attempt: depth>2 hash_group=1
- Change: depth>2 uses hash_group=1 (others 3).
- Cycles: 1657 (no change). Reverted.


## 2026-01-29 Attempt: aggressive_schedule=False in fast path
- Cycles: 1706 (worse). Reverted.


## 2026-01-29 Attempt: reverse start order
- Change: starts=31..0 for all depths.
- Cycles: 1661 (worse). Reverted.


## 2026-01-29 Attempt: UNROLL_MAIN=24 + tail group of 8
- Change: UNROLL_MAIN=24, emit_group(24) + emit_group(8) with count guard.
- Cycles: 1697 (worse). Reverted.


## 2026-01-29 Attempt: use quarter-interleave starts for all depths
- Change: starts order set to depth>2 quarter-interleave for all depths.
- Cycles: 1664 (worse). Reverted.


## 2026-01-29 Attempt: depth3 full cache (correct vselect chain)
- Change: preloaded v_level3[7..14], added depth==3 vselect chain with temp vectors.
- Correctness: OK.
- Cycles: 1757 (worse). Reverted.


## 2026-01-29 Attempt: depth>2 starts by bit1 blocks
- Change: starts=0,8,16,24,2,10,18,26,4,12,20,28,6,14,22,30,1,9,17,25,3,11,19,27,5,13,21,29,7,15,23,31.
- Cycles: 1657 (no change). Reverted.


## 2026-01-29 Attempt: depth2 bit-extract from relative idx
- Change: depth2 b0/b1 from (v_idx - forest_base).
- Result: correctness failed (Incorrect output values). Reverted.


## 2026-01-29 Attempt: round-pair unroll (r0+r1 inside start loop)
- Change: emit depth0+depth1 back-to-back per start.
- Cycles: 1668 (worse). Reverted.


## 2026-01-29 Attempt: depth3 partial hot cache with mask + manual loads
- Change: preload hot nodes 7-10; compute hot_mask and vselect; manual load_offset per lane at depth3.
- Correctness: OK.
- Cycles: 1799 (worse). Reverted.


## 2026-01-29 Simplification: remove unused emit_idx_update + precompute starts tuples
- Change: removed unused helper, made round_depths/starts tuples.
- Cycles: 1657 (no change). Kept.


## 2026-01-29 Simplification: remove depth2 cache (use pipeline loads)
- Change: deleted v_level2 loads and depth2 vselects; depth2 now uses build_hash_pipeline_addr.
- Cycles: 1707 (worse). Reverted.


## 2026-01-29 Simplification: remove level1 cache (pipeline loads)
- Change: removed level1 cache and depth1 vselect; depth1 uses pipeline loads.
- Result: IndexError in frozen simulator (load_offset out of range). Reverted.


## 2026-01-29 Simplification: depth1 pipeline + init idx + update at depth0
- Change: removed level1 cache, initialized v_idx to forest base, enabled idx update at depth0, depth1 uses pipeline loads.
- Result: IndexError in frozen simulator (load_offset out of range). Reverted.


## 2026-01-29 Simplification: UNROLL_MAIN=16 + two groups
- Change: UNROLL_MAIN=16, start orders adjusted to 16-wide, emit two groups.
- Cycles: 1696 (worse). Reverted.


## 2026-01-29 Simplification attempt: remove depth0 fast path + init idx
- Change: depth0 uses pipeline; init v_idx to forest base (introduced v_zero).
- Result: NameError (v_zero missing) during build; reverted to 1657 baseline.


## 2026-01-29 Simplification: depth0 via pipeline + init idx
- Change: removed root cache and used build_hash_pipeline_addr at depth0; init v_idx to forest base.
- Result: correctness failed (Incorrect output values). Reverted.


## 2026-01-29 Simplification: sequential starts
- Change: starts=0..vec_count-1 for all depths (removes interleaving order).
- Cycles: 1661. Snapshot: perf_takehome_1661_sequential_starts.py


## 2026-01-29 Simplification: sequential starts + chunk=2
- Change: chunk=2, starts=0,2,4,...
- Cycles: 1663 (worse). Reverted to sequential starts (chunk=1).


## 2026-01-29 Simplification: depth>2 single chunk
- Change: for depth>2, chunk=vec_count, starts=(0,), hash_group=vec_count (no interleaving).
- Cycles: 1717 (worse). Reverted to sequential starts (1661).


## 2026-01-29 Simplification: remove depth2 cache (sequential starts)
- Change: removed v_level2 loads and depth2 vselects; depth2 uses pipeline loads.
- Cycles: 1708. Snapshot: perf_takehome_1708_simplified_nol2.py


## 2026-01-29 Simplification: remove depth1 + depth2 caches (pipeline for depth1/2)
- Change: removed level1 cache; depth1 computes v_idx then pipeline loads. depth2 already pipeline.
- Cycles: 1934. Snapshot: perf_takehome_1934_simplified_nol1_l2.py


## 2026-01-29 Simplification: merge depth1/2/else branches
- Change: depth>0 all use build_hash_pipeline_addr (single else).
- Cycles: 2114 (no change). Kept.


## 2026-01-29 Simplification: UNROLL_MAIN=16 with two groups
- Change: UNROLL_MAIN=16, emit two groups; pipeline for all depth>0; depth0 pipeline.
- Cycles: 2115. Snapshot: perf_takehome_2115_simplified_unroll16.py


## 2026-01-29 Simplification: UNROLL_MAIN=8 with four groups
- Change: UNROLL_MAIN=8, emit four groups.
- Cycles: 2117. Snapshot: perf_takehome_2117_simplified_unroll8.py


## 2026-01-29 Simplification: UNROLL_MAIN=4 with eight groups
- Change: UNROLL_MAIN=4, emit eight groups.
- Cycles: 2128. Snapshot: perf_takehome_2128_simplified_unroll4.py


## 2026-01-29 Simplification: UNROLL_MAIN=2 with sixteen groups
- Change: UNROLL_MAIN=2, emit sixteen groups.
- Cycles: 4135. Snapshot: perf_takehome_4135_simplified_unroll2.py


## 2026-01-29 Simplification: hash_group=1 for pipeline
- Change: all build_hash_pipeline_addr use hash_group=1.
- Cycles: 2115 (no change). Kept.


## 2026-01-29 Simplification: single chunk per round
- Change: chunk=vec_count, starts=(0,) to simplify per-round scheduling.
- Cycles: 2115 (no change). Snapshot: perf_takehome_2115_simplified_unroll16_chunkall.py


## 2026-01-29 Simplification attempt: UNROLL_MAIN=8 (two groups) from chunkall
- Change: UNROLL_MAIN=8, two groups of 128 elements.
- Result: Incorrect output values (submission_tests). Reverted.


## 2026-01-29 Simplification: hash_group=count (single group per chunk)
- Change: build_hash_pipeline_addr uses hash_group=count (no interleave).
- Cycles: 2115 (no change). Snapshot: perf_takehome_2115_simplified_hashgroup_count.py


## 2026-01-29 Simplification: manual load+hash (no pipeline helper)
- Change: inline load_offset per lane + build_hash_vec_multi for all depths.
- Cycles: 2115 (no change). Snapshot: perf_takehome_2115_simplified_manual_load_hash.py


## 2026-01-29 Simplification: remove unused temps/helpers
- Change: drop unused v_tmp3_shared and emit_idx_update helper (no assembly change).
- Cycles: 2115 (no change). Snapshot: perf_takehome_2115_simplified_remove_unused.py


## 2026-01-29 Simplification on 1657 base: hash_group=count for depth>2
- Change: build_hash_pipeline_addr uses hash_group=count for depth>2 pipeline.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup_count_depthgt2.py


## 2026-01-29 Simplification on 1657 base: single chunk per round
- Change: chunk=vec_count, starts=(0,) (no interleave).
- Cycles: 1717 (worse). Snapshot: perf_takehome_1717_chunkall_on_1657.py


## 2026-01-29 Simplification on 1657 base: manual load+hash for depth>2
- Change: inline load_offset + build_hash_vec_multi for depth>2 (no pipeline helper).
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_manual_load_hash_depthgt2.py


## 2026-01-29 Simplification on 1657 base: chunk=2 paired starts
- Change: chunk=2 with even-only starts (paired vectors) for all depths.
- Cycles: 1663 (worse). Snapshot: perf_takehome_1663_chunk2_pairs.py


## 2026-01-29 Simplification on 1657 base: hash_group=2 for depth>2
- Change: build_hash_pipeline_addr uses hash_group=2 for depth>2.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup2_depthgt2.py


## 2026-01-29 Simplification on 1657 base: hash_group=4 for depth>2
- Change: build_hash_pipeline_addr uses hash_group=4 for depth>2.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup4_depthgt2.py


## 2026-01-29 Simplification on 1657 base: chunk=2 for depth<=2 only
- Change: depth<=2 uses chunk=2 (even-only starts), depth>2 unchanged.
- Cycles: 1663 (worse). Snapshot: perf_takehome_1663_chunk2_shallow_only.py


## 2026-01-29 Simplification on 1657 base: reverse depth>2 starts
- Change: depth>2 starts order reversed.
- Cycles: 1687 (worse). Snapshot: perf_takehome_1687_depthgt2_reverse_starts.py


## 2026-01-29 Idea #6: halves-interleave starts for depth>2
- Change: depth>2 starts [0,8,1,9,...,23,31] (halves interleave).
- Cycles: 1659 (worse). Snapshot: perf_takehome_1659_depthgt2_halves_interleave.py


## 2026-01-29 Simplification: idx_update helper inlined per range
- Change: replace inline idx update loop with helper emit_idx_update_range.
- Cycles: 1662 (worse). Snapshot: perf_takehome_1662_idxupdate_helper.py


## 2026-01-29 Simplification on 1657 base: manual load+hash for depth>2 (v2)
- Change: inline load_offset + build_hash_vec_multi for depth>2 (redo on baseline).
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_manual_load_hash_depthgt2_v2.py


## 2026-01-29 Idea: depth>2 4-way stride starts
- Change: depth>2 starts [0,8,16,24,1,9,...,23,31].
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_depthgt2_4way_stride.py


## 2026-01-29 Idea: manual load+hash for depth<=4 (invalid)
- Change: depth<=4 uses manual load_offset + hash to avoid pipeline.
- Result: IndexError in load_offset (out-of-range memory access). Reverted.


## 2026-01-29 Idea: store values before indices
- Change: swap store order (values first, then indices).
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_store_values_first.py


## 2026-01-29 Idea: hash_group by depth
- Change: depth 3-5 uses hash_group=2; depth 6-10 uses hash_group=4.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup_by_depth.py


## 2026-01-29 Additional Idea #1: depth2 pipeline (remove cache)
- Change: depth==2 uses build_hash_pipeline_addr instead of cached vselect.
- Cycles: 1723 (worse). Snapshot: perf_takehome_1723_depth2_pipeline.py


## 2026-01-29 Additional Idea #2: depth1 pipeline (remove cache)
- Change: depth==1 uses build_hash_pipeline_addr instead of cached vselect.
- Cycles: 1698 (worse). Snapshot: perf_takehome_1698_depth1_pipeline.py


## 2026-01-29 Additional Idea #3: depth1+2 pipeline (remove caches)
- Change: depth==1 and depth==2 use build_hash_pipeline_addr.
- Cycles: 1954 (worse). Snapshot: perf_takehome_1954_depth1_2_pipeline.py


## 2026-01-29 Additional Idea #4: quarter-interleave starts for shallow depths
- Change: depth<=2 starts use quarter-interleave order (same as depth>2).
- Cycles: 1664 (worse). Snapshot: perf_takehome_1664_shallow_quarter_interleave.py


## 2026-01-29 Additional Idea #5: hash_group=1 for depth>2
- Change: build_hash_pipeline_addr uses hash_group=1 for depth>2.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup1_depthgt2.py


## 2026-01-29 Additional Idea #6: reorder depth2 vselect chain (invalid)
- Change: select by b1 first then b0 for depth==2.
- Result: Incorrect output values. Reverted.


## 2026-01-29 Additional Idea #7: idx_update after all starts per round
- Change: move idx_update out of per-start loop (after all hashes for the round).
- Cycles: 1709 (worse). Snapshot: perf_takehome_1709_idxupdate_after_round.py


## 2026-01-29 Additional Idea #8: even-odd starts for depth>2
- Change: depth>2 starts order even indices then odd indices.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_depthgt2_evenodd_starts.py


## 2026-01-29 Additional Idea #9: mod-4 starts for depth>2
- Change: depth>2 starts order by i % 4 (0s,1s,2s,3s).
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_depthgt2_mod4_starts.py


## 2026-01-29 Additional Idea #10: shallow sequential starts
- Change: depth<=2 starts sequential order 0..31.
- Cycles: 1662 (worse). Snapshot: perf_takehome_1662_shallow_sequential_starts.py


## 2026-01-29 Additional Idea #11: round-pair with same starts
- Change: process r and r+1 per start (shared starts order).
- Cycles: 1668 (worse). Snapshot: perf_takehome_1668_roundpair_same_starts.py


## 2026-01-29 Additional Idea #12: block store values then indices
- Change: store all values first, then all indices.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_store_block_values_then_indices.py


## 2026-01-29 Additional Idea #13: stagger depth>2 starts by round parity
- Change: depth>2 starts alternate between quarter-interleave and even-odd by round.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_depthgt2_staggered_rounds.py


## 2026-01-29 Additional Idea #14: hash_group by round
- Change: depth>2 hash_group=2 for rounds 0-7; hash_group=4 for rounds 8-15.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup_by_round.py


## 2026-01-29 Additional Idea #15: depth>2 starts split by depth
- Change: depth 3-6 uses quarter-interleave; depth 7-10 uses even-odd.
- Cycles: 1660 (worse). Snapshot: perf_takehome_1660_depthgt2_split_by_depth.py


## 2026-01-29 Additional Idea #16: bit-reversal starts for depth>2
- Change: depth>2 starts in 5-bit bit-reversal order.
- Cycles: 1659 (worse). Snapshot: perf_takehome_1659_depthgt2_bitreverse.py


## 2026-01-29 Additional Idea #17: round-mod4 start patterns for depth>2
- Change: depth>2 starts switch among 4 patterns by round_idx%4.
- Cycles: 1659 (worse). Snapshot: perf_takehome_1659_depthgt2_roundmod4_starts.py


## 2026-01-29 Additional Idea #18: hash_group matrix by depth/round
- Change: depth<=6 uses hash_group 2/3 by round parity; depth>6 uses 4/2.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup_depth_round_matrix.py


## 2026-01-29 Additional Idea #19: idx_update muladd reorder
- Change: idx_update uses muladd with b0 then add base_minus1.
- Cycles: 1662 (worse). Snapshot: perf_takehome_1662_idxupdate_reorder_muladd.py


## 2026-01-29 Additional Idea #20: depth3 hot4 cache with fallback
- Change: depth==3 uses hot4 vselect with rel>>2 check + fallback load.
- Cycles: 1854 (worse). Snapshot: perf_takehome_1854_depth3_hot4_fallback.py


## 2026-01-29 Additional Idea #21: chunk=2 for depth>6
- Change: depth>6 uses chunk=2 and even-only starts.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_chunk2_depthgt6.py


## 2026-01-29 Additional Idea #22: block-of-4 starts for depth>2
- Change: depth>2 starts in 4-wide blocks per half.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_depthgt2_block4_starts.py


## 2026-01-29 Additional Idea #23: hash_group by start parity
- Change: depth>2 hash_group=2 for even starts, 4 for odd.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup_by_startparity.py


## 2026-01-29 Additional Idea #24: UNROLL_MAIN=16 (invalid)
- Change: UNROLL_MAIN=16 with 16-wide start lists.
- Result: Incorrect output values. Snapshot: perf_takehome_unroll16_incorrect.py


## 2026-01-29 Additional Idea #25: disable aggressive_schedule
- Change: fast path aggressive_schedule=False.
- Cycles: 1706 (worse). Snapshot: perf_takehome_1706_no_aggr_schedule.py


## 2026-01-29 Additional Idea #26: vliw=False
- Change: build body without VLIW scheduling.
- Cycles: 11225 (much worse). Snapshot: perf_takehome_11225_vliw_off.py


## 2026-01-29 Additional Idea #27: zigzag halves starts for depth>2
- Change: depth>2 starts [0,16,1,17,...,15,31].
- Cycles: 1659 (worse). Snapshot: perf_takehome_1659_depthgt2_zigzag_halves.py


## 2026-01-29 Additional Idea #28: hash_group=6 for depth>2
- Change: build_hash_pipeline_addr uses hash_group=6 for depth>2.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup6_depthgt2.py


## 2026-01-29 Additional Idea #29: shallow zigzag halves starts
- Change: depth<=2 starts [0,16,1,17,...,15,31].
- Cycles: 1659 (worse). Snapshot: perf_takehome_1659_shallow_zigzag_halves.py


## 2026-01-29 Additional Idea #30: reverse within quarter groups
- Change: depth>2 starts reversed within quarter blocks.
- Cycles: 1659 (worse). Snapshot: perf_takehome_1659_depthgt2_reverse_within_quarters.py


## 2026-01-29 Additional Idea #31: depth1 temp swap
- Change: depth==1 uses v_tmp2 for b0 and v_tmp1 for selected value.
- Cycles: 1659 (worse). Snapshot: perf_takehome_1659_depth1_tmp_swap.py


## 2026-01-29 Additional Idea #32: depth2 temp swap
- Change: depth==2 swaps b0/b1 temp usage and vselect inputs.
- Cycles: 1661 (worse). Snapshot: perf_takehome_1661_depth2_tmp_swap.py


## 2026-01-29 Additional Idea #33: depth1 manual load_offset
- Change: depth==1 uses load_offset instead of L1 cache vselect.
- Cycles: 1698 (worse). Snapshot: perf_takehome_1698_depth1_manual_load.py


## 2026-01-29 Additional Idea #34: depth2 manual load_offset
- Change: depth==2 uses load_offset instead of L2 cache vselect.
- Cycles: 1723 (worse). Snapshot: perf_takehome_1723_depth2_manual_load.py


## 2026-01-29 Additional Idea #35: chunk=2 for depth>2
- Change: depth>2 uses chunk=2 with even starts.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_chunk2_depthgt2.py


## 2026-01-29 Additional Idea #36: hash_group=8 for depth>2
- Change: build_hash_pipeline_addr uses hash_group=8 for depth>2.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup8_depthgt2.py


## 2026-01-29 Additional Idea #37: shallow starts by round parity
- Change: depth<=2 uses sequential starts on even rounds, even-odd on odd rounds.
- Cycles: 1658 (worse). Snapshot: perf_takehome_1658_shallow_roundparity_starts.py


## 2026-01-29 Additional Idea #38: depth>2 round-parity sequential
- Change: depth>2 uses sequential starts on odd rounds, quarter-interleave on even rounds.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_depthgt2_roundparity_seq.py


## 2026-01-29 Additional Idea #39: shallow reverse starts
- Change: depth<=2 starts reversed order.
- Cycles: 1687 (worse). Snapshot: perf_takehome_1687_shallow_reverse_starts.py


## 2026-01-29 Additional Idea #40: chunk=4 for depth>2
- Change: depth>2 uses chunk=4 with 8 start positions.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_chunk4_depthgt2.py


## 2026-01-29 Additional Idea #41: chunk=2 for depth<=2
- Change: depth<=2 uses chunk=2 with even starts.
- Cycles: 1663 (worse). Snapshot: perf_takehome_1663_chunk2_shallow.py


## 2026-01-29 Additional Idea #42: chunk=8 for depth>2 (invalid)
- Change: depth>2 uses chunk=8 with starts (0,16) only.
- Result: Incorrect output values. Snapshot: perf_takehome_chunk8_incorrect.py


## 2026-01-29 Additional Idea #43: chunk=8 for depth>2 (full coverage)
- Change: depth>2 uses chunk=8 with starts (0,8,16,24).
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_chunk8_depthgt2.py


## 2026-01-29 Additional Idea #44: chunk=16 for depth>2
- Change: depth>2 uses chunk=16 with starts (0,16).
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_chunk16_depthgt2.py


## 2026-01-29 Additional Idea #45: round-parity bit-reverse for depth>2
- Change: depth>2 alternates quarter-interleave and bit-reverse by round parity.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_depthgt2_roundparity_bitreverse.py


## 2026-01-29 Additional Idea #46: reverse vload order
- Change: reverse vload order for v_val vectors.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_reverse_vload_order.py


## 2026-01-29 Additional Idea #47: depth>2 group by bit3
- Change: depth>2 starts grouped by bit3 (0-7,16-23,8-15,24-31).
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_depthgt2_bit3_group.py


## 2026-01-29 Additional Idea #48: round-cycle-2 starts for depth>2
- Change: depth>2 alternates quarter-interleave and zigzag-halves every 2 rounds.
- Cycles: 1659 (worse). Snapshot: perf_takehome_1659_depthgt2_roundcycle2.py


## 2026-01-29 Additional Idea #49: level1 broadcast inside depth1
- Change: move v_level1_left/right vbroadcast into depth==1 path.
- Cycles: 1783 (worse). Snapshot: perf_takehome_1783_level1_broadcast_in_depth1.py


## 2026-01-29 Additional Idea #50: level2 broadcast inside depth2
- Change: move v_level2 vbroadcast into depth==2 path.
- Cycles: 1767 (worse). Snapshot: perf_takehome_1767_level2_broadcast_in_depth2.py


## 2026-01-29 Additional Idea #51: depth1 load_offset after idx update
- Change: depth==1 uses load_offset from node address (correct idx).
- Cycles: 1698 (worse). Snapshot: perf_takehome_1698_depth1_load_offset_correct_idx.py


## 2026-01-29 Additional Idea #52: depth2 load_offset (repeat)
- Change: depth==2 uses load_offset for node values.
- Cycles: 1723 (worse). Snapshot: perf_takehome_1723_depth2_load_offset_again.py


## 2026-01-29 Additional Idea #53: use build_hash_vec_multi_stages in depth<=2 (invalid)
- Change: depth==1/2 uses build_hash_vec_multi_stages directly.
- Result: build() failed (invalid slot format). Reverted.


## 2026-01-29 Additional Idea #54: depth0 pipeline load (invalid)
- Change: depth==0 uses build_hash_pipeline_addr (uses v_zero not defined).
- Result: NameError in build_kernel. Reverted.


## 2026-01-29 Additional Idea #55: depth0 pipeline load (fixed)
- Change: add v_zero and use build_hash_pipeline_addr for depth==0.
- Cycles: 1740 (worse). Snapshot: perf_takehome_1740_depth0_pipeline.py


## 2026-01-29 Additional Idea #56: idx_update pre-add
- Change: idx_update uses tmp2=b0+base_minus1 then muladd.
- Cycles: 1662 (worse). Snapshot: perf_takehome_1662_idxupdate_preadd.py


## 2026-01-29 Additional Idea #57: depth>2 4-way stride alt
- Change: depth>2 starts [0,8,16,24,4,12,20,28,...].
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_depthgt2_4way_stride_alt.py


## 2026-01-29 Additional Idea #58: hash_group=5 for depth>2
- Change: build_hash_pipeline_addr uses hash_group=5 for depth>2.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup5_depthgt2.py


## 2026-01-29 Additional Idea #59: add_imm for addr increments (invalid)
- Change: use flow add_imm for tmp_val/tmp_idx addr increments.
- Result: Incorrect output values. Snapshot: perf_takehome_addimm_addr_incorrect.py


## 2026-01-29 Additional Idea #60: hash_group=7 for depth>2
- Change: build_hash_pipeline_addr uses hash_group=7 for depth>2.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup7_depthgt2.py


## 2026-01-29 Additional Idea #61: reverse level2 init order
- Change: initialize v_level2 in reverse order.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_reverse_level2_init.py


## 2026-01-29 Additional Idea #62: swap level1 load order
- Change: load level1 right then left.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_swap_level1_load_order.py


## 2026-01-29 Additional Idea #63: swap level1 broadcast order
- Change: vbroadcast right then left.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_swap_level1_broadcast_order.py


## 2026-01-29 Additional Idea #64: move root vbroadcast after level1
- Change: v_root_val broadcast moved after v_level1 broadcasts.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_move_root_broadcast.py


## 2026-01-29 Additional Idea #65: shallow 4-way stride starts
- Change: depth<=2 starts [0,8,16,24,1,9,...,7,15,23,31].
- Cycles: 1661 (worse). Snapshot: perf_takehome_1661_shallow_4way_stride.py


## 2026-01-29 Additional Idea #66: hash_group by half
- Change: depth>2 hash_group=2 for starts <16, 4 for starts >=16.
- Cycles: 1657 (no change). Snapshot: perf_takehome_1657_hashgroup_by_half.py

2026-01-30: Idea #67 depth<=2 chunk=2 starts even-only -> incorrect output (round 0). Reverted to baseline 1657.
2026-01-30: Idea #68 depth2 b1 via shift+and instead of &v_two -> incorrect output (round 0). Reverted to baseline 1657.
2026-01-30: Idea #69 UNROLL_MAIN=8 with 4 groups (offsets 0/64/128/192) + 8-start orders -> incorrect output (round 0). Reverted to baseline 1657.
2026-01-30: Idea #70 depth>2 alternate starts by round parity -> incorrect output (round 0). Reverted to baseline 1657.
2026-01-30: Idea #71 round-pair interleave (zip starts0/starts1) -> incorrect output (round 0). Reverted to baseline 1657.
2026-01-30: Idea #72 idx_update split into 3 passes (&, multiply_add, +) -> incorrect output (round 0). Reverted to baseline 1657.
2026-01-30: Idea #73 store values then indices (two loops) -> incorrect output (round 0). Reverted to baseline 1657.
2026-01-30: Idea #74 hash_group dynamic (depth>=8 ->2 else 3) -> incorrect output (round 0). Reverted to baseline 1657.
2026-01-30: Idea #75 depth>2 starts 4-way stride (0,8,16,24...) -> incorrect output (round 0). Reverted to baseline 1657.
2026-01-30: Idea #76 idx_update preadd (tmp2=tmp1+base_minus1, multiply_add with tmp2) -> incorrect output (round 0). Reverted to baseline 1657.
2026-01-30: Idea #77 depth>2 chunk=2 with even-only starts (also tested perf_takehome_1657_chunk2_depthgt2.py) -> incorrect output (round 0). Reverted to baseline 1657.
2026-01-30: Idea #78 hash_group=2 for depth>2 -> incorrect output (round 0). Reverted to baseline 1657.
2026-01-30: Idea #79 aggressive_schedule=False in fast path -> incorrect output (round 0). Reverted to baseline 1657.
