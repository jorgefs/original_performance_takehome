"""
Test: Optimize depth 2 with 2 vselects instead of 3
Current: 3 vselects per vector (96 per round, 192 total)
Target: 2 vselects per vector (64 per round, 128 total)
Savings: 64 flow cycles
"""
from perf_takehome import *

# Patch the build_kernel method
original_build_kernel = KernelBuilder.build_kernel

def patched_build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
    """
    Modified kernel with optimized depth 2 vselect.

    For depth 2, idx_mem is 10, 11, 12, or 13.
    We want tree values at indices 3, 4, 5, 6 (level2_vals[0..3]).

    Current approach uses (idx_mem & 1) and (idx_mem & 2) as selectors.

    Alternative: Use (idx_mem - 10) directly as a 2-bit selector.
    But vselect only does pairwise selection...

    Actually, can we restructure to use 2 independent vselects then XOR?

    New idea: Instead of vselect tree, use load_offset from a precomputed
    contiguous block in scratch... but load_offset reads from MEM, not scratch.

    Let's try: Cache level2 values in memory (copy to extra_room), then load.
    """
    # For now, just run the original
    return original_build_kernel(self, forest_height, n_nodes, batch_size, rounds)

if __name__ == "__main__":
    # Just verify baseline
    import random
    random.seed(123)
    do_kernel_test(10, 16, 256)
