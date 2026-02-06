# Outline of the current algorithm (1399-cycle fast path) and the generic path
# NOTE: This is explanatory Python, not runnable against the simulator.

from typing import List


def kernel_outline_fast_path(mem, forest_height=10, rounds=16, batch_size=256):
    # Fast path is specialized for the benchmark case.
    assert forest_height == 10 and rounds == 16 and batch_size == 256

    # 1) Base pointers (layout is fixed in build_mem_image)
    forest_values_p = 7
    inp_indices_p = forest_values_p + (2 ** forest_height - 1)
    inp_values_p = inp_indices_p + batch_size

    # 2) Cache depth 0-2 nodes in scratch (scalar -> vector broadcast)
    # These are constant for all lanes in a round and avoid gathers for early depths.
    root = mem[forest_values_p + 0]
    level1_left = mem[forest_values_p + 1]
    level1_right = mem[forest_values_p + 2]
    level2 = [mem[forest_values_p + 3 + i] for i in range(4)]

    # 3) Load initial values and indices into vector registers
    # v_idx[u], v_val[u] are VLEN-wide vectors for u in 0..31 (32*8=256 lanes)
    v_idx = [mem[inp_indices_p + u * 8 : inp_indices_p + (u + 1) * 8] for u in range(32)]
    v_val = [mem[inp_values_p + u * 8 : inp_values_p + (u + 1) * 8] for u in range(32)]

    # 4) Fixed schedule for 16 rounds: depths [0..10,0..4]
    round_depths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]

    # 5) Per-round processing (order is chosen to improve ILP)
    for round_idx in range(rounds):
        depth = round_depths[round_idx]
        for u in range(32):
            # 5.1) node value selection
            if depth == 0:
                node_val = [root] * 8
            elif depth == 1:
                bit0 = [x & 1 for x in v_val[u]]
                node_val = [level1_right if b else level1_left for b in bit0]
                v_idx[u] = [forest_values_p + 1 + b for b in bit0]
            elif depth == 2:
                # Two-bit selection via a vselect tree (in SIMD code)
                b0 = [x & 1 for x in v_idx[u]]
                b1 = [x & 2 for x in v_idx[u]]
                # Mapping matches the SIMD vselect tree
                node_val = [level2[(bb1 >> 1) * 2 + bb0] for bb0, bb1 in zip(b0, b1)]
            else:
                # Gather for depth >= 3
                node_val = [mem[forest_values_p + idx] for idx in v_idx[u]]

            # 5.2) hash update (exact stages)
            v_val[u] = hash_pipeline(v_val[u], node_val, round_idx)

            # 5.3) index update (except last store-only round)
            if depth != forest_height:
                bit0 = [x & 1 for x in v_val[u]]
                v_idx[u] = [2 * idx + (1 + (1 - b)) for idx, b in zip(v_idx[u], bit0)]

    # 6) Store results back
    for u in range(32):
        mem[inp_values_p + u * 8 : inp_values_p + (u + 1) * 8] = v_val[u]
        mem[inp_indices_p + u * 8 : inp_indices_p + (u + 1) * 8] = v_idx[u]

    return mem


def kernel_outline_generic(mem, forest_height, rounds, batch_size):
    # Generic fallback (correct for any size). Not micro-optimized.
    forest_values_p = mem[4]
    inp_indices_p = mem[5]
    inp_values_p = mem[6]
    n_nodes = mem[1]

    # Preload depth 0-2 nodes (indices 0..6) to reuse when possible.
    cache_nodes = [mem[forest_values_p + i] for i in range(7)]

    for h in range(rounds):
        for i in range(batch_size):
            idx = mem[inp_indices_p + i]
            val = mem[inp_values_p + i]

            # Use cached node values for indices 0..6; otherwise load from memory.
            if 0 <= idx < 7:
                node_val = cache_nodes[idx]
            else:
                node_val = mem[forest_values_p + idx]

            # Hash and index update (same as reference)
            val = myhash(val ^ node_val, h, i)
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= n_nodes else idx

            mem[inp_values_p + i] = val
            mem[inp_indices_p + i] = idx

    return mem


def hash_pipeline(val_vec, node_vec, round_idx):
    # Placeholder: in real code this is the HASH_STAGES pipeline.
    # The kernel uses SIMD-friendly multiply_add where possible.
    return [((v ^ n) * 0x9E3779B1) & 0xFFFFFFFF for v, n in zip(val_vec, node_vec)]


def myhash(val, h, i):
    # Placeholder for the exact HASH_STAGES sequence.
    return (val + 0x9E3779B1 + h + i) & 0xFFFFFFFF
