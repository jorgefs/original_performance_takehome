"""
Create a modified version that uses build_hash_pipeline_addr_with_idx
and skips the separate idx update for depth > 2.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Step 1: Add the new function after build_hash_pipeline_addr
old_build_hash_end = '''            if next_g < num_groups:
                while load_progress[next_g] < next_total:
                    emit_load_for_group(next_g, next_start, next_vecs)

        return slots

    def build_kernel('''

new_build_hash_end = '''            if next_g < num_groups:
                while load_progress[next_g] < next_total:
                    emit_load_for_group(next_g, next_start, next_vecs)

        return slots

    def build_hash_pipeline_addr_with_idx(
        self,
        v_idx,
        v_val,
        v_tmp1,
        v_tmp2,
        round,
        i_base,
        emit_debug,
        vec_count,
        v_one,
        v_two,
        v_base_minus1,
        forest_height,
        depth,
        hash_group=3,
    ):
        """Like build_hash_pipeline_addr but includes idx update interleaved with hash.

        The key optimization: emit multiply_add for previous group during current
        group's hash stages, creating truly independent work.
        """
        slots = []
        group_size = min(hash_group, vec_count)
        num_groups = (vec_count + group_size - 1) // group_size
        load_progress = [0] * num_groups

        # Skip idx update for leaf depth
        skip_idx_update = (depth == forest_height)

        def emit_load_for_group(g, group_start, group_vecs):
            idx = load_progress[g]
            u = group_start + idx // VLEN
            lane = idx % VLEN
            slots.append(("load", ("load_offset", v_tmp1[u], v_idx[u], lane)))
            load_progress[g] = idx + 1

        prev_group_start = None
        prev_group_vecs = None
        idx_update_done = {}

        for g in range(num_groups):
            group_start = g * group_size
            group_vecs = min(group_size, vec_count - group_start)
            total_loads = group_vecs * VLEN
            while load_progress[g] < total_loads:
                emit_load_for_group(g, group_start, group_vecs)
            if emit_debug:
                for u in range(group_start, group_start + group_vecs):
                    base = i_base + u * VLEN
                    keys = [(round, base + lane, "node_val") for lane in range(VLEN)]
                    slots.append(("debug", ("vcompare", v_tmp1[u], keys)))

            for u in range(group_start, group_start + group_vecs):
                slots.append(("valu", ("^", v_val[u], v_val[u], v_tmp1[u])))

            stages = self.build_hash_vec_multi_stages(
                v_val[group_start : group_start + group_vecs],
                v_tmp1[group_start : group_start + group_vecs],
                v_tmp2[group_start : group_start + group_vecs],
                round,
                i_base + group_start * VLEN,
                emit_debug,
            )

            next_g = g + 1
            if next_g < num_groups:
                next_start = next_g * group_size
                next_vecs = min(group_size, vec_count - next_start)
                next_total = next_vecs * VLEN
                remaining = next_total - load_progress[next_g]
                loads_per_stage = (
                    (remaining + (len(stages) * 2) - 1) // (len(stages) * 2)
                    if remaining > 0
                    else 0
                )
            else:
                next_start = next_vecs = next_total = loads_per_stage = 0

            for si, stage_slots in enumerate(stages):
                if next_g < num_groups and loads_per_stage:
                    for _ in range(loads_per_stage):
                        if load_progress[next_g] >= next_total:
                            break
                        emit_load_for_group(next_g, next_start, next_vecs)
                slots.extend(stage_slots)

                # Interleave idx update for PREVIOUS group during current hash
                if not skip_idx_update and prev_group_start is not None:
                    # After stage 0: emit multiply_add for previous group
                    if si == 0:
                        for u in range(prev_group_start, prev_group_start + prev_group_vecs):
                            if u not in idx_update_done:
                                slots.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            v_idx[u],
                                            v_idx[u],
                                            v_two,
                                            v_base_minus1,
                                        ),
                                    )
                                )
                    # After stage 2: emit bit extraction
                    if si == 2:
                        for u in range(prev_group_start, prev_group_start + prev_group_vecs):
                            if u not in idx_update_done:
                                slots.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                    # After stage 4: emit final add
                    if si == 4:
                        for u in range(prev_group_start, prev_group_start + prev_group_vecs):
                            if u not in idx_update_done:
                                slots.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))
                                idx_update_done[u] = True

                if next_g < num_groups and loads_per_stage:
                    for _ in range(loads_per_stage):
                        if load_progress[next_g] >= next_total:
                            break
                        emit_load_for_group(next_g, next_start, next_vecs)

            if next_g < num_groups:
                while load_progress[next_g] < next_total:
                    emit_load_for_group(next_g, next_start, next_vecs)

            prev_group_start = group_start
            prev_group_vecs = group_vecs

        # Finish idx update for the last group
        if not skip_idx_update and prev_group_start is not None:
            for u in range(prev_group_start, prev_group_start + prev_group_vecs):
                if u not in idx_update_done:
                    slots.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                    slots.append(
                        (
                            "valu",
                            (
                                "multiply_add",
                                v_idx[u],
                                v_idx[u],
                                v_two,
                                v_base_minus1,
                            ),
                        )
                    )
                    slots.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))
                    idx_update_done[u] = True

        return slots

    def build_kernel('''

if old_build_hash_end in content:
    content = content.replace(old_build_hash_end, new_build_hash_end)
    print("Added build_hash_pipeline_addr_with_idx function")
else:
    print("ERROR: Could not find insertion point")

# Step 2: Modify emit_hash_only_range to use the new function for depth > 2
# and return True if it handles idx update

old_emit_hash_else = '''                else:
                    body.extend(
                        self.build_hash_pipeline_addr(
                            v_idx_l,
                            v_val_l,
                            v_tmp1_l,
                            v_tmp2_l,
                            round_idx,
                            start * VLEN,
                            emit_debug,
                            count,
                            hash_group=3,
                        )
                    )

            def emit_idx_update(vec_count):'''

new_emit_hash_else = '''                else:
                    body.extend(
                        self.build_hash_pipeline_addr(
                            v_idx_l,
                            v_val_l,
                            v_tmp1_l,
                            v_tmp2_l,
                            round_idx,
                            start * VLEN,
                            emit_debug,
                            count,
                            hash_group=3,
                        )
                    )

            def emit_hash_with_idx_range(round_idx: int, depth: int, start: int, count: int):
                """Emit hash AND idx update for depth > 2, with interleaved operations."""
                v_idx_l = v_idx[start : start + count]
                v_val_l = v_val[start : start + count]
                v_tmp1_l = v_tmp1[start : start + count]
                v_tmp2_l = v_tmp2[start : start + count]
                body.extend(
                    self.build_hash_pipeline_addr_with_idx(
                        v_idx_l,
                        v_val_l,
                        v_tmp1_l,
                        v_tmp2_l,
                        round_idx,
                        start * VLEN,
                        emit_debug,
                        count,
                        v_one,
                        v_two,
                        v_base_minus1,
                        forest_height,
                        depth,
                        hash_group=3,
                    )
                )

            def emit_idx_update(vec_count):'''

if old_emit_hash_else in content:
    content = content.replace(old_emit_hash_else, new_emit_hash_else)
    print("Added emit_hash_with_idx_range function")
else:
    print("ERROR: Could not find emit_hash else clause")

# Step 3: Modify the round loop to use emit_hash_with_idx_range for depth > 2
old_round_loop = '''                for round_idx in range(rounds):
                    depth = round_depths[round_idx]
                    chunk = 1
                    if depth > 2:
                        starts = (
                            0, 16, 4, 20, 8, 24, 12, 28, 2, 18, 6, 22, 10, 26, 14, 30,
                            1, 17, 5, 21, 9, 25, 13, 29, 3, 19, 7, 23, 11, 27, 15, 31,
                        )
                    else:
                        starts = (
                            0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                            1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31,
                        )
                    for start in starts:
                        count = min(chunk, vec_count - start)
                        emit_hash_only_range(round_idx, depth, start, count)
                        # Skip idx update for depth 0 and forest_height
                        if depth != 0 and depth != forest_height:
                            for u in range(start, start + count):
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                body.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            v_idx[u],
                                            v_idx[u],
                                            v_two,
                                            v_base_minus1,
                                        ),
                                    )
                                )
                                body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))'''

new_round_loop = '''                for round_idx in range(rounds):
                    depth = round_depths[round_idx]
                    chunk = 1
                    if depth > 2:
                        starts = (
                            0, 16, 4, 20, 8, 24, 12, 28, 2, 18, 6, 22, 10, 26, 14, 30,
                            1, 17, 5, 21, 9, 25, 13, 29, 3, 19, 7, 23, 11, 27, 15, 31,
                        )
                    else:
                        starts = (
                            0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                            1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31,
                        )

                    # For depth > 2, use the interleaved hash+idx function
                    if depth > 2:
                        # Process ALL vectors together so idx update can interleave
                        emit_hash_with_idx_range(round_idx, depth, 0, vec_count)
                    else:
                        for start in starts:
                            count = min(chunk, vec_count - start)
                            emit_hash_only_range(round_idx, depth, start, count)
                            # Skip idx update for depth 0 and forest_height
                            if depth != 0 and depth != forest_height:
                                for u in range(start, start + count):
                                    body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                    body.append(
                                        (
                                            "valu",
                                            (
                                                "multiply_add",
                                                v_idx[u],
                                                v_idx[u],
                                                v_two,
                                                v_base_minus1,
                                            ),
                                        )
                                    )
                                    body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))'''

if old_round_loop in content:
    content = content.replace(old_round_loop, new_round_loop)
    print("Modified round loop to use interleaved hash+idx")
else:
    print("ERROR: Could not find round loop")

with open('perf_takehome_hash_idx_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_hash_idx_v2.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"Hash with interleaved idx: {match.group(1)} cycles (baseline: 1615)")
elif 'OK' in output:
    print("Test passed")
else:
    print("Test result:")
    if 'AssertionError' in output:
        print("Correctness FAILED!")
    print(output[-1500:])
