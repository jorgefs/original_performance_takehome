"""
Parallel multiply_add: Store result in temp to avoid clobbering v_idx.

Problem: multiply_add writes to v_idx, but load_offset reads v_idx.
Solution:
  1. temp = idx * 2 + base_minus1  (can run during loads!)
  2. ... all loads read idx (unchanged)
  3. idx = temp + bit  (after hash completes)

This allows multiply_add to run in PARALLEL with loads, filling VALU slots
during load latency.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Modify build_hash_pipeline_addr to include multiply_add during loads
old_pipeline = '''    def build_hash_pipeline_addr(
        self,
        v_idx,
        v_val,
        v_tmp1,
        v_tmp2,
        round,
        i_base,
        emit_debug,
        vec_count,
        hash_group=3,
    ):
        slots = []
        group_size = min(hash_group, vec_count)
        num_groups = (vec_count + group_size - 1) // group_size
        load_progress = [0] * num_groups

        def emit_load_for_group(g, group_start, group_vecs):
            idx = load_progress[g]
            u = group_start + idx // VLEN
            lane = idx % VLEN
            slots.append(("load", ("load_offset", v_tmp1[u], v_idx[u], lane)))
            load_progress[g] = idx + 1

        for g in range(num_groups):
            group_start = g * group_size
            group_vecs = min(group_size, vec_count - group_start)
            total_loads = group_vecs * VLEN
            while load_progress[g] < total_loads:
                emit_load_for_group(g, group_start, group_vecs)'''

new_pipeline = '''    def build_hash_pipeline_addr(
        self,
        v_idx,
        v_val,
        v_tmp1,
        v_tmp2,
        round,
        i_base,
        emit_debug,
        vec_count,
        hash_group=3,
        v_two=None,
        v_base_minus1=None,
        skip_idx_update=False,
    ):
        slots = []
        group_size = min(hash_group, vec_count)
        num_groups = (vec_count + group_size - 1) // group_size
        load_progress = [0] * num_groups
        muladd_emitted = [False] * vec_count  # Track which vectors have muladd done

        def emit_load_for_group(g, group_start, group_vecs):
            idx = load_progress[g]
            u = group_start + idx // VLEN
            lane = idx % VLEN
            slots.append(("load", ("load_offset", v_tmp1[u], v_idx[u], lane)))
            load_progress[g] = idx + 1

        for g in range(num_groups):
            group_start = g * group_size
            group_vecs = min(group_size, vec_count - group_start)
            total_loads = group_vecs * VLEN

            # Interleave multiply_add with loads (they can run in parallel!)
            # Store result in v_tmp2 to avoid clobbering v_idx during loads
            load_count = 0
            muladd_count = 0
            while load_count < total_loads or (not skip_idx_update and v_two is not None and muladd_count < group_vecs):
                # Emit 2 loads (we have 2 load slots)
                for _ in range(2):
                    if load_count < total_loads:
                        idx = load_count
                        u = group_start + idx // VLEN
                        lane = idx % VLEN
                        slots.append(("load", ("load_offset", v_tmp1[u], v_idx[u], lane)))
                        load_count += 1

                # Emit multiply_add if we have slots available and v_two is provided
                if not skip_idx_update and v_two is not None and muladd_count < group_vecs:
                    u = group_start + muladd_count
                    if not muladd_emitted[u]:
                        # Store in v_tmp2[u], but v_tmp2 is used by hash...
                        # Actually we can't use v_tmp2 safely.
                        # Let's skip this optimization for now.
                        pass
                    muladd_count += 1

            # Fall back to original behavior
            while load_progress[g] < total_loads:
                emit_load_for_group(g, group_start, group_vecs)'''

# Actually this is getting too complex and might break things.
# Let me try a simpler approach: just test if the problem is load latency
# by adjusting the number of loads we issue before starting hash.

print("This approach requires too much scratch space or is too complex.")
print("Let me try a simpler diagnostic: adjust load interleaving.")

# Simpler test: what if we don't interleave loads with hash at all?
# This would show if load-hash interleaving is actually helping.

old_interleave = '''            for stage_slots in stages:
                if next_g < num_groups and loads_per_stage:
                    for _ in range(loads_per_stage):
                        if load_progress[next_g] >= next_total:
                            break
                        emit_load_for_group(next_g, next_start, next_vecs)
                slots.extend(stage_slots)
                if next_g < num_groups and loads_per_stage:
                    for _ in range(loads_per_stage):
                        if load_progress[next_g] >= next_total:
                            break
                        emit_load_for_group(next_g, next_start, next_vecs)

            if next_g < num_groups:
                while load_progress[next_g] < next_total:
                    emit_load_for_group(next_g, next_start, next_vecs)'''

# Test removing all load interleaving
new_no_interleave = '''            # No load interleaving - just emit hash stages
            for stage_slots in stages:
                slots.extend(stage_slots)

            # Then emit loads for next group all at once
            if next_g < num_groups:
                while load_progress[next_g] < next_total:
                    emit_load_for_group(next_g, next_start, next_vecs)'''

if old_interleave in content:
    content_no_interleave = content.replace(old_interleave, new_no_interleave)
    with open('perf_takehome_no_interleave.py', 'w', encoding='utf-8') as f:
        f.write(content_no_interleave)

    import subprocess
    result = subprocess.run(
        ['python', 'perf_takehome_no_interleave.py', 'Tests.test_kernel_cycles'],
        capture_output=True, text=True, timeout=180
    )
    output = result.stdout + result.stderr
    import re
    match = re.search(r'CYCLES:\s*(\d+)', output)
    if match:
        print(f"No load interleaving: {match.group(1)} cycles (baseline: 1615)")
    else:
        print("No interleave test failed")
        if 'AssertionError' in output:
            print("Correctness failed")
else:
    print("ERROR: Could not find interleave pattern")
