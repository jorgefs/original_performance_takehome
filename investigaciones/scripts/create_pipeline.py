"""
Software Pipelining: Process vectors in waves where each wave is at a different stage.

Current flow (for depth > 2):
  For each vector u:
    load_offset -> XOR -> hash(6 stages) -> idx_update

This creates a critical path of ~8-10 operations per vector.

New flow (pipeline of 3 stages):
  Stage 0: load_offset for vectors [u, u+1, ...]
  Stage 1: XOR + hash stages 0-2 for vectors [u-K, u-K+1, ...]
  Stage 2: hash stages 3-5 + idx_update for vectors [u-2K, u-2K+1, ...]

By having different vectors at different stages, we create more independent work.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace build_hash_pipeline_addr with a pipelined version
old_func = '''    def build_hash_pipeline_addr(
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

            for stage_slots in stages:
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
                    emit_load_for_group(next_g, next_start, next_vecs)

        return slots'''

# New pipelined version: more aggressive interleaving with 2 groups active at once
new_func = '''    def build_hash_pipeline_addr(
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
        """
        Pipelined version: keep 2 groups active at different stages.
        Group N does hash while Group N+1 does loads.
        """
        slots = []

        # Use smaller groups for better pipelining (2 vectors per group)
        PIPE_GROUP = 2
        group_size = min(PIPE_GROUP, vec_count)
        num_groups = (vec_count + group_size - 1) // group_size

        # Track state per group
        load_progress = [0] * num_groups  # How many load_offset done
        hash_stage = [-1] * num_groups     # Which hash stage (-1 = not started, 6 = done)
        xor_done = [False] * num_groups

        def group_vecs(g):
            start = g * group_size
            return min(group_size, vec_count - start)

        def group_start(g):
            return g * group_size

        def emit_loads(g, count):
            """Emit up to 'count' load_offset for group g"""
            gs = group_start(g)
            gv = group_vecs(g)
            total = gv * VLEN
            emitted = 0
            while load_progress[g] < total and emitted < count:
                idx = load_progress[g]
                u = gs + idx // VLEN
                lane = idx % VLEN
                slots.append(("load", ("load_offset", v_tmp1[u], v_idx[u], lane)))
                load_progress[g] += 1
                emitted += 1
            return emitted

        def emit_xor(g):
            """Emit XOR for group g"""
            if xor_done[g]:
                return
            gs = group_start(g)
            gv = group_vecs(g)
            for u in range(gs, gs + gv):
                slots.append(("valu", ("^", v_val[u], v_val[u], v_tmp1[u])))
            xor_done[g] = True

        def emit_hash_stage(g):
            """Emit next hash stage for group g"""
            if hash_stage[g] >= 5:
                return False  # Already done
            hash_stage[g] += 1
            hs = hash_stage[g]
            gs = group_start(g)
            gv = group_vecs(g)

            op1, val1, op2, op3, val3 = HASH_STAGES[hs]
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mult = 1 + (1 << val3)
                for u in range(gs, gs + gv):
                    slots.append((
                        "valu",
                        ("multiply_add", v_val[u], v_val[u],
                         self.vector_const(mult), self.vector_const(val1))
                    ))
            else:
                for u in range(gs, gs + gv):
                    slots.append((
                        "valu",
                        (op1, v_tmp1[u], v_val[u], self.vector_const(val1))
                    ))
                for u in range(gs, gs + gv):
                    slots.append((
                        "valu",
                        (op3, v_tmp2[u], v_val[u], self.vector_const(val3))
                    ))
                for u in range(gs, gs + gv):
                    slots.append(("valu", (op2, v_val[u], v_tmp1[u], v_tmp2[u])))
            return True

        def is_group_done(g):
            return hash_stage[g] >= 5

        def is_group_loads_done(g):
            return load_progress[g] >= group_vecs(g) * VLEN

        # Main pipeline loop: keep 2 groups active
        active_groups = []
        next_group = 0

        while next_group < num_groups or active_groups:
            # Start new group if we have capacity
            while len(active_groups) < 2 and next_group < num_groups:
                active_groups.append(next_group)
                next_group += 1

            # For each active group, do one step
            for g in list(active_groups):
                if not is_group_loads_done(g):
                    # Do loads (2 per cycle to saturate load engine)
                    emit_loads(g, 2)
                elif not xor_done[g]:
                    emit_xor(g)
                elif not is_group_done(g):
                    emit_hash_stage(g)
                else:
                    active_groups.remove(g)

        return slots'''

if old_func in content:
    content = content.replace(old_func, new_func)
    print("Replaced build_hash_pipeline_addr with pipelined version")
else:
    print("ERROR: Could not find build_hash_pipeline_addr")
    # Try to find partial match
    if "def build_hash_pipeline_addr" in content:
        print("Found function header but body didn't match")

with open('perf_takehome_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Test
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_pipeline.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"Pipelined version: {match.group(1)} cycles (baseline: 1615)")
elif 'OK' in output:
    print("Test passed but no cycle count found")
    print(output[-300:])
else:
    print("Test failed:")
    print(output[-800:])
