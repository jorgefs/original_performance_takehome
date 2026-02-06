"""
FASE 1 - Cache depth 3 con vselect tree.

Para depth 3, idx ∈ {7..14} (8 valores posibles del árbol).
v_idx = forest_values_p + tree_idx, donde tree_idx ∈ {7..14}

Implementación:
1. En init: cargar los 8 valores de forest[7..14] en escalares
2. Broadcast a vectores v_level3[0..7]
3. En depth 3: usar 7 vselects por vector para elegir el valor correcto

vselect tree para 8 valores:
- bit0 = (tree_idx - 7) & 1
- bit1 = ((tree_idx - 7) >> 1) & 1
- bit2 = ((tree_idx - 7) >> 2) & 1

Nivel 1: sel01 = bit0 ? v[1] : v[0], sel23 = bit0 ? v[3] : v[2], ...
Nivel 2: sel0123 = bit1 ? sel23 : sel01, ...
Nivel 3: result = bit2 ? sel4567 : sel0123
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Agregar scratch para level 3 values
old_level2 = '''            level2_vals = [self.alloc_scratch(f"level2_val_{i}") for i in range(4)]'''
new_level2 = '''            level2_vals = [self.alloc_scratch(f"level2_val_{i}") for i in range(4)]

            # Level 3 cached values (tree indices 7-14)
            level3_vals = [self.alloc_scratch(f"level3_val_{i}") for i in range(8)]'''

if old_level2 in content:
    content = content.replace(old_level2, new_level2)
    print("1. Added level3_vals scratch")
else:
    print("ERROR: Could not find level2_vals")

# 2. Agregar v_level3 vectors
old_v_level2 = '''            v_level2 = [self.alloc_scratch(f"v_level2_{i}", length=VLEN) for i in range(4)]'''
new_v_level2 = '''            v_level2 = [self.alloc_scratch(f"v_level2_{i}", length=VLEN) for i in range(4)]
            v_level3 = [self.alloc_scratch(f"v_level3_{i}", length=VLEN) for i in range(8)]'''

if old_v_level2 in content:
    content = content.replace(old_v_level2, new_v_level2)
    print("2. Added v_level3 vectors")
else:
    print("ERROR: Could not find v_level2")

# 3. Agregar v_four para shifts
old_v_base = '''            v_base_plus1 = self.alloc_scratch("v_base_plus1", length=VLEN)'''
new_v_base = '''            v_base_plus1 = self.alloc_scratch("v_base_plus1", length=VLEN)
            v_four = self.alloc_scratch("v_four", length=VLEN)
            v_seven = self.alloc_scratch("v_seven", length=VLEN)'''

if old_v_base in content:
    content = content.replace(old_v_base, new_v_base)
    print("3. Added v_four and v_seven")
else:
    print("ERROR: Could not find v_base_plus1")

# 4. Cargar valores de level 3 en init
old_level2_loads = '''            # Level 2 addresses and loads
            init.append(("alu", ("+", tmp1, forest_values_p, const_3)))
            init.append(("load", ("load", level2_vals[0], tmp1)))
            init.append(("alu", ("+", tmp1, forest_values_p, const_4)))
            init.append(("load", ("load", level2_vals[1], tmp1)))
            init.append(("alu", ("+", tmp1, forest_values_p, const_5)))
            init.append(("load", ("load", level2_vals[2], tmp1)))
            init.append(("alu", ("+", tmp1, forest_values_p, const_6)))
            init.append(("load", ("load", level2_vals[3], tmp1)))'''

new_level2_loads = '''            # Level 2 addresses and loads
            init.append(("alu", ("+", tmp1, forest_values_p, const_3)))
            init.append(("load", ("load", level2_vals[0], tmp1)))
            init.append(("alu", ("+", tmp1, forest_values_p, const_4)))
            init.append(("load", ("load", level2_vals[1], tmp1)))
            init.append(("alu", ("+", tmp1, forest_values_p, const_5)))
            init.append(("load", ("load", level2_vals[2], tmp1)))
            init.append(("alu", ("+", tmp1, forest_values_p, const_6)))
            init.append(("load", ("load", level2_vals[3], tmp1)))

            # Level 3 addresses and loads (tree indices 7-14)
            const_7 = self.scratch_const(7)
            const_8 = self.scratch_const(8)
            const_9 = self.scratch_const(9)
            const_10 = self.scratch_const(10)
            const_11 = self.scratch_const(11)
            const_12 = self.scratch_const(12)
            const_13 = self.scratch_const(13)
            const_14 = self.scratch_const(14)
            for i, c in enumerate([const_7, const_8, const_9, const_10, const_11, const_12, const_13, const_14]):
                init.append(("alu", ("+", tmp1, forest_values_p, c)))
                init.append(("load", ("load", level3_vals[i], tmp1)))'''

if old_level2_loads in content:
    content = content.replace(old_level2_loads, new_level2_loads)
    print("4. Added level 3 loads")
else:
    print("ERROR: Could not find level2 loads")

# 5. Agregar broadcasts de level 3
old_broadcasts = '''            init.append(("valu", ("vbroadcast", v_level2[0], level2_vals[0])))
            init.append(("valu", ("vbroadcast", v_level2[1], level2_vals[1])))
            init.append(("valu", ("vbroadcast", v_level2[2], level2_vals[2])))
            init.append(("valu", ("vbroadcast", v_level2[3], level2_vals[3])))'''

new_broadcasts = '''            init.append(("valu", ("vbroadcast", v_level2[0], level2_vals[0])))
            init.append(("valu", ("vbroadcast", v_level2[1], level2_vals[1])))
            init.append(("valu", ("vbroadcast", v_level2[2], level2_vals[2])))
            init.append(("valu", ("vbroadcast", v_level2[3], level2_vals[3])))
            # Level 3 broadcasts
            for i in range(8):
                init.append(("valu", ("vbroadcast", v_level3[i], level3_vals[i])))
            # Additional constants
            init.append(("valu", ("vbroadcast", v_four, self.scratch_const(4))))
            init.append(("valu", ("vbroadcast", v_seven, self.scratch_const(7))))'''

if old_broadcasts in content:
    content = content.replace(old_broadcasts, new_broadcasts)
    print("5. Added level 3 broadcasts")
else:
    print("ERROR: Could not find level2 broadcasts")

# 6. Agregar caso depth==3 en emit_hash_only_range
# Necesitamos insertar antes del "else:" que maneja depth > 2
old_depth2_end = '''                    body.extend(
                        self.build_hash_vec_multi(
                            v_val_l,
                            v_tmp1_l,
                            v_tmp2_l,
                            round_idx,
                            start * VLEN,
                            emit_debug,
                        )
                    )
                else:
                    body.extend(
                        self.build_hash_pipeline_addr('''

new_depth2_end = '''                    body.extend(
                        self.build_hash_vec_multi(
                            v_val_l,
                            v_tmp1_l,
                            v_tmp2_l,
                            round_idx,
                            start * VLEN,
                            emit_debug,
                        )
                    )
                elif depth == 3:
                    # Depth 3: Use cached values with vselect tree
                    # tree_idx = v_idx - forest_values_p ∈ {7..14}
                    # offset = tree_idx - 7 ∈ {0..7}
                    for u in range(count):
                        # Compute tree_idx = v_idx - forest_values_p
                        # Using v_neg_forest = 1 - forest_values_p, so v_idx + v_neg_forest = v_idx - forest_values_p + 1
                        # Actually v_neg_forest = -forest_values_p + 1
                        # So tree_idx = v_idx + v_neg_forest - 1
                        # But we need offset = tree_idx - 7 = v_idx - forest_values_p - 7 = v_idx - 14
                        # Hmm, let's compute directly: offset = v_idx - v_forest_values_p - v_seven

                        # Step 1: tree_idx = v_idx - forest_values_p  (use addition with v_neg_forest and subtract 1)
                        # v_neg_forest = 1 - forest_values_p
                        # tree_idx = v_idx + (1 - forest_values_p) - 1 = v_idx - forest_values_p
                        body.append(("valu", ("+", v_tmp1_l[u], v_idx_l[u], v_neg_forest)))
                        body.append(("valu", ("-", v_tmp1_l[u], v_tmp1_l[u], v_one)))
                        # Now v_tmp1_l[u] = tree_idx ∈ {7..14}

                        # Step 2: offset = tree_idx - 7
                        body.append(("valu", ("-", v_tmp2_l[u], v_tmp1_l[u], v_seven)))
                        # Now v_tmp2_l[u] = offset ∈ {0..7}

                        # Step 3: Extract bits
                        # bit0 = offset & 1
                        body.append(("valu", ("&", v_tmp1_l[u], v_tmp2_l[u], v_one)))
                        # bit1 = (offset >> 1) & 1
                        body.append(("valu", (">>", v_tmp3_shared, v_tmp2_l[u], v_one)))
                        body.append(("valu", ("&", v_tmp3_shared, v_tmp3_shared, v_one)))
                        # bit2 = (offset >> 2) & 1
                        body.append(("valu", (">>", v_tmp2_l[u], v_tmp2_l[u], v_two)))
                        body.append(("valu", ("&", v_tmp2_l[u], v_tmp2_l[u], v_one)))

                        # Step 4: vselect tree (7 vselects)
                        # First level: select between pairs using bit0 (v_tmp1_l[u])
                        # sel01 = bit0 ? v_level3[1] : v_level3[0]
                        # sel23 = bit0 ? v_level3[3] : v_level3[2]
                        # sel45 = bit0 ? v_level3[5] : v_level3[4]
                        # sel67 = bit0 ? v_level3[7] : v_level3[6]

                        # We need 4 temps for first level, but we only have v_tmp1, v_tmp2, v_tmp3_shared
                        # Let's use a different approach: chain vselects reusing temps

                        # Actually, we can use v_val_l[u] as temp since we'll overwrite it with XOR result anyway
                        # But that's risky. Let's just do fewer vselects by being clever.

                        # Alternative: use a cascade where we select at each bit level
                        # Start with pairs (using bit0):
                        # Then combine (using bit1):
                        # Then final (using bit2):

                        # Reuse v_idx_l[u] temporarily (we don't need idx anymore at this point!)
                        # NO - we need idx for idx update later

                        # Use v_node_val which is not allocated in fast path... wait, it's not allocated
                        # Let's allocate more scratch or be clever

                        # Simplest: Just use a longer sequence with fewer temps
                        # bit0 is in v_tmp1_l[u]
                        # bit1 is in v_tmp3_shared
                        # bit2 is in v_tmp2_l[u]

                        # First level - we need 4 intermediate results but only have v_tmp1, v_tmp2, v_tmp3_shared
                        # Process in stages, reusing temps

                        # Stage 1: sel01 = bit0 ? v_level3[1] : v_level3[0]  -> store in some temp
                        # Let's use v_val_l[u] temporarily (we'll fix it with XOR at the end)
                        body.append(("flow", ("vselect", v_val_l[u], v_tmp1_l[u], v_level3[1], v_level3[0])))
                        # sel01 now in v_val_l[u]

                        # sel23 needs another temp... can we chain?
                        # Let's compute final result progressively

                        # Actually, we need to be more careful. Let me use a different structure.
                        # Store bit0, bit1, bit2 first, then do all vselects

                        # bits stored in: bit0=v_tmp1_l[u], bit1=v_tmp3_shared, bit2=v_tmp2_l[u]

                        # We can allocate v_tmp3 per-vector instead of shared. Let's skip that for now.

                        # Simpler approach for now: use the bit tests inline
                        # This will take more cycles but work

                        # ACTUALLY: let's just not do this optimization yet and fall back
                        # For now, use original load_offset

                        # FALL THROUGH TO LOAD_OFFSET (remove the depth==3 specific code)

                    # Fall back to original
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
                else:
                    body.extend(
                        self.build_hash_pipeline_addr('''

# This is too complex. Let me just test if the setup works first
print("Depth 3 vselect tree is complex - needs more temp scratch.")
print("Testing with just the initialization changes...")

with open('perf_takehome_d3cache.py', 'w', encoding='utf-8') as f:
    f.write(content)

import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_d3cache.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"With depth 3 cache setup: {match.group(1)} cycles (baseline: 1615)")
elif 'OK' in output:
    print("Test passed")
else:
    print("ERROR:")
    if 'AssertionError' in output:
        print("Correctness or scratch FAILED")
    print(output[-2000:])
