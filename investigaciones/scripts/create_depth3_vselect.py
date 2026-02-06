"""
FASE 1 - Software gather para depth 3 usando vselect tree.

En depth 3, idx ∈ {7..14} (8 valores posibles).
En lugar de 256 load_offsets, hacemos:
1. 8 scalar loads para forest_values[7..14]
2. 8 vbroadcasts
3. 3 vselects por vector para elegir el valor correcto basado en bits de idx

Trade-off:
- Original: 256 load_offset = 128 load cycles + latencia
- Nuevo: 8 loads + 8 vbroadcasts + 32*3=96 vselects = 4 load + 8 valu + 96 flow

Dado que load y valu pueden correr en paralelo con flow, y flow es el cuello
(1 slot/cycle), esto debería ahorrar ~32 cycles si la latencia de load era
el limitante.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Agregar scratch para valores de depth 3 (índices 7-14)
old_alloc = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)'''
new_alloc = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)

            # Depth 3 cached values: forest_values[7..14]
            level3_vals = [self.alloc_scratch(f"level3_val_{i}") for i in range(8)]
            v_level3 = [self.alloc_scratch(f"v_level3_{i}", length=VLEN) for i in range(8)]'''

if old_alloc in content:
    content = content.replace(old_alloc, new_alloc)
    print("Added depth 3 scratch")
else:
    print("ERROR: Could not find scratch allocation")

# Agregar carga de valores level 3 en inicialización
old_init_broadcasts = '''            # Broadcasts (can run in parallel, 6 valu slots)
            init.append(("valu", ("vbroadcast", v_root_val, root_val)))'''

new_init_broadcasts = '''            # Load level 3 values (indices 7-14)
            for i in range(8):
                init.append(("alu", ("+", tmp1, forest_values_p, self.scratch_const(7 + i))))
                init.append(("load", ("load", level3_vals[i], tmp1)))

            # Broadcasts (can run in parallel, 6 valu slots)
            init.append(("valu", ("vbroadcast", v_root_val, root_val)))'''

if old_init_broadcasts in content:
    content = content.replace(old_init_broadcasts, new_init_broadcasts)
    print("Added level 3 loads in init")

# Agregar broadcasts de level 3 después de otros broadcasts
old_broadcasts_end = '''            init.append(("valu", ("vbroadcast", v_forest_values_p, forest_values_p)))'''

new_broadcasts_end = '''            init.append(("valu", ("vbroadcast", v_forest_values_p, forest_values_p)))

            # Broadcast level 3 values
            for i in range(8):
                init.append(("valu", ("vbroadcast", v_level3[i], level3_vals[i])))'''

if old_broadcasts_end in content:
    content = content.replace(old_broadcasts_end, new_broadcasts_end)
    print("Added level 3 vbroadcasts")

# Agregar constantes para bits de idx
old_const_map = '''            # Update const_map for body to use
            self.const_map[0] = zero_const'''

new_const_map = '''            # Compute constants for level 3 bit selection
            const_7 = self.scratch_const(7)
            v_four = self.vector_const(4, "v_four")
            v_seven = self.vector_const(7, "v_seven")

            # Update const_map for body to use
            self.const_map[0] = zero_const'''

if old_const_map in content:
    content = content.replace(old_const_map, new_const_map)
    print("Added bit selection constants")

# Modificar emit_hash_only_range para manejar depth 3 con vselect
old_depth_else = '''                else:
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
                    )'''

new_depth_else = '''                elif depth == 3:
                    # Depth 3: Use vselect tree instead of load_offset
                    # idx ∈ {7..14}, so (idx - 7) ∈ {0..7} has 3 bits
                    for u in range(count):
                        # Extract bits of (idx - 7)
                        # bit0 = (idx - 7) & 1 = idx & 1 (since 7 is odd)
                        # bit1 = ((idx - 7) >> 1) & 1 = (idx >> 1) & 1 (since 7>>1 = 3 is odd)
                        # Actually: idx ∈ {7,8,9,10,11,12,13,14}
                        # idx-7 ∈ {0,1,2,3,4,5,6,7}
                        # We need to select from v_level3[0..7] based on idx-7

                        # Compute offset = idx & 7 (works because forest_values_p = 7)
                        # idx = forest_values_p + tree_idx, so idx & 7 = tree_idx & 7 when tree_idx < 8
                        # Actually idx here is the MEMORY address, not tree index!
                        # idx = forest_values_p + tree_index
                        # For depth 3, tree_index ∈ {7..14}
                        # So idx = 7 + {7..14} = {14..21}

                        # We need to use (idx - forest_values_p - 7) = tree_index - 7
                        # tree_index - 7 ∈ {0..7}

                        # Extract: offset = (idx - v_forest_values_p - v_seven) & 7
                        # But simpler: use v_idx - v_forest_values_p to get tree_index
                        # Then extract 3 bits: bit0, bit1, bit2

                        # Step 1: compute tree_idx = idx - forest_values_p
                        body.append(("valu", ("+", v_tmp1_l[u], v_idx_l[u], v_neg_forest)))

                        # Now v_tmp1_l[u] contains tree_idx ∈ {7..14}
                        # We want bits 0,1,2 of (tree_idx - 7) = bits 0,1,2 of tree_idx XOR 7

                        # Simpler: tree_idx = 7 + offset where offset ∈ {0..7}
                        # tree_idx & 1 = (7 + offset) & 1 = (1 + offset) & 1 = ~(offset & 1)? No...
                        # 7 = 0b111, so tree_idx & 7 = (7 + offset) & 7 = (7 & 7) + offset if no carry
                        # But there IS carry for some values!

                        # Let's be explicit:
                        # tree_idx=7:  0b0111 -> want v_level3[0]
                        # tree_idx=8:  0b1000 -> want v_level3[1]
                        # tree_idx=9:  0b1001 -> want v_level3[2]
                        # ...
                        # tree_idx=14: 0b1110 -> want v_level3[7]

                        # So we need (tree_idx - 7) & 7 as the selector

                        body.append(("valu", ("-", v_tmp2_l[u], v_tmp1_l[u], v_seven)))
                        # Now v_tmp2_l[u] = (tree_idx - 7) ∈ {0..7}

                        # Extract bit 0
                        body.append(("valu", ("&", v_tmp1_l[u], v_tmp2_l[u], v_one)))
                        # bit0 in v_tmp1_l[u]

                        # Select between pairs based on bit0
                        # If bit0=0: v_level3[0,2,4,6], if bit0=1: v_level3[1,3,5,7]
                        body.append(
                            ("flow", ("vselect", v_tmp3_shared, v_tmp1_l[u], v_level3[1], v_level3[0]))
                        )  # sel01 = bit0 ? v_level3[1] : v_level3[0]

                        # We need separate temps for each pair selection...
                        # This is getting complex. Let me use a cleaner approach.

                        # Actually, use a tree of vselects:
                        # Level 1: bit2 selects between (0-3) vs (4-7)
                        # Level 2: bit1 selects within each half
                        # Level 3: bit0 selects within each quarter

                        # Extract bit2 = (offset >> 2) & 1
                        body.append(("valu", (">>", v_tmp1_l[u], v_tmp2_l[u], v_two)))
                        body.append(("valu", ("&", v_tmp1_l[u], v_tmp1_l[u], v_one)))
                        # bit2 in v_tmp1_l[u]

                        # Extract bit1 = (offset >> 1) & 1
                        body.append(("valu", (">>", v_tmp3_shared, v_tmp2_l[u], v_one)))
                        body.append(("valu", ("&", v_tmp3_shared, v_tmp3_shared, v_one)))
                        # bit1 in v_tmp3_shared

                        # Extract bit0 = offset & 1
                        # v_tmp2_l[u] still has offset, so:
                        body.append(("valu", ("&", v_tmp2_l[u], v_tmp2_l[u], v_one)))
                        # bit0 in v_tmp2_l[u]

                        # Now do vselect tree (need temp space for intermediate results)
                        # This requires more scratch... Let me simplify.

                        # For now, let's just use 3 vselects in a chain:
                        # This selects among 8 values using 3 bits
                        # But vselect only picks between 2 values based on condition

                        # SKIP this optimization for now - it's too complex for the current structure
                        pass

                    # Fall back to original load_offset for now
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
                    )'''

# Actually let me take a simpler approach - just test with depth 3 using
# the existing build_hash_pipeline_addr but with a modified version that
# does scalar loads instead of load_offset

print("Depth 3 vselect approach is complex. Let me try a different approach...")
print("Testing simpler bucketing strategy...")

# Simpler test: just run the baseline to confirm cycles
with open('perf_takehome_d3test.py', 'w', encoding='utf-8') as f:
    f.write(content)

import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_d3test.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"With depth 3 prep (no change yet): {match.group(1)} cycles")
elif 'AssertionError' in output:
    print("Correctness FAILED!")
    print(output[-1500:])
else:
    print("ERROR")
    print(output[-1500:])
