"""
FASE 1 - Cache depth 3 con vselect tree completo.

Implementación cuidadosa que reutiliza temporales:
- Procesar el vselect tree en etapas
- Reutilizar registros cuando ya no se necesitan
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Primero, agregar 3 vectores temporales extra para depth 3 vselect
old_tmp3 = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)'''
new_tmp3 = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)
            # Extra temps for depth 3 vselect tree
            v_d3_tmp1 = self.alloc_scratch("v_d3_tmp1", length=VLEN)
            v_d3_tmp2 = self.alloc_scratch("v_d3_tmp2", length=VLEN)
            v_d3_tmp3 = self.alloc_scratch("v_d3_tmp3", length=VLEN)
            v_d3_tmp4 = self.alloc_scratch("v_d3_tmp4", length=VLEN)'''

if old_tmp3 in content:
    content = content.replace(old_tmp3, new_tmp3)
    print("1. Added extra temps for depth 3")
else:
    print("ERROR: Could not find v_tmp3_shared")

# Agregar level 3 scalar/vector allocations
old_level2_alloc = '''            level2_vals = [self.alloc_scratch(f"level2_val_{i}") for i in range(4)]'''
new_level2_alloc = '''            level2_vals = [self.alloc_scratch(f"level2_val_{i}") for i in range(4)]
            level3_vals = [self.alloc_scratch(f"level3_val_{i}") for i in range(8)]'''

if old_level2_alloc in content:
    content = content.replace(old_level2_alloc, new_level2_alloc)
    print("2. Added level3_vals")
else:
    print("ERROR 2")

old_v_level2_alloc = '''            v_level2 = [self.alloc_scratch(f"v_level2_{i}", length=VLEN) for i in range(4)]'''
new_v_level2_alloc = '''            v_level2 = [self.alloc_scratch(f"v_level2_{i}", length=VLEN) for i in range(4)]
            v_level3 = [self.alloc_scratch(f"v_level3_{i}", length=VLEN) for i in range(8)]'''

if old_v_level2_alloc in content:
    content = content.replace(old_v_level2_alloc, new_v_level2_alloc)
    print("3. Added v_level3")
else:
    print("ERROR 3")

# Agregar constante v_seven
old_v_neg = '''            v_neg_forest = self.alloc_scratch("v_neg_forest", length=VLEN)'''
new_v_neg = '''            v_neg_forest = self.alloc_scratch("v_neg_forest", length=VLEN)
            v_seven = self.alloc_scratch("v_seven", length=VLEN)'''

if old_v_neg in content:
    content = content.replace(old_v_neg, new_v_neg)
    print("4. Added v_seven")
else:
    print("ERROR 4")

# Cargar valores level 3 en init
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
            for i in range(8):
                addr_const = self.scratch_const(7 + i)
                init.append(("alu", ("+", tmp1, forest_values_p, addr_const)))
                init.append(("load", ("load", level3_vals[i], tmp1)))'''

if old_level2_loads in content:
    content = content.replace(old_level2_loads, new_level2_loads)
    print("5. Added level 3 loads")
else:
    print("ERROR 5")

# Broadcasts level 3
old_level2_bc = '''            init.append(("valu", ("vbroadcast", v_level2[0], level2_vals[0])))
            init.append(("valu", ("vbroadcast", v_level2[1], level2_vals[1])))
            init.append(("valu", ("vbroadcast", v_level2[2], level2_vals[2])))
            init.append(("valu", ("vbroadcast", v_level2[3], level2_vals[3])))'''

new_level2_bc = '''            init.append(("valu", ("vbroadcast", v_level2[0], level2_vals[0])))
            init.append(("valu", ("vbroadcast", v_level2[1], level2_vals[1])))
            init.append(("valu", ("vbroadcast", v_level2[2], level2_vals[2])))
            init.append(("valu", ("vbroadcast", v_level2[3], level2_vals[3])))
            # Level 3 broadcasts
            for i in range(8):
                init.append(("valu", ("vbroadcast", v_level3[i], level3_vals[i])))
            # v_seven constant
            seven_const = self.scratch_const(7)
            init.append(("valu", ("vbroadcast", v_seven, seven_const)))'''

if old_level2_bc in content:
    content = content.replace(old_level2_bc, new_level2_bc)
    print("6. Added level 3 broadcasts")
else:
    print("ERROR 6")

# Ahora la parte crítica: reemplazar el manejo de depth > 2 con depth == 3 especial
# Buscar el bloque else que maneja depth > 2

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
                    )

            def emit_idx_update(vec_count):'''

new_depth_else = '''                elif depth == 3:
                    # Depth 3: Use vselect tree with cached values
                    # tree_idx ∈ {7..14}, offset = tree_idx - 7 ∈ {0..7}
                    for u in range(count):
                        uu = start + u  # actual vector index in scratch

                        # Step 1: Compute offset = tree_idx - 7
                        # tree_idx = v_idx - forest_values_p
                        # We have v_neg_forest = 1 - forest_values_p
                        # So tree_idx = v_idx + v_neg_forest - 1
                        body.append(("valu", ("+", v_d3_tmp1, v_idx[uu], v_neg_forest)))
                        body.append(("valu", ("-", v_d3_tmp1, v_d3_tmp1, v_one)))
                        # tree_idx in v_d3_tmp1

                        # offset = tree_idx - 7
                        body.append(("valu", ("-", v_d3_tmp1, v_d3_tmp1, v_seven)))
                        # offset in v_d3_tmp1

                        # Step 2: Extract bits
                        # bit0 = offset & 1
                        body.append(("valu", ("&", v_d3_tmp2, v_d3_tmp1, v_one)))
                        # bit0 in v_d3_tmp2

                        # bit1 = (offset >> 1) & 1
                        body.append(("valu", (">>", v_d3_tmp3, v_d3_tmp1, v_one)))
                        body.append(("valu", ("&", v_d3_tmp3, v_d3_tmp3, v_one)))
                        # bit1 in v_d3_tmp3

                        # bit2 = (offset >> 2) & 1
                        body.append(("valu", (">>", v_d3_tmp4, v_d3_tmp1, v_two)))
                        body.append(("valu", ("&", v_d3_tmp4, v_d3_tmp4, v_one)))
                        # bit2 in v_d3_tmp4

                        # Step 3: vselect tree for 8 values
                        # Level 1: select between pairs using bit0
                        # sel01 = bit0 ? v_level3[1] : v_level3[0]
                        body.append(("flow", ("vselect", v_d3_tmp1, v_d3_tmp2, v_level3[1], v_level3[0])))
                        # sel23 = bit0 ? v_level3[3] : v_level3[2]
                        body.append(("flow", ("vselect", v_tmp1[uu], v_d3_tmp2, v_level3[3], v_level3[2])))
                        # sel45 = bit0 ? v_level3[5] : v_level3[4]
                        body.append(("flow", ("vselect", v_tmp2[uu], v_d3_tmp2, v_level3[5], v_level3[4])))
                        # sel67 = bit0 ? v_level3[7] : v_level3[6]
                        body.append(("flow", ("vselect", v_tmp3_shared, v_d3_tmp2, v_level3[7], v_level3[6])))

                        # Level 2: select between pairs using bit1
                        # sel0123 = bit1 ? sel23 : sel01
                        body.append(("flow", ("vselect", v_d3_tmp1, v_d3_tmp3, v_tmp1[uu], v_d3_tmp1)))
                        # sel4567 = bit1 ? sel67 : sel45
                        body.append(("flow", ("vselect", v_d3_tmp2, v_d3_tmp3, v_tmp3_shared, v_tmp2[uu])))

                        # Level 3: select between halves using bit2
                        # result = bit2 ? sel4567 : sel0123
                        body.append(("flow", ("vselect", v_tmp1[uu], v_d3_tmp4, v_d3_tmp2, v_d3_tmp1)))
                        # node_val now in v_tmp1[uu]

                        # XOR with val
                        body.append(("valu", ("^", v_val[uu], v_val[uu], v_tmp1[uu])))

                    # Hash (reuse tmp vectors)
                    body.extend(
                        self.build_hash_vec_multi(
                            v_val[start : start + count],
                            v_tmp1[start : start + count],
                            v_tmp2[start : start + count],
                            round_idx,
                            start * VLEN,
                            emit_debug,
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
                    )

            def emit_idx_update(vec_count):'''

if old_depth_else in content:
    content = content.replace(old_depth_else, new_depth_else)
    print("7. Added depth 3 vselect tree")
else:
    print("ERROR 7: Could not find depth else block")

with open('perf_takehome_d3full.py', 'w', encoding='utf-8') as f:
    f.write(content)

import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_d3full.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    cycles = int(match.group(1))
    diff = 1615 - cycles
    print(f"Depth 3 vselect tree: {cycles} cycles (baseline: 1615, diff: {diff:+d})")
elif 'OK' in output:
    print("Test passed")
else:
    print("ERROR:")
    if 'AssertionError' in output:
        print("Correctness FAILED!")
    elif 'Out of scratch' in output:
        print("Scratch overflow!")
    print(output[-2500:])
