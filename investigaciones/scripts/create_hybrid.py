# Script para crear version hibrida del kernel
# 28 vectores VALU + 32 escalares ALU intercalados

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Modificaciones principales:

# 1. Cambiar UNROLL_MAIN de 32 a 28
content = content.replace('UNROLL_MAIN = 32', 'UNROLL_MAIN = 28')

# 2. Simplificar starts a secuencial para 28 vectores
old_starts1 = '''                    if depth > 2:
                        starts = (
                            0, 16, 4, 20, 8, 24, 12, 28, 2, 18, 6, 22, 10, 26, 14, 30,
                            1, 17, 5, 21, 9, 25, 13, 29, 3, 19, 7, 23, 11, 27, 15, 31,
                        )
                    else:
                        starts = (
                            0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                            1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31,
                        )'''

new_starts1 = '''                    if depth > 2:
                        starts = tuple(range(28))
                    else:
                        starts = tuple(range(28))'''

content = content.replace(old_starts1, new_starts1)

# 3. Agregar procesamiento escalar despues de emit_group
old_emit = '''            emit_group(UNROLL_MAIN, zero_const, base_is_zero=True)

            body_instrs = self.build(body, vliw=True)'''

new_emit = '''            emit_group(UNROLL_MAIN, zero_const, base_is_zero=True)

            # === HYBRID: Process 32 scalar elements with ALU ===
            SCALAR_COUNT = 32
            s_idx = [self.alloc_scratch(f"s_idx{i}") for i in range(SCALAR_COUNT)]
            s_val = [self.alloc_scratch(f"s_val{i}") for i in range(SCALAR_COUNT)]
            s_tmp1 = [self.alloc_scratch(f"s_tmp1_{i}") for i in range(SCALAR_COUNT)]
            s_tmp2 = [self.alloc_scratch(f"s_tmp2_{i}") for i in range(SCALAR_COUNT)]
            s_val_addr = [self.alloc_scratch(f"s_val_addr{i}") for i in range(SCALAR_COUNT)]
            s_idx_addr = [self.alloc_scratch(f"s_idx_addr{i}") for i in range(SCALAR_COUNT)] if store_indices else []

            # Load scalar values (elements 224-255)
            for i in range(SCALAR_COUNT):
                offset_const = self.scratch_const(224 + i)
                body.append(("alu", ("+", s_val_addr[i], self.scratch["inp_values_p"], offset_const)))
                body.append(("load", ("load", s_val[i], s_val_addr[i])))
                body.append(("alu", ("+", s_idx[i], zero_const, zero_const)))
                if store_indices:
                    body.append(("alu", ("+", s_idx_addr[i], self.scratch["inp_indices_p"], offset_const)))

            # Scalar hash for all rounds
            round_depths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]
            for round_idx in range(rounds):
                depth = round_depths[round_idx]
                for i in range(SCALAR_COUNT):
                    if depth == 0:
                        body.append(("alu", ("^", s_val[i], s_val[i], root_val)))
                    elif depth == 1:
                        body.append(("alu", ("&", s_tmp1[i], s_val[i], one_const)))
                        body.append(("alu", ("+", s_idx[i], s_tmp1[i], one_const)))
                        body.append(("flow", ("select", s_tmp1[i], s_tmp1[i], level1_right, level1_left)))
                        body.append(("alu", ("^", s_val[i], s_val[i], s_tmp1[i])))
                    else:
                        # depth >= 2: load from memory (simpler than nested selects)
                        body.append(("alu", ("+", s_tmp1[i], forest_values_p, s_idx[i])))
                        body.append(("load", ("load", s_tmp2[i], s_tmp1[i])))
                        body.append(("alu", ("^", s_val[i], s_val[i], s_tmp2[i])))

                    # Scalar hash
                    body.extend(self.build_hash(s_val[i], s_tmp1[i], s_tmp2[i], round_idx, 224+i, False))

                    # Update idx
                    if depth != 0 and depth != forest_height:
                        body.append(("alu", ("&", s_tmp1[i], s_val[i], one_const)))
                        body.append(("alu", ("*", s_idx[i], s_idx[i], two_const)))
                        body.append(("alu", ("+", s_idx[i], s_idx[i], one_const)))
                        body.append(("alu", ("+", s_idx[i], s_idx[i], s_tmp1[i])))

            # Store scalar results
            for i in range(SCALAR_COUNT):
                if store_indices:
                    body.append(("alu", ("-", s_tmp1[i], s_idx[i], forest_values_p)))
                    body.append(("store", ("store", s_idx_addr[i], s_tmp1[i])))
                body.append(("store", ("store", s_val_addr[i], s_val[i])))

            body_instrs = self.build(body, vliw=True)'''

content = content.replace(old_emit, new_emit)

with open('perf_takehome_hybrid.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Archivo creado: perf_takehome_hybrid.py')
