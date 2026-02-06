# Hybrid v4: 31 vectores (248 elementos) + 8 escalares intercalados
# El objetivo es que las operaciones ALU llenen los ciclos con VALU < 6

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Cambiar UNROLL_MAIN a 31
content = content.replace('UNROLL_MAIN = 32', 'UNROLL_MAIN = 31')

# 2. Simplificar starts
old_starts = '''                    if depth > 2:
                        starts = (
                            0, 16, 4, 20, 8, 24, 12, 28, 2, 18, 6, 22, 10, 26, 14, 30,
                            1, 17, 5, 21, 9, 25, 13, 29, 3, 19, 7, 23, 11, 27, 15, 31,
                        )
                    else:
                        starts = (
                            0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                            1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31,
                        )'''

new_starts = '''                    starts = tuple(range(31))'''

content = content.replace(old_starts, new_starts)

# 3. Agregar inicializacion de escalares despues de "body = []"
old_body_init = '''            body = []

            UNROLL_MAIN = 31'''

new_body_init = '''            body = []

            UNROLL_MAIN = 31

            # Scalar elements (248-255)
            SCALAR_COUNT = 8
            s_idx = [self.alloc_scratch(f"s_idx{i}") for i in range(SCALAR_COUNT)]
            s_val = [self.alloc_scratch(f"s_val{i}") for i in range(SCALAR_COUNT)]
            s_tmp1 = [self.alloc_scratch(f"s_tmp1_{i}") for i in range(SCALAR_COUNT)]
            s_tmp2 = [self.alloc_scratch(f"s_tmp2_{i}") for i in range(SCALAR_COUNT)]
            s_node_val = [self.alloc_scratch(f"s_node{i}") for i in range(SCALAR_COUNT)]'''

content = content.replace(old_body_init, new_body_init)

# 4. Agregar procesamiento escalar intercalado despues de emit_group
old_emit_group = '''            emit_group(UNROLL_MAIN, zero_const, base_is_zero=True)

            body_instrs = self.build(body, vliw=True)'''

new_emit_group = '''            emit_group(UNROLL_MAIN, zero_const, base_is_zero=True)

            # === SCALAR ELEMENTS (248-255) INTERLEAVED ===
            # Initialize scalar values
            for i in range(SCALAR_COUNT):
                offset = self.scratch_const(248 + i)
                s_addr = self.alloc_scratch(f"s_addr_{i}")
                body.append(("alu", ("+", s_addr, self.scratch["inp_values_p"], offset)))
                body.append(("load", ("load", s_val[i], s_addr)))
                body.append(("alu", ("+", s_idx[i], zero_const, zero_const)))

            # Process scalar rounds - interleave with vector operations
            round_depths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]
            for round_idx in range(rounds):
                depth = round_depths[round_idx]

                # Process 2 scalar elements per chunk to interleave better
                for chunk_start in range(0, SCALAR_COUNT, 2):
                    for i in range(chunk_start, min(chunk_start + 2, SCALAR_COUNT)):
                        if depth == 0:
                            body.append(("alu", ("^", s_val[i], s_val[i], root_val)))
                        elif depth == 1:
                            body.append(("alu", ("&", s_tmp1[i], s_val[i], one_const)))
                            body.append(("alu", ("+", s_idx[i], s_tmp1[i], one_const)))
                            body.append(("flow", ("select", s_node_val[i], s_tmp1[i], level1_right, level1_left)))
                            body.append(("alu", ("^", s_val[i], s_val[i], s_node_val[i])))
                        else:
                            body.append(("alu", ("+", s_tmp1[i], forest_values_p, s_idx[i])))
                            body.append(("load", ("load", s_node_val[i], s_tmp1[i])))
                            body.append(("alu", ("^", s_val[i], s_val[i], s_node_val[i])))

                        # Scalar hash
                        body.extend(self.build_hash(s_val[i], s_tmp1[i], s_tmp2[i], round_idx, 248+i, False))

                        # Update idx
                        if depth != 0 and depth != forest_height:
                            body.append(("alu", ("&", s_tmp1[i], s_val[i], one_const)))
                            body.append(("alu", ("*", s_idx[i], s_idx[i], two_const)))
                            body.append(("alu", ("+", s_idx[i], s_idx[i], one_const)))
                            body.append(("alu", ("+", s_idx[i], s_idx[i], s_tmp1[i])))

            # Store scalar results
            for i in range(SCALAR_COUNT):
                offset = self.scratch_const(248 + i)
                s_val_addr = self.alloc_scratch(f"s_val_addr_{i}")
                body.append(("alu", ("+", s_val_addr, self.scratch["inp_values_p"], offset)))
                body.append(("store", ("store", s_val_addr, s_val[i])))
                if store_indices:
                    s_idx_addr = self.alloc_scratch(f"s_idx_addr_{i}")
                    body.append(("alu", ("+", s_idx_addr, self.scratch["inp_indices_p"], offset)))
                    body.append(("alu", ("-", s_tmp1[i], s_idx[i], forest_values_p)))
                    body.append(("store", ("store", s_idx_addr, s_tmp1[i])))

            body_instrs = self.build(body, vliw=True)'''

content = content.replace(old_emit_group, new_emit_group)

with open('perf_takehome_hybrid_v4.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Archivo creado: perf_takehome_hybrid_v4.py')
