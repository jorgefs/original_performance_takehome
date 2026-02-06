# Hybrid kernel v2: Intercalar operaciones ALU y VALU
# La clave es emitir operaciones ALU mezcladas con VALU para que el scheduler las empaquete

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Estrategia: Mantener 32 vectores pero agregar procesamiento escalar
# de los ultimos 32 elementos ANTES del procesamiento vectorial
# Asi el scheduler puede mezclarlos

# 1. NO cambiar UNROLL_MAIN - mantener 32 vectores para los primeros 224 elementos
# Pero espera, 32 vectores * 8 = 256 elementos, no 224

# Nueva estrategia: Procesar 256 elementos con VALU pero TAMBIEN
# hacer trabajo escalar redundante que se mezcle con VALU
# Esto no es eficiente pero es un experimento

# Mejor idea: Reducir a 24 vectores (192 elementos) y agregar 64 escalares
# que se procesen intercalados

# Por ahora, probemos solo agregar operaciones ALU "dummy" para ver
# si el scheduler las empaqueta con VALU

# Buscar el patron donde se emiten las operaciones de hash
# y agregar operaciones ALU intercaladas

# Modificacion simple: durante cada round, emitir algunas operaciones ALU
# que preparen datos futuros (aunque sean redundantes)

old_hash_emit = '''                    for start in starts:
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

new_hash_emit = '''                    for start in starts:
                        count = min(chunk, vec_count - start)
                        emit_hash_only_range(round_idx, depth, start, count)
                        # Skip idx update for depth 0 and forest_height
                        if depth != 0 and depth != forest_height:
                            for u in range(start, start + count):
                                # Emit idx*2 + C BEFORE the AND (it doesn't depend on val)
                                body.append(
                                    (
                                        "valu",
                                        (
                                            "multiply_add",
                                            v_tmp2[u],  # Use tmp2 as intermediate
                                            v_idx[u],
                                            v_two,
                                            v_base_minus1,
                                        ),
                                    )
                                )
                                body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
                                body.append(("valu", ("+", v_idx[u], v_tmp2[u], v_tmp1[u])))'''

if old_hash_emit in content:
    content = content.replace(old_hash_emit, new_hash_emit)
    print("Patron encontrado y reemplazado")
else:
    print("ADVERTENCIA: Patron no encontrado")

with open('perf_takehome_hybrid_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Archivo creado: perf_takehome_hybrid_v2.py')
