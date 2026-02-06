# Hybrid v3: Procesar 4 elementos escalares intercalados con 252 vectoriales
# 252 = 31.5 vectores, redondeamos a 248 = 31 vectores
# Los 8 elementos restantes (256-248) se procesan con ALU intercalado

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Cambiar UNROLL_MAIN de 32 a 31 (248 elementos vectoriales)
content = content.replace('UNROLL_MAIN = 32', 'UNROLL_MAIN = 31')

# Ajustar starts para 31 vectores
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

new_starts = '''                    # 31 vectores: intercalar para mejor pipelining
                    starts = tuple(range(31))'''

content = content.replace(old_starts, new_starts)

# Agregar procesamiento escalar de 8 elementos INTERCALADO con cada round
# Buscar donde se hace emit_hash_only_range y agregar operaciones ALU

old_emit_pattern = '''                    for start in starts:
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

# Por ahora, solo cambiar a 31 vectores y ver el impacto base
# Luego agregaremos el procesamiento escalar

with open('perf_takehome_hybrid_v3.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Archivo creado: perf_takehome_hybrid_v3.py')
print('Solo 31 vectores (248 elementos) - sin escalares aun')
