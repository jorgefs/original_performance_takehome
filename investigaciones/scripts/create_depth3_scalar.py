"""
Enfoque alternativo para depth 3: usar loads escalares con patrón conocido.

Observación clave de la instrumentación:
- Depth 3 tiene solo 8 índices únicos
- Los 8 valores son forest_values[7..14]
- Están precargados en level3_vals[0..7]

Nuevo enfoque: En lugar de 256 load_offsets (que leen memoria principal),
hacer 8 scalar loads desde SCRATCH (donde ya están los valores).

Pero wait - los valores ya están en vectors v_level3[0..7].
El problema es seleccionar cuál valor aplicar a cada lane.

Enfoque más simple: ¿Qué tal si usamos el hecho de que depth 3
siempre sigue después de depth 2, y depth 2 tiene solo 4 índices?

Después de depth 2:
- idx ∈ {3,4,5,6}
- El idx update hace: new_idx = 2*idx + 1 + bit

Entonces para depth 3:
- Si idx era 3: new_idx ∈ {7,8}
- Si idx era 4: new_idx ∈ {9,10}
- Si idx era 5: new_idx ∈ {11,12}
- Si idx era 6: new_idx ∈ {13,14}

Esto significa que podemos predecir los posibles índices basados en el
estado de depth 2, y usar menos vselects!

En depth 2, usamos 3 vselects. Si recordamos el resultado intermedio
(cuál de los 4 valores se seleccionó), podemos reducir depth 3 a solo
1 vselect adicional por vector!

Estructura:
- En depth 2: además de XOR+hash, guardamos bit1 (la selección de nivel 2)
- En depth 3: usamos bit1 guardado + bit0 nuevo para seleccionar entre 8 valores
  con solo 1 vselect adicional (porque ya sabemos de qué grupo de 4 viene)

NO - esto no funciona porque depth 2 también tiene hash que cambia val,
así que el bit0 de depth 3 es diferente.

Volvamos al enfoque original pero optimicemos el número de vselects.
"""

# Por ahora, revertir a baseline y probar otras ideas
print("El enfoque depth 3 con vselects es fundamentalmente costoso.")
print("Probando enfoque alternativo: eliminar redundancia en idx update.")
print()

# Idea: el idx update hace 3 ops por vector:
# 1. bit = val & 1
# 2. tmp = idx * 2 + base_minus1 (multiply_add)
# 3. idx = tmp + bit

# La op 2 (multiply_add) es INDEPENDIENTE del hash result!
# Podemos mover multiply_add ANTES del hash para crear más ILP.

# Pero ya probamos esto y no ayudó. El VLIW scheduler ya lo maneja.

# Otra idea: reducir operaciones combinando pasos.
# Actualmente: bit = val & 1, tmp = idx*2 + base_minus1, idx = tmp + bit
# Equivalente: idx = idx*2 + base_minus1 + (val & 1)
#            = idx*2 + 1 - forest_p + (val & 1)

# Con multiply_add: idx = idx*2 + c donde c = 1 - forest_p + bit
# Pero c depende de bit, así que no podemos precomputar c.

# Sin embargo, podemos hacer:
# idx = idx*2 + 1 - forest_p  (multiply_add, independiente)
# idx = idx + bit  (add, depende de val)

# Esto reduce de 3 ops a 2 ops por vector para idx update.
# Ahorro: 32 vectors * (16-2) rounds que usan idx update = 32 * 14 = 448 ops menos
# Pero espera, el código actual hace exactamente esto para algunos casos.

# Revisemos el código actual de idx update más cuidadosamente...

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Buscar el idx update code
import re
idx_pattern = r'# Skip idx update.*?v_idx\[u\], v_idx\[u\], v_tmp1\[u\]'
matches = re.findall(idx_pattern, content, re.DOTALL)
print(f"Found {len(matches)} idx update patterns")

# El idx update actual hace:
# body.append(("valu", ("&", v_tmp1[u], v_val[u], v_one)))
# body.append(("valu", ("multiply_add", v_idx[u], v_idx[u], v_two, v_base_minus1)))
# body.append(("valu", ("+", v_idx[u], v_idx[u], v_tmp1[u])))

# Esto es: bit = val & 1, idx = idx*2 + base_minus1, idx = idx + bit
# Total: 3 VALU ops

# El multiply_add computa idx*2 + base_minus1 = idx*2 + (1 - forest_p)
# Luego sumamos bit.

# Podríamos reordenar:
# idx = idx*2 + base_minus1  (multiply_add)
# bit = val & 1
# idx = idx + bit

# Pero el scheduler ya maneja esto.

# La verdadera optimización sería NO hacer idx update en algunos casos.
# Pero necesitamos idx para el siguiente depth...

# A menos que... cambiemos la representación de idx.
# En lugar de mantener idx actualizado, podríamos mantener solo los bits
# y reconstruir idx cuando lo necesitemos.

# Para depth d, idx tiene d bits significativos.
# Si mantenemos los bits en un formato packed, podríamos reconstruir idx
# solo cuando lo necesitemos para el gather.

# Pero esto requiere más operaciones para reconstruir, no menos.

print()
print("Conclusión: Las optimizaciones de scheduling ya están bien aplicadas.")
print("Para mejoras significativas necesitamos cambios algorítmicos más profundos.")
print()
print("Ideas pendientes:")
print("1. Procesar paths en orden diferente para maximizar cache locality")
print("2. Precomputar patrones de acceso cuando sea posible")
print("3. Usar especulación selectiva para paths con alta probabilidad")
