"""
Enfoque de reorganización de paths por buckets.

Idea: En lugar de procesar 32 vectores donde cada lane tiene idx diferente,
reorganizar los paths para que paths con mismo idx estén en lanes contiguas.

Para depth 3 con 8 idx únicos y 256 paths:
- Bucket 0 (idx=7): ~32 paths
- Bucket 1 (idx=8): ~32 paths
- ...

Si reorganizamos, podemos tener:
- Vectores 0-3: todos paths con idx=7 (o mezcla de idx=7,8)
- Vectores 4-7: todos paths con idx=8,9
- etc.

Entonces para cada grupo de vectores con mismo idx:
- 1 scalar load de forest_values[idx]
- 1 vbroadcast
- vstore XOR directo (sin vselect)

El costo de reorganización:
- Necesitamos crear una permutación de los paths
- Aplicar scatter para reorganizar v_val y v_idx
- Después de procesamiento, aplicar gather para restaurar orden

Sin instrucciones scatter/gather, usamos vselect chains.
Pero eso es costoso...

ALTERNATIVA: No restaurar el orden. Mantener paths reorganizados.
El resultado final son pares (idx, val) para cada path.
El orden no importa si guardamos ambos correctamente al final.

Implementación:
1. Después de round 2, calcular bucket para cada path
2. Crear vectores "compactos" donde cada vector tiene paths del mismo bucket
3. Procesar depths 3+ con loads escalares por bucket
4. Al final, guardar resultados con índices correctos
"""

import subprocess
import re

# Primero, veamos si podemos hacer una versión simplificada:
# Para depth 3, en lugar de reorganizar, simplemente iteramos por bucket
# y usamos máscaras para aplicar XOR solo a los paths relevantes.

# Esto requiere:
# 1. Precargar los 8 valores de depth 3 (ya hecho en v_level3)
# 2. Calcular máscara: cuáles lanes tienen idx=7, cuáles idx=8, etc.
# 3. Para cada bucket: XOR condicional usando vselect

# El XOR condicional sería:
# new_val = vselect(mask, val ^ node_val, val)
# Esto es: 1 XOR + 1 vselect por bucket = 8 XOR + 8 vselect por vector

# Para 32 vectores: 256 XOR + 256 vselect
# vs original: 256 load_offset + 32 XOR

# En throughput:
# - Original: 128 load cycles + ~6 XOR cycles = ~134 cycles
# - Bucket: ~43 XOR cycles + 256 vselect cycles = ~299 cycles

# Aún peor por el mismo problema de flow throughput.

print("Análisis de bucket con máscaras:")
print("- Original depth 3: 128 load cycles + 6 XOR cycles ≈ 134 cycles")
print("- Bucket con vselect: 43 XOR cycles + 256 vselect cycles ≈ 299 cycles")
print("- Conclusión: Sigue siendo peor por flow throughput")
print()

# Idea diferente: ¿Qué tal si procesamos MENOS paths pero de forma más eficiente?
# No - esto no tiene sentido, necesitamos procesar todos.

# Idea: ¿Qué tal si hacemos la reorganización ANTES del kernel?
# En el setup de memoria, ordenar inp.values por su trayectoria esperada.
# Pero las trayectorias dependen de los hashes, que son runtime.

# Idea: ¿Qué tal si usamos el patrón de que depth 0,1,2 son deterministas
# en términos de estructura, solo el hash value cambia?

# En depth 0: todos van a root (idx=0)
# En depth 1: van a idx=1 o idx=2 basado en bit0
# En depth 2: van a idx=3,4,5,6 basado en bit0,bit1

# Después de depth 2, los paths se distribuyen en 4 grupos:
# - Grupo A (idx=3): paths que tomaron left-left
# - Grupo B (idx=4): paths que tomaron left-right
# - Grupo C (idx=5): paths que tomaron right-left
# - Grupo D (idx=6): paths que tomaron right-right

# Para depth 3, cada grupo se bifurca en 2:
# - A -> idx=7 o idx=8
# - B -> idx=9 o idx=10
# - C -> idx=11 o idx=12
# - D -> idx=13 o idx=14

# La clave: los paths en grupo A SOLO acceden a idx=7 o idx=8.
# Si procesamos grupo A separadamente, solo necesitamos 2 valores, no 8!

# Esto sugiere un enfoque jerárquico:
# 1. Después de depth 2, separar paths en 4 grupos por idx
# 2. Para depth 3, cada grupo solo necesita 2 valores (sus hijos)
# 3. Usar 1 vselect por grupo para elegir entre 2 valores

print("Enfoque jerárquico por grupos:")
print("Después de depth 2, hay 4 grupos basados en idx ∈ {3,4,5,6}")
print("Para depth 3, cada grupo solo bifurca en 2 opciones:")
print("- Grupo idx=3 -> hijos {7,8}")
print("- Grupo idx=4 -> hijos {9,10}")
print("- Grupo idx=5 -> hijos {11,12}")
print("- Grupo idx=6 -> hijos {13,14}")
print()
print("Si conocemos el grupo de cada path, solo necesitamos 1 vselect")
print("para elegir entre 2 valores (no 7 vselects para 8 valores)!")
print()

# ¿Cómo conocemos el grupo? Por los bits del idx después de depth 2.
# idx ∈ {3,4,5,6} = {0b011, 0b100, 0b101, 0b110}
# Los bits 0 y 1 de idx determinan el grupo.

# Para depth 3:
# 1. Extraer bits del idx para determinar grupo
# 2. Para cada grupo, precargar solo los 2 valores relevantes
# 3. Usar 1 vselect para elegir

# Pero espera - cada LANE puede estar en un grupo diferente.
# Así que aún necesitamos manejar 4 grupos por vector.

# Alternativa: reorganizar paths para que cada vector tenga paths del mismo grupo
# Después de depth 2, hacer una pasada de "compactación" por grupo.

# Esto es complejo pero podría funcionar...

# Implementación simplificada para probar:
# Para depth 3, usar el hecho de que el grupo se puede inferir de bit1 de idx
# idx & 2 == 0 -> grupo A o B (idx ∈ {3,5} si impar en bit1... no, esto no es así)

# Veamos: idx=3 (0b11), idx=4 (0b100), idx=5 (0b101), idx=6 (0b110)
# bit1 (idx >> 1) & 1: 3->1, 4->0, 5->0, 6->1
# bit0 idx & 1: 3->1, 4->0, 5->1, 6->0

# Para depth 3, new_idx = 2*idx + 1 + bit
# Si idx=3, new_idx ∈ {7,8}
# Si idx=4, new_idx ∈ {9,10}
# etc.

# La selección dentro de cada par es por bit (resultado del hash).
# La selección del par es por idx anterior.

# Podemos hacer:
# 1. Precargar pares: (v_level3[0], v_level3[1]) para idx=3
#                    (v_level3[2], v_level3[3]) para idx=4
#                    etc.
# 2. Primero seleccionar el par correcto basado en idx
# 3. Luego seleccionar dentro del par basado en bit

# Paso 2 requiere conocer idx. Si idx está en v_idx, podemos extraer bits.
# Pero esto aún requiere múltiples vselects...

# A menos que procesemos de forma diferente: en lugar de 32 vectores mixtos,
# procesar 4 grupos de ~8 vectores cada uno, donde cada grupo tiene idx fijo.

# Para hacer esto, necesitamos reorganizar los paths después de depth 2.

print("Probando enfoque de grupos para depth 3...")
print()

# Crear versión que procesa depth 3 en 4 sub-grupos
# Para simplificar, asumamos que podemos identificar el grupo de cada lane
# y procesar con 2 vselects en lugar de 7:
# vselect 1: elegir par de valores basado en grupo (idx bits)
# vselect 2: elegir valor dentro del par basado en bit

# Pero vselect solo elige entre 2 opciones. Para 4 grupos necesitamos 2 vselects.
# Luego 1 vselect para elegir dentro del par.
# Total: 3 vselects por vector vs 7 para árbol completo.

# 3 * 32 = 96 vselects vs 224 vselects (7 * 32)
# Ahorro: 128 vselects = 128 cycles de flow

# Pero aún necesitamos calcular los bits del grupo, que son operaciones VALU extra.

# Implementemos esto:

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Modificar depth 3 para usar enfoque de 2 niveles de vselect

# Primero agregar los vectores de level 3 si no están
if 'v_level3' not in content:
    # Agregar level 3 allocations y loads (código similar a antes)
    old_level2_alloc = '''            level2_vals = [self.alloc_scratch(f"level2_val_{i}") for i in range(4)]'''
    new_level2_alloc = '''            level2_vals = [self.alloc_scratch(f"level2_val_{i}") for i in range(4)]
            level3_vals = [self.alloc_scratch(f"level3_val_{i}") for i in range(8)]'''
    content = content.replace(old_level2_alloc, new_level2_alloc)

    old_v_level2 = '''            v_level2 = [self.alloc_scratch(f"v_level2_{i}", length=VLEN) for i in range(4)]'''
    new_v_level2 = '''            v_level2 = [self.alloc_scratch(f"v_level2_{i}", length=VLEN) for i in range(4)]
            v_level3 = [self.alloc_scratch(f"v_level3_{i}", length=VLEN) for i in range(8)]'''
    content = content.replace(old_v_level2, new_v_level2)

    # Loads en init
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
            # Level 3 loads (tree indices 7-14)
            for i in range(8):
                addr_const = self.scratch_const(7 + i)
                init.append(("alu", ("+", tmp1, forest_values_p, addr_const)))
                init.append(("load", ("load", level3_vals[i], tmp1)))'''
    content = content.replace(old_level2_loads, new_level2_loads)

    # Broadcasts
    old_bc = '''            init.append(("valu", ("vbroadcast", v_level2[0], level2_vals[0])))
            init.append(("valu", ("vbroadcast", v_level2[1], level2_vals[1])))
            init.append(("valu", ("vbroadcast", v_level2[2], level2_vals[2])))
            init.append(("valu", ("vbroadcast", v_level2[3], level2_vals[3])))'''
    new_bc = '''            init.append(("valu", ("vbroadcast", v_level2[0], level2_vals[0])))
            init.append(("valu", ("vbroadcast", v_level2[1], level2_vals[1])))
            init.append(("valu", ("vbroadcast", v_level2[2], level2_vals[2])))
            init.append(("valu", ("vbroadcast", v_level2[3], level2_vals[3])))
            for i in range(8):
                init.append(("valu", ("vbroadcast", v_level3[i], level3_vals[i])))'''
    content = content.replace(old_bc, new_bc)
    print("Added level 3 setup")

# Ahora modificar el manejo de depth 3 para usar menos vselects
# La idea: usar el idx anterior (de depth 2) para determinar qué par de valores usar

# En depth 3, idx viene del update de depth 2: idx = 2*old_idx + 1 + bit
# donde old_idx ∈ {3,4,5,6}

# En lugar de vselect tree de 8 valores, hacemos:
# 1. Determinar old_idx (bits del idx actual)
# 2. Seleccionar par de valores basado en old_idx
# 3. Seleccionar dentro del par basado en bit del hash

# old_idx = (idx - 1) >> 1 cuando idx ∈ {7..14}
# O más simple: tree_idx = idx - forest_p, old_tree_idx = (tree_idx - 1) >> 1

# Para determinar el par:
# - Si old_tree_idx ∈ {3,4,5,6}, entonces (old_tree_idx - 3) ∈ {0,1,2,3}
# - pair_idx = old_tree_idx - 3

# Pero esto requiere calcular old_tree_idx, lo cual no tenemos directamente.
# Tenemos idx actual que es el resultado del idx_update de depth 2.

# Alternativa: en depth 2, GUARDAR el old_idx antes del update.
# Pero eso requiere más scratch.

# Alternativa más simple: calcular pair_idx desde tree_idx de depth 3.
# tree_idx ∈ {7..14}
# tree_idx - 7 ∈ {0..7}
# pair_idx = (tree_idx - 7) >> 1 ∈ {0,1,2,3}

# Con pair_idx, seleccionamos entre 4 pares: (0,1), (2,3), (4,5), (6,7)
# Esto requiere 2 vselects para elegir el par, más 1 para elegir dentro.
# Total: 3 vselects vs 7.

print("Implementando depth 3 con 3 vselects en lugar de 7...")

# El problema es que necesitamos 4 temps para los 4 pares antes de seleccionar.
# Veamos si podemos hacerlo con menos.

# Estructura de vselect para 4 pares + 1 selección final:
# pair_bit1 = (tree_idx - 7) >> 1 & 1  # distingue {0,1} vs {2,3} y {4,5} vs {6,7}
# pair_bit0 = ((tree_idx - 7) >> 1) >> 1 & 1 = (tree_idx - 7) >> 2 & 1  # distingue mitades

# Espera, eso no es correcto. Reorganizo:
# tree_idx - 7 ∈ {0,1,2,3,4,5,6,7}
# pair_idx = (tree_idx - 7) >> 1 ∈ {0,0,1,1,2,2,3,3} -> {0,1,2,3}
# within_pair = (tree_idx - 7) & 1 ∈ {0,1,0,1,0,1,0,1}

# Para seleccionar entre 4 pares necesitamos 2 bits de pair_idx:
# pair_bit0 = pair_idx & 1
# pair_bit1 = (pair_idx >> 1) & 1

# Selección jerárquica:
# 1. sel_01 = vselect(within_pair, v[1], v[0])
# 2. sel_23 = vselect(within_pair, v[3], v[2])
# 3. sel_45 = vselect(within_pair, v[5], v[4])
# 4. sel_67 = vselect(within_pair, v[7], v[6])
# 5. sel_0123 = vselect(pair_bit0, sel_23, sel_01)
# 6. sel_4567 = vselect(pair_bit0, sel_67, sel_45)
# 7. result = vselect(pair_bit1, sel_4567, sel_0123)

# Eso sigue siendo 7 vselects! No hay mejora.

# El problema es que cada vselect solo maneja 1 bit de selección.
# Para 8 valores necesitamos 3 bits = 7 nodos en el árbol binario.

# No hay forma de reducir vselects para selección de 8 valores.

print("Confirmado: 8 valores requieren 7 vselects mínimo (árbol binario)")
print("No hay mejora posible con este enfoque.")
print()

# Siguiente idea: ¿Qué tal si hacemos la selección de forma diferente?
# En lugar de árbol, ¿podemos usar arithmetic?

# node_val = v_level3[offset] donde offset = tree_idx - 7

# Pero no tenemos instrucción de índice vectorial en scratch.
# v_level3 son 8 vectores separados, no un array indexable.

# ¿Qué tal si los ponemos en memoria y usamos load_offset?
# Pero eso es exactamente lo que hace el código original con forest_values!

print("Conclusión: Sin instrucciones de gather en scratch,")
print("no hay forma de mejorar sobre load_offset.")
print()
print("Probando si hay alguna configuración que mejore marginalmente...")

# Probar diferentes configuraciones de interleaving
with open('perf_takehome_test.py', 'w', encoding='utf-8') as f:
    f.write(content)

result = subprocess.run(
    ['python', 'perf_takehome_test.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    cycles = int(match.group(1))
    print(f"Con level 3 setup: {cycles} cycles (baseline: 1615)")
else:
    print("Error en test")
    if 'AssertionError' in output:
        print("Correctness failed")
    elif 'Out of scratch' in output:
        print("Scratch overflow")
