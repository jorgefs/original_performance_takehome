"""
Optimizar idx update: reducir de 3 ops a 2 ops.

Actual:
  bit = val & 1                          # VALU 1
  idx = idx * 2 + base_minus1            # VALU 2 (multiply_add)
  idx = idx + bit                        # VALU 3

Observación: base_minus1 = 1 - forest_p

Nueva fórmula:
  idx = idx * 2 + 1 - forest_p + (val & 1)
      = idx * 2 + (2 - forest_p) + (val & 1) - 1

Pero necesitamos (val & 1), así que al menos 2 ops.

Alternativa usando propiedades del árbol:
  new_idx = 2*old_idx + 1 + bit
  Si tenemos v_base_plus1 = forest_p + 1, entonces:
  idx_mem = forest_p + tree_idx
  new_tree_idx = 2*tree_idx + 1 + bit
  new_idx_mem = forest_p + 2*tree_idx + 1 + bit
              = 2*(forest_p + tree_idx) + 1 + bit - forest_p
              = 2*idx_mem + 1 + bit - forest_p
              = 2*idx_mem + (1 - forest_p) + bit  <- fórmula actual

Con multiply_add podemos hacer: idx = idx * 2 + (1 - forest_p) en 1 op
Luego: idx = idx + bit en 1 op

Total: 2 ops si movemos bit extraction a otro lugar o la hacemos inline.

Pero bit = val & 1 es 1 op, así que mínimo es 2 ops si podemos fusionar algo.

¿Podemos usar multiply_add de forma diferente?
multiply_add(dest, a, b, c) = a*b + c

¿Qué tal si computamos idx*2 + bit directamente, luego sumamos base_minus1?
  tmp = idx * 2 + bit    # necesitaríamos multiply_add(tmp, idx, 2, bit) pero bit no es constante
  idx = tmp + base_minus1

No, multiply_add requiere que c sea un registro, no una expresión.

¿Qué tal si precomputamos idx*2 antes del hash y guardamos?
  pre_idx = idx * 2  # antes del hash, independiente
  # ... hash ...
  bit = val & 1
  idx = pre_idx + base_minus1 + bit  # 2 ops: add, add

Esto es 3 ops total (mul antes, 2 adds después), no mejor.

¿Qué tal usar la estructura del problema?
Para depth d, tree_idx ∈ [2^d - 1, 2^(d+1) - 2]
idx_mem = forest_p + tree_idx

Después del update:
new_tree_idx = 2*tree_idx + 1 + bit
new_idx_mem = forest_p + new_tree_idx

Hmm, no veo una simplificación obvia.

Probemos otra cosa: mover el multiply_add fuera del loop crítico.
El multiply_add no depende del hash result, solo de idx actual.
Si lo emitimos ANTES de build_hash_pipeline_addr, puede correr durante los loads.

Ya intenté esto antes sin éxito. El scheduler ya lo maneja.

Última idea: ¿Qué tal si NO actualizamos idx en cada round, sino que lo reconstruimos cuando lo necesitamos?

Para un path, el tree_idx en depth d es determinado por la secuencia de bits de las decisiones left/right.
Si guardamos estos bits (d bits por path), podemos reconstruir tree_idx como:
  tree_idx = sum(bit[i] * 2^i for i in 0..d-1) + (2^d - 1)

Pero esto requiere d operaciones para reconstruir, vs 3 operaciones para actualizar incrementalmente.
Solo vale la pena si d es pequeño... pero para depths grandes (7-10), sería más costoso.

No hay optimización obvia aquí. Probemos otra cosa.
"""

# Probar sin cambios significativos, solo verificar el baseline
import subprocess
result = subprocess.run(
    ['python', 'perf_takehome.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
import re
match = re.search(r'CYCLES:\s*(\d+)', output)
if match:
    print(f"Baseline verificado: {match.group(1)} cycles")

print()
print("Análisis de posibles optimizaciones:")
print()
print("1. Idx update: 3 VALU ops × ~448 updates = 1344 ops")
print("   - No hay forma obvia de reducir sin cambiar el algoritmo")
print()
print("2. Hash: 12 VALU ops × 512 hashes = 6144 ops")
print("   - Ya optimizado con multiply_add para 3 stages")
print()
print("3. XOR: 1 VALU op × 512 = 512 ops")
print("   - Mínimo posible")
print()
print("4. Load_offset: 2560 loads")
print("   - Principal oportunidad: reducir para depths 3-6")
print("   - Pero vselect tree es más costoso que load_offset")
print()
print("Conclusión: Las optimizaciones de software gather requieren")
print("instrucciones que no están disponibles (scatter/gather en scratch).")
print()
print("Para alcanzar 1363 cycles, probablemente se necesita un")
print("enfoque algorítmico completamente diferente que no hemos explorado.")
