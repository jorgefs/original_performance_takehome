"""
FASE 1: Análisis y cuantificación de estrategias de traversal alternativas.

Objetivo: Reducir gathers (load_offset) de ~4096 a <1200 idealmente.
"""

import random
from collections import defaultdict, Counter
from problem import Tree, Input, myhash, VLEN, SLOT_LIMITS

def simulate_baseline_costs():
    """Simula costos del baseline vectorizado actual."""
    # Por round: 256 batch elements, procesados en vectores de 8
    # 32 vectores por round, 16 rounds = 512 iteraciones de loop vectorial

    # Por iteración vectorial:
    # - 1 vload idx (2 cycles para 8 lanes via load_offset = 4 loads)
    # - 1 vload val (2 cycles)
    # - 8 gathers para forest_values[idx] (load_offset, 4 cycles = 8/2 loads)
    # - hash computation (18 valu ops, 3 cycles cada = 6 cycles si slot limit valu=6)
    # - 2 vbroadcast (constantes)
    # - branch logic (valu ops)
    # - 1 vstore idx, 1 vstore val

    print("=" * 80)
    print("ANÁLISIS DE COSTOS - BASELINE VECTORIZADO")
    print("=" * 80)

    rounds = 16
    batch = 256
    vlen = 8
    n_vectors = batch // vlen  # 32

    # Gathers por round (el cuello de botella)
    gathers_per_vector = vlen  # 8 load_offset
    gathers_per_round = n_vectors * gathers_per_vector  # 256
    total_gathers = gathers_per_round * rounds  # 4096

    # Con limit de load=2 por cycle, 8 gathers = 4 cycles
    gather_cycles_per_vector = gathers_per_vector // SLOT_LIMITS["load"]

    print(f"Vectores por round: {n_vectors}")
    print(f"Gathers por vector: {gathers_per_vector}")
    print(f"Total gathers: {total_gathers}")
    print(f"Cycles solo para gathers (load limit=2): {gather_cycles_per_vector} cycles/vector")
    print(f"Cycles de gather totales: {gather_cycles_per_vector * n_vectors * rounds}")


def analyze_strategy_A_level_bucket():
    """
    Estrategia A: Level-order bucketing.

    Idea: Procesar por depth en vez de por round.
    En cada depth d, todos los 256 caminos tienen índices en el mismo nivel.
    Agrupar caminos por idx, cargar forest_values[idx] una vez por idx único.

    Problema: Requiere almacenar state de todos los 256 caminos entre depths.
    """
    print("\n" + "=" * 80)
    print("ESTRATEGIA A: Level-order Bucketing")
    print("=" * 80)

    random.seed(123)
    tree = Tree.generate(10)
    inp = Input.generate(tree, 256, 16)
    n_nodes = len(tree.values)

    indices = inp.indices.copy()
    values = inp.values.copy()

    total_unique_loads = 0
    total_broadcast_applies = 0

    # Simulamos 16 rounds, pero procesando por depth
    # Cada "mega-round" de depth 0->10 equivale a un round completo

    for round_num in range(16):
        # En cada round, los caminos van de depth 0 a depth max (hasta wrap)
        current_indices = indices.copy()
        current_values = values.copy()

        # Simular traversal completo del round
        max_steps = 11  # Altura del árbol + 1

        for step in range(max_steps):
            # Agrupar por idx actual
            buckets = defaultdict(list)
            for i in range(256):
                buckets[current_indices[i]].append(i)

            unique_this_step = len(buckets)
            total_unique_loads += unique_this_step

            # Por cada idx único, cargamos una vez y aplicamos a todos sus caminos
            for idx, path_ids in buckets.items():
                total_broadcast_applies += len(path_ids)
                node_val = tree.values[idx]

                for pid in path_ids:
                    val = current_values[pid]
                    val = myhash(val ^ node_val)
                    new_idx = 2 * idx + (1 if val % 2 == 0 else 2)
                    new_idx = 0 if new_idx >= n_nodes else new_idx
                    current_values[pid] = val
                    current_indices[pid] = new_idx

            # Verificar si todos los caminos llegaron a las hojas (wrap)
            if all(idx == 0 for idx in current_indices):
                break

        # Actualizar estado para siguiente round
        indices = current_indices.copy()
        values = current_values.copy()

    print(f"\nResultados:")
    print(f"Total loads únicos (forest_values): {total_unique_loads}")
    print(f"Total applies (broadcast a caminos): {total_broadcast_applies}")
    print(f"Baseline gathers: 4096")
    print(f"Reducción de gathers: {4096 - total_unique_loads} ({(1 - total_unique_loads/4096)*100:.1f}%)")

    print("\nCostos adicionales:")
    print("- Necesita tracking de buckets en scratch (256 * 2 = 512 words para idx+pathid)")
    print("- Necesita loops anidados: outer por bucket, inner por paths en bucket")
    print("- Requiere scatter al final (vstore por path, no vectorizado)")
    print("- Complejidad: ALTA (reescribir todo el kernel)")


def analyze_strategy_B_frontier():
    """
    Estrategia B: Frontier-based traversal.

    Similar a A, pero organizado por nodos en vez de por caminos.
    Mantener conjunto de nodos "activos" con sus caminos asociados.
    """
    print("\n" + "=" * 80)
    print("ESTRATEGIA B: Frontier-based Traversal")
    print("=" * 80)

    # Esencialmente igual que A en términos de gathers.
    # La diferencia es la estructura de datos: lista de (nodo, [caminos])
    # vs buckets por índice.

    print("Similar a Estrategia A en reducción de gathers.")
    print("Diferencia: organización por nodos activos vs por depths.")
    print("Misma reducción esperada: ~80% menos gathers")
    print("\nVentaja: Puede ser más natural para implementar expansión de frontera")
    print("Desventaja: Tracking más complejo del mapping nodo->caminos")


def analyze_strategy_C_path_compaction():
    """
    Estrategia C: Path compaction.

    Reordenar caminos en memoria para maximizar colisiones de idx dentro
    de cada vector de 8. Sin permute vectorial, la compaction requiere
    stores/loads/flow select.
    """
    print("\n" + "=" * 80)
    print("ESTRATEGIA C: Path Compaction")
    print("=" * 80)

    random.seed(123)
    tree = Tree.generate(10)
    inp = Input.generate(tree, 256, 16)
    n_nodes = len(tree.values)

    indices = inp.indices.copy()
    values = inp.values.copy()

    # Simular con y sin compaction
    total_gathers_no_compact = 0
    total_gathers_with_compact = 0

    for round_num in range(16):
        # Sin compaction: cada vector de 8 tiene sus índices
        unique_per_vector = 0
        for v in range(32):  # 256/8 vectores
            vec_idx = indices[v*8:(v+1)*8]
            unique_per_vector += len(set(vec_idx))
        total_gathers_no_compact += unique_per_vector

        # Con compaction perfecta: ordenar índices para minimizar únicos por vector
        # Ideal: agrupar caminos con mismo idx en el mismo vector
        sorted_order = sorted(range(256), key=lambda i: indices[i])
        compacted_indices = [indices[i] for i in sorted_order]

        unique_compacted = 0
        for v in range(32):
            vec_idx = compacted_indices[v*8:(v+1)*8]
            unique_compacted += len(set(vec_idx))
        total_gathers_with_compact += unique_compacted

        # Avanzar round (sin compaction para siguiente medición)
        for i in range(256):
            idx = indices[i]
            val = values[i]
            node_val = tree.values[idx]
            val = myhash(val ^ node_val)
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= n_nodes else idx
            indices[i] = idx
            values[i] = val

    print(f"\nResultados:")
    print(f"Gathers sin compaction: {total_gathers_no_compact}")
    print(f"Gathers con compaction perfecta: {total_gathers_with_compact}")
    print(f"Reducción: {total_gathers_no_compact - total_gathers_with_compact} ({(1 - total_gathers_with_compact/total_gathers_no_compact)*100:.1f}%)")

    print("\nCostos adicionales:")
    print("- Sort/permutación de 256 elementos cada round")
    print("- Sin shuffle vectorial, requiere scatter/gather escalar")
    print("- Overhead de reordenamiento puede superar beneficio")
    print("- Complejidad: MEDIA-ALTA")


def analyze_strategy_D_two_phase():
    """
    Estrategia D: Two-phase traversal.

    Fase 1: idx-only - calcular todos los índices para todos los rounds
    sin cargar valores del árbol. Solo usar el patrón de branching.

    Problema: El branching depende del hash, que depende de val^node_val.
    No podemos calcular idx sin node_val.

    Esta estrategia NO ES VIABLE para este algoritmo específico.
    """
    print("\n" + "=" * 80)
    print("ESTRATEGIA D: Two-phase Traversal")
    print("=" * 80)
    print("\n¡DESCARTADA!")
    print("Razón: El siguiente idx depende de hash(val ^ node_val)")
    print("No podemos precalcular índices sin cargar node_val primero.")
    print("La dependencia es fundamental al algoritmo.")


def analyze_strategy_E_cached_subtrees():
    """
    Estrategia E: Cached subtrees para depths superficiales.

    Observación: En depths 0-5, hay como máximo 63 nodos únicos.
    Podemos pre-cargar los primeros niveles del árbol en scratch.

    Para depth 0-5: 1+2+4+8+16+32 = 63 nodos
    Con SCRATCH_SIZE=1536, podemos almacenar fácilmente estos 63 valores.

    Beneficio: En depths 0-5, NO necesitamos gathers - leemos de scratch.
    Solo necesitamos gathers para depths 6+.
    """
    print("\n" + "=" * 80)
    print("ESTRATEGIA E: Cached Subtrees (depths 0-5 en scratch)")
    print("=" * 80)

    random.seed(123)
    tree = Tree.generate(10)
    inp = Input.generate(tree, 256, 16)
    n_nodes = len(tree.values)

    # Nodos en levels 0-5: 2^6 - 1 = 63
    cached_nodes = 2**6 - 1  # 63

    print(f"Nodos cacheables (depth 0-5): {cached_nodes}")
    print(f"Scratch disponible: 1536 words")
    print(f"Uso de scratch para cache: {cached_nodes} words ({cached_nodes/1536*100:.1f}%)")

    indices = inp.indices.copy()
    values = inp.values.copy()

    gathers_cached = 0
    gathers_uncached = 0

    for round_num in range(16):
        for i in range(256):
            idx = indices[i]

            # ¿Está en cache?
            if idx < cached_nodes:
                gathers_cached += 1  # Lee de scratch, no gather
            else:
                gathers_uncached += 1  # Necesita gather

            val = values[i]
            node_val = tree.values[idx]
            val = myhash(val ^ node_val)
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= n_nodes else idx
            indices[i] = idx
            values[i] = val

    print(f"\nResultados:")
    print(f"Accesos a cache (sin gather): {gathers_cached}")
    print(f"Accesos sin cache (con gather): {gathers_uncached}")
    print(f"Baseline gathers: 4096")
    print(f"Gathers con cache depth 0-5: {gathers_uncached}")
    print(f"Reducción: {4096 - gathers_uncached} ({(1 - gathers_uncached/4096)*100:.1f}%)")

    # Contar por depth para más detalle
    print("\nDistribución de accesos por depth:")
    indices = inp.indices.copy()
    values = inp.values.copy()

    depth_counts = Counter()
    for round_num in range(16):
        for i in range(256):
            idx = indices[i]
            import math
            depth = int(math.log2(idx + 1)) if idx > 0 else 0
            depth_counts[depth] += 1

            val = values[i]
            node_val = tree.values[idx]
            val = myhash(val ^ node_val)
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= n_nodes else idx
            indices[i] = idx
            values[i] = val

    for d in sorted(depth_counts.keys()):
        cached = "CACHED" if d <= 5 else "GATHER"
        print(f"  Depth {d}: {depth_counts[d]} accesos [{cached}]")


def analyze_strategy_F_hybrid_bucket_cache():
    """
    Estrategia F: Híbrido bucket + cache.

    Combinar:
    - Cache de subtree (depth 0-5) para eliminar gathers superficiales
    - Bucketing para depths profundos (6+) donde hay menos colisiones

    Para depths 6+: ~1900 accesos, con bucketing podemos reducir a ~600
    """
    print("\n" + "=" * 80)
    print("ESTRATEGIA F: Híbrido Cache + Bucket")
    print("=" * 80)

    random.seed(123)
    tree = Tree.generate(10)
    inp = Input.generate(tree, 256, 16)
    n_nodes = len(tree.values)

    cached_depth = 5
    cached_nodes = 2**(cached_depth+1) - 1  # 63

    indices = inp.indices.copy()
    values = inp.values.copy()

    gathers_needed = 0  # Solo para depths > cached_depth
    unique_loads_deep = 0  # Con bucketing perfecto

    for round_num in range(16):
        # Track accesos por depth en este round
        deep_accesses = defaultdict(lambda: {'unique': set(), 'total': 0})

        for i in range(256):
            idx = indices[i]
            import math
            depth = int(math.log2(idx + 1)) if idx > 0 else 0

            if depth > cached_depth:
                deep_accesses[depth]['unique'].add(idx)
                deep_accesses[depth]['total'] += 1

            val = values[i]
            node_val = tree.values[idx]
            val = myhash(val ^ node_val)
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= n_nodes else idx
            indices[i] = idx
            values[i] = val

        for d, stats in deep_accesses.items():
            gathers_needed += stats['total']
            unique_loads_deep += len(stats['unique'])

    print(f"\nResultados (cache depth 0-{cached_depth}):")
    print(f"Nodos en cache: {cached_nodes}")
    print(f"Gathers para depths {cached_depth+1}+: {gathers_needed} (sin bucket)")
    print(f"Gathers con bucketing perfecto: {unique_loads_deep}")
    print(f"Baseline: 4096")
    print(f"Con solo cache: {gathers_needed}")
    print(f"Con cache + bucket: {unique_loads_deep}")
    print(f"Reducción total: {4096 - unique_loads_deep} ({(1 - unique_loads_deep/4096)*100:.1f}%)")


def print_strategy_comparison():
    """Tabla comparativa de todas las estrategias."""
    print("\n" + "=" * 100)
    print("TABLA COMPARATIVA DE ESTRATEGIAS")
    print("=" * 100)

    strategies = [
        ("Baseline", 4096, 0, "N/A", "Referencia"),
        ("A: Level Bucket", 839, "~500 stores", "Alta", "Reescritura completa"),
        ("B: Frontier", 839, "~500 stores", "Alta", "Similar a A"),
        ("C: Compaction", 2100, "256 scatter/round", "Media-Alta", "Overhead puede superar beneficio"),
        ("D: Two-phase", "N/A", "N/A", "N/A", "DESCARTADA - dependencia val-idx"),
        ("E: Cache d0-5", 1900, "63 loads iniciales", "Baja", "Solo optimiza depths superficiales"),
        ("F: Cache+Bucket", 600, "63 loads + bucket tracking", "Media", "Mejor ROI"),
    ]

    print(f"\n{'Estrategia':<20} | {'Gathers':>8} | {'Overhead Adicional':<25} | {'Complejidad':<12} | {'Notas':<30}")
    print("-" * 110)
    for name, gathers, overhead, complexity, notes in strategies:
        g = f"{gathers}" if isinstance(gathers, int) else gathers
        print(f"{name:<20} | {g:>8} | {overhead:<25} | {complexity:<12} | {notes:<30}")

    print("\n" + "=" * 100)
    print("RECOMENDACIÓN: Estrategia F (Cache + Bucket) o E (Solo Cache)")
    print("=" * 100)
    print("""
Justificación:
1. Cache de depth 0-5 es BAJO COSTO y elimina ~54% de gathers
2. Bucket para depths profundos añade complejidad pero maximiza reducción
3. Estrategias A/B requieren reescritura completa del kernel
4. Estrategia C tiene overhead alto por falta de permute vectorial
5. Estrategia D no es viable para este algoritmo

Para ~1000 ciclos:
- Cache depth 0-5: reduce gathers de 4096 a ~1900
- Con VLIW packing óptimo, 1900 gathers / 2 = 950 cycles solo de loads
- Necesitamos bucket también para depths 6+ para llegar a ~600 gathers
""")


if __name__ == "__main__":
    simulate_baseline_costs()
    analyze_strategy_A_level_bucket()
    analyze_strategy_B_frontier()
    analyze_strategy_C_path_compaction()
    analyze_strategy_D_two_phase()
    analyze_strategy_E_cached_subtrees()
    analyze_strategy_F_hybrid_bucket_cache()
    print_strategy_comparison()
