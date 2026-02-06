"""
Instrumentación para analizar patrones de acceso en tree traversal.
Mide por depth: U_d (únicos), histograma buckets, Uchild_d, % colisión.
"""

import random
from collections import defaultdict, Counter
from problem import Tree, Input, myhash, VLEN

def analyze_traversal(forest_height: int, rounds: int, batch_size: int, seed: int = 123):
    """
    Simula el traversal y recolecta estadísticas por depth.

    En este problema, "depth" se refiere a la profundidad del nodo en el árbol.
    El árbol es binario perfecto con índices:
    - Raíz: idx=0 (depth=0)
    - Hijos de idx: 2*idx+1 (left), 2*idx+2 (right)
    - depth de idx: floor(log2(idx+1))
    """
    random.seed(seed)
    tree = Tree.generate(forest_height)
    inp = Input.generate(tree, batch_size, rounds)

    n_nodes = len(tree.values)

    # Estadísticas por round y por depth del nodo visitado
    stats_by_round = []

    # Copia local de índices y valores
    indices = inp.indices.copy()
    values = inp.values.copy()

    for round_num in range(rounds):
        round_stats = {
            'round': round_num,
            'by_depth': defaultdict(lambda: {
                'unique_idx': set(),
                'idx_counts': Counter(),  # Para histograma de buckets
                'unique_children': set(),
                'accesses': 0
            })
        }

        for i in range(batch_size):
            idx = indices[i]
            val = values[i]

            # Calcular depth del nodo actual
            depth = 0
            if idx > 0:
                import math
                depth = int(math.log2(idx + 1))

            # Registrar estadísticas
            stats = round_stats['by_depth'][depth]
            stats['unique_idx'].add(idx)
            stats['idx_counts'][idx] += 1
            stats['accesses'] += 1

            # Calcular hijos candidatos (antes de saber cuál elegiremos)
            child_left = 2 * idx + 1
            child_right = 2 * idx + 2
            if child_left < n_nodes:
                stats['unique_children'].add(child_left)
            if child_right < n_nodes:
                stats['unique_children'].add(child_right)

            # Ejecutar el paso del algoritmo
            node_val = tree.values[idx]
            val = myhash(val ^ node_val)
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= n_nodes else idx

            indices[i] = idx
            values[i] = val

        stats_by_round.append(round_stats)

    return stats_by_round, tree, indices, values


def print_stats_table(stats_by_round, batch_size: int, max_rounds: int = None):
    """Imprime tabla de estadísticas por round y depth."""

    if max_rounds is None:
        max_rounds = len(stats_by_round)

    print("=" * 100)
    print(f"ANÁLISIS DE TRAVERSAL - batch_size={batch_size}")
    print("=" * 100)

    for round_stats in stats_by_round[:max_rounds]:
        round_num = round_stats['round']
        print(f"\n{'='*50}")
        print(f"ROUND {round_num}")
        print(f"{'='*50}")
        print(f"{'Depth':>6} | {'U_d':>6} | {'%Colision':>10} | {'Uchild':>8} | {'Top-5 Buckets (idx:count)':<40}")
        print("-" * 90)

        for depth in sorted(round_stats['by_depth'].keys()):
            stats = round_stats['by_depth'][depth]
            U_d = len(stats['unique_idx'])
            accesses = stats['accesses']
            collision_pct = (1 - U_d / accesses) * 100 if accesses > 0 else 0
            Uchild = len(stats['unique_children'])

            # Top-5 buckets por tamaño
            top_buckets = stats['idx_counts'].most_common(5)
            bucket_str = ", ".join([f"{idx}:{cnt}" for idx, cnt in top_buckets])

            print(f"{depth:>6} | {U_d:>6} | {collision_pct:>9.1f}% | {Uchild:>8} | {bucket_str:<40}")


def print_aggregate_stats(stats_by_round, batch_size: int):
    """Imprime estadísticas agregadas sobre todos los rounds."""

    print("\n" + "=" * 100)
    print("ESTADÍSTICAS AGREGADAS (todos los rounds)")
    print("=" * 100)

    # Agregar por depth
    agg_by_depth = defaultdict(lambda: {
        'total_unique': 0,
        'total_accesses': 0,
        'rounds_with_depth': 0,
        'max_bucket_size': 0,
        'total_children': 0
    })

    for round_stats in stats_by_round:
        for depth, stats in round_stats['by_depth'].items():
            agg = agg_by_depth[depth]
            agg['total_unique'] += len(stats['unique_idx'])
            agg['total_accesses'] += stats['accesses']
            agg['rounds_with_depth'] += 1
            max_bucket = max(stats['idx_counts'].values()) if stats['idx_counts'] else 0
            agg['max_bucket_size'] = max(agg['max_bucket_size'], max_bucket)
            agg['total_children'] += len(stats['unique_children'])

    print(f"\n{'Depth':>6} | {'Avg U_d':>8} | {'Avg %Coll':>10} | {'Max Bucket':>10} | {'Avg Uchild':>10} | {'Potential Gather Reduction':>25}")
    print("-" * 95)

    total_gathers_baseline = 0
    total_gathers_optimized = 0

    for depth in sorted(agg_by_depth.keys()):
        agg = agg_by_depth[depth]
        n_rounds = agg['rounds_with_depth']
        avg_unique = agg['total_unique'] / n_rounds
        avg_accesses = agg['total_accesses'] / n_rounds
        avg_collision = (1 - avg_unique / avg_accesses) * 100 if avg_accesses > 0 else 0
        avg_children = agg['total_children'] / n_rounds

        # Gather reduction: baseline usa batch_size gathers, optimizado usa U_d gathers
        baseline_gathers = avg_accesses
        optimized_gathers = avg_unique
        reduction = baseline_gathers - optimized_gathers
        reduction_pct = (reduction / baseline_gathers) * 100 if baseline_gathers > 0 else 0

        total_gathers_baseline += agg['total_accesses']
        total_gathers_optimized += agg['total_unique']

        print(f"{depth:>6} | {avg_unique:>8.1f} | {avg_collision:>9.1f}% | {agg['max_bucket_size']:>10} | {avg_children:>10.1f} | {reduction:.0f} ({reduction_pct:.1f}%)")

    print("-" * 95)
    total_reduction = total_gathers_baseline - total_gathers_optimized
    total_reduction_pct = (total_reduction / total_gathers_baseline) * 100
    print(f"{'TOTAL':>6} | {total_gathers_optimized:>8} | {total_reduction_pct:>9.1f}% | {'':>10} | {'':>10} | {total_reduction:.0f} gathers eliminables")

    print(f"\nBaseline gathers totales: {total_gathers_baseline}")
    print(f"Gathers mínimos (con bucketing perfecto): {total_gathers_optimized}")
    print(f"Reducción potencial: {total_reduction_pct:.1f}%")


def analyze_vector_utilization(stats_by_round, batch_size: int):
    """Analiza cuántas colisiones hay dentro de vectores de VLEN=8."""

    print("\n" + "=" * 100)
    print("ANÁLISIS DE UTILIZACIÓN VECTORIAL (VLEN=8)")
    print("=" * 100)

    # Simular cómo quedarían los índices si los procesamos en vectores de 8
    random.seed(123)
    tree = Tree.generate(10)  # forest_height=10
    inp = Input.generate(tree, batch_size, 16)
    n_nodes = len(tree.values)

    indices = inp.indices.copy()
    values = inp.values.copy()

    print(f"\n{'Round':>6} | {'Vectors con 8 únicos':>20} | {'Vectors con <8 únicos':>22} | {'Avg únicos/vector':>18}")
    print("-" * 80)

    for round_num in range(16):
        vectors_all_unique = 0
        vectors_with_collisions = 0
        total_unique_in_vectors = 0
        n_vectors = batch_size // VLEN

        for v in range(n_vectors):
            vec_indices = indices[v*VLEN:(v+1)*VLEN]
            unique_in_vec = len(set(vec_indices))
            total_unique_in_vectors += unique_in_vec
            if unique_in_vec == VLEN:
                vectors_all_unique += 1
            else:
                vectors_with_collisions += 1

        avg_unique = total_unique_in_vectors / n_vectors
        print(f"{round_num:>6} | {vectors_all_unique:>20} | {vectors_with_collisions:>22} | {avg_unique:>18.2f}")

        # Avanzar un round
        for i in range(batch_size):
            idx = indices[i]
            val = values[i]
            node_val = tree.values[idx]
            val = myhash(val ^ node_val)
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= n_nodes else idx
            indices[i] = idx
            values[i] = val


def estimate_load_offset_counts():
    """Estima número de load_offset en diferentes escenarios."""

    print("\n" + "=" * 100)
    print("ESTIMACIÓN DE LOAD_OFFSET POR ESTRATEGIA")
    print("=" * 100)

    # Baseline: 256 batch * 16 rounds = 4096 accesos
    # Cada acceso es 1 gather (load_offset por lane o load escalar)

    # Con vectorización actual (sin bucketing):
    # 256/8 = 32 vectores * 16 rounds = 512 vloads para idx, 512 para val
    # Pero forest_values[idx] requiere gather: 32 vectores * 8 lanes * 16 rounds = 4096 gathers

    baseline_gathers = 256 * 16

    print(f"\nBaseline (escalar): {baseline_gathers} gathers totales")
    print(f"Vectorizado sin optimizar: {baseline_gathers} gathers (1 load_offset por lane)")

    # Analizar reducción real con diferentes forest_heights
    for fh in [8, 9, 10]:
        random.seed(123)
        tree = Tree.generate(fh)
        inp = Input.generate(tree, 256, 16)
        n_nodes = len(tree.values)

        indices = inp.indices.copy()
        values = inp.values.copy()

        total_unique = 0
        for round_num in range(16):
            unique_this_round = len(set(indices))
            total_unique += unique_this_round

            for i in range(256):
                idx = indices[i]
                val = values[i]
                node_val = tree.values[idx]
                val = myhash(val ^ node_val)
                idx = 2 * idx + (1 if val % 2 == 0 else 2)
                idx = 0 if idx >= n_nodes else idx
                indices[i] = idx
                values[i] = val

        reduction = baseline_gathers - total_unique
        reduction_pct = (reduction / baseline_gathers) * 100
        print(f"\nforest_height={fh} (n_nodes={n_nodes}):")
        print(f"  Gathers con bucketing perfecto: {total_unique}")
        print(f"  Reducción: {reduction} ({reduction_pct:.1f}%)")


if __name__ == "__main__":
    # Configuración del benchmark real
    configs = [
        (8, 16, 256),   # forest_height=8
        (9, 16, 256),   # forest_height=9
        (10, 16, 256),  # forest_height=10 (benchmark oficial)
    ]

    for forest_height, rounds, batch_size in configs:
        print("\n" + "#" * 100)
        print(f"# CONFIGURACIÓN: forest_height={forest_height}, rounds={rounds}, batch_size={batch_size}")
        print("#" * 100)

        stats, tree, final_indices, final_values = analyze_traversal(
            forest_height, rounds, batch_size
        )

        # Mostrar solo primeros 4 rounds en detalle
        print_stats_table(stats, batch_size, max_rounds=4)

        # Estadísticas agregadas
        print_aggregate_stats(stats, batch_size)

    # Análisis de utilización vectorial
    analyze_vector_utilization(None, 256)

    # Estimaciones de load_offset
    estimate_load_offset_counts()
