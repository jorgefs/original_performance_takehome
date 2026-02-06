"""
FASE 0 - Instrumentación

Mide por depth d:
1) U_d = número de idx únicos en los 256 caminos
2) Uchild_d = número de índices únicos considerando ambos hijos (2*idx+1, 2*idx+2)
3) Histograma del tamaño de buckets
4) Porcentaje de colisiones: 1 - U_d/256
"""

import random
from collections import Counter
from problem import Tree, Input, HASH_STAGES

def myhash(val):
    """Replica exacta de la función hash del problema."""
    for op1, val1, op2, op3, val3 in HASH_STAGES:
        if op1 == "+":
            tmp1 = (val + val1) % (2**32)
        else:  # op1 == "^"
            tmp1 = val ^ val1

        if op3 == "<<":
            tmp2 = (val << val3) % (2**32)
        else:  # op3 == ">>"
            tmp2 = val >> val3

        if op2 == "+":
            val = (tmp1 + tmp2) % (2**32)
        else:  # op2 == "^"
            val = tmp1 ^ tmp2
    return val

def simulate_paths(forest_height, rounds, batch_size, seed=123):
    """
    Simula los caminos de todos los batch elements a través del árbol.
    Retorna: lista de diccionarios, uno por round, con info por depth.
    """
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)

    # Estado inicial: idx=0, val=inp.values[i] para cada camino
    current_idx = [0] * batch_size
    current_val = list(inp.values)

    results = []
    period = forest_height + 1

    for r in range(rounds):
        depth = r % period

        # Collect idx values at this depth (before update)
        idx_at_depth = list(current_idx)

        # Get node values for current indices
        node_vals = [forest.values[idx] for idx in current_idx]

        # XOR and hash
        for i in range(batch_size):
            current_val[i] = myhash(current_val[i] ^ node_vals[i])

        # Update idx based on hash result (bit 0)
        new_idx = []
        for i in range(batch_size):
            bit = current_val[i] & 1
            if depth == forest_height:
                # Wrap to root
                new_idx.append(0)
            else:
                # idx = 2*idx + 1 + bit
                new_idx.append(2 * current_idx[i] + 1 + bit)

        # Analyze this depth
        idx_counter = Counter(idx_at_depth)
        U_d = len(idx_counter)  # Unique indices

        # Children indices: for each unique idx, compute 2*idx+1 and 2*idx+2
        child_indices = set()
        for idx in idx_counter.keys():
            child_indices.add(2 * idx + 1)
            child_indices.add(2 * idx + 2)
        Uchild_d = len(child_indices)

        # Bucket histogram
        bucket_sizes = list(idx_counter.values())
        max_bucket = max(bucket_sizes) if bucket_sizes else 0

        # Top-5 buckets by size
        top_buckets = idx_counter.most_common(5)

        # Collision percentage
        collision_pct = 1.0 - U_d / batch_size

        results.append({
            'round': r,
            'depth': depth,
            'U_d': U_d,
            'Uchild_d': Uchild_d,
            'max_bucket': max_bucket,
            'top_buckets': top_buckets,
            'collision_pct': collision_pct,
            'bucket_histogram': Counter(bucket_sizes),
        })

        current_idx = new_idx

    return results

def main():
    # Parámetros del benchmark real
    forest_height = 10
    rounds = 16
    batch_size = 256

    print(f"Instrumentación: forest_height={forest_height}, rounds={rounds}, batch_size={batch_size}")
    print("=" * 80)

    results = simulate_paths(forest_height, rounds, batch_size)

    # Agrupar por depth
    by_depth = {}
    for r in results:
        d = r['depth']
        if d not in by_depth:
            by_depth[d] = []
        by_depth[d].append(r)

    print("\nPor depth (promediando sobre todas las apariciones en rounds):")
    print("-" * 80)

    for d in sorted(by_depth.keys()):
        rounds_at_d = by_depth[d]

        # Promedios
        avg_U = sum(r['U_d'] for r in rounds_at_d) / len(rounds_at_d)
        avg_Uchild = sum(r['Uchild_d'] for r in rounds_at_d) / len(rounds_at_d)
        avg_max_bucket = sum(r['max_bucket'] for r in rounds_at_d) / len(rounds_at_d)
        avg_collision = sum(r['collision_pct'] for r in rounds_at_d) / len(rounds_at_d)

        # Top buckets del primer round en este depth (representativo)
        top = rounds_at_d[0]['top_buckets']

        print(f"depth {d:2d}: U_d={avg_U:6.1f}, Uchild_d={avg_Uchild:6.1f}, "
              f"max_bucket={avg_max_bucket:5.1f}, collision={avg_collision*100:5.1f}%")
        print(f"          top={([(idx, cnt) for idx, cnt in top[:5]])}")

        # Bucket histogram para el primer round
        hist = rounds_at_d[0]['bucket_histogram']
        hist_str = ", ".join(f"size={k}:count={v}" for k, v in sorted(hist.items())[:10])
        print(f"          bucket_hist: {hist_str}")
        print()

    print("\n" + "=" * 80)
    print("RESUMEN PARA DECISIONES:")
    print("-" * 80)

    # Análisis de viabilidad de bucketing
    print("\nBucketing recomendado para depths con U_d << 256:")
    for d in sorted(by_depth.keys()):
        rounds_at_d = by_depth[d]
        avg_U = sum(r['U_d'] for r in rounds_at_d) / len(rounds_at_d)
        if avg_U < 64:  # Significativamente menos que 256
            savings = 256 - avg_U
            print(f"  depth {d}: U_d={avg_U:.0f} -> ahorra ~{savings:.0f} gathers")

    print("\nEspeculación de hijos recomendada donde Uchild_d << 2*U_d:")
    for d in sorted(by_depth.keys()):
        rounds_at_d = by_depth[d]
        avg_U = sum(r['U_d'] for r in rounds_at_d) / len(rounds_at_d)
        avg_Uchild = sum(r['Uchild_d'] for r in rounds_at_d) / len(rounds_at_d)
        expected_children = 2 * avg_U
        if expected_children > 0 and avg_Uchild < expected_children * 0.9:
            savings = expected_children - avg_Uchild
            print(f"  depth {d}: Uchild_d={avg_Uchild:.0f} vs 2*U_d={expected_children:.0f} "
                  f"-> {savings:.0f} hijos compartidos")

if __name__ == "__main__":
    main()
