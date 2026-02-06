"""
Analizar el verdadero bottleneck: desglose de operaciones por tipo.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Contar operaciones emitidas
# Insertar código de conteo antes de build()

count_code = '''
# Conteo de operaciones antes de build
from collections import Counter
op_counts = Counter()
for engine, slot in body:
    op_type = f"{engine}:{slot[0]}"
    op_counts[op_type] += 1

print("\\n=== OPERACIONES POR TIPO ===")
for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
    print(f"{op}: {count}")
print(f"\\nTotal ops: {sum(op_counts.values())}")

# Desglose por depth
print("\\n=== DESGLOSE POR PROFUNDIDAD ===")
'''

# No modifiquemos el archivo, hagamos un análisis estático
print("Análisis estático de operaciones:")
print()
print("Por round (16 rounds, forest_height=10):")
print("Round depths: [0,1,2,3,4,5,6,7,8,9,10,0,1,2,3,4]")
print()

VLEN = 8
VEC_COUNT = 32

# Operaciones por depth
for depth in range(11):
    if depth == 0:
        # XOR + hash + idx update simple
        valu = VEC_COUNT * (1 + 12 + 2)  # XOR, 12 hash ops, 2 idx ops
        loads = 0
        flows = 0
        load_offset = 0
    elif depth == 1:
        # vselect + XOR + hash + idx update
        valu = VEC_COUNT * (3 + 12 + 3)  # bit extract, XOR, hash, idx
        loads = 0
        flows = VEC_COUNT  # 1 vselect per vec
        load_offset = 0
    elif depth == 2:
        # 3 vselects + XOR + hash + idx update
        valu = VEC_COUNT * (3 + 12 + 3)  # bits, XOR, hash, idx
        loads = 0
        flows = VEC_COUNT * 3
        load_offset = 0
    else:
        # load_offset + XOR + hash + idx update (if not leaf)
        valu = VEC_COUNT * (1 + 12 + (3 if depth < 10 else 0))
        loads = 0
        flows = 0
        load_offset = VEC_COUNT * VLEN

    # Cuántas veces aparece este depth en 16 rounds
    depths = [d % 11 for d in range(16)]
    count = depths.count(depth)

    print(f"Depth {depth:2d}: appears {count}x, per round: {valu:4d} valu, {load_offset:4d} load_offset, {flows:3d} flow")
    print(f"          total: {valu*count:5d} valu, {load_offset*count:5d} load_offset, {flows*count:4d} flow")
    print()

# Totales
total_valu = 0
total_load_offset = 0
total_flows = 0

for depth in range(11):
    depths = [d % 11 for d in range(16)]
    count = depths.count(depth)

    if depth == 0:
        valu = VEC_COUNT * 15
    elif depth <= 2:
        valu = VEC_COUNT * 18
    else:
        valu = VEC_COUNT * (13 + (3 if depth < 10 else 0))

    load_offset = VEC_COUNT * VLEN if depth > 2 else 0
    flows = VEC_COUNT * (1 if depth == 1 else 3 if depth == 2 else 0)

    total_valu += valu * count
    total_load_offset += load_offset * count
    total_flows += flows * count

print("=" * 50)
print(f"TOTALES: {total_valu} valu, {total_load_offset} load_offset, {total_flows} flow")
print()
print(f"Min VALU cycles (6/cycle): {total_valu / 6:.0f}")
print(f"Min load_offset cycles (2/cycle): {total_load_offset / 2:.0f}")
print(f"Min flow cycles (1/cycle): {total_flows}")
print()
print("Critical path analysis:")
print(f"  If VALU-bound: {total_valu / 6:.0f} cycles")
print(f"  If load-bound: {total_load_offset / 2:.0f} cycles (but loads overlap with VALU)")
print(f"  If flow-bound: {total_flows} cycles (but flow overlaps with VALU)")
print()
print("Current actual: 1615 cycles")
print("Overhead from theoretical: 1615 - 1339 = 276 cycles")
print()
print("Key insight: load_offset has LATENCY, not just throughput.")
print("Each load_offset takes multiple cycles to return data.")
print("The hash depends on load results, creating stalls.")
