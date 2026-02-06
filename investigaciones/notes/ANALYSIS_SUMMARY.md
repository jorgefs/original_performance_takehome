# Análisis de Optimización VLIW/SIMD

## Estado Actual
- **Ciclos actuales**: 1615
- **Target próximo**: 1579 (opus45_2hr)
- **Target ambicioso**: 1363 (opus45_improved_harness)
- **Mínimo teórico VALU**: 1381 cycles (8288 ops / 6 slots)

## Desglose de Ciclos (body = 1598, init = 16)

### Operaciones por Tipo
- **VALU**: 8288 ops → 1381 cycles mínimo
  - Hash: 6144 ops (12 ops × 512 hashes)
  - Idx update: 1344 ops (3 ops × 448 updates)
  - XOR + misc: 800 ops
- **Load_offset**: 2560 ops → 1280 cycles @ 2/cycle (overlap con VALU)
- **Flow (vselect)**: 256 ops → 256 cycles @ 1/cycle (parcial overlap)

## Instrumentación de Bucketing (instrument.py)

| Depth | U_d (únicos) | Uchild_d | Max Bucket | Colisión |
|-------|--------------|----------|------------|----------|
| 0     | 1            | 2        | 256        | 99.6%    |
| 1     | 2            | 4        | 138        | 99.2%    |
| 2     | 4            | 8        | 75         | 98.4%    |
| 3     | 8            | 16       | 44         | 96.9%    |
| 4     | 16           | 32       | 24         | 93.8%    |
| 5     | 32           | 64       | 13         | 87.5%    |
| 6     | 63           | 126      | 8          | 75.4%    |
| 7     | 108          | 216      | 6          | 57.8%    |
| 8     | 159          | 318      | 6          | 37.9%    |
| 9     | 191          | 382      | 4          | 25.4%    |
| 10    | 224          | 448      | 2          | 12.5%    |

**Conclusión**: Depths 3-6 tienen pocos índices únicos → alto potencial de ahorro.

## Enfoques Probados

### 1. Vselect Tree para Depth 3
- **Resultado**: 1788 cycles (PEOR)
- **Problema**: 7 vselects/vector × 32 vectors = 224 flow ops
- Flow tiene 1 slot/cycle, más costoso que load_offset (2/cycle)

### 2. Diferentes hash_group
- Probado: 1, 2, 3, 4, 6, 8, 16, 32
- **Resultado**: Todos dan 1615 cycles
- El scheduler VLIW ya optimiza el interleaving

### 3. Diferentes ordenamientos de starts
- Sequential, reversed, original interleaved
- **Resultado**: Original es mejor (1615 vs 1619)

### 4. Chunk > 1
- chunk=2: 1621 cycles (peor)
- chunk=4: 1623 cycles (peor)

### 5. Reordenamiento de idx update
- Mover multiply_add antes del hash
- **Resultado**: 1615 cycles (igual)
- El scheduler ya maneja esto

### 6. Eliminar interleaving de loads
- **Resultado**: 1615 cycles (igual)
- El scheduler compensa

## Análisis del Cuello de Botella

Según análisis previo (analyze_dependencies.py):
- 76% de ciclos VALU=4 bloqueados por **hash RAW chain**
- 24% bloqueados por **load latency**
- 0% por flow/vselect

**Conclusión**: El hash RAW chain es el limitante fundamental, no puede romperse sin cambiar el algoritmo.

## Enfoques No Viables

1. **Bucketing con vselect**: Flow (1/cycle) es más costoso que load_offset (2/cycle)
2. **Scatter/gather en scratch**: No hay instrucciones disponibles
3. **Especulación de hijos**: Duplica loads sin beneficio en throughput
4. **Procesamiento híbrido ALU/VALU**: El usuario lo prohibió

## Análisis de Trade-off Flow vs Load

**Hallazgo clave**: vselect (flow) a 1 slot/cycle SIEMPRE es peor que load_offset a 2 slots/cycle para software gather.

Para depth 3 (8 valores únicos):
- Original: 256 load_offsets = 128 cycles de issue
- vselect tree: 7 vselects × 32 vecs × 2 rounds = 448 cycles de flow
- **Resultado**: +192 cycles (PEOR)

## Bloqueo Fundamental

El hash RAW chain (76% del overhead) es ineludible porque:
1. Cada stage del hash depende del anterior
2. No hay trabajo independiente suficiente para llenar los slots vacíos
3. Diferentes vectores ya están interleaved al máximo

## Posibles Direcciones Futuras (No Probadas)

1. **Reorganización de paths**: Ordenar paths por idx predicho antes de procesamiento
   - Costo: scatter/gather inicial costoso
   - Beneficio: loads contiguos en depths bajos

2. **Procesamiento por buckets literal**:
   - Para cada idx único, procesar todos los paths con ese idx
   - Requiere máscaras de selección precalculadas

3. **Explorar el simulador**:
   - ¿Hay instrucciones o características no documentadas?
   - ¿Hay latencia de load que podamos medir/explotar?

4. **Reducción de rounds/depths**:
   - ¿El algoritmo permite fusionar operaciones entre rounds consecutivos?

## Estado Actual

- **Baseline**: 1615 cycles
- **Mejor intento**: 1615 cycles (varias variantes probadas)
- **Peor intento**: 1788 cycles (depth 3 vselect tree)
- **Gap a target**: 252 cycles (1615 - 1363)

## Archivos Relevantes

- `instrument.py`: Análisis de bucketing por depth
- `perf_takehome_d3full.py`: Intento de vselect tree (1788 cycles)
- `analyze_bottleneck.py`: Análisis de operaciones
- `perf_takehome.py`: Baseline actual (1615 cycles)
