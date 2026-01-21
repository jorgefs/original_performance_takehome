# MEMORIA

Resumen de optimizaciones aplicadas hasta ahora en `perf_takehome.py` y su motivacion.

## Optimizations aplicadas

1) Reuso de direcciones de entrada por ronda
   - Cambio: mantener punteros `tmp_idx_addr` y `tmp_val_addr` que avanzan con `+1` (o `+VLEN`) y eliminar el `base + i` en cada iteracion.
   - Motivacion: reducir ALU por iteracion y reutilizar direcciones ya calculadas.

2) Empaquetado VLIW (bundle packing)
   - Cambio: `KernelBuilder.build(..., vliw=True)` agrupa slots por ciclo respetando dependencias RAW y limites por motor.
   - Motivacion: explotar paralelismo VLIW del simulador para reducir ciclos sin cambiar la semantica.

3) Vectorizacion del batch (SIMD)
   - Cambio: usar `vload/valu/vstore` para procesar bloques de `VLEN`, con `load_offset` para el gather del arbol.
   - Motivacion: aprovechar SIMD para operar 8 entradas en paralelo y reducir ciclos por elemento.

4) Mantener valores en scratch entre rondas
   - Cambio: cargar `idx/val` una sola vez, ejecutar todas las rondas en scratch y escribir al final.
   - Motivacion: evitar loads/stores repetidos por ronda.

5) Unrolling de bloques
   - Cambio: unroll de `UNROLL=6` (6 vectores a la vez).
   - Motivacion: llenar los 6 slots de `valu` y mejorar el throughput.

6) Hash optimizado con `multiply_add`
   - Cambio: reemplazar etapas lineales del hash por `valu multiply_add` (a * (1+2^k) + c).
   - Motivacion: reducir instrucciones por etapa del hash.

7) Evitar cargas de `idx` iniciales
   - Cambio: `idx` inicial siempre es 0, se setea con `v_zero`/`zero_const` en lugar de `vload/load`.
   - Motivacion: eliminar loads innecesarios.

8) Especializacion de la ronda 0 y de niveles del arbol
   - Cambio: en profundidad 0 se usa `root_val` broadcast en vez de gather; en hojas se evita actualizar `idx` salvo si es la ultima ronda.
   - Motivacion: eliminar loads y ALU cuando el indice es determinista por nivel.

9) Paridad con bitwise
   - Cambio: `val & 1` + `+1` en vez de `% 2` y comparaciones.
   - Motivacion: menos operaciones ALU y mejor packing.

10) Lectura minima del header
    - Cambio: en modo no-debug solo se cargan offsets de memoria necesarios (forest_values_p, inp_indices_p, inp_values_p) desde indices 4/5/6.
    - Motivacion: reducir instrucciones de setup.

## Ideas para el futuro

- Software pipelining real: solapar `load_offset` del siguiente vector con el hash actual, con reordenamiento por etapas.
- Unroll por etapas del hash (interleaving de varios vectores) para llenar slots de `load` y `valu` en paralelo.
- Ajustar heuristicas del empaquetador VLIW para permitir mas mezcla de `load`/`store` y reducir burbujas.
- Experimentar con distintos valores de `UNROLL` segun el balance real de `load` vs `valu`.
- Cache manual de nodos superiores del arbol en scratch si hay patrones repetitivos (nivel raiz o niveles bajos).
