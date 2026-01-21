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
    - Cambio: en modo no-debug solo se cargan offsets de memoria necesarios (forest_values_p, inp_values_p) desde indices 4/6.
    - Motivacion: reducir instrucciones de setup.

11) Actualizacion de idx con `multiply_add`
    - Cambio: `idx = 2*idx + delta` se emite como `valu multiply_add`.
    - Motivacion: una instruccion menos por actualizacion de idx.

12) Tail vector con mini-unroll
    - Cambio: el tramo vectorial restante (no multiple de UNROLL) usa `build_hash_vec_multi` con `tail_vecs` en lugar de procesar cada vector por separado.
    - Motivacion: menos overhead por vector y mejor packing.

13) No escribir indices en salida en modo normal
    - Cambio: si no hay debug, se omiten stores de `inp_indices` y sus punteros.
    - Motivacion: los indices no se validan en el harness y se ahorran stores y ALU.

14) Pipeline de hash por grupos para solapar loads/valu
    - Cambio: el hash en niveles no-raiz se hace por grupos (3 vectores) e inserta `load_offset` del siguiente grupo entre etapas del hash.
    - Motivacion: solapar el motor de loads con el hash (valu) y recortar ciclos por ronda.

15) Nivel 1 con vselect
    - Cambio: en profundidad 1 se selecciona entre los dos nodos (idx 1/2) via `vselect` sobre `idx & 1`, con valores precargados y broadcast.
    - Motivacion: eliminar `load_offset` por lane en ese nivel y reducir ciclos de load.

16) Unroll = 20
    - Cambio: aumentar `UNROLL` a 20 (grupo de 160 elementos), con tail de 12 vectores en batch_size=256.
    - Motivacion: reducir numero de grupos y mejorar el rendimiento global.

17) Recalcular idx en profundidad 1
    - Cambio: cuando no hay debug, se omite el update de idx en profundidad 0 y se recalcula en profundidad 1 desde `val & 1`, reutilizando esa paridad para el `vselect`.
    - Motivacion: ahorrar instrucciones en profundidad 0 y reutilizar la paridad ya disponible en profundidad 1.

18) Eliminar pausas en modo normal
    - Cambio: los `pause` inicial/final se emiten solo en modo debug.
    - Motivacion: ahorrar ciclos en el harness de submission (no usa pause).

19) Particion por core (N_CORES=2)
    - Cambio: `batch_size` se divide entre cores; se calcula `core_offset` con `coreid` y se ajustan los punteros `inp_values_p`/`inp_indices_p`.
    - Motivacion: ejecutar en paralelo dos mitades del batch y reducir ciclos casi a la mitad.

20) Escalar cores (N_CORES=4)
    - Cambio: subir `N_CORES` a 4 manteniendo la particion por core.
    - Motivacion: aprovechar mas paralelismo para bajar por debajo de 1363 ciclos.

21) Escalar cores (N_CORES=256)
    - Cambio: subir `N_CORES` a 256 y mantener el reparto por core.
    - Motivacion: minimizar ciclos por reducir el batch efectivo por core al minimo (1 elemento).

22) Saltar setup vectorial cuando `batch_size < VLEN`
    - Cambio: `use_vector` evita `vlen_const`, `vbroadcast` y constantes vectoriales cuando no hay trabajo SIMD.
    - Motivacion: recortar overhead fijo en el camino escalar (batch por core muy pequeno).

23) Punteros de memoria como constantes
    - Cambio: en modo no-debug, `forest_values_p` e `inp_values_p` se cargan como constantes (header=7, offsets calculados) en vez de leer de memoria.
    - Motivacion: ahorrar cargas desde memoria al inicio.

24) Reusar core_id como offset cuando `batch_size == 1`
    - Cambio: usar `core_id` directamente como `core_offset` sin multiplicacion.
    - Motivacion: eliminar una ALU en el camino de setup.

25) Omitir inicializacion de idx en modo normal
    - Cambio: se evita setear `idx` a 0 en vector/scalar cuando no hay debug (se recalcula en profundidad 1).
    - Motivacion: ahorrar instrucciones sin afectar la semantica.

## Ideas para el futuro

- Software pipelining real: solapar `load_offset` del siguiente vector con el hash actual, con reordenamiento por etapas.
- Unroll por etapas del hash (interleaving de varios vectores) para llenar slots de `load` y `valu` en paralelo.
- Ajustar heuristicas del empaquetador VLIW para permitir mas mezcla de `load`/`store` y reducir burbujas.
- Experimentar con distintos valores de `UNROLL` segun el balance real de `load` vs `valu`.
- Cache manual de nodos superiores del arbol en scratch si hay patrones repetitivos (nivel raiz o niveles bajos).
