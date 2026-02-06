# Kernel Optimization Master Synthesis (2026-02-06)

## 1) Estado real verificado hoy

- Archivo activo: `perf_takehome.py`
- Ciclos actuales (medido): `1397`
- Comando: `python perf_takehome.py Tests.test_kernel_cycles`
- Submission tests: `8/9` pasan; falla solo `test_opus45_improved_harness` (`<1363`).
- Comando: `python tests/submission_tests.py`

Observaciones de codigo (estado actual):

- Prebuilt por defecto desactivado: `USE_PREBUILT=0`.
- Scheduler por defecto mas agresivo que antes:
  - `SCHED_SEARCH=8`
  - `HASH_BIAS=1`
- `idx_update` mixto por ronda habilitado por defecto:
  - `IDX_UPDATE_ARITH=0`
  - `IDX_UPDATE_ARITH_ROUNDS=5`
- Fast path guarda indices (`store_indices=True`) para cumplir submission correctness.

Referencias:
- `perf_takehome.py:903`
- `perf_takehome.py:926`
- `perf_takehome.py:930`
- `perf_takehome.py:1147`

## 2) Contraste con el resumen de la sesion cortada

Tu bloque pegado es consistente con el estado actual en lo esencial:

- Coincide que el modo aritmetico global de idx_update empeora.
- Coincide que modo aritmetico por ronda (`round 5`) mejora.
- Coincide que el baseline activo quedo en `1397`.
- Coincide que la frontera `<1363` sigue lejos.

Pequenas diferencias con notas antiguas:

- Varias notas en `investigaciones/notes/*.md` siguen ancladas en la etapa `1615`.
- `SUMMARY.md` contiene referencias antiguas a `FAST_SCHED`; el codigo actual ya no usa ese flag como control principal.

## 3) Cronologia sintetica (hitos confirmados)

- ~1883: baseline inicial de la rama de experimentacion.
- 1803 -> 1782 -> 1766 -> 1752 -> 1739 -> 1728: mejoras por simplificacion de early depths + idx path.
- 1723 -> 1708 -> 1699 -> 1697 -> 1694: ganancias por orden/chunking/interleaving.
- 1399: salto grande (candidato v15 / tuning scheduler + forma final fast path).
- 1397: mejora final estable por combinacion de scheduler (`HASH_BIAS=1`) + idx_update aritmetico selectivo en ronda 5.

Fuente principal: `investigaciones/logs/EXPERIMENT_LOG.md`.

## 4) Inventario de lineas probadas, por que fallaron y resquicio restante

Nota: `EXPERIMENT_LOG.md` contiene el detalle fino por intento individual. Aqui se sintetiza "todas las familias" de pasos probados, con su patron de fallo y posible ultimo resquicio.

### A. Reordenacion de starts/chunks/unroll en fast path

- Probado: decenas de variantes de orden (secuencial, reverse, even-odd, bitrev, mod4, round-parity, etc), chunk sizes (1/2/3/4/8/16), unroll alternativo.
- Resultado dominante: neutro o peor; algunos incorrectos por cobertura incompleta o aliasing/indices fuera de rango.
- Causa raiz:
  - El scheduler ya extrae casi todo el ILP disponible para esa estructura.
  - Cambios de orden suelen romper overlap load/hash o introducen dependencias mas duras.
- Resquicio:
  - Busqueda automatizada estructurada (auto-tuner) solo para orden por profundidad/ronda bajo constraints de correccion, con pruning fuerte.

### B. hash_group, pipeline shape, load interleave

- Probado: hash_group 1..32, variantes por profundidad/ronda/paridad, pipeline simple/no simple, bulk vs interleave.
- Resultado dominante: casi siempre neutro; algunas peores.
- Causa raiz:
  - Techo impuesto por RAW chain del hash.
  - Ajustar solo granularidad de emision no reduce dependencias fundamentales.
- Resquicio:
  - Superopt de micro-scheduling solo en ventanas conflictivas (round/depth especificos), no global.

### C. idx_update reformulations (flow vs valu)

- Probado:
  - Reorders de operaciones.
  - split en fases.
  - variantes muladd + preadd.
  - conversion completa a forma aritmetica.
  - forma aritmetica selectiva por ronda.
- Resultado:
  - Global aritmetico: peor.
  - Selectivo por ronda: mejora real en `round 5` (base actual 1397).
- Causa raiz:
  - Tradeoff FLOW vs VALU depende del punto exacto de presion por ronda.
  - Beneficio local existe, global no.
- Resquicio:
  - Auto-busqueda de subset optimo de rondas para modo aritmetico (espacio 2^16, pero reducible por heuristica).

### D. Caches tempranas (depth 0/1/2/3), vselect trees, diffs

- Probado:
  - Cache L1/L2/L3 por broadcast/vselect.
  - variantes con diffs/muladd para sustituir flow.
  - depth3 hot caches parciales.
- Resultado:
  - L1/L2 bien en la linea final (parte del exito historico).
  - L3 y superiores: generalmente peor o incorrecto.
- Causa raiz:
  - FLOW slot (1/cycle) se convierte en cuello si aumenta vselect tree.
  - Presion de scratch/temps y aliasing del scheduler.
- Resquicio:
  - Replantear L3/L4 con representacion compacta y seleccion hibrida muy acotada (solo subcasos de alta colision).

### E. Bucketing/frontier/scatter-unscatter ASM-first

- Probado:
  - Materializacion por buckets con loops escalares, prefix sum, scatter/unscatter, procesamiento por bucket.
- Resultado:
  - Empeoramiento severo (kernel mucho mas largo y lento).
- Causa raiz:
  - Sobrecoste escalar/FLOW/permuta domina ahorro de gathers.
  - Baja densidad de VLIW packing en loops raw.
- Resquicio:
  - Solo seria viable con bucketing vectorizado de verdad (sin scatter/unscatter completo por depth), o manteniendo orden bucket varias rondas antes de restaurar.

### F. Depth6-focused strategies (chunked vselect, frontier K)

- Probado:
  - Atacar explicitamente el "quiebre" en depth 6 con chunking/select.
- Resultado:
  - Correctness fragil bajo VLIW (aliasing/interleaving), o sin mejora neta.
- Causa raiz:
  - Presion de registros + reorder agresivo del scheduler en bloques con temps reutilizados.
- Resquicio:
  - Version correcta requeriria aislamiento fuerte de temporales por bloque y/o menor unroll efectivo local.

### G. speculative/prefetch/predictive ideas

- Probado:
  - Prefetch de lanes, speculation de hijos, caches predictivos.
- Resultado:
  - Neutro o peor; en varios casos incorrecto.
- Causa raiz:
  - Carga extra supera ahorro real.
  - Distribucion estadistica no sostiene reuse suficiente en profundidades altas.
- Resquicio:
  - Solo potencial en estrategia offline/especializada al bosque fijo (si se permite precompute fuerte).

### H. Cambios de seguridad/correccion que parecian "gratis"

- Probado:
  - no guardar indices, saltar idx update final, reorder stores, etc.
- Resultado:
  - Algun ahorro aparente, pero rompe correctness contractual o empeora al validar completo.
- Causa raiz:
  - submission exige semantica completa, no solo values.
- Resquicio:
  - Ninguno util para version de entrega salvo pruebas internas.

## 5) Sobre la hipotesis "el gran problema es solo memoria"

Lo que confirman los artefactos actuales:

- Es una mezcla: memoria dispersa + dependencia RAW del hash + slot FLOW limitado.
- No basta optimizar loads de forma aislada.
- Los mejores saltos vinieron de:
  - reducir gathers en early depths,
  - mejorar empaquetado/scheduling,
  - ajustar idx_update en puntos locales de presion.

## 6) Lista corta de "resquicios reales" aun abordables

1. Auto-tuner de `IDX_UPDATE_ARITH_ROUNDS` (busqueda guiada por rondas candidatas) para intentar `1396`/`1395` sin cambiar semantica.
2. Auto-tuner conjunto pequeno: `{ORDER_VARIANT, HASH_GROUP, SCHED_SEARCH, HASH_BIAS}` + `idx rounds`, con validacion de correccion.
3. Micro-opt focal en depth de mayor presion (5/6) con temporales dedicados por bloque para evitar aliasing VLIW.
4. Si se permite costo de build mayor: explorar scheduler random/local-search mas profundo, conservando misma semantica.

## 7) Archivos fuente de verdad para seguimiento

- Estado de codigo: `perf_takehome.py`
- Historial detallado de intentos: `investigaciones/logs/EXPERIMENT_LOG.md`
- Resumenes previos: `SUMMARY.md`, `REPORT_ANALYSIS.md`
- Notas de etapa 1615 (historicas): `investigaciones/notes/SESSION_STATE.md`, `investigaciones/notes/OPTIMIZATION_PLAN.md`, `investigaciones/notes/FRONTIER_STATUS.md`

## 8) Conclusiones operativas

- El resumen pegado de la sesion cortada es util y, en lo esencial, correcto para el estado actual.
- La fotografia actual consistente es: `1397`, correcto funcionalmente, frontera `<1363` aun no alcanzada.
- El siguiente paso con mejor retorno esperado y menor riesgo es la busqueda automatica de combinaciones por ronda para `idx_update` (sobre el baseline actual), no reabrir bucketing escalar.
