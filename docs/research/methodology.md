# Protocolo de evaluación

Reglas comunes a todos los experimentos desde la Fase 0. Un resultado que no las cumpla no es
comparable con el registro y no debe usarse para decidir.

## 1. Datos y tiempo

- **Unidad:** velas de 4h de BTCUSDT spot (Binance). La fila `t` describe la vela que abre en
  `open_time` y cierra en `open_time + 4h`; la **decisión** se toma al cierre.
- **Etiqueta a h velas:** `y_h[t] = 1` si `close[t+h] > close[t]`; se conoce en
  `open_time + (h+1)·4h`. La etiqueta histórica del proyecto es `h = 1`
  ([labels.py](../../ml/src/features/labels.py)).
- **Disponibilidad:** toda feature se alinea al instante de decisión con `merge_asof` hacia
  atrás y una antigüedad máxima. Los retrasos por fuente están en
  [data-sources.md](./data-sources.md). Cada familia nueva requiere un test que altere los
  datos posteriores a T y verifique que las features ≤ T no cambian (ver
  `test_long_horizon_features_never_use_unavailable_data`).
- **Dataset congelado:** `btcusdt-20200901-20261001-v1` (13,126 filas, 2020-10-04 a
  2026-09-30) con manifiesto y huellas. Las fuentes nuevas van en un snapshot auxiliar
  inmutable (`ml/artifacts/aux-sources/<id>/`), nunca dentro del dataset.
- **Corte de desarrollo:** `2026-10-01T00:00Z`. Ningún experimento de desarrollo usa datos
  posteriores.

## 2. Particiones

- **Estimación causal (única válida para decidir):** walk-forward mensual expansivo de **48
  folds** (sep-2022 a ago-2026, 8,766 predicciones OOS). El train de cada fold solo contiene
  filas con `open_time + (h+1)·4h <= inicio del fold` (purga por horizonte) y al menos 365
  días. Todos los horizontes comparten exactamente las mismas filas de validación.
- **CPCV** (12 grupos, 2 de prueba, 11 trayectorias, purga `h+1`, embargo 1%): solo como
  diagnóstico de estabilidad, porque entrena con datos posteriores al bloque de prueba.
- **Hiperparámetros:** fijos y preespecificados (p. ej. logística `C = 0.01`; LightGBM
  profundidad 3, lr 0.03, 200 árboles). La selección entre configuraciones se paga en la
  inferencia (White, límites simultáneos, PBO), no con búsquedas internas.

## 3. Baselines (por horizonte)

| Baseline | Definición |
|---|---|
| `majority` | Tasa de subida del train (climatología); en la práctica, "siempre sube" |
| `conditional_previous_1` | P(sube a h \| la última vela subió/bajó), estimada en el train |
| `conditional_previous_h` | P(sube a h \| el retorno de las últimas h velas fue +/−) |

El modelo debe superar **a los tres a la vez**. `previous_direction` (repetir la última
dirección) se informa en la Fase 0, pero es sistemáticamente peor (≈46%).

## 4. Métricas

- Accuracy sobre **todas** las filas contra el signo del retorno a h (no solo los movimientos
  grandes), AUC, log-loss, Brier, ECE, accuracy por año y desviación entre folds.
- **Skill** de Brier y de log-loss frente a la climatología del train.
- **Cobertura:** accuracy por margen fijo `|p − 0.5| ≥ m` (causal) y por cuantil de confianza
  (solo descriptivo).
- **Economía:** libro continuo de retornos de 4h. Con `h > 1`, la cartera es solapada (cada
  señal compromete 1/h del capital durante h velas). Costo: 7 pb por unidad de rotación
  (sensibilidad 0 y 15 pb) más **funding** del perpetuo (tasa/2 por vela). El margen de
  abstención se elige al inicio de cada fold con el Sharpe neto del historial OOS ya realizado
  (mínimo seis meses). Referencias: *buy & hold* spot y perpetuo.

## 5. Inferencia y gate

- Bootstrap por bloques móviles de **180 velas** (30 días) para cualquier intervalo; Wilson
  solo como referencia descriptiva.
- **Gate de aceptación (Fases 1 y 2):** sobre la matriz de diferencias de acierto
  (configuración − baseline) para todos los pares, se calculan límites inferiores
  simultáneos al 95%. Pasa si la configuración con mayor límite en el peor caso cumple:
  1. límite inferior simultáneo > 0 frente a los tres baselines, y
  2. diferencia media en el peor caso ≥ +3 pp.
- Complementos (no sustituyen al gate): prueba de White por baseline, PBO maximin (CSCV de 16
  bloques), White económico contra *buy & hold* y Sharpe deflactado (no inferencial).
- **Ablación de una familia (Fase 2):** se conserva solo si la mejora en log-loss **y** en
  accuracy frente al conjunto de referencia tiene límite inferior simultáneo > 0 en todas las
  comparaciones a la vez.

## 6. Registro y reproducibilidad

- Cada ejecución escribe `report.json` y `predictions.parquet` en un directorio nuevo y añade
  una línea a `ml/artifacts/experiments.jsonl` (append-only), incluidos los fallos. Ese
  archivo es el recuento de ensayos para corregir comparaciones múltiples.
- El informe guarda las huellas del dataset, del snapshot auxiliar, de las predicciones y del
  código (`implementation`).
- **Integridad de holdouts:** no modificar los archivos de `IMPLEMENTATION_FILES` en
  [provenance.py](../../ml/src/research/provenance.py). La reserva de noviembre de 2026
  compara sus huellas. Formatear solo archivos nuevos y explícitos; nunca `ruff format src`
  completo. Los comandos de desarrollo rechazan datos de holdouts reservados.

## 7. Cómo ejecutar

Todos los estudios se ejecutan en la imagen `btc-predictor-ml-worker:latest` montando el código
local. Los comandos exactos están en el [README](../../README.md#línea-base-de-evaluación-fase-0-del-plan-de-mejora).
Las pruebas requieren pytest dentro de la imagen; por ejemplo, crear una imagen temporal con
`pip install --user pytest pytest-asyncio pytest-cov ruff mypy` y ejecutar
`python -m pytest -q` con `PYTHONPATH=/work`.
