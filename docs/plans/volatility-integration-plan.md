# Plan: integrar la predicción de volatilidad en el backend y el frontend

> **Rama de trabajo:** `feature/prediccion-volatilidad`.
> **Objetivo:** mostrar, junto a la predicción direccional del plan original, un **pronóstico
> de volatilidad validado**: volatilidad esperada, rango de precio probable, régimen de riesgo
> y una calculadora de tamaño de posición. Todo con la misma API (FastAPI) y el mismo dashboard
> (Next.js).
> **Estado:** F0–F5 implementadas; F6 preparada (infraestructura y evaluador listos, reloj sin iniciar); F7 pendiente.

## 1. Punto de partida (evidencia)

| Hecho | Fuente |
|---|---|
| La dirección de BTC no supera a reglas simples con datos gratuitos (≈54% a 4h, igual que la regla condicional) | [E-01 a E-11](../research/experiment-log.md) |
| La volatilidad **sí** es predecible: HAR-X (HAR + DVOL, VIX, funding, OI, drawdown, …) mejora a HAR y al ingenuo con IC > 0 (QLIKE a 7d: 0.237 vs 0.262 vs 0.419) | [E-12](../research/experiment-log.md#e-12-pronóstico-de-volatilidad-y-exposición-ajustada-por-riesgo) |
| Usar el pronóstico para dimensionar automáticamente la exposición **no** mejoró el Sharpe en 2022–2026 | E-12 |
| Las bandas «normales» (±kσ con k gaussiano) **no** están calibradas: colas gruesas | Tabla siguiente |

**Cobertura real de las bandas a 7 días** (8,766 pronósticos fuera de muestra, sep-2022 a
ago-2026, retorno logarítmico real frente a ±kσ pronosticada):

| Modelo | ±1σ (ideal 68%) | ±1.645σ (ideal 90%) | ±1.96σ (ideal 95%) | k empírico 68 / 90 / 95% |
|---|---|---|---|---|
| Ingenuo | 69.7% | 85.0% | 88.9% | 0.95 / 2.09 / 3.01 |
| EWMA | 72.2% | 87.5% | 90.8% | 0.90 / 1.88 / 2.76 |
| HAR | 76.9% | 90.0% | 92.8% | 0.80 / 1.64 / 2.38 |
| **HAR-X** | 74.1% | 88.7% | 91.8% | **0.86 / 1.77 / 2.52** |
| LightGBM | 74.0% | 88.3% | 91.3% | 0.85 / 1.82 / 2.50 |

Conclusión: el rango que se muestre **debe** usar multiplicadores empíricos (calibración
conforme sobre residuos fuera de muestra), no los gaussianos. Con el gaussiano, la banda
«del 95%» fallaría 1 de cada 12 semanas en lugar de 1 de cada 20.

## 2. Qué verá el usuario

| Elemento | Contenido | Estado de evidencia |
|---|---|---|
| Volatilidad esperada | Volatilidad anualizada pronosticada a 1 y 7 días, y su equivalente diario y semanal | Validado (E-12) |
| Rango probable de precio | Bandas del 68%, 90% y 95% a 1 y 7 días, calibradas empíricamente; superpuestas en el gráfico | En validación hasta completar la sombra (F6) |
| Régimen de riesgo | Bajo / Normal / Alto / Extremo según el percentil histórico del pronóstico | Descriptivo |
| Pronóstico vs realizado | Historial de la volatilidad pronosticada contra la realizada | Validación continua |
| Calibración | Cobertura real de las bandas en las últimas N semanas | Validación continua |
| Calculadora de posición | Tamaño de posición según capital, riesgo y stop basado en volatilidad | Herramienta educativa |
| Predicción direccional | Se mantiene, marcada como **experimental** y comparada con la regla simple | Sin ventaja demostrada |

Umbrales iniciales de régimen (percentiles de la volatilidad anual pronosticada por HAR-X a 7d,
2022–2026): **bajo** < 38.6% (p25), **normal** 38.6–51.0%, **alto** 51.0–59.5% (p90),
**extremo** > 59.5%. Se recalculan con cada reentrenamiento y se guardan con el modelo.

## 3. Estrategia de ramas (crítica)

`feature/prediccion-volatilidad` no contiene el backend ni el frontend de la Fase 4. Hay que
traerlos con un merge de `feature/fase4-desarrollo-API`. **Un merge normal es destructivo:** la
rama de la Fase 4 contiene los dos commits de *revert* de `main` (`1813058` y `dd979b6`), y
como el ancestro común es `d54b8c3`, git aplica esos reverts y **borra en silencio
`docs/research` y todos los estudios**. Se verificó en un worktree desechable.

Procedimiento probado con la punta `09d3d27` de la rama de la Fase 4 (2 conflictos, ambos en el
frontend). Si esa rama avanza, el número de conflictos puede cambiar; el orden de los reverts
(primero `dd979b6`, luego `1813058`, del más reciente al más antiguo) no:

```bash
git checkout feature/prediccion-volatilidad
git merge --no-ff feature/fase4-desarrollo-API      # trae backend/frontend y los reverts
git revert --no-edit dd979b6                         # revierte el revert de 22b3a31 (limpio)
git revert 1813058                                   # revierte el revert de d54b8c3
#   conflictos: frontend/src/app/page.tsx y frontend/src/lib/api.ts
#   resolución: conservar el dashboard de la Fase 4 y portar a sus componentes el estado
#   «experimental» (modelStatus.ts, status/beats_baseline/baseline_overall en api.ts)
git revert --continue
```

Comprobaciones después del merge: existen `docs/research/`, `ml/src/research/volatility_study.py`,
`ml/src/features/context_store.py` y `ml/src/training/activation.py`; pasan los tests de `ml/`,
`backend/` y el build del frontend; las 25 huellas del holdout de noviembre siguen intactas.

Regla para el futuro: los arreglos del plan original se hacen en la rama de la Fase 4 y se
traen a esta rama con merge (ya sin reverts pendientes); nunca al revés.

## 4. Arquitectura

```text
ml-worker (cada 4h, tras ingesta y features)
  ├─ ingesta nueva: DVOL de Deribit (horario), OI de Binance (5 min), calendario FOMC
  ├─ features de volatilidad (mismo código que la investigación: volatility_frame + HAR-X)
  ├─ inferencia del modelo de volatilidad activo ─► volatility_forecasts (1d y 7d)
  └─ resolución: al cumplirse el horizonte, guarda la volatilidad y el retorno realizados
backend (FastAPI, hexagonal)
  └─ lee volatility_forecasts y volatility_models; expone /api/v1/volatility/* y /api/v1/risk/*;
     difunde «volatility» por /ws/live
frontend (Next.js)
  └─ panel de riesgo, bandas en el gráfico, historial, calibración y calculadora
```

**Decisión: la inferencia de volatilidad vive en el ML worker, no en el backend.** El modelo
(una regresión lineal sobre log-varianzas) es trivial, pero sus entradas requieren pandas y el
pipeline de features. Calcularlas en el backend duplicaría código y abriría la puerta a
diferencias entre entrenamiento y producción. El backend solo lee resultados. La predicción
direccional sigue en el backend (ONNX), como en la Fase 4.

## 5. Modelo de datos (migraciones 007–009)

**Decisión: registro de modelos de volatilidad separado.** Reutilizar `model_registry` con una
columna `task` obligaría a cambiar `predictions_log`, la activación y el repositorio del
backend direccional, y no impediría que un pronóstico apunte a un modelo de otra tarea. Una
tabla propia deja intacto el stack direccional y permite restricciones estrictas.

| Migración | Tabla | Columnas clave y restricciones |
|---|---|---|
| 007 | `implied_volatility` | PK (`currency`, `open_time`); `close`, `recorded_at` (momento de la ingesta) |
| 007 | `open_interest` | PK (`symbol`, `create_time`); `sum_open_interest`, `sum_open_interest_value`, `recorded_at` |
| 007 | `event_calendar` | PK (`event_type`, `event_time`); `source_url`, `captured_at` |
| 008 | `volatility_models` | `version` PK; `kind` (`har_x_live` \| `har`), `role` (`primary` \| `fallback`), `horizon_candles` NOT NULL, `artifact_path`, `feature_names`, métricas de validación, `is_active`, `notes`, `trained_at`. Único (`version`, `horizon_candles`) para la FK compuesta. Índice único parcial (`horizon_candles`, `role`) WHERE `is_active` |
| 009 | `volatility_forecasts` | `candle_id`; FK compuesta (`model_version`, `horizon_candles`) → `volatility_models` con `ON DELETE RESTRICT` (el historial de evaluación no se borra); `requested_model_version`; `fallback_reason` (NULL si sirvió el primario); `features_as_of` (JSON con la marca de tiempo de cada fuente exógena usada); `forecast_var_annual`, `forecast_vol_annual`, `k68`, `k90`, `k95`, `calibration_window_end`, `regime`, `reference_price`, `created_at`. Resolución: `realized_var_annual`, `realized_log_return`, `inside_68`, `inside_90`, `inside_95`, `qlike`, `resolved_at`. Único (`candle_id`, `horizon_candles`) |

Un modelo por horizonte: HAR-X ajusta coeficientes distintos a 1 y 7 días. Cada horizonte tiene
un modelo primario (`har_x_live`) y uno de respaldo (`har`), ambos registrados y activables por
separado.

## 6. ML worker

1. **Ingesta en vivo** (`ml/src/ingestion/`):
   - `deribit_client.py`: DVOL horario (`public/get_volatility_index_data`), con el mismo
     paginado de `aux_sources._capture_dvol`; upsert idempotente por PK.
   - `open_interest_client.py`: `futures/data/openInterestHist` (solo 30 días; suficiente para
     el cambio a 7 días). Upsert idempotente por (`symbol`, `create_time`).
   - Calendario del FOMC: reutilizar los parsers de `aux_sources`; refresco semanal.
2. **Disponibilidad idéntica en entrenamiento y producción.** La paridad de fórmulas no basta;
   cada exógena debe tener la misma regla de disponibilidad y frescura en ambos lados:

   | Fuente | Disponible desde | Antigüedad máxima | Si está vencida |
   |---|---|---|---|
   | DVOL | `open_time + 1h` de la vela horaria | 6h | respaldo HAR |
   | VIX | cierre del día D + 1 día | 5 días | respaldo HAR |
   | Funding | `funding_time + 4h` | 36h | respaldo HAR |
   | OI | ver nota | — | — |

   **Nota sobre el OI:** la investigación lo usó con 48h de demora simulada porque el histórico
   de archivos no es verificable *point-in-time*; en vivo está disponible casi al instante.
   Entrenar y servir con reglas distintas introduce sesgo. Decisión: **`har_x_live` excluye el
   OI**. Se repite E-12 sin OI (E-12b) y se acepta si la pérdida de QLIKE a 7d frente a HAR-X
   es < 0.005 y sigue mejorando a HAR con IC > 0. El OI se ingesta igualmente para reevaluarlo
   cuando haya ≥ 1 año de datos en vivo.

   Los datos rellenados retroactivamente (backfill) solo se usan con la regla de disponibilidad
   de la tabla, nunca con su `recorded_at`.
3. **Features compartidas:** extraer de `research/volatility_study.py` las funciones
   `volatility_frame` y `model_inputs` a `ml/src/volatility/features.py`, usadas por la
   investigación y por producción. Test de paridad: sobre un mismo rango histórico, las features
   calculadas desde PostgreSQL y desde el dataset congelado coinciden (tolerancia 1e-9).
4. **Entrenamiento** (`python -m src.volatility.train --horizon 6|42`): walk-forward mensual
   como E-12; guarda coeficientes (JSON), varianza residual y umbrales de régimen; registra el
   primario y el respaldo **inactivos**. Reentrenamiento mensual.
5. **Calibración de bandas sin fuga ni tautología:**
   - Estadístico: `z = |log(P[t+h] / P[t])| / σ̂_h`.
   - Para un pronóstico emitido en `t`, los multiplicadores `k_q` son el cuantil `q` de `z` de
     los pronósticos **cuya resolución ocurrió antes de `t`** (`t' + h ≤ t`), en una ventana
     móvil de 2 años. Se guardan en cada fila (`k68`, `k90`, `k95`, `calibration_window_end`).
   - La cobertura se evalúa en periodos **posteriores** a la ventana de ajuste (walk-forward),
     nunca sobre los mismos residuos.
6. **Activación:** comando propio `python -m src.volatility.activation --version N --reason ...`
   que valida los artefactos de volatilidad (no los ONNX direccionales) y solo desactiva la
   versión activa del mismo horizonte y rol. `src.training.activation` no cambia.
7. **Inferencia** (al final de `run_features`): para la última vela cerrada, pronóstico a 6 y
   42 velas con el primario; si alguna exógena incumple su regla de frescura, se usa el
   respaldo y se guardan `fallback_reason` y `features_as_of`. Banda de precio =
   `reference_price · exp(±k·σ_h)`, con `σ_h = σ_anual · √(h·4h / 1 año)`. Idempotente por
   (vela, horizonte).
8. **Resolución** — invariante explícito y probado:
   - Varianza realizada: media de los cuadrados de los retornos log de las velas `t+1 … t+h`
     (es decir, `log c[t+i] − log c[t+i−1]`, i = 1…h), anualizada. Es la misma definición que
     `target_var_*` en `volatility_study`.
   - Cobertura de bandas: `close[t+h] / close[t]`.
   - Solo se resuelve cuando existen las `h` velas siguientes sin huecos.

## 7. Backend (hexagonal, misma estructura de la Fase 4)

- **Dominio:** `VolatilityForecast`, `PriceBand`, `VolatilityRegime`, `VolatilityMetrics`;
  puertos `VolatilityRepositoryPort` y `VolatilityPort`.
- **Salida:** `PostgresVolatilityRepository` (lecturas paginadas, último pronóstico por
  horizonte, agregados de calibración).
- **Aplicación:** `VolatilityService`, `RiskService` (calculadora pura, sin estado).
- **Endpoints:**

| Método | Ruta | Respuesta (resumen) |
|---|---|---|
| GET | `/api/v1/volatility/current?horizon=1d\|7d` | `{ horizon, as_of, reference_price, vol_annual, vol_horizon_pct, regime, bands: {p68,p90,p95: {low, high}}, dvol, realized_vol_7d, realized_vol_30d, model_version, fallback }` |
| GET | `/api/v1/volatility/history?horizon=&limit=&offset=` | Pronóstico vs realizado por vela, paginado |
| GET | `/api/v1/metrics/volatility?horizon=` | `{ resolved, coverage_68, coverage_90, coverage_95, qlike, qlike_naive, mz_slope, last_12w }` |
| GET | `/api/v1/risk/position-size?capital=&risk_pct=&stop_k=&horizon=` | `{ stop_distance_pct, stop_price, position_usd, position_btc, exposure_pct, warnings[] }` (rechaza `risk_pct > 5`) |
| WS | `/ws/live` | Nuevo mensaje `{ type: "volatility", data: { horizon, vol_annual, regime, bands } }` |

- `/api/v1/metrics/model` y `/predictions/current` añaden `status`, `beats_baseline` y
  `baseline_overall` (ya especificado en el plan de plataforma).
- **Pruebas:** unitarias de servicio y calculadora (casos límite: capital 0, stop 0, riesgo
  fuera de rango), routers con dobles de puertos e integración del repositorio contra
  PostgreSQL. Cobertura ≥ 80%, igual que hoy.

## 8. Frontend

Nuevo «Panel de riesgo» en la columna lateral y una superposición en el gráfico:

| Componente | Contenido |
|---|---|
| `VolatilityCard` | Volatilidad anual y semanal esperada, régimen (insignia de color neutro con texto), DVOL y realizada como referencia |
| `ExpectedRangeCard` | Rango de precio a 1 y 7 días (68/90/95%), con la advertencia de que 1 de cada 20 semanas cae fuera del 95% |
| `CandlestickChart` (ampliado) | Líneas de banda 68% y 95% a 7 días proyectadas desde la última vela |
| `VolatilityHistoryChart` | Pronóstico vs realizado (Recharts), últimas 26 semanas |
| `CalibrationPanel` | Cobertura real de cada banda vs objetivo; alerta si se desvía más de 5 pp |
| `PositionSizeCalculator` | Capital, % de riesgo (0.25–2%), multiplicador de stop → tamaño, precio de stop y exposición; enlaza a la guía de trading |
| `CurrentPredictionCard` | Añade la etiqueta «Experimental — no supera a reglas simples» y la comparación con la regla |

Requisitos: estados de carga y error como los componentes actuales, sin colores de
compra/venta para la volatilidad (no es una señal direccional) y textos que no prometan
resultados.

## 9. Criterios de validación antes de mostrarlo como «validado»

1. **Offline (E-12b):** `har_x_live` mejora a HAR en QLIKE a 7d con IC por bloques de 180 velas
   > 0, con las reglas de disponibilidad de producción.
2. **Calibración offline:** con los multiplicadores de la sección 6.5, cobertura de 68/90/95% en
   periodos posteriores al ajuste, con IC por bloques (bloque ≥ 42 velas para 7d) que contenga
   el nivel nominal; también por año.
3. **Paridad:** el pronóstico de producción coincide con el de investigación en el mismo rango
   (diferencia relativa < 1e-6).
4. **Sombra prospectiva de 12 semanas,** sin mostrarse en la UI:
   - Muestreo **sin solapamiento**: el pronóstico a 1 día emitido en la vela de 00:00 UTC
     (84 observaciones). Pasa si el IC binomial del 95% de la cobertura del 68% y del 90%
     contiene el nivel nominal. Con 84 datos el IC es de ±10 pp: es una prueba de cordura, no
     de precisión.
   - A 7 días, una observación semanal sin solapamiento (12 en 12 semanas): solo se monitorea;
     el horizonte de 7 días sigue «en validación» hasta acumular 26 semanas.
   - Ninguna falla de ingesta mayor a 6h sin alerta, y uso del respaldo < 5% del tiempo.
5. Hasta cumplir el punto 4, el panel se muestra con la etiqueta «en validación».

## 10. Fases

| Fase | Contenido | Esfuerzo | Criterio de salida |
|---|---|---|---|
| F0 ✅ | Merge de ramas según la sección 3 (punta probada `09d3d27`) | 0.5 d | Tests de las tres partes en verde; huellas intactas |
| F1 🟡 | Ingesta en vivo (DVOL, OI, FOMC), migración 007, health checks | 2–3 d | Datos en vivo ≤ 3h de antigüedad durante 48h (código y pruebas listos; falta la observación de 48 h en un stack propio) |
| F2 ✅ | Features compartidas, paridad, E-12b sin OI con reglas de producción, calibración, entrenamiento, registro (008) y activación | 3–4 d | Criterios 1–3 de la sección 9 (cumplidos a 7d; ver [E-12b](../research/experiment-log.md#e-12b-har_x_live-pronóstico-sin-oi-calibración-de-bandas-y-paridad-con-producción)) |
| F3 ✅ | Inferencia y resolución en el ML worker (009) | 1–2 d | Pronósticos idempotentes cada 4h; resolución correcta en test (probado también de extremo a extremo con datos reales en una base temporal) |
| F4 ✅ | Backend: dominio, repositorio, servicios, endpoints, WS, pruebas | 2–3 d | Cobertura ≥ 80% (96%); contratos documentados en `/docs` |
| F5 ✅ | Frontend: panel de riesgo, bandas, historial, calibración, calculadora | 3–4 d | tsc, lint y build en verde; verificación en el navegador (hecha con datos reales) |
| F6 🟡 | Sombra prospectiva (12 semanas de calendario) | Bajo | Criterio 4 de la sección 9 (stack `infra/docker-compose.shadow.yml`, `shadow.sh` y `src.volatility.shadow` listos y probados; falta iniciar y esperar 12 semanas) |
| F7 | Documentación: runbook, `docs/trading`, registro E-13 | 1 d | Documentos actualizados con cifras reales de producción |

Total de esfuerzo: ≈ 14–18 días de desarrollo más 12 semanas de observación.

## 11. Riesgos

| Riesgo | Mitigación |
|---|---|
| El merge borra la investigación | Procedimiento de la sección 3 y comprobaciones posteriores |
| Diferencias entre investigación y producción | Código de features compartido, test de paridad y las mismas reglas de disponibilidad (OI excluido) |
| Pronósticos de respaldo indistinguibles | `fallback_reason` y `features_as_of` en cada fila; la UI lo muestra |
| La API de Deribit falla o cambia | Respaldo HAR sin exógenas, marcado como `fallback` en la respuesta y en la UI |
| Las bandas se interpretan como garantía | Textos explícitos, cobertura visible y guía de uso en `docs/trading` |
| Deriva de régimen (la calibración de 2022–2026 no se mantiene) | Multiplicadores en ventana móvil, alerta de cobertura y reentrenamiento mensual |
| Revisiones históricas de datos (CoinMetrics, métricas de Binance) | HAR-X-live no usa CoinMetrics; OI en vivo se guarda al momento de su lectura |

## 12. Fuera de alcance

- Ejecución automática de órdenes (ver [docs/trading/08-automatizacion-segura.md](../trading/08-automatizacion-segura.md)).
- Cambios al modelo direccional.
- Estrategias combinadas (filtro de tendencia + tamaño por volatilidad + stops por rango): se
  evaluarán como E-13 antes de recomendarlas con cifras.
