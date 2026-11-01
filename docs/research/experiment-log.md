# Registro de experimentos

Registro cronológico de todo lo intentado para mejorar la predicción de dirección de BTC.
Cada entrada es evidencia: hipótesis, método, resultado, artefacto y decisión. Los artefactos
viven en `ml/artifacts/` (no versionados); las cifras se copiaron de sus `report.json`,
`ablation.json` o `walk_forward.json`. El recuento de ensayos desde la Fase 0 está en
`ml/artifacts/experiments.jsonl`.

**Estado global (2026-10-31):**
- **Dirección:** ningún modelo supera el gate; el modelo registrado permanece **inactivo**.
  La única regularidad robusta es una leve reversión de una vela a 4h (≈54% de acierto), que
  una regla condicional trivial iguala y que los costos anulan. La línea direccional queda
  **cerrada** con las fuentes gratuitas (E-01 a E-11).
- **Volatilidad:** **sí es predecible** (E-12). HAR-X con DVOL mejora a HAR y al pronóstico
  ingenuo con significancia. No mejora el Sharpe al usarse para dimensionar la exposición,
  pero es un insumo válido para mostrar riesgo.

| ID | Fecha | Pregunta | Resultado | Gate |
|---|---|---|---|---|
| E-01 | 2026-10-29 | ¿XGBoost o LSTM superan a los baselines a 4h? | 49.7% / 49.9% vs baseline 53.0% | ✗ |
| E-02 | 2026-10-29 | ¿Una búsqueda de 18 candidatos encuentra señal? | Desarrollo 53.1% (+0.99 pp); holdout oct 55.4% (IC 47.8–62.7%) | ✗ |
| E-03 | 2026-10-29 | ¿Aporta el flujo agresor? | −0.24 pp | ✗ |
| E-04 | 2026-10-29 | ¿Aporta el funding? ¿Y LightGBM? | +0.22 pp (ruido); LightGBM peor | ✗ |
| E-05 | 2026-10-30 | ¿Aportan OI y posicionamiento (dataset 2020+)? | −0.16 / −0.28 pp | ✗ |
| E-06 | 2026-10-30 | Igual que E-05 con bootstrap pareado; reserva de holdout | Base 53.6% vs condicional 53.5%; holdout de nov **pendiente** | ✗ (desarrollo) |
| E-07 | 2026-10-30 | Línea base de evaluación rigurosa (Fase 0) | 54.0% = regla condicional; Sharpe neto −2.3 | ✗ |
| E-08 | 2026-10-30 | ¿Horizonte más largo, zona muerta o abstención? (Fase 1) | 12h–3d: 49–52% < "siempre sube" | ✗ |
| E-09 | 2026-10-30 | ¿Features de horizonte largo? (Fase 2) | Mejor: 52.9% a 3d (+0.17 pp vs mayoría); ninguna familia conservada | ✗ |
| E-10 | 2026-10-31 | ¿Aporta la volatilidad implícita de opciones (DVOL)? | Ninguna mejora; PBO 0.78 | ✗ |
| E-11 | 2026-10-31 | ¿Una regla clásica de tendencia iguala al ML? | TSMOM 180d largo/plano: Sharpe 0.94, DD −33%, igual que el mejor ML; no supera a *buy & hold* | ✗ (referencia) |
| E-12 | 2026-10-31 | ¿Es predecible la volatilidad? ¿Sirve para dimensionar la exposición? | Sí es predecible (QLIKE con IC > 0); el *vol targeting* no mejora el Sharpe | ✓ pronóstico / ✗ economía |

---

## E-01 Walk-forward XGBoost vs LSTM

- **Artefacto:** `ml/artifacts/run-20261029T065504Z-cc98c8c8/` (`walk_forward.json`,
  `model.onnx`, `calibration.json`). Comando: `python -m src.training.run`.
- **Hipótesis:** un modelo no lineal sobre los 46 indicadores (`existing`) predice la vela
  siguiente de 4h mejor que los baselines.
- **Método:** 10 folds mensuales (dic-2025 a sep-2026), datos desde 2023-08 (7,000 filas),
  selección de features por correlación ≥ 0.02 y MI, búsqueda en rejilla, Platt sobre un
  mes. El último fold se reserva.
- **Resultado:** XGBoost 49.7%, LSTM 49.9%; mejor baseline promedio por fold 53.0%; último
  fold 53.3% = baseline. Pendiente de Platt 0.11 (probabilidades casi planas). En XGBoost,
  `sp500_close` concentró 0.28 de la importancia.
- **Conclusión:** sin señal; la selección de features por correlación es inestable y los
  niveles absolutos introducen señales espurias.
- **No repetir:** modelos más complejos sobre el mismo conjunto, el mismo target y 10 folds
  (potencia insuficiente: ±3.7 pp por fold).

## E-02 Búsqueda de 18 candidatos con holdout de octubre

- **Artefacto:** `ml/artifacts/research-20261029T070840Z-73abf8e4/` (`development.json`,
  `holdout.json`, `comparison.json`, `holdout.lock`). Comando:
  `python -m src.training.research develop|evaluate`.
- **Método:** XGBoost/logística × features legacy/existing/expanded × ventana expansiva o de
  540 días × calibración Platt o ninguna; 10 folds de desarrollo; holdout 2026-10-01 a
  2026-10-29 (168 velas), evaluado una vez.
- **Resultado:** seleccionado `logistic_expanded_expanding_raw` (desarrollo 53.15% contra
  baseline 52.16%, +0.99 pp). Holdout: 55.36%, IC de Wilson [47.8%, 62.7%]; "siempre baja"
  53.6%.
- **Conclusión:** el holdout es demasiado corto para concluir. Platt sobre un mes empeora en
  todos los candidatos.
- **No repetir:** holdouts de un mes como evidencia; Platt sobre un mes de calibración. El
  período de octubre de 2026 ya está consumido.

## E-03 Flujo agresor (taker)

- **Artefactos:** `ml/artifacts/taker-ablation-20261029T074256Z-2c2712df/`, `...074514Z-4705bcdf/`,
  `...075518Z-9b08a2e7/`. Comando: `python -m src.training.taker_ablation`.
- **Resultado (logística, 10 folds):** sin taker 53.15%; todas las variables 52.91%; solo el
  ratio 53.15%; multiescala 52.73%; LightGBM sin y con taker 52.54% y 52.42%.
- **Nota:** los subconjuntos se añadieron *después* de ver la primera ablación (selección a
  posteriori).
- **Conclusión:** el flujo agresor spot no aporta información direccional a 4h.
- **Reabrir solo si:** se usa el flujo de futuros con resolución intradía y otro horizonte.

## E-04 Funding y LightGBM

- **Artefacto:** `ml/artifacts/taker-ablation-20261029T221109Z-1fa0bcb8/`.
- **Resultado:** con funding 53.37% vs 53.15% sin él (+0.22 pp); con funding y taker 52.30%;
  LightGBM con funding 51.70%.
- **Conclusión:** diferencia dentro del ruido; el funding se mantuvo en `base` por su bajo
  costo. En la Fase 2, el funding acumulado (C3) tampoco aportó.

## E-05 Interés abierto (primera corrida, dataset 2020+)

- **Artefacto:** `ml/artifacts/oi-ablation-20261030T164049Z-8ffbdb8f/`. Dataset
  `btcusdt-20200901-20261001-v1`. Comando: `python -m src.research.oi_experiment`.
- **Método:** logística con C anidado en {0.003, 0.01, 0.03, 0.1} en 3 meses internos; 10
  folds (nov-2025 a ago-2026); métricas de futuros con 48h de demora.
- **Resultado:** base 53.62%, +OI 53.46%, +OI+posicionamiento 53.33%; mejor baseline
  retrospectivo por fold 55.57%.

## E-06 Interés abierto con comparación pareada y reserva de noviembre

- **Artefacto:** `ml/artifacts/oi-ablation-20261030T171037Z-b9bda9ee/` (`ablation.json`,
  `frozen.json`). Imagen `btc-predictor-ml-worker:oi-frozen-20261030`.
- **Resultado:** mismas cifras que E-05. Diferencias pareadas (bootstrap por fold): OI−base
  −0.16 pp [−0.49, +0.12]; posicionamiento−base −0.28 pp [−1.11, +0.37]; base−condicional
  +0.11 pp [−1.76, +2.15].
- **Reserva:** variante `base` congelada para el holdout **2026-11-01 a 2026-12-01**. Evaluar
  con `python -m src.research.oi_holdout evaluate` a partir del 2026-12-02 (ver README).
  **No modificar los archivos de `IMPLEMENTATION_FILES`**: invalidaría la reserva.
- **Conclusión:** OI y posicionamiento no aportan a 4h.

## E-07 Línea base de evaluación (Fase 0)

- **Artefacto:** `ml/artifacts/eval-baseline-20261030T223358Z-27c9963a/`. Comando:
  `python -m src.research.evaluation_harness`. Código:
  [evaluation_harness.py](../../ml/src/research/evaluation_harness.py) y `ml/src/evaluation/*`.
- **Método:** 48 folds mensuales (sep-2022 a ago-2026, 8,766 predicciones OOS); 10
  candidatos fijos (logística sobre existing/base/base_oi × C {0.003, 0.01, 0.03} y LightGBM
  sobre base); par primario preespecificado `logistic_base_C0.01` contra
  `conditional_direction`.
- **Resultados:**

  | Métrica | Valor |
  |---|---|
  | Accuracy primaria | 54.02% (Wilson 53.0–55.1%); por año: 2022 53.4%, 2023 56.9%, 2024 54.1%, 2025 51.5%, 2026 53.6% |
  | Regla condicional | 54.04%; diferencia −0.02 pp, IC por bloques [−1.0, +1.1] pp |
  | White (10 candidatos) | p = 0.75 |
  | existing / base / base_oi | 52.0% / 54.0% / 53.9% |
  | Cobertura | margen 0.05 → 52% de las velas con 55.0%; 10% más confiable 57.9% (descriptivo) |
  | Economía (largo-corto) | Sharpe 0 pb +0.70; 7 pb −2.29; 15 pb −5.65; rotación 0.89 por vela; *buy & hold* 0.98 |
  | Calibración online | ECE cruda 2.0% → Platt expansiva 0.8% / último año 0.4%; isotónica sin mejora |
  | Sobreajuste | PBO accuracy 0.24; PBO Sharpe 0.37; Sharpe deflactado ≈ 0 |
  | CPCV (11 trayectorias) | base 54.4% ± 0.2 pp; Sharpe medio −0.76 |

- **Conclusión:** existe una señal pequeña, estable y real, pero es reversión de una vela, ya
  capturada por la regla condicional; los costos la anulan.

## E-08 Estudio de objetivo (Fase 1)

- **Artefacto:** `ml/artifacts/target-study-20261030T230014Z-6953a53e/`. Comando:
  `python -m src.research.target_study`. Código:
  [target_study.py](../../ml/src/research/target_study.py).
- **Hipótesis:** a horizontes más largos la señal supera los costos; excluir movimientos
  pequeños del entrenamiento reduce ruido; abstenerse con baja confianza mejora la precisión.
- **Método:** 24 configuraciones: h ∈ {1, 3, 6, 18} velas × zona muerta k ∈ {0, 0.25,
  0.5}·σ·√h (σ = desviación de 42 velas) × {logística `base` C=0.01, LightGBM}. Tres
  baselines por horizonte; gate de límites simultáneos; estrategias en perpetuo con funding y
  margen elegido causalmente. Primario preespecificado: `h6_k0.25_logistic_base_C0.01`.
- **Resultados:**

  | Horizonte | Mejor accuracy | "Siempre sube" | Mejor condicional |
  |---|---|---|---|
  | 4h | 54.02% (k=0) | 51.2% | 54.04% |
  | 12h | 51.56% (LightGBM k=0.25) | 51.5% | 52.1% |
  | 24h | 51.06% | 52.0% | 52.1% |
  | 3d | 49.40% | 52.7% | 52.7% |

  - Zona muerta: empeora a 4h (k=0.5 → 52.9%) y no corrige los horizontes largos.
  - Primario (24h, k=0.25): 50.9%; predice "sube" solo el 36% de las veces.
  - Gate: mejor límite simultáneo en el peor caso −5.4 pp; White p = 0.28 / 0.98 / 0.99;
    PBO maximin 0.27.
  - Economía: mejor estrategia `h1_k0.5_logistic_long_flat` con Sharpe neto 0.79 y 8% de
    exposición, contra *buy & hold* 0.99; White económico p = 0.997. Largo-corto: Sharpe de
    −1.39 a −0.12 en todos los horizontes.
- **Conclusión:** el target no era el problema; las features técnicas no contienen
  información de horizonte largo.
- **Reabrir solo si:** existen features que sí la contengan (ver E-09).

## E-09 Ablación de familias de features (Fase 2)

- **Artefactos:** snapshot `ml/artifacts/aux-sources/aux-20190901-20261001-v1/` e informe
  `ml/artifacts/feature-study-20261030T234559Z-00e2a9e9/`. Comandos:
  `python -m src.research.aux_sources capture|verify` y `python -m src.research.feature_study`.
  Código: [long_horizon.py](../../ml/src/features/long_horizon.py),
  [aux_sources.py](../../ml/src/research/aux_sources.py) y
  [feature_study.py](../../ml/src/research/feature_study.py).
- **Hipótesis:** la falta de señal a 24h–3d se debe a features de corto plazo; familias con
  información de régimen, derivados, on-chain, eventos y otros activos la aportarían.
- **Método:** h ∈ {6, 18}; conjuntos `base`, `stationary` (C1), `stationary` + cada familia
  C2–C6 y `all` (111 columnas); logística L2 C=0.01 en todos, más L1 C=0.05 (selección,
  área D) y LightGBM sobre `all`. Son 20 configuraciones con los mismos 48 folds, el mismo
  gate y la misma economía que E-08. Ablación pareada contra `stationary` con límites
  simultáneos sobre 14 comparaciones. Primario preespecificado: `h6_all_logistic_l2`.
- **Resultados:**

  | Configuración | 24h | 3d |
  |---|---|---|
  | `base` | 51.06% | 48.15% |
  | `stationary` | 51.23% | 50.94% |
  | `+regime` | 50.89% | 51.65% |
  | `+derivatives` | 50.76% | 50.19% |
  | `+onchain` | 50.72% | 50.22% |
  | `+events` | 50.74% | 51.37% |
  | `+cross_asset` | 51.38% | 50.44% |
  | `all` L2 / L1 / LightGBM | 50.82% / 51.11% / 51.21% | 50.39% / 50.92% / **52.87%** |
  | "Siempre sube" | 52.01% | 52.70% |

  - Ablación: **ninguna familia conservada**. La mayor mejora fue `stationary` frente a `base`
    a 3d (+2.78 pp, IC por bloques [−1.0, +6.6]), que corrige el sesgo de los niveles
    absolutos pero no supera a "siempre sube". Añadir todas las familias empeora el log-loss
    (−0.043 a 3d).
  - Gate: mejor `h18_all_lightgbm`, con +0.17 pp en el peor caso y límite simultáneo
    −4.6 pp; White p = 0.82 / 0.84 / 0.93. **PBO maximin 0.63**: la configuración que gana
    dentro de muestra tiende a perder fuera.
  - `h18_all_lightgbm` por año: 2022 61.6%, 2023 49.3%, 2024 49.9%, 2025 56.4%, 2026 53.1%.
    Acierta en los mercados bajistas y falla en los alcistas.
  - Primario (24h, `all`, L2): 50.8%; peor caso −1.23 pp.
  - Economía: mejor estrategia `h18_all_lightgbm_long_flat`, Sharpe neto 0.94 con drawdown
    máximo −32%, contra *buy & hold* Sharpe 0.99 con drawdown −53%. White p = 0.986. Reduce el
    riesgo, pero no mejora el retorno ajustado por riesgo.
  - Estabilidad: los coeficientes principales mantienen el signo en los 48 folds (volatilidad,
    retorno de 90 días, drawdown, funding de 7 días, ETH/BTC, FOMC) y aun así no generalizan.
- **Conclusión:** con las fuentes gratuitas disponibles no hay información direccional
  explotable a 24h–3d. Lo poco que aparece depende del régimen (bajista contra alcista).
- **No repetir:** añadir más features de las mismas familias, ajustar ventanas o hacer
  búsquedas de hiperparámetros sobre este conjunto (PBO 0.63 indica que la selección ya está
  sobreajustando).
- **Reabrir solo si:** hay fuentes cualitativamente nuevas (opciones o volatilidad implícita,
  flujos de ETF con más historia, calendario macro completo, datos de órdenes) o un objetivo
  distinto (ver «Direcciones abiertas»).

---

## E-10 Volatilidad implícita de opciones (DVOL de Deribit)

- **Artefactos:** snapshot `ml/artifacts/aux-sources/aux-20190901-20261001-v2/` (v1 más el
  DVOL horario) e informe `ml/artifacts/feature-study-20261031T004437Z-937bbb9c/` (se repitió
  con el código final y dio resultados idénticos a `...003830Z-a369db2f`). Comando:
  `python -m src.research.feature_study --aux .../aux-20190901-20261001-v2`.
- **Hipótesis:** el mercado de opciones (volatilidad implícita, prima de varianza) contiene
  información direccional que no está en precio, derivados ni on-chain.
- **Método:** familia `options` (DVOL, cambios de 1/7 días, z-score de 90 días, prima de
  varianza de 30 días, ratio implícita/realizada de 7 días) añadida como séptima extensión al
  estudio de E-09; mismos folds, gate y regla de ablación. El skew y el put/call **no** tienen
  historia gratuita.
- **Resultados:**
  - `stationary_options` frente a `stationary`: 24h −0.51 pp (IC [−1.9, +0.6]); 3d +0.67 pp
    (IC [−1.3, +2.9]); el log-loss empeora en ambos horizontes. No se conserva.
  - `all` con opciones: 24h 50.9% (L2), 3d 52.5% (LightGBM; peor caso −0.24 pp).
  - Gate: límite simultáneo −5.0 pp; **PBO maximin 0.78**; White económico p = 0.98.
  - Nota: `all` cambia respecto de E-09 porque ahora incluye la familia de opciones.
- **Conclusión:** tampoco hay información direccional en la volatilidad implícita. Con esto
  se cierra la línea direccional.
- **Reabrir solo si:** hay skew o put/call históricos (de pago, p. ej. Tardis), flujo de
  órdenes de alta frecuencia o un cambio estructural del mercado.

## E-11 Reglas clásicas de tendencia (referencia sin ML)

- **Artefacto:** `ml/artifacts/trend-study-20261031T004114Z-f5cc35c7/`. Comando:
  `python -m src.research.trend_study`. Código:
  [trend_study.py](../../ml/src/research/trend_study.py).
- **Hipótesis:** el momentum de series de tiempo, documentado en cripto, ofrece una referencia
  económica tan buena o mejor que los modelos de ML, sin riesgo de sobreajuste.
- **Método:** 5 reglas preespecificadas (TSMOM de 7/30/90/180 días y cruce de medias de
  50/200 días) × largo/plano o largo/corto × con o sin escalado a 50% de volatilidad = 20
  estrategias. Perpetuo con 7 pb y funding. Ventanas: 48 meses OOS (primaria) y la ventana
  evaluable de E-08/E-09.
- **Resultados (48 meses; *buy & hold* spot: Sharpe 0.98, drawdown −53%):**

  | Estrategia | Sharpe | Drawdown | Exposición | IC de ΔSharpe vs B&H |
  |---|---|---|---|---|
  | TSMOM 180d largo/plano | 0.94 | −33% | 64% | [−0.66, +0.57] |
  | TSMOM 30d largo/plano | 0.85 | −40% | 56% | [−0.91, +0.56] |
  | TSMOM 90d largo/plano | 0.77 | −34% | 56% | [−0.76, +0.52] |
  | TSMOM 180d largo/corto | 0.71 | −40% | 100% | [−1.39, +0.94] |
  | Cruce 50/200 largo/plano | 0.49 | −41% | 58% | [−1.19, +0.27] |

  - White contra *buy & hold*: p = 0.96; PBO Sharpe 0.54. El escalado por volatilidad no
    cambia nada (la exposición casi nunca se reduce). Las versiones largo/corto pierden frente
    a las largo/plano.
- **Conclusión:** una regla de una línea (largo si el retorno de 180 días es positivo)
  reproduce el perfil de la mejor estrategia de ML de E-09 (Sharpe 0.94, drawdown −32%). Los
  modelos no aportan valor sobre ella. Ninguna regla supera a *buy & hold* en Sharpe; solo
  reducen el drawdown a costa del retorno.

## E-12 Pronóstico de volatilidad y exposición ajustada por riesgo

- **Artefacto:** `ml/artifacts/volatility-study-20261031T004234Z-fdba3c4a/` (idéntico a la
  corrida `...003616Z-20027603`). Comando: `python -m src.research.volatility_study`. Código:
  [volatility_study.py](../../ml/src/research/volatility_study.py).
- **Hipótesis:** aunque la dirección no sea predecible, la volatilidad sí lo es (agrupamiento
  de volatilidad), y un mejor pronóstico permite dimensionar la exposición.
- **Método:**
  - Objetivo: varianza realizada anualizada de las próximas 6 velas (1 día) y 42 velas (7
    días, primario) con retornos de 4h.
  - Modelos: ingenuo (varianza de la ventana previa), EWMA (vida media de 42 velas), HAR
    (log-varianzas de 1/7/30 días), **HAR-X** (primario: HAR + DVOL, VIX, funding de 7 días,
    OI de 7 días, drawdown de 90 días, |retorno de 1 día|, volumen relativo y FOMC ±72h) y
    LightGBM con todas las features.
  - Pérdida QLIKE con corrección de sesgo lognormal; los mismos 48 folds con purga por
    horizonte; bootstrap por bloques de 180 velas.
  - Economía: posición larga spot `min(1, 0.5/σ̂)` con banda de rebalanceo de 0.1 y 7 pb,
    comparada con *buy & hold* y con una exposición constante igual a su media.
- **Resultados del pronóstico (QLIKE, menor es mejor):**

  | Modelo | 1 día | 7 días | R² log frente al ingenuo (7d) |
  |---|---|---|---|
  | Ingenuo | 2.051 | 0.419 | 0 |
  | EWMA | 0.716 | 0.309 | +0.145 |
  | HAR | 0.682 | 0.262 | +0.022 |
  | **HAR-X** | 0.667 | **0.237** | **+0.185** |
  | LightGBM | **0.577** | 0.257 | +0.025 |

  - HAR frente al ingenuo (7d): mejora 0.157, IC [0.063, 0.266]. **Volatilidad predecible.**
  - HAR-X frente a HAR (7d): mejora 0.025, IC [0.011, 0.037]. **Las variables exógenas
    aportan.**
  - A 1 día gana LightGBM (frente a HAR-X: IC [0.047, 0.124]); a 7 días LightGBM no supera a
    HAR-X (IC [−0.051, +0.002]).
  - Mincer-Zarnowitz de HAR-X a 7d: pendiente 0.76 (pronósticos algo sobredispersos);
    volatilidad media pronosticada 45.7% contra 42.7% realizada.
- **Resultados económicos (preespecificados, objetivo 50%):** HAR-X 7d tiene Sharpe 0.89 con
  drawdown −53% y exposición media 0.93, contra *buy & hold* 0.98 y −53%; IC de ΔSharpe
  [−0.20, −0.01]. El pronóstico supera 50% solo el 28% del tiempo, así que la regla casi no
  actúa.
- **Sensibilidad *post hoc* (no preespecificada; no cuenta para el gate):** con objetivo de
  30%, HAR-X 7d logra Sharpe 0.90, drawdown −43% y exposición 0.68. Una exposición constante
  de 0.68 logra Sharpe 0.98 y drawdown −39%. El *timing* por volatilidad **no** añade valor
  frente a simplemente invertir menos (IC de ΔSharpe [−0.30, +0.11]).
- **Conclusión:** la volatilidad de BTC es predecible y HAR-X (con DVOL) es el mejor modelo
  a 7 días. En 2022–2026 eso no se tradujo en mejor retorno ajustado por riesgo: los periodos
  de alta volatilidad también fueron alcistas. El valor está en **comunicar el riesgo**
  (rango esperado, régimen de volatilidad) y en el dimensionamiento que decida el usuario,
  no en una estrategia automática.

## E-12b `har_x_live`: pronóstico sin OI, calibración de bandas y paridad con producción

- **Artefacto:** `ml/artifacts/volatility-live-20261101T190031Z-9cae2305/` · Comando:
  `python -m src.volatility.study` · Código: [study.py](../../ml/src/volatility/study.py),
  [features.py](../../ml/src/volatility/features.py),
  [model.py](../../ml/src/volatility/model.py),
  [calibration.py](../../ml/src/volatility/calibration.py). Mismos datos y 48 folds que E-12.
- **Hipótesis:** el modelo servible en producción (`har_x_live`: HAR-X **sin** interés abierto,
  porque el OI histórico solo es utilizable con 48 h de demora simulada y en vivo estaría
  disponible al instante) conserva la ventaja de E-12; las bandas de precio calibradas de forma
  conforme tienen la cobertura nominal; y la ruta de producción reproduce a la de investigación.
- **Método (preespecificado en el plan, sección 9):**
  - Criterio 1: `har_x_live` mejora a HAR en QLIKE a 7d con IC 95% por bloques de 180 velas > 0
    y su QLIKE medio supera al de HAR-X en menos de 0.005.
  - Criterio 2: multiplicadores `k_q` = cuantil conforme de `z = |log(P[t+h]/P[t])|/σ̂_h` entre
    los pronósticos ya resueltos (`t' + h ≤ t`) de los últimos 2 años, con al menos 1,080
    resueltos; cobertura de 68/90/95% evaluada solo donde hay multiplicadores y con IC por
    bloques de 180 velas que debe contener el nivel nominal.
  - Criterio 3: las entradas de `live_inputs` (tablas crudas) coinciden con las de la
    investigación sobre todo el historial (1e-9) y sobre ventanas finales de 600 velas (25
    puntos), y el pronóstico difiere menos de 1e-6 en términos relativos.
  - Código compartido: `volatility_frame` y `model_inputs` pasaron a `src/volatility/features.py`
    (E-12 los reutiliza). HAR y `har_x_live` se ajustan con el artefacto JSON de producción
    (float64); HAR-X conserva el código de E-12.
- **Resultados (QLIKE, menor es mejor):**

  | Modelo | 1 día | 7 días |
  |---|---|---|
  | Ingenuo | 2.051 | 0.419 |
  | HAR | 0.682 | 0.262 |
  | HAR-X (con OI) | 0.667 | 0.237 |
  | **HAR-X-live (sin OI)** | **0.666** | **0.236** |

  - 7d: `har_x_live` frente a HAR mejora 0.026, IC [0.012, 0.037] y frente a HAR-X pierde
    −0.0004 (es decir, no pierde): **criterio 1 cumplido.** Quitar el OI no cuesta nada.
  - 1d: `har_x_live` frente a HAR mejora 0.016 con IC [−0.013, 0.039]: **a 1 día no se demuestra
    ventaja sobre HAR**. El primario se registra igual, pero la activación a 1d es una decisión
    explícita.
- **Calibración (cobertura fuera de muestra, desde 2023-03):**

  | Horizonte | Nivel | Conforme (IC 95%) | Gaussiano | ¿IC contiene el nominal? |
  |---|---|---|---|---|
  | 7d | 68% | 68.4% [63.8, 73.4] | 73.9% | sí |
  | 7d | 90% | 91.1% [87.9, 94.6] | 88.9% | sí |
  | 7d | 95% | 96.0% [94.1, 98.3] | 92.2% | sí |
  | 1d | 68% | 67.5% [65.8, 69.2] | 72.9% | sí |
  | 1d | 90% | 91.1% [90.1, 92.4] | 88.5% | **no** (conservadora) |
  | 1d | 95% | 96.0% [95.2, 97.0] | 92.4% | **no** (conservadora) |

  - **Criterio 2 cumplido a 7d (horizonte primario).** A 1d las bandas cubren ~1 pp de más
    (conservadoras: más anchas de lo necesario); no se ajustan para no sobreajustar.
  - Con los multiplicadores gaussianos la banda «del 95%» cubre 92.2% a 7d; los conformes
    (k medios a 7d: 0.87 / 1.87 / 2.68) corrigen esa subcobertura. Por año a 7d el 68% oscila
    entre 61.7% (2024) y 72.3% (2025): con ~52 semanas independientes por año el ruido es de
    ±6 pp.
- **Paridad:** diferencia máxima de entradas 0 en todo el historial y 5e-14 en ventanas
  finales; diferencia relativa máxima del pronóstico 1.6e-15. **Criterio 3 cumplido.**
- **Conclusión:** `har_x_live` queda **aceptado** para producción a 7d: sin OI iguala a HAR-X,
  mejora a HAR y sus bandas conformes tienen la cobertura nominal. A 1d la ventaja sobre HAR
  no está demostrada y las bandas son algo conservadoras. Pendiente: sombra prospectiva de 12
  semanas (criterio 4) antes de llamarlo «validado».

---

## Direcciones abiertas

1. **Producto de riesgo basado en E-12:** exponer el pronóstico HAR-X de volatilidad a 7 días
   y el rango esperado de precio (±σ̂·√(7/365)). Antes de mostrarlo como "validado", evaluar
   la cobertura de esos intervalos (p. ej. que el 68% de las semanas caiga dentro de ±1σ̂) y
   recalibrar la pendiente de Mincer-Zarnowitz (0.76).
2. **Holdout de noviembre (E-06):** evaluarlo a partir del 2026-12-02 tal como está congelado.
3. **Datos de pago o de alta frecuencia** (skew histórico, libro de órdenes, liquidaciones):
   la única vía direccional que no se ha probado.
4. **No priorizar:** ensembles, deep learning o más features gratuitas para dirección (E-07 a
   E-11 cubren ese espacio, con PBO de 0.63–0.78).

## Plantilla para nuevas entradas

```markdown
## E-XX Título

- **Artefacto:** `ml/artifacts/...` · Comando: `python -m ...` · Código: [archivo](...)
- **Hipótesis:** ...
- **Método:** horizonte(s), conjuntos, modelos, folds, baselines, primario preespecificado.
- **Resultados:** tabla con accuracy, diferencias contra baselines, límites simultáneos, White,
  PBO, economía neta y cobertura.
- **Conclusión:** ...
- **No repetir / reabrir solo si:** ...
```
