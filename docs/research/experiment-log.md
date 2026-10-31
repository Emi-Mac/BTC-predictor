# Registro de experimentos

Registro cronológico de todo lo intentado para mejorar la predicción de dirección de BTC.
Cada entrada es evidencia: hipótesis, método, resultado, artefacto y decisión. Los artefactos
viven en `ml/artifacts/` (no versionados); las cifras se copiaron de sus `report.json`,
`ablation.json` o `walk_forward.json`. El recuento de ensayos desde la Fase 0 está en
`ml/artifacts/experiments.jsonl`.

**Estado global (2026-10-30):** ningún modelo supera el gate; el modelo registrado permanece
**inactivo**. La única regularidad robusta es una leve reversión de una vela a 4h (≈54% de
acierto), que una regla condicional trivial iguala y que los costos anulan.

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

## Direcciones abiertas (no probadas)

1. **Predecir volatilidad en lugar de dirección.** La volatilidad sí es persistente y
   predecible. Las estrategias largo/plano de E-08 y E-09 reducen el drawdown (−32% contra
   −53%) con un Sharpe similar a *buy & hold*. Un modelo de volatilidad para dimensionar la
   exposición (*volatility targeting*) es la hipótesis con mejor fundamento.
2. **Clasificación de régimen** (bajista o alcista) como variable de control, en lugar de
   predecir la dirección vela a vela. E-09 muestra un desempeño muy dependiente del régimen.
3. **Holdout de noviembre (E-06):** evaluarlo después del 2026-12-02 tal como está congelado,
   aunque se espera ≈ baseline.
4. **Fase 3 del plan (ensembles, meta-labeling):** baja probabilidad previa; sin señal en las
   features, combinar modelos no la crea. No priorizar.

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
