# Plan de mejora del modelo de predicción BTC

> **Alcance:** todas las áreas (A–I). Horizonte: comparar 4h / 12h / 24h / 3d.
> **Principio rector:** ninguna mejora se acepta sin evidencia walk-forward con potencia estadística suficiente y sin tocar el holdout congelado.

## 1. Diagnóstico (estado a 2026-10-30)

| Experimento | Accuracy | Mejor baseline | Lectura |
|---|---|---|---|
| Walk-forward producción (XGB / LSTM) — `run-20261029T065504Z` | 49.7% / 49.9% | 53.0% | Por debajo del baseline; criterios no superados |
| Research logística — `research-20261029T070840Z` (desarrollo) | 53.1% | 52.2% | +0.99 pp, lejos de +3 pp |
| Holdout octubre (168 velas) | 55.4% | 48.2% | IC Wilson 95% [47.8%, 62.7%]: no concluyente |
| Ablaciones OI / taker / funding | −0.3 a +0.2 pp | — | Sin efecto medible con el target actual |

Causas raíz identificadas:

1. **Target de máximo ruido:** `close[t+1] > close[t]` a 4h sin zona muerta ([labels.py](../../ml/src/features/labels.py)). Un +0.01% pesa igual que un +3%.
2. **Features no estacionarias:** niveles absolutos (`sp500_close`, `obv`, `ema_*`, `bb_*`, `vwap_daily`, `sma_200`, `accumulation_distribution`) en [compute.py](../../ml/src/features/compute.py). `sp500_close` es la feature más importante en XGB (0.28): señal espuria de régimen.
3. **Selección frágil:** filtro de correlación lineal ≥ 0.02 + MI en [selection.py](../../ml/src/features/selection.py); inestable entre folds.
4. **Datos subutilizados:** producción entrena con ~7,000 velas (desde 2023-08); el dataset versionado cubre desde 2020-09 (13,126 filas). Calibración con ~179 muestras.
5. **Probabilidades planas:** pendiente Platt = 0.11. LSTM con 6 épocas y sin early stopping.
6. **Evaluación sin potencia:** folds de ~180 velas ⇒ error estándar ≈ 3.7 pp por fold. Más de 18 candidatos comparados ⇒ riesgo de selección por azar.

## 2. Redefinir "eficacia"

Perseguir >60% de accuracy en todas las velas de 4h no es realista (probable leakage). Métricas objetivo:

- **Accuracy @ cobertura** (p. ej. accuracy en el 30% de velas con mayor confianza).
- **Brier skill score** y log loss contra el baseline climatológico.
- **Métricas económicas:** profit factor y Sharpe netos de comisiones y slippage, max drawdown.
- Accuracy global se sigue reportando para continuidad con el criterio de Fase 3.

## 3. Áreas de mejora

### A. Rediseño del target (impacto alto · esfuerzo bajo)
- **A1 Horizontes:** 4h, 12h, 24h, 3d. Para horizontes de varias velas, las etiquetas se solapan ⇒ purga y embargo de `h` velas en todos los cortes.
- **A2 Zona muerta por volatilidad:** UP si `r > k·σ`, DOWN si `r < −k·σ` (σ = ATR o volatilidad realizada conocida en `t`). Las velas intermedias se excluyen del entrenamiento o forman una clase "neutral". Barrido de `k ∈ {0, 0.25, 0.5}` dentro del desarrollo.
- **A3 Triple barrier:** take-profit / stop-loss en múltiplos de σ y barrera temporal.
- **A4 Meta-labeling:** un modelo o regla primaria decide la dirección; un modelo secundario decide si tomar la señal.

### B. Predicción selectiva (impacto alto · esfuerzo bajo)
- Curva accuracy vs cobertura por fold.
- Umbral de abstención elegido dentro del walk-forward (ventana de tuning), nunca con el holdout.
- API y UI exponen el estado "sin señal".

### C. Features (impacto medio-alto · esfuerzo medio)
- **C1 Estacionariedad:** sustituir niveles por ratios, z-scores móviles, log-retornos; diferenciación fraccional opcional.
- **C2 Régimen:** contexto 1D/1W, percentil de volatilidad realizada, drawdown desde máximos, tendencia contra rango.
- **C3 Derivados:** basis spot-perp, funding acumulado y en z-score, liquidaciones, skew de opciones. Repetir las ablaciones OI/taker con los nuevos targets.
- **C4 On-chain y flujos:** netflows a exchanges, suministro de stablecoins, MVRV/SOPR, flujos de ETF spot.
- **C5 Eventos:** calendario FOMC/CPI/NFP, sesiones de mercado.
- **C6 Cross-asset:** ETH/BTC, dominancia BTC, Nasdaq.
- Toda fuente nueva requiere verificación point-in-time (retraso de publicación documentado).

### D. Selección de features (impacto medio · esfuerzo bajo)
- Permutation importance o MDA dentro de cada fold.
- Stability selection: conservar features elegidas en ≥ 70% de los folds.
- Conjuntos más pequeños y estables.

### E. Datos (impacto medio · esfuerzo bajo)
- Entrenar con historia desde 2020-09.
- Ponderación temporal (mayor peso a lo reciente) y por unicidad de etiqueta en horizontes solapados.

### F. Modelos (impacto medio · esfuerzo medio-alto)
- GBM con regularización fuerte, early stopping y Optuna en CV anidado.
- Stacking: logística + LightGBM + regla técnica → meta-modelo logístico.
- Modelos por régimen (HMM o clusters de volatilidad) o régimen como feature.
- Deep learning (TCN, PatchTST, TFT) solo tras A y C; alternativa: embeddings de modelos fundacionales (Chronos, TimesFM) como features.

### G. Calibración (impacto alto en uso · esfuerzo bajo)
- Isotónica o Platt ajustada con predicciones fuera de muestra acumuladas de folds previos.
- ECE y diagramas de fiabilidad por fold.

### H. Rigor estadístico (requisito transversal)
- Combinatorial Purged CV (CPCV); holdouts de 3–6 meses.
- Probability of Backtest Overfitting (PBO) y Deflated Sharpe Ratio.
- Registro de todos los experimentos, incluidos los fallidos, para corrección por comparaciones múltiples.
- Mantener el pre-registro existente (`holdout.lock`, `frozen.json`).
- Backtest económico con comisiones y slippage.

### I. Arquitectura MLOps
- Experimentos declarativos (YAML: target, features, modelo, CV) en lugar de candidatos codificados en [research_models.py](../../ml/src/training/research_models.py).
- Tracking de experimentos (MLflow o índice sobre los artefactos actuales).
- Feature sets versionados (`feature_set@version`) desacoplados del dataset.
- Champion/challenger con shadow mode (1–2 meses) antes de promover.
- Monitoreo de drift (PSI), accuracy rolling contra baseline y reentrenamiento condicionado a mejora significativa.

## 4. Hoja de ruta

| Fase | Áreas | Criterio de salida |
|---|---|---|
| 0. Fundamentos de evaluación (1 sem) | H, G, E | El pipeline reporta accuracy @ cobertura, Brier skill, PBO y profit factor con datos 2020+ |
| 1. Target (1–2 sem) | A1, A2, B | Existe un horizonte/umbral con > +3 pp sobre baseline en desarrollo e IC que excluye 0 |
| 2. Features (2–3 sem) | C1, D, luego C2–C6 | Cada familia se conserva solo si mejora log loss y accuracy en CPCV (ablación pareada) |
| 3. Modelos (2 sem) | F, A3, A4 | El ensemble supera al mejor modelo individual en CPCV |
| 4. Producción (1–2 sem) | I | Holdout prospectivo de 2 meses confirma el resultado en shadow mode |

## 4.1 Estado de la Fase 0 (completada 2026-10-30)

Implementación en `ml/src/evaluation/{selective,economic,recalibration,robustness}.py` y
`ml/src/research/evaluation_harness.py`. Informe: `ml/artifacts/eval-baseline-20261030T223358Z-27c9963a/`.
Walk-forward de 48 meses (8,766 predicciones OOS, sep-2022 a ago-2026), objetivo actual a 4h.

| Hallazgo | Evidencia |
|---|---|
| Hay señal débil pero real frente al 50% | Logística `base`: 54.0% (Wilson 53.0–55.1%); estable en CPCV (54.4% ± 0.2 pp entre trayectorias) |
| Esa señal es casi solo reversión a la media de 4h | Baseline condicional P(sube \| dirección previa): 54.0%; diferencia del primario −0.02 pp, IC bloques [−1.0, +1.1] pp; prueba de White p = 0.75 |
| Las features de contexto de precio aportan; OI no | `existing` 52.0% vs `base` 54.0%; `base_oi` 53.9% |
| La selectividad mejora el acierto | Margen causal 0.05 → 52% de cobertura con 55.0%; el 10% más confiable, 57.9% (descriptivo) |
| No es rentable con costos | Sharpe 0 pb ≈ +0.7, con 7 pb ≈ −2.3 (rotación 0.89 por vela); *buy & hold* Sharpe 0.98; DSR ≈ 0 |
| Platt online mejora la calibración; isotónica no | ECE 2.0% → 0.8% (expansiva) / 0.4% (último año) |
| Sobreajuste por selección moderado | PBO accuracy 0.24, PBO Sharpe 0.37 |

**Implicación para la Fase 1:** a 4h la ventaja se consume en costos por la rotación. Se
prioriza un horizonte más largo (12h–3d), la zona muerta por volatilidad y los márgenes de
abstención. La regla condicional es el baseline a batir.

## 4.2 Estado de la Fase 1 (ejecutada 2026-10-30): criterio de salida NO cumplido

Implementación en `ml/src/research/target_study.py` (con `overlapping_positions`,
`causal_margin_selection`, `simultaneous_lower_bounds` y `maximin_pbo`). Informe:
`ml/artifacts/target-study-20261030T230014Z-6953a53e/`. Son 24 configuraciones
(h ∈ {4h, 12h, 24h, 3d} × zona muerta k ∈ {0, 0.25, 0.5}σ√h × {logística, LightGBM}) en los
mismos 48 folds, contra tres baselines por horizonte (mayoría, condicional a la vela previa y
condicional al retorno previo de h velas).

| Hallazgo | Evidencia |
|---|---|
| Los horizontes largos **no** tienen señal con las features actuales | Mejor accuracy frente a "siempre sube": 12h 51.6% vs 51.5%, 24h 51.1% vs 52.0%, 3d 49.4% vs 52.7% |
| La zona muerta no ayuda | A 4h: k=0 54.0%, k=0.5 52.9%; en horizontes largos no cambia la conclusión |
| La única señal sigue siendo la reversión de 4h | Mejor config `h1_k0_logistic`: −0.02 pp frente al condicional; límite simultáneo −5.4 pp |
| Primario preespecificado (24h, k=0.25) | 50.9%; IC frente a los baselines incluye 0 o es negativo; sesgo bajista (36% de predicciones "sube") |
| La abstención causal solo rinde en largo/plano y no supera a *buy & hold* | Mejor: `h1_k0.5_logistic_long_flat`, Sharpe neto 0.79, 8% de exposición, 60% de acierto en señales; *buy & hold* 0.99; White p = 0.997 |
| Largo-corto pierde dinero en todos los horizontes una vez descontado el funding | Sharpe neto de −1.39 a −0.12 |
| Sobreajuste por selección | PBO maximin 0.27, PBO Sharpe 0.29 |

**Conclusión:** cambiar el target no crea señal. Las features actuales son técnicas de corto
plazo y solo capturan reversión de 4h; a 12h–3d no anticipan la dirección. Esto descarta la
expectativa de 54–56% a 24h *con este conjunto de features*.

**Siguiente paso (Fase 2):** antes de nuevos modelos hacen falta features con información de
horizonte largo: estacionarias (C1), de régimen y tendencia en 1D/1W (C2), de derivados
acumulados (C3), on-chain y flujos de ETF (C4) y eventos macro (C5). Evaluarlas con
`target_study` en h ∈ {6, 18} y con el mismo gate. Mientras tanto, la referencia operativa
honesta sigue siendo *buy & hold* o una estrategia largo/plano con muy baja exposición.

## 4.3 Estado de la Fase 2 (ejecutada 2026-10-30): criterio de salida NO cumplido

Implementación:
- `ml/src/research/aux_sources.py`: snapshot inmutable de Binance (perpetuo, prima, ETHBTC,
  historia spot desde 2019), CoinMetrics y el calendario del FOMC.
- `ml/src/features/long_horizon.py`: familias C1–C6 con disponibilidad explícita y un test de
  no-anticipación.
- `ml/src/research/feature_study.py`: ablación pareada, L1 (área D), LightGBM y el mismo gate.

Informe: `ml/artifacts/feature-study-20261030T234559Z-00e2a9e9/`. Detalle completo en
[E-09](../research/experiment-log.md#e-09-ablación-de-familias-de-features-fase-2) y en el
[catálogo de features](../research/feature-catalog.md).

| Hallazgo | Evidencia |
|---|---|
| Ninguna familia (C2–C6) mejora log-loss y accuracy de forma significativa | 14 comparaciones; todos los límites simultáneos < 0 |
| C1 (estacionariedad) corrige el sesgo de los niveles absolutos | 3d: 48.2% → 50.9% (+2.8 pp, IC [−1.0, +6.6]); sin superar a "siempre sube" (52.7%) |
| Mejor configuración: `h18_all_lightgbm` | 52.9%, +0.17 pp en el peor caso; límite simultáneo −4.6 pp; muy dependiente del régimen (2022 61.6%, 2023 49.3%) |
| La selección sobreajusta | PBO maximin 0.63 |
| Economía | Mejor largo/plano: Sharpe 0.94 con drawdown −32% vs *buy & hold* 0.99 con −53%; White p = 0.986 |
| Fuentes no disponibles | Calendario del CPI (BLS 403, FRED sin conexión) y flujos de ETF (sin API ni historia suficiente) |

**Conclusión de las Fases 0–2:** con datos gratuitos no se encontró información direccional
explotable a 4h, 24h ni 3 días. La Fase 3 (ensembles y meta-labeling) tiene baja
probabilidad previa: combinar modelos sin señal no la crea.

**Recomendación:** reorientar el producto de "predecir la dirección" a **gestión de riesgo**.
1. Pronosticar volatilidad, que sí es persistente.
2. Usarla para dimensionar la exposición (*volatility targeting*) y como filtro de régimen.
3. Mantener la predicción direccional como indicador experimental, mostrando de forma
   transparente que no supera a los baselines.

Las direcciones abiertas están en el
[registro de experimentos](../research/experiment-log.md#direcciones-abiertas).

## 4.4 Seguimiento (2026-10-31): cierre de la línea direccional y volatilidad

| Paso | Resultado | Evidencia |
|---|---|---|
| Opciones de Deribit (DVOL) como familia direccional | No aporta (24h −0.5 pp, 3d +0.7 pp, sin significancia; PBO 0.78). **Línea direccional cerrada** | [E-10](../research/experiment-log.md#e-10-volatilidad-implícita-de-opciones-dvol-de-deribit) |
| Reglas clásicas de tendencia | TSMOM 180d largo/plano: Sharpe 0.94, drawdown −33%; igual que el mejor ML y sin superar a *buy & hold* (0.98) | [E-11](../research/experiment-log.md#e-11-reglas-clásicas-de-tendencia-referencia-sin-ml) |
| Pronóstico de volatilidad | **Predecible**: HAR-X (con DVOL) mejora a HAR y al ingenuo con IC > 0 (QLIKE 7d 0.237 vs 0.262 vs 0.419) | [E-12](../research/experiment-log.md#e-12-pronóstico-de-volatilidad-y-exposición-ajustada-por-riesgo) |
| Exposición ajustada por volatilidad | No mejora el Sharpe frente a *buy & hold* ni frente a una exposición constante equivalente | E-12 |
| Producto | Predicción direccional marcada como **experimental** en el frontend (`frontend/src/lib/modelStatus.ts`, `page.tsx`) y en el contrato de tipos (`status`, `beats_baseline`, `baseline_overall`); especificación de las Fases 4 y 5 corregida en [btc-prediction-platform-plan.md](./btc-prediction-platform-plan.md) | — |

**Recomendación vigente:**
1. Mantener la dirección como indicador experimental y transparente.
2. Construir como siguiente funcionalidad un **panel de riesgo**: volatilidad pronosticada a
   7 días con HAR-X y rango esperado de precio. Antes, validar la cobertura de sus intervalos.
   Plan detallado: [volatility-integration-plan.md](./volatility-integration-plan.md). Uso en
   trading real: [docs/trading](../trading/README.md).
3. Evaluar el holdout de noviembre a partir del 2026-12-02.

## 5. Expectativas realistas

- Todas las velas, horizonte 24h: 54–56% de accuracy *(no alcanzado en la Fase 1 con las features actuales: 51.1%)*.
- Cobertura selectiva del 25–40%: 57–62% de accuracy.
- Resultados superiores a esos rangos deben tratarse como sospecha de leakage hasta demostrar lo contrario.
- La viabilidad depende del profit factor neto, no solo del accuracy.
