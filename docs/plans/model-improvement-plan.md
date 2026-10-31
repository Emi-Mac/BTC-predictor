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

## 5. Expectativas realistas

- Todas las velas, horizonte 24h: 54–56% de accuracy.
- Cobertura selectiva del 25–40%: 57–62% de accuracy.
- Resultados superiores a esos rangos deben tratarse como sospecha de leakage hasta demostrar lo contrario.
- La viabilidad depende del profit factor neto, no solo del accuracy.
