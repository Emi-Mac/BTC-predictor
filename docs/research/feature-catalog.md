# Catálogo de features

Todas se calculan con información disponible al cierre de la vela `t` (instante de decisión
`open_time + 4h`). En el código, las columnas están en
[compute.py](../../ml/src/features/compute.py) (`FEATURE_COLUMNS`),
[price_context.py](../../ml/src/features/price_context.py),
[taker_flow.py](../../ml/src/features/taker_flow.py),
[funding_context.py](../../ml/src/features/funding_context.py),
[metrics_features.py](../../ml/src/research/metrics_features.py) y
[long_horizon.py](../../ml/src/features/long_horizon.py) (`FAMILIES`).

## Conjuntos usados en los experimentos

| Conjunto | Columnas | Definición |
|---|---|---|
| `existing` / `legacy` | 46 | `FEATURE_COLUMNS`: indicadores técnicos, macro diaria, Fear & Greed, calendario |
| `expanded` | 67 | `existing` + `PRICE_CONTEXT_COLUMNS` |
| `base` | 70 | `expanded` + funding (`funding_rate`, `funding_mean_3`, `funding_change_1`) |
| `base_oi` | 75 | `base` + OI de 4h/24h (E-06) |
| `stationary` | 59 | `base` sin 16 niveles absolutos + 5 sustitutos estacionarios (Fase 2) |
| `stationary_<familia>` | 65–75 | `stationary` + una familia de la Fase 2 |
| `all` | 111 | `stationary` + las cinco familias de la Fase 2 |

## Indicadores originales (`FEATURE_COLUMNS`)

RSI 14, MACD (línea, señal e histograma), bandas de Bollinger (superior, media, inferior y
ancho), EMA 9/21/50, SMA 200, ATR 14, volatilidad histórica 20, rango alto-bajo, OBV, ratio
de volumen contra su SMA, VWAP diario, acumulación/distribución, ROC 5/10, Williams %R,
Stoch-RSI, distancia a EMA 50, pendiente de EMA 21, máximos crecientes y mínimos decrecientes
en 20 velas, DXY (nivel y cambio), S&P 500 (nivel y cambio), VIX, cambio del oro, correlación
BTC-S&P de 20 días, Fear & Greed (valor, SMA 7 y momentum 7), hora, fin de semana y día de la
semana.

**Observado:** los niveles absolutos (`sp500_close`, `obv`, `ema_*`) dominaban la importancia
en XGBoost (`sp500_close` con 0.28 en E-01): señal espuria de régimen. Desde la Fase 2 se
excluyen del conjunto `stationary`.

## Contexto de precio (`PRICE_CONTEXT_COLUMNS`, 21)

Retornos de 1, 3, 6, 12 y 24 velas; rezagos del retorno de 1 vela (1, 2, 3 y 6); cuerpo,
posición del cierre y mechas; cambio de volumen; ratio de volatilidad 6/24; distancia a VWAP,
EMA 9 y EMA 21; MACD y ATR relativos; posición en Bollinger.
**Observado:** es lo único que movió la accuracy a 4h (+2 pp en E-07: `existing` 52.0% →
`base` 54.0%). Su contenido es reversión de corto plazo y no supera a la regla condicional.

## Flujo agresor (`TAKER_FEATURE_COLUMNS`) — E-04

Ratio de compra agresora (1, 6 y 24 velas), en cantidad base y en quote, y desequilibrio.
**Observado:** −0.24 pp de accuracy media; sin aporte.

## Funding (`FUNDING_FEATURE_COLUMNS`) — E-04

Tasa vigente, media de 3 eventos y cambio. **Observado:** +0.22 pp (53.37% contra 53.15%),
exploratorio y dentro del ruido. Se mantiene en `base`.

## Interés abierto y posicionamiento — E-06

`oi_log_change_4h/24h`, `oi_value_log_change_4h/24h`, interacción OI×retorno, ratios
long/short (cuentas, *top traders* por cuentas y por posiciones) y ratio agresor de futuros,
con 48h de demora. **Observado:** −0.16 pp (OI) y −0.28 pp (posicionamiento) frente a `base`.

## Familias de la Fase 2 (`long_horizon.py`) — E-09

Resultado de la ablación pareada (frente a `stationary`, logística L2 con C=0.01, mismos 48
folds). Ninguna familia cumplió la regla de conservación: límites simultáneos positivos en
log-loss **y** accuracy.

| Familia | Columnas | Disponibilidad | Δ accuracy 24h | Δ accuracy 3d | Δ log-loss 3d | Conservada |
|---|---|---|---|---|---|---|
| C1 `stationary` (vs `base`) | `distance_sma200`, `obv_flow_42`, `ad_flow_42`, `relative_macd_signal`, `relative_macd_histogram` | Velas del dataset | +0.17 pp | **+2.78 pp** (IC [−1.0, +6.6]) | mejora 0.0073 | No (límite < 0) |
| C2 `regime` | Retornos de 3/7/14/30/90 días, distancia a MA de 50/200 días, spread de MA, volatilidad realizada de 7/30 días, ratio y percentil anual, drawdown de 90 días/1 año, eficiencia de 30 días, posición en el rango de 30 días | Velas spot 4h (historia desde 2019-09) | −0.34 pp | +0.72 pp | peor | No |
| C3 `derivatives` | Funding acumulado de 3/7/30 días y z-score de 90 días; cambio de OI de 3/7/30 días; z-scores de long/short; ratio agresor de 7 días; basis spot-perpetuo (nivel, media de 7 días, z de 30 días); prima media de 7 días | Funding +4h; métricas +48h; perpetuo al cierre | −0.47 pp | −0.74 pp | peor | No |
| C4 `onchain` | MVRV y su cambio de 30 días; cambios de 30 días en direcciones activas, transacciones, hash rate y comisiones; ratio de flujo neto a exchanges de 7 días; cambio de oferta en exchanges; oferta de stablecoins a 7/30 días | CoinMetrics D+2 | −0.51 pp | −0.72 pp | peor | No |
| C5 `events` | Días al próximo FOMC y desde el último; FOMC en ±72h; días al vencimiento mensual de opciones; semana de vencimiento trimestral | Calendario programado | −0.49 pp | +0.43 pp | peor | No |
| C6 `cross_asset` | Retorno ETH/BTC de 7/30 días; retornos de 20 días del S&P 500, DXY y oro; z-score de 60 días del VIX | ETHBTC al cierre; macro D+1 | +0.15 pp | −0.49 pp | peor | No |
| Todas (`all`) | 111 columnas | — | −0.41 pp | −0.55 pp | mucho peor (−0.043) | No |

| C3b `options` (E-10, snapshot v2) | `dvol`, `dvol_change_1d`, `dvol_change_7d`, `dvol_zscore_90d`, `variance_risk_premium_30d` (DVOL − volatilidad realizada de 30 días), `implied_realized_ratio_7d` | DVOL horario al cerrar la hora | −0.51 pp | +0.67 pp | peor | No |

**Uso en volatilidad (E-12):** el DVOL y las variables de HAR-X (VIX, funding de 7 días, OI de
7 días, drawdown, |retorno de 1 día|, volumen relativo, FOMC) **sí** mejoran el pronóstico de
la volatilidad a 7 días frente a HAR (QLIKE −0.025, IC > 0). Las features que no anticipan la
dirección sí informan sobre el riesgo.

**Estabilidad de coeficientes (área D):** con todas las familias, los coeficientes de mayor
magnitud conservan el signo en los 48 folds. A 3d destacan el percentil anual de volatilidad
(+), la volatilidad de 30 días (−), el retorno de 90 días (+), el drawdown de 90 días (−), el
funding de 7 días (−), ETH/BTC de 7 días (−) y FOMC en las últimas 72h (−). Aun así fracasan
fuera de muestra. Con ventanas expansivas casi idénticas entre folds, la estabilidad del signo
**no** indica poder predictivo; refleja relaciones del período inicial que no persisten.

## Ideas de features aún no probadas

- Calendario del CPI, NFP y otros datos macro (fuente bloqueada; ver
  [data-sources.md](./data-sources.md)).
- Flujos de ETF spot (historia insuficiente).
- Skew y volatilidad implícita de opciones (Deribit DVOL): no se capturaron.
- Liquidaciones agregadas: sin fuente histórica gratuita identificada.
- Dominancia de BTC: CoinMetrics Community no la ofrece directamente.
