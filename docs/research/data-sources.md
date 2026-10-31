# Fuentes de datos

Todas las fuentes son gratuitas. Ninguna tiene verificación point-in-time completa: las
conclusiones pueden estar sesgadas si la fuente revisó valores históricos.

## Dataset congelado `btcusdt-20200901-20261001-v1`

Ubicación: `ml/artifacts/datasets/btcusdt-20200901-20261001-v1/`, verificable con
`python -m src.research.dataset verify`. Filas de features: 2020-10-04 a 2026-09-30.

| Fuente | Contenido | Disponibilidad asumida | Limitaciones |
|---|---|---|---|
| Binance spot klines BTCUSDT 4h | OHLCV, volumen agresor | Al cierre de la vela | Sin velas sintéticas; completitud 100% |
| yfinance diario | DXY, S&P 500, VIX, oro | Cierre del día D desde D+1 00:00 UTC; propagación ≤ 5 días | Fines de semana y festivos sin dato |
| Alternative.me | Fear & Greed diario | Mismo día | Publicación histórica no verificada |
| Binance funding BTCUSDT | Tasa cada 8h | Una vela de 4h tras la liquidación; antigüedad ≤ 36h | — |
| Binance futures metrics (archivo diario) | OI, ratios long/short, ratio agresor | Fin de intervalo de 5 min + **48h** de demora simulada; antigüedad ≤ 10 min (E-06) o 12h (Fase 2) | 71 días con huecos; 1,913 de 2,221 archivos modificados después del plazo simulado |

## Snapshots auxiliares `aux-20190901-20261001-v1` (E-09) y `-v2` (E-10 a E-12)

Ubicación: `ml/artifacts/aux-sources/<id>/`. La v2 recaptura todas las fuentes de la v1 y
añade el DVOL de Deribit; sus Parquet de Binance son idénticos a los de la v1. Se captura con
`python -m src.research.aux_sources capture` y se verifica con `... verify`. Es inmutable:
tiene un manifiesto con huellas de cada archivo crudo y de cada Parquet; los ZIP de Binance se
validan con su `.CHECKSUM` oficial. Capturado el 2026-10-30.

| Fuente | Rango | Uso | Disponibilidad asumida |
|---|---|---|---|
| Binance spot BTCUSDT 4h (data.binance.vision) | 2019-09 → 2026-09 | Historia previa para ventanas de hasta 1 año; se valida que coincide con el dataset en el solapamiento | Al cierre de la vela |
| Binance perpetuo BTCUSDT 4h | 2020-01 → 2026-09 (no hay archivos 2019-09…12) | Basis spot-perpetuo | Al cierre de la vela |
| Binance premium index 4h | 2020-01 → 2026-09 | Prima media de 7 días | Al cierre de la vela |
| Binance spot ETHBTC 4h | 2019-09 → 2026-09 | Retornos relativos ETH/BTC | Al cierre de la vela |
| CoinMetrics Community API (diario) | 2019-09 → 2026-09 | MVRV, direcciones activas, transacciones, hash rate, comisiones, flujos y oferta en exchanges; oferta de USDT+USDC | Valor del día D desde **D+2 00:00 UTC**; antigüedad ≤ 5 días. **Revisiones históricas desconocidas** |
| Deribit DVOL (`public/get_volatility_index_data`, solo v2) | 2021-03-24 → 2026-09-30, horario | Volatilidad implícita a 30 días; familia `options` y modelo HAR-X | Cierre de la hora disponible al terminar la hora; antigüedad ≤ 6h. Antes de 2021-03-24 se imputa la mediana del train |
| Reserva Federal: calendario FOMC | Reuniones programadas 2020–2027 | Días al próximo FOMC y desde el último | Calendario publicado con antelación; decisión a las 14:00 de Nueva York (18:00/19:00 UTC). Se excluyen votaciones por notación y reuniones extraordinarias o canceladas |

## Fuentes intentadas y no disponibles

| Fuente | Motivo | Cuándo reintentar |
|---|---|---|
| Calendario de publicación del CPI (bls.gov) | HTTP 403 desde este entorno | Con un proxy, una descarga manual del calendario o una clave de la API de FRED (`release/dates`) |
| FRED / ALFRED (fechas de vintage del CPI) | Sin conexión (timeout) desde este entorno | Igual que arriba |
| Flujos de ETF spot de BTC | Sin API gratuita estable; historia solo desde ene-2024 (~1/3 del período) | Cuando haya ≥ 3 años de historia o una fuente con descarga estable (p. ej. un CSV de Farside guardado como snapshot) |
| Skew, put/call y superficie de volatilidad histórica de Deribit | La API pública solo da valores actuales; la historia es de pago (p. ej. Tardis) | Si se contrata un proveedor de datos de opciones |
| Interés abierto reciente vía API REST de Binance | El endpoint rechaza fechas antiguas | Se sustituyó por el archivo diario de métricas |

## Cambios de disponibilidad que invalidarían resultados

- Usar CoinMetrics con una demora menor de D+2 sin evidencia de publicación.
- Reducir la demora de 48h de las métricas de futuros sin verificar las fechas de publicación.
- Rellenar huecos con valores posteriores (prohibido en todas las fuentes).
