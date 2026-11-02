# 3. Cómo leer y usar los datos

## 3.1 Conceptos de tiempo

- Las velas son de **4 horas** y se alinean en UTC: 00:00, 04:00, 08:00, 12:00, 16:00 y 20:00
  UTC (en Ciudad de México, UTC−6: 18:00, 22:00, 02:00, 06:00, 10:00 y 14:00).
- Una predicción usa la vela **ya cerrada** que abrió a la hora `T` y se refiere a la vela
  siguiente, que cierra en `T + 8h`. Ejemplo: la vela de 12:00–16:00 UTC produce una predicción
  para 16:00–20:00 UTC.
- La predicción aparece entre 5 y 90 minutos después del cierre (depende de cuándo se ingesta
  y se calculan las features).
- La API devuelve horas en UTC (`...Z`); el dashboard las muestra en tu hora local.

## 3.2 Paneles del dashboard

> El dashboard y la API descritos aquí están integrados en esta rama (ver
> `docs/runbooks/full-stack-runbook.md` para levantarlos). El panel de riesgo requiere el
> backend con `VOLATILITY_ENABLED=true`; si no, se reduce a un aviso.

| Panel | Qué muestra | Cómo usarlo | Error común |
|---|---|---|---|
| Encabezado | Precio de la última vela y cambio porcentual; estado de conexión | Contexto | Tomarlo como precio en tiempo real: es el cierre de la última vela de 4h |
| Gráfico | Velas de 4h y flechas de predicción con su confianza | Ver el historial de señales sobre el precio | Juzgar el modelo por unas cuantas flechas vistosas |
| Predicción actual | Dirección (UP/DOWN), confianza y tiempo al cierre que predice | Solo observación (señal experimental) | Pensar que 55% es «casi seguro» |
| Métricas | Acierto de las predicciones resueltas en 7 días, 30 días y total | Vigilar si el desempeño en vivo se aleja de ≈54% | Sacar conclusiones con menos de 100 predicciones |
| Features actuales | Valores de entrada del modelo | Transparencia y depuración | Usarlas como indicadores de trading por separado |
| Historial | Tabla de predicciones con resultado real | Calcular tu propio acierto por periodo | Ignorar las rachas: 5 fallos seguidos son normales con 54% |

**Panel de riesgo** (sección inferior del dashboard; etiqueta «En validación» hasta cerrar la
sombra prospectiva):

| Panel | Qué muestra | Cómo usarlo |
|---|---|---|
| Volatilidad esperada | Volatilidad anual y su equivalente diario y semanal; régimen; DVOL y volatilidad realizada como referencia | Ajustar tamaño de posición y distancia de stops |
| Rango probable (y líneas en el gráfico) | Bandas del 68/90/95% a 1 y 7 días; en el gráfico, 68% y 95% a 7 días desde la última vela | Poner stops fuera de la banda del 68% del horizonte de tu operación |
| Pronosticada vs realizada | Volatilidad pronosticada a 7 días frente a la que ocurrió después | Confiar más o menos en el pronóstico según su desempeño reciente |
| Calibración | Cobertura real de cada banda frente a su nivel nominal (histórica y últimas 12 semanas), QLIKE frente al ingenuo | Si la cobertura se desvía más de 5 pp (alerta ámbar), tómalas con cautela y amplía los stops |
| Calculadora | Tamaño de posición según capital, riesgo (0.25–2%) y stop (k·σ) | Ver [documento 5](./05-ordenes-y-tamano-de-posicion.md) |

Si el panel avisa de **pronóstico de respaldo**, faltaba una fuente (DVOL, funding, etc.) y se usó
un modelo más simple; trátalo con más cautela. Ningún elemento del panel es una señal de compra
o venta, y por eso no usa los colores verde y rojo.

## 3.3 Cómo interpretar la predicción direccional

- **Confianza** es la probabilidad estimada de que la siguiente vela cierre por encima (UP) o
  por debajo (DOWN) del cierre actual. Está calibrada: cuando el modelo dice 55%, acierta cerca
  del 55% de las veces. No dice **cuánto** subirá o bajará.
- El modelo casi nunca supera el 60%. Una confianza alta del modelo no es una oportunidad
  excepcional.
- Para ser rentable a 4h necesitarías un acierto de 58–66% (ver [documento 2](./02-viabilidad-en-trading-real.md)).
- **Uso aceptable:** como dato secundario, por ejemplo para no ejecutar una compra ya planeada
  justo cuando el modelo predice DOWN con confianza alta, sabiendo que la evidencia de que eso
  ayude es nula. **Uso no aceptable:** abrir o cerrar posiciones por la flecha.

## 3.4 Cómo interpretar la volatilidad

La volatilidad se expresa **anualizada**. Para convertirla a tu horizonte:

```text
σ(horizonte) = σ_anual × √(días / 365)
```

| σ anual | 1 día | 7 días | 30 días |
|---|---|---|---|
| 34% (régimen bajo, p10) | 1.78% | 4.71% | 9.75% |
| 44.4% (mediana) | 2.32% | 6.15% | 12.73% |
| 59.5% (régimen alto, p90) | 3.11% | 8.24% | 17.06% |
| 69% (extremo, p97) | 3.61% | 9.56% | 19.78% |

**Rango probable** alrededor del precio `P`:

```text
límite inferior = P × exp(−k × σ(horizonte))
límite superior = P × exp(+k × σ(horizonte))
```

Usa los multiplicadores **conformes** de BTC (cuantiles de los resultados reales ya resueltos,
en ventana móvil de 2 años; E-12b), no los de una distribución normal. El dashboard los calcula
en vivo y los recalcula con cada pronóstico; la tabla muestra su valor medio fuera de muestra:

| Cobertura deseada | k gaussiano | k conforme 1 día | k conforme 7 días |
|---|---|---|---|
| 68% | 1.00 | 0.87 | 0.87 |
| 90% | 1.645 | 1.85 | 1.87 |
| 95% | 1.96 | 2.50 | 2.68 |

El 95% empírico es mucho más ancho que el gaussiano: BTC tiene **colas gruesas** (movimientos
extremos más frecuentes de lo que predice la campana de Gauss).

**Ejemplo completo** (precio 77,582 USD, σ anual 44.4%, multiplicadores de la tabla):

| Horizonte | 68% | 90% | 95% |
|---|---|---|---|
| 1 día | 76,029 – 79,167 | 74,317 – 80,990 | 73,203 – 82,223 |
| 7 días | 73,541 – 81,845 | 69,155 – 87,036 | 65,795 – 91,480 |

Cobertura fuera de muestra (desde 2023-03, conforme): a 7 días 68.4% / 91.1% / 96.0%; a 1 día
67.5% / 91.1% / 96.0% (a 1 día las bandas de 90% y 95% son ligeramente conservadoras: cubren
~1 pp de más).

## 3.5 Datos históricos útiles para calibrar expectativas

Movimientos reales de BTC entre sep-2022 y ago-2026 (retornos logarítmicos):

| Horizonte | Movimiento medio | Mediana | 90% de las veces menor a | Peor | Mejor | % de veces que subió |
|---|---|---|---|---|---|---|
| 4 horas | 0.63% | 0.39% | 1.50% | −8.4% | +8.1% | 51.2% |
| 1 día | 1.68% | 1.15% | 3.87% | −17.1% | +16.4% | 52.0% |
| 7 días | 4.72% | 3.35% | 10.60% | −30.4% | +32.1% | 53.9% |
| 30 días | 10.62% | 8.11% | 23.94% | −39.9% | +46.2% | 58.1% |

El 11.5% de las semanas BTC se movió más de 10%; el 1.5%, más de 20%. La mayor caída desde un
máximo en el periodo fue −53%.

## 3.6 Usar la API

La API está en `http://localhost:8000` (documentación interactiva en `/docs`).

| Endpoint | Para qué |
|---|---|
| `GET /api/v1/candles?limit=1000&offset=0` | Velas históricas de 4h (máx. 1000 por página) |
| `GET /api/v1/candles/latest` | Última vela y su variación |
| `GET /api/v1/predictions/current` | Predicción vigente |
| `GET /api/v1/predictions/history?limit=500&offset=0` | Predicciones y resultados (máx. 500 por página) |
| `GET /api/v1/metrics/accuracy` | Acierto agregado y por mes |
| `GET /api/v1/metrics/model` | Modelo activo y sus métricas de validación |
| `GET /api/v1/features/latest` | Features de la última vela |
| `GET /api/v1/volatility/current?horizon=1d\|7d` | Volatilidad, régimen, bandas 68/90/95% (`null` sin calibración suficiente), DVOL, realizada 7d/30d, modelo y motivo de respaldo, `validation_status` |
| `GET /api/v1/volatility/history?horizon=&limit=&offset=` | Pronóstico frente a realizado por vela |
| `GET /api/v1/metrics/volatility?horizon=` | Cobertura real, QLIKE (y del ingenuo), pendiente de Mincer-Zarnowitz y últimas 12 semanas |
| `GET /api/v1/risk/position-size?capital=&risk_pct=&stop_k=&horizon=&side=` | Calculadora de tamaño (rechaza riesgo > 5% y valores no positivos) |
| WebSocket `/ws/live` | Mensaje `volatility` con cada pronóstico nuevo |

Los endpoints de volatilidad responden 404 si el backend tiene `VOLATILITY_ENABLED=false` o aún
no hay pronósticos. `/metrics/model` y `/predictions/*` incluyen `status` («experimental»),
`beats_baseline` y la comparación contra la regla trivial.

Ejemplo en Python para medir tu propio acierto por mes:

```python
import httpx
import pandas as pd

API = "http://localhost:8000/api/v1/predictions/history"
rows, offset = [], 0
while True:  # la API devuelve como máximo 500 por página
    page = httpx.get(API, params={"limit": 500, "offset": offset}).json()
    rows += page["data"]
    if not page["pagination"]["has_more"]:
        break
    offset += 500
df = pd.DataFrame(rows).dropna(subset=["is_correct"])
df["month"] = pd.to_datetime(df["candle_open_time"]).dt.to_period("M")
print(df.groupby("month")["is_correct"].agg(["mean", "size"]))
```

Para exportar a Excel o Google Sheets: `df.to_csv("predicciones.csv", index=False)`.
