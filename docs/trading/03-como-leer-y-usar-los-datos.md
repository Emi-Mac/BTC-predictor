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

> El dashboard y la API descritos aquí están en la rama `feature/fase4-desarrollo-API` (ver su
> `docs/runbooks/full-stack-runbook.md` para levantarlos). Llegarán a esta rama con la fase F0
> del [plan de integración](../plans/volatility-integration-plan.md).

| Panel | Qué muestra | Cómo usarlo | Error común |
|---|---|---|---|
| Encabezado | Precio de la última vela y cambio porcentual; estado de conexión | Contexto | Tomarlo como precio en tiempo real: es el cierre de la última vela de 4h |
| Gráfico | Velas de 4h y flechas de predicción con su confianza | Ver el historial de señales sobre el precio | Juzgar el modelo por unas cuantas flechas vistosas |
| Predicción actual | Dirección (UP/DOWN), confianza y tiempo al cierre que predice | Solo observación (señal experimental) | Pensar que 55% es «casi seguro» |
| Métricas | Acierto de las predicciones resueltas en 7 días, 30 días y total | Vigilar si el desempeño en vivo se aleja de ≈54% | Sacar conclusiones con menos de 100 predicciones |
| Features actuales | Valores de entrada del modelo | Transparencia y depuración | Usarlas como indicadores de trading por separado |
| Historial | Tabla de predicciones con resultado real | Calcular tu propio acierto por periodo | Ignorar las rachas: 5 fallos seguidos son normales con 54% |

**Panel de riesgo (tras la integración planificada):**

| Panel | Qué muestra | Cómo usarlo |
|---|---|---|
| Volatilidad esperada | Volatilidad anual y su equivalente diario y semanal; régimen | Ajustar tamaño de posición y distancia de stops |
| Rango probable | Bandas del 68/90/95% a 1 y 7 días | Poner stops fuera de la banda del 68% del horizonte de tu operación |
| Pronóstico vs realizado | Historial del error del pronóstico | Confiar más o menos en el pronóstico según su desempeño reciente |
| Calibración | Porcentaje real de veces que el precio quedó dentro de cada banda | Si la banda del 68% contiene el precio mucho menos del 68%, amplía los stops |
| Calculadora | Tamaño de posición según capital, riesgo y stop | Ver [documento 5](./05-ordenes-y-tamano-de-posicion.md) |

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

Usa los multiplicadores **empíricos** de BTC (medidos sobre 8,766 pronósticos de 2022–2026; son
descriptivos hasta completar la validación prospectiva), no los de una distribución normal:

| Cobertura deseada | k gaussiano | k empírico (HAR-X, 7d) |
|---|---|---|
| 68% | 1.00 | 0.86 |
| 90% | 1.645 | 1.77 |
| 95% | 1.96 | 2.52 |

El 95% empírico es mucho más ancho que el gaussiano: BTC tiene **colas gruesas** (movimientos
extremos más frecuentes de lo que predice la campana de Gauss).

**Ejemplo completo** (precio 77,582 USD, σ anual 44.4%):

| Horizonte | 68% | 90% | 95% |
|---|---|---|---|
| 1 día | 76,047 – 79,148 | 74,455 – 80,840 | 73,169 – 82,261 |
| 7 días | 73,586 – 81,795 | 69,582 – 86,502 | 66,446 – 90,585 |

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
| `GET /api/v1/volatility/current?horizon=7d` | *(planificado)* Volatilidad, régimen y bandas |
| `GET /api/v1/risk/position-size?...` | *(planificado)* Calculadora de tamaño |

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
