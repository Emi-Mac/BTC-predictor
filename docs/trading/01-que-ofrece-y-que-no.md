# 1. Qué ofrece el proyecto y qué no

## Lo que sí puedes obtener

### 1.1 Cuánto puede moverse BTC (volatilidad pronosticada)

- **Qué es:** la volatilidad anualizada esperada para el próximo día y la próxima semana.
  Ejemplo: 44% anual equivale a un movimiento típico de ±2.3% en un día y ±6.2% en una semana.
- **Qué tan confiable es:** es la única señal del proyecto que supera a sus alternativas con
  significancia estadística. HAR-X redujo el error del pronóstico semanal (QLIKE) de 0.419
  (suponer que la próxima semana se parecerá a la anterior) a 0.237, con un intervalo de
  confianza que excluye el cero.
- **Para qué sirve:** decidir **cuánto** comprar, **dónde** poner el stop y **cuándo** reducir
  la exposición. No dice si el precio subirá o bajará.

### 1.2 Rango probable de precio

- **Qué es:** el intervalo donde, históricamente, cae el precio en el 68%, 90% o 95% de los
  casos. Con BTC en 77,582 USD y 44.4% de volatilidad anual, el rango a 7 días era:

  | Probabilidad histórica | Rango | Distancia |
  |---|---|---|
  | 68% | 73,586 – 81,795 | ±5.3% |
  | 90% | 69,582 – 86,502 | ±10.9% |
  | 95% | 66,446 – 90,585 | ±15.5% |

- **Advertencia:** «95%» significa que **1 de cada 20 semanas** el precio cierra fuera del
  rango. Los movimientos extremos ocurren: en 2022–2026, la peor semana fue −30.4% y la mejor
  +32.1%.

### 1.3 Régimen de riesgo

Clasifica el momento según el percentil histórico de la volatilidad pronosticada:

| Régimen | Volatilidad anual | Qué implica |
|---|---|---|
| Bajo | < 38.6% | Movimientos pequeños; stops más cercanos, posiciones algo mayores |
| Normal | 38.6–51.0% | Situación habitual |
| Alto | 51.0–59.5% | Reducir tamaño; ampliar stops |
| Extremo | > 59.5% | Ocurre el 10% del tiempo; posiciones pequeñas o fuera del mercado |

### 1.4 Datos y transparencia

- Precio, velas de 4h, features usadas por el modelo y el historial de predicciones con su
  resultado real.
- Todo se expone en la API (ver [cómo leer y usar los datos](./03-como-leer-y-usar-los-datos.md)),
  así que puedes analizarlo en una hoja de cálculo o en Python.

### 1.5 Evidencia documentada de qué funciona y qué no

Los 12 experimentos de [docs/research](../research/experiment-log.md) te ahorran probar ideas
que ya fallaron: más indicadores técnicos, on-chain, opciones, horizontes más largos, modelos
más complejos.

## Lo que no ofrece

| Expectativa | Realidad según la evidencia |
|---|---|
| «Me dice si BTC va a subir» | La predicción direccional acierta ≈54% a 4h, igual que la regla «después de una vela verde, la siguiente tiende a bajar un poco». Con comisiones, cada operación pierde en promedio (ver documento 2). |
| «Con más confianza del modelo, más seguro» | Una confianza de 55% significa 55%, no «muy probable». El modelo casi nunca sale de 45–60%. |
| «Las bandas son un soporte o resistencia» | Son un rango estadístico de movimiento, no niveles donde el precio «rebota». |
| «Puedo automatizarlo y ganar» | No hay ninguna estrategia del proyecto que supere a comprar y mantener en retorno ajustado por riesgo. |
| «La volatilidad baja significa que es seguro» | Los periodos tranquilos suelen terminar con movimientos bruscos. Bajo riesgo pronosticado no es ausencia de riesgo. |

## Para quién es útil

- **Inversionista de largo plazo** que compra y mantiene BTC: para dimensionar compras
  periódicas, decidir cuánto de su patrimonio exponer y entender qué caída es «normal».
- **Trader discrecional:** para calibrar stops y tamaño con datos en lugar de intuición.
- **Desarrollador o investigador:** como base reproducible para probar ideas nuevas con un
  protocolo riguroso.

No es útil para quien busca una señal de compra/venta lista para seguir a ciegas.
