# 1. Qué ofrece el proyecto y qué no

## Lo que sí puedes obtener

### 1.1 Cuánto puede moverse BTC (volatilidad pronosticada)

- **Qué es:** la volatilidad anualizada esperada para el próximo día y la próxima semana.
  Ejemplo: 44% anual equivale a un movimiento típico de ±2.3% en un día y ±6.2% en una semana.
- **Qué tan confiable es:** es la única señal del proyecto que supera a sus alternativas con
  significancia estadística. El modelo en producción (HAR-X-live, sin interés abierto) redujo el
  error del pronóstico semanal (QLIKE) de 0.419 (suponer que la próxima semana se parecerá a la
  anterior) a 0.236, y mejora al modelo base HAR con un intervalo de confianza que excluye el
  cero (E-12b). **A 1 día** también supera al ingenuo, pero no se demostró ventaja sobre el
  modelo base HAR. En vivo sigue **en validación** (sombra prospectiva de 12 semanas).
- **Para qué sirve:** decidir **cuánto** comprar, **dónde** poner el stop y **cuándo** reducir
  la exposición. No dice si el precio subirá o bajará.

### 1.2 Rango probable de precio

- **Qué es:** el intervalo donde, históricamente, cae el precio en el 68%, 90% o 95% de los
  casos, con multiplicadores calibrados de forma conforme sobre resultados reales (no una
  campana de Gauss). Con BTC en 77,582 USD y 44.4% de volatilidad anual, el rango a 7 días era:

  | Probabilidad histórica | Rango | Distancia |
  |---|---|---|
  | 68% | 73,541 – 81,845 | ±5.3% |
  | 90% | 69,155 – 87,036 | ±11.5% |
  | 95% | 65,795 – 91,480 | ±16.5% |

  Fuera de muestra (desde 2023-03) a 7 días cubrieron 68.4%, 91.1% y 96.0%; con multiplicadores
  gaussianos la banda «del 95%» habría cubierto solo 92.2% (E-12b).

- **Advertencia:** «95%» significa que **1 de cada 20 semanas** el precio cierra fuera del
  rango. Los movimientos extremos ocurren: en 2022–2026, la peor semana fue −30.4% y la mejor
  +32.1%.

### 1.3 Régimen de riesgo

Clasifica el momento según el percentil histórico de la volatilidad pronosticada:

| Régimen | Volatilidad anual | Qué implica |
|---|---|---|
| Bajo | < 38.7% | Movimientos pequeños; stops más cercanos, posiciones algo mayores |
| Normal | 38.7–51.0% | Situación habitual |
| Alto | 51.0–59.3% | Reducir tamaño; ampliar stops |
| Extremo | > 59.3% | Ocurre el 10% del tiempo; posiciones pequeñas o fuera del mercado |

Los umbrales se guardan con cada modelo y se recalculan al reentrenar; los de arriba son los del
modelo a 7 días entrenado con datos hasta 2026-10-01.

### 1.4 Datos y transparencia

- Precio, velas de 4h, features usadas por el modelo y el historial de predicciones con su
  resultado real.
- Todo se expone en la API (ver [cómo leer y usar los datos](./03-como-leer-y-usar-los-datos.md)),
  así que puedes analizarlo en una hoja de cálculo o en Python.

### 1.5 Evidencia documentada de qué funciona y qué no

Los 13 experimentos (E-01 a E-12b) de [docs/research](../research/experiment-log.md) te ahorran probar ideas
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
