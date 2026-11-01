# 2. Viabilidad en trading real

Una señal solo es útil si, **después de costos**, deja más dinero que no hacer nada. Este
documento hace esas cuentas con datos reales de BTC (sep-2022 a ago-2026).

## 2.1 Costos reales de operar

Comisiones de referencia de Binance para cuentas sin nivel VIP (verifícalas en la tabla oficial
vigente antes de operar; cambian con el tiempo y por país):

| Producto | Maker (orden límite que espera) | Taker (orden que se ejecuta al instante) |
|---|---|---|
| Spot | 0.10% (0.075% pagando con BNB) | 0.10% (0.075% con BNB) |
| Futuros perpetuos USDⓈ-M | 0.02% | 0.05% |

Además de la comisión:

- **Deslizamiento (*slippage*):** diferencia entre el precio que ves y el que obtienes. En BTC
  suele ser pequeño (≈0.01–0.05%) para montos de particulares, pero crece en momentos de pánico.
- **Funding (solo futuros perpetuos):** pago periódico entre largos y cortos. En 2020–2026 el
  promedio fue 0.0101% cada 8h a cargo de los largos, **≈11% anual**. Mantener un largo en
  perpetuos cuesta mucho más que en spot: en las pruebas, *buy & hold* en spot tuvo Sharpe 0.98
  y en perpetuo 0.82.
- **Impuestos:** en muchos países, incluido México, las ganancias por compraventa de
  criptomonedas son gravables. Cada operación cerrada puede generar una obligación. Consulta a
  un contador y guarda el registro de todas tus operaciones.

**Costo de ida y vuelta típico** (entrar y salir): 0.04% (futuros, ambas órdenes maker) a 0.20%
(spot, ambas taker). Los estudios del proyecto usan 0.14% (7 pb por lado: taker de futuros más
deslizamiento).

## 2.2 El acierto mínimo que necesitas

Si cada operación gana o pierde en promedio el movimiento típico `m` y pagas un costo de ida y
vuelta `c`, necesitas un acierto mayor que:

```text
acierto mínimo = 0.5 + c / (2 · m)
```

Con el movimiento medio real de BTC en cada horizonte:

| Horizonte | Movimiento medio | c = 0.10% | c = 0.14% | c = 0.20% | c = 0.30% |
|---|---|---|---|---|---|
| 4 horas | 0.63% | 57.9% | 61.0% | 65.8% | 73.7% |
| 1 día | 1.68% | 53.0% | 54.2% | 56.0% | 58.9% |
| 7 días | 4.72% | 51.1% | 51.5% | 52.1% | 53.2% |

Lectura: **operar cada 4 horas exige un acierto altísimo**, porque los movimientos son
pequeños frente a la comisión. A 7 días bastaría poco más del 51%, pero el proyecto no encontró
ninguna señal que lo logre (E-08 a E-10).

## 2.3 Qué pasa con el modelo direccional actual

Valor esperado por operación a 4h (acierto × ganancia − fallo × pérdida − costo):

| Acierto | Costo 0.10% | Costo 0.14% | Costo 0.20% |
|---|---|---|---|
| 54.0% (walk-forward de 4 años) | −0.050% | −0.090% | −0.150% |
| 55.4% (holdout de octubre) | −0.032% | −0.072% | −0.132% |

Siempre negativo. Con 6 operaciones al día, −0.09% por operación equivale a perder ≈0.5% del
capital operado **cada día**. En el backtest, la estrategia largo-corto con este modelo tuvo
Sharpe −2.3 con 7 pb de costo, contra +0.98 de comprar y mantener.

## 2.4 Qué sí resultó viable (y con qué límites)

| Enfoque | Sharpe neto | Caída máxima | Retorno anual | Comentario |
|---|---|---|---|---|
| Comprar y mantener (spot) | 0.98 | −53% | 41% | Referencia; el periodo fue alcista |
| Regla de tendencia de 180 días, largo/fuera | 0.94 | −33% | 33% | Casi el mismo retorno ajustado con caídas mucho menores; 64% del tiempo invertido |
| Mejor estrategia de ML (3 días, largo/fuera) | 0.94 | −32% | 31% | Mismo perfil que la regla simple: el ML no aporta |
| Exposición ajustada por volatilidad | 0.89 | −53% | 32% | No mejoró el Sharpe; útil para controlar el riesgo, no para ganar más |
| Predicción direccional a 4h, largo-corto | −2.3 | −99% | −99% (pérdida total) | Inviable con costos |

Conclusiones prácticas:

1. **Ninguna estrategia superó a comprar y mantener** en retorno ajustado por riesgo con
   significancia estadística.
2. Lo que sí se logra es **cambiar el perfil de riesgo**: mismas ganancias ajustadas con caídas
   de −33% en lugar de −53%. Para mucha gente eso es la diferencia entre aguantar o vender en
   pánico.
3. **Operar con frecuencia destruye valor** con estas señales. Menos operaciones, horizontes
   más largos y órdenes límite (maker) son el camino viable.
4. 2022–2026 fue un periodo con tendencia alcista. En un mercado bajista prolongado, comprar y
   mantener pierde y los filtros de tendencia suelen ayudar más.

## 2.5 Requisitos mínimos para operar en serio

- **Capital que puedas perder** sin afectar tu vida: nunca dinero para gastos, deudas ni fondo
  de emergencia.
- **Montos mínimos de exchange:** las órdenes en Binance tienen un nocional mínimo (del orden
  de 5–10 USDT) y pasos de cantidad; con capital muy pequeño las comisiones pesan más en
  proporción.
- **Tiempo:** revisar el plan a horas fijas (ver [plan operativo](./07-plan-operativo-y-diario.md)),
  no mirar el precio todo el día.
- **Registro:** un diario de operaciones desde el primer día; es imprescindible para mejorar y
  para los impuestos.
