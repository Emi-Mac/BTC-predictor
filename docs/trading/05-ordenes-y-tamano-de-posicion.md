# 5. Órdenes y tamaño de posición

Este documento responde tres preguntas antes de cada operación: **qué tipo de orden**,
**cuánto** y **dónde salir**. Las cifras de ejemplo usan BTC a 77,582 USD y una volatilidad
pronosticada de 44.4% anual (σ de 1 día = 2.32%, σ de 7 días = 6.15%).

## 5.1 Tipos de orden

| Tipo | Qué hace | Costo | Cuándo usarla | Riesgo |
|---|---|---|---|---|
| **Mercado** | Ejecuta ya al mejor precio disponible | Taker (más cara) | Salidas urgentes; montos pequeños en horas de buena liquidez | Deslizamiento en momentos de pánico |
| **Límite** | Ejecuta solo a tu precio o mejor | Maker si queda en el libro | Entradas planificadas, rebalanceos | Puede no ejecutarse |
| **Límite *post-only*** | Se cancela si se ejecutaría como taker | Siempre maker | Para garantizar la comisión baja | Cancelación si el precio se mueve |
| **Stop-market** | Cuando el precio toca el disparador, envía una orden de mercado | Taker | Stop de pérdida: **prioriza salir** | Puede ejecutarse bastante peor en caídas bruscas |
| **Stop-limit** | Al tocar el disparador, envía una orden límite | Taker o maker | Cuando prefieres no vender por debajo de cierto precio | **Puede no ejecutarse** y dejarte expuesto en un desplome |
| **OCO** (*one cancels the other*) | Un objetivo límite y un stop a la vez; si se ejecuta uno, se cancela el otro | Según el que se ejecute | Proteger una posición con objetivo y stop sin vigilarla | Ninguno adicional; es la forma recomendada de proteger posiciones en spot |
| **Trailing stop** | El stop sigue al precio a una distancia fija | Taker | Dejar correr ganancias en tendencia | Con una distancia muy corta, el ruido te saca |

Vigencia (*time in force*): **GTC** (hasta cancelar), **IOC** (ejecuta lo posible y cancela el
resto), **FOK** (todo o nada). Para entradas planificadas, usa GTC con una fecha de revisión.

**Regla práctica:** entra con órdenes límite (maker) y protege con stop-market. Pagar taker al
salir de una pérdida es preferible a quedarte atrapado con un stop-limit sin ejecutar.

## 5.2 Lista de verificación antes de enviar una orden

1. ¿La operación está en tu plan escrito (estrategia, horizonte, motivo)?
2. ¿Calculaste el tamaño con la fórmula de la sección 5.3, **no a ojo**?
3. ¿El stop está definido **antes** de entrar y fuera del ruido normal (sección 5.4)?
4. ¿Revisaste lado (compra/venta), tipo de orden, precio y cantidad? Un cero de más es el error
   más caro.
5. ¿La cantidad respeta el mínimo nocional y el paso de cantidad del exchange?
6. ¿Hay un FOMC en las próximas 72 horas o es fin de semana con poca liquidez?
7. ¿Esta operación respeta tus límites diarios, semanales y de exposición total ([documento 6](./06-gestion-de-riesgo.md))?
8. ¿La registraste en tu diario?

## 5.3 Cuánto comprar

### Método 1: riesgo fijo por operación (recomendado para operaciones con stop)

Arriesgas un porcentaje fijo del capital **si se ejecuta el stop**:

```text
tamaño (USD) = capital × riesgo% / (distancia del stop % + costos de ida y vuelta %)
```

- **Riesgo por operación:** 0.5% para empezar; nunca más de 1–2%.
- Con 1% por operación, 10 pérdidas seguidas (posibles con cualquier sistema) cuestan ≈9.6%
  del capital, no la mitad.

**Ejemplo (operación de 1 día):** capital 10,000 USD, riesgo 1% = 100 USD, stop a 1.5σ de un
día = 3.49% → stop en **74,877**. Costo de ida y vuelta 0.20% (spot taker):

```text
tamaño = 10,000 × 1% / (3.49% + 0.20%) = 2,713 USD ≈ 0.03497 BTC (27% del capital)
```

Si ignoras las comisiones, comprarías 2,869 USD y tu riesgo real sería 106 USD, no 100.

### Método 2: exposición por volatilidad objetivo (para el bloque de largo plazo)

```text
exposición = min(100%, volatilidad objetivo / volatilidad pronosticada)
```

**Ejemplo:** patrimonio invertible de 50,000 USD, bloque de BTC del 20% = 10,000 USD,
volatilidad objetivo de 30%:

| Volatilidad pronosticada | Exposición | En BTC | En stablecoins | Acción |
|---|---|---|---|---|
| 44.4% | 68% | 6,800 USD | 3,200 USD | Posición inicial |
| 59.5% (sube el riesgo) | 50% | 5,000 USD | 5,000 USD | Vender 1,800 USD (cambio de 18 pp > 10 pp) |
| 51.0% | 59% | — | — | No rebalancear (cambio de 9 pp < 10 pp) |

### Método 3: Kelly (por qué no usarlo aquí)

La fracción de Kelly `f* = (p·b − q) / b` maximiza el crecimiento **si conoces tu ventaja**.
Con el modelo direccional (p = 54%, b = 1) daría 8% *antes* de costos; **después de costos la
ventaja es negativa y Kelly es 0**: no apostar. Aun con una ventaja demostrada, usa como máximo
un cuarto de Kelly, porque los errores en la estimación de `p` son enormes.

### Límites que nunca se superan

- Exposición total a BTC ≤ tu asignación máxima (por ejemplo, 20% del patrimonio invertible).
- Una sola operación ≤ 30% del capital de trading en spot, aunque la fórmula dé más (stops muy
  cercanos generan tamaños enormes).
- Riesgo total abierto (suma de todas las pérdidas posibles hasta los stops) ≤ 3% del capital.

## 5.4 Dónde poner el stop

El stop debe estar **fuera del ruido normal** del horizonte de tu operación. Si está dentro de
la banda del 68%, el movimiento aleatorio te sacará con frecuencia aunque tu idea sea correcta.

| Horizonte de la operación | Distancia mínima recomendada | Ejemplo (77,582 USD) |
|---|---|---|
| 1 día | 1.5 σ de 1 día ≈ 3.5% (fuera del 68% y cerca del 90%) | 74,877 |
| 7 días, ajustado | 1 σ de 7 días = 6.15% (fuera del 68%) | 72,812 |
| 7 días, holgado | 1.87 σ de 7 días = 11.5% (borde del 90%) | 68,660 |

> La calculadora del dashboard usa un stop log-simétrico, `P·exp(∓k·σ)` (5.97% para 1σ en lugar
> de 6.15%); las cifras de esta tabla son aproximaciones lineales y difieren unos decimales.

Más reglas:

- **Ajusta el stop a la volatilidad, no a una cifra fija.** «Siempre 5%» es demasiado cerca en
  régimen alto y demasiado lejos en régimen bajo.
- Evita números redondos (75,000): muchos stops se acumulan ahí. Colócalo unos dólares más allá.
- Un stop más lejano **no** es más riesgoso si reduces el tamaño con la fórmula del método 1: la
  pérdida en USD sigue siendo la misma.
- **Nunca alejes el stop** una vez abierta la operación. Solo puedes acercarlo a favor
  (por ejemplo, a precio de entrada tras alcanzar 1R).

## 5.5 Objetivos y salidas

- Mide las ganancias en múltiplos del riesgo (**R** = distancia de entrada a stop). Busca
  operaciones con objetivo ≥ 1.5R. Con 2R, incluso un 40% de acierto es rentable antes de
  costos.
- **Toma parcial:** vende 50% en 1.5R y mueve el stop del resto a precio de entrada.
- **Salida por tiempo:** si la operación no se movió en el horizonte previsto, ciérrala.

| Ejemplo | Entrada | Stop | 1.5R | 2R |
|---|---|---|---|---|
| 1 día (stop 3.49%) | 77,582 | 74,877 | 81,639 | 82,991 |
| 7 días (stop 6.15%) | 77,582 | 72,812 | — | 87,123 |

## 5.6 Entradas escalonadas

En lugar de comprar todo a mercado, divide la entrada en 2–3 órdenes límite dentro de la mitad
inferior de la banda del 68% del horizonte. Ejemplo a 1 día: 50% a 77,582, 25% a 76,811 y
25% a 76,047. Si las órdenes inferiores no se ejecutan, aceptas una posición menor: **no
persigas el precio**. El stop y el riesgo se calculan sobre el tamaño total planeado.

## 5.7 Apalancamiento y futuros

Movimiento adverso aproximado que liquida una posición con margen aislado (mantenimiento ≈0.4%):

| Apalancamiento | Liquidación aprox. | Contexto histórico (sep-2022 a ago-2026) |
|---|---|---|
| 2x | −49.6% | Peor semana: −30.4%; caída máxima desde un máximo: −53% |
| 3x | −32.9% | Muy cerca de la peor semana |
| 5x | −19.6% | El 1.5% de las semanas se movió más de 20% |
| 10x | −9.6% | El 11.5% de las semanas se movió más de 10% |
| 20x | −4.6% | El 10% de los días se movió más de 3.9% |

Recomendaciones:

- **Prefiere spot.** Si usas futuros, máximo 2x, margen **aislado** y el stop siempre muy antes
  del precio de liquidación.
- El apalancamiento no cambia el cálculo del tamaño: el riesgo en USD sigue siendo
  capital × riesgo%. Solo cambia cuánto margen inmovilizas.
- Recuerda el funding (≈11% anual en contra de los largos en promedio).

## 5.8 Errores comunes

| Error | Consecuencia | Corrección |
|---|---|---|
| Calcular el tamaño después de entrar | Riesgo arbitrario | Primero tamaño y stop, después la orden |
| Stop dentro de la banda del 68% | Salidas por ruido | Usar la tabla de la sección 5.4 |
| Stop-limit en un desplome | La orden no se ejecuta | Stop-market para protección |
| Orden de mercado grande en fin de semana | Deslizamiento | Límites escalonadas |
| Ignorar comisiones y funding | El riesgo real es mayor | Incluirlos en la fórmula |
