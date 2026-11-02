# 4. Estrategias

Cada estrategia indica su **nivel de evidencia** en este proyecto:

- ✅ **Probada:** evaluada fuera de muestra con el protocolo de [docs/research](../research/methodology.md).
- 🟡 **Parcial:** sus piezas están probadas, pero no la combinación.
- ⚪ **No evaluada:** práctica habitual sin prueba en el proyecto.
- 🔴 **Desaconsejada:** la evidencia indica que pierde dinero.

Ninguna superó a comprar y mantener en retorno ajustado por riesgo con significancia. Lo que
ofrecen es **otro perfil de riesgo**. Simula cualquiera antes de usarla con dinero real.

---

## A. Núcleo de largo plazo con compras periódicas (DCA) ⚪

**Idea:** comprar una cantidad fija cada semana o mes, sin intentar adivinar el momento.

**Reglas:**
1. Define un presupuesto fijo por periodo (por ejemplo, 5% de tu ahorro mensual).
2. Compra el mismo día y hora con orden límite cerca del precio actual (ver documento 5).
3. No vendas por caídas que estén dentro del rango histórico: una caída de 30% en un mes ocurrió
   en el peor caso del periodo, y caídas de 50% desde máximos son parte de la historia de BTC.

**Uso del proyecto:** el rango probable y el régimen sirven para **saber qué esperar** y no
reaccionar al ruido. No cambian la regla de compra.

**Cuándo falla:** mercados bajistas largos (2018, 2022) en los que el precio tarda años en
recuperarse.

## B. Filtro de tendencia de 180 días, largo o fuera ✅

**Idea:** mantener BTC solo cuando su retorno de los últimos 180 días es positivo.

**Reglas:**
1. Una vez al día (por ejemplo, al cierre de 00:00 UTC), compara el precio actual con el de
   hace 180 días.
2. Si es mayor → mantener BTC (spot). Si es menor → pasar a stablecoins o efectivo.
3. Cambia de estado solo en la revisión diaria, no dentro del día.

**Evidencia (E-11, sep-2022 a ago-2026, neto de 7 pb y funding):**

| | Sharpe | Caída máxima | Retorno anual | Tiempo invertido |
|---|---|---|---|---|
| Comprar y mantener | 0.98 | −53% | 41% | 100% |
| Filtro 180 días | 0.94 | −33% | 33% | 64% |

**Qué esperar:** caídas mucho menores a cambio de algo menos de retorno. La diferencia de
Sharpe no es significativa (IC −0.66 a +0.57).

**Cuándo falla:** mercados laterales con cambios frecuentes de tendencia (entradas y salidas
seguidas con pérdida). Cada salida en spot puede generar impuestos.

**Variantes probadas y peores:** 30 días (Sharpe 0.85), 90 días (0.77), cruce de medias de
50/200 días (0.49) y versiones largo-corto (0.71 la mejor). La versión largo-corto pierde porque
los cortos en mercado alcista y el funding cuestan caro.

## C. Exposición ajustada por volatilidad 🟡

**Idea:** mantener constante el riesgo de tu bloque de BTC, no la cantidad: invertir menos
cuando la volatilidad esperada es alta y más cuando es baja.

**Reglas:**
1. Elige una volatilidad objetivo para tu bloque de BTC (por ejemplo, 30% anual).
2. Exposición = `min(100%, objetivo / volatilidad pronosticada)`.
3. Rebalancea solo si la exposición deseada cambia más de 10 puntos (evita comisiones).

| Volatilidad pronosticada | Objetivo 20% | Objetivo 30% |
|---|---|---|
| 34% | 59% | 88% |
| 44.4% | 45% | 68% |
| 59.5% | 34% | 50% |
| 69% | 29% | 43% |

**Evidencia (E-12):** el pronóstico es bueno, pero esta regla **no mejoró el Sharpe** frente a
mantener siempre la misma exposición media. Sirve para que tu riesgo sea predecible, no para
ganar más.

## D. Operaciones con tesis propia, stops y tamaño basados en volatilidad 🟡

**Idea:** tú decides la dirección (análisis propio, fundamentales, noticias); el proyecto
decide **cuánto** y **dónde salir si te equivocas**.

**Reglas:**
1. Define el horizonte de la operación (1 día o 7 días).
2. Stop: fuera de la banda del 68% de ese horizonte, idealmente entre 1× y 1.87× σ del
   horizonte (ver documento 5).
3. Tamaño: riesgo de 0.5–1% del capital total si el stop se ejecuta.
4. Objetivo: al menos 1.5–2 veces la distancia del stop (relación beneficio/riesgo ≥ 1.5).
5. Si la operación no avanza dentro del horizonte, ciérrala: tu tesis tenía un plazo.

**Evidencia:** el cálculo de tamaño y stop está respaldado por la calibración de las bandas
(E-12b: cobertura nominal fuera de muestra a 7 días; la sombra prospectiva sigue en curso). Los
tamaños los calcula la calculadora del dashboard.
La ventaja de la dirección depende de **tu** tesis y no está probada.

## E. Reducción de riesgo por régimen y eventos ⚪

**Reglas sugeridas:**
- Régimen **extremo** (> 59.3% anual): no abrir posiciones nuevas o hacerlo a la mitad del
  tamaño normal.
- **72 horas antes y después de una decisión del FOMC:** evitar entradas nuevas o reducir
  tamaño; ampliar stops a la banda del 90%.
- Fines de semana: menor liquidez; evitar órdenes de mercado grandes.

**Evidencia:** son reglas de control de riesgo; no se probó que mejoren el retorno.

## F. Seguir la predicción direccional de 4h 🔴

Acierto ≈54%, por debajo del 58–66% necesario con comisiones; Sharpe −2.3 en el backtest.
Si quieres experimentar, hazlo **solo en simulación** y registra cada operación.

## G. Operar dentro del rango (*grid*) ⚪

Comprar en la parte baja del rango esperado y vender en la alta. No se evaluó en el proyecto.
Funciona en mercados laterales y pierde mucho cuando el precio rompe el rango (11.5% de las
semanas se mueve más de 10%). Si lo pruebas: solo en régimen bajo, con stop fuera de la banda
del 95% y primero en simulación.

---

## Plan base sugerido para empezar

Combina las piezas con mejor respaldo, sin apalancamiento:

1. **Decide tu asignación máxima a BTC** como porcentaje de tu patrimonio invertible (por
   ejemplo, 5–20%, según tu tolerancia al riesgo). Ese es tu «bloque de BTC».
2. **Filtro de tendencia (B):** invertido solo si el retorno de 180 días es positivo.
3. **Exposición por volatilidad (C):** dentro del bloque, exposición = `min(100%, 30% / σ)`.
4. **Reglas de evento (E):** sin cambios de posición en las 72h alrededor de un FOMC, salvo
   para reducir.
5. **Revisión diaria** a una hora fija; rebalanceo solo si el cambio supera 10 puntos.
6. **Spot, no perpetuos,** para posiciones que se mantienen días o semanas (el funding cuesta
   ≈11% anual).

⚠️ Esta combinación **no se ha probado completa** (está planeada como experimento E-13). Sus
piezas sí. Simúlala al menos 3 meses (ver [plan operativo](./07-plan-operativo-y-diario.md)).

## Lo que nunca debes hacer

- **Martingala** o duplicar después de perder.
- **Promediar a la baja** sin un límite fijado de antemano.
- **Apalancamiento alto:** a 10x, un movimiento adverso de ≈9.6% te liquida, y eso pasó en más
  del 11% de las semanas.
- Operar porque «el modelo dio 60%».
- Mover el stop para no aceptar una pérdida.
- Invertir dinero que necesitarás en los próximos años.
