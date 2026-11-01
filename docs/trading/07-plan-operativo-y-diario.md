# 7. Plan operativo y diario de trading

## 7.1 Fases para pasar a dinero real

| Fase | Duración mínima | Qué haces | Criterio para avanzar |
|---|---|---|---|
| 1. Estudio | 2 semanas | Leer esta guía y [docs/research](../research/README.md); observar el dashboard sin operar; escribir tu plan | Plan escrito con estrategia, tamaño, stops y cortacircuitos |
| 2. Simulación (*paper trading*) | 3 meses **y** ≥ 50 operaciones (o 12 rebalanceos del bloque) | Registrar cada decisión como si fuera real, con comisiones | KPIs de la sección 7.4 cumplidos y cero violaciones del plan en el último mes |
| 3. Dinero real mínimo | 3 meses | 10–25% del capital de trading previsto | KPIs similares a la simulación (diferencia de expectativa < 30%) |
| 4. Escalado | Gradual | Aumentar el capital en pasos de ≤ 25% por trimestre | Mantener los KPIs; ningún cortacircuito mensual activado |

Si en cualquier fase se activa el cortacircuito de caída máxima, vuelve a la fase 2.

## 7.2 Rutina

**Cada 4 horas (solo si haces operaciones de corto plazo), 10 minutos:**
1. Verificar que el dashboard está actualizado (última vela de hace < 8h).
2. Revisar régimen de volatilidad y eventos próximos.
3. Gestionar órdenes abiertas según el plan; no abrir nada fuera del plan.

**Diaria (a hora fija, por ejemplo 00:30 UTC), 15 minutos:**
1. Filtro de tendencia de 180 días: ¿cambió el estado?
2. Exposición objetivo por volatilidad: ¿el cambio supera 10 puntos?
3. Revisar límites diarios y semanales.
4. Actualizar el diario.

**Semanal, 30 minutos:**
1. Cobertura de las bandas en la semana (¿el precio quedó dentro de la del 68%?).
2. Resultado de la semana frente a los límites.
3. Calendario de la semana siguiente (FOMC, datos macro).

**Mensual, 1 hora:**
1. KPIs del mes y acumulados (sección 7.4).
2. Revisión del diario: errores repetidos y violaciones del plan.
3. Comparación contra comprar y mantener en el mismo periodo.
4. Decidir si se cambia algo (solo con evidencia, ver [documento 6](./06-gestion-de-riesgo.md#65-riesgo-del-propio-sistema)).

## 7.3 Plantilla del diario

| Campo | Ejemplo |
|---|---|
| Fecha y hora (UTC) | 2026-11-02 00:30 |
| Estrategia | D (tesis propia, 7 días) |
| Motivo de la entrada | Ruptura del rango semanal con volumen; tendencia de 180 días positiva |
| Régimen de volatilidad / σ anual | Normal / 44.4% |
| Predicción direccional (solo registro) | UP 55% |
| Entrada / stop / objetivo | 77,582 / 72,812 / 87,123 |
| Distancia del stop / R | 6.15% / 4,770 USD por BTC |
| Tamaño (USD y BTC) | 1,626 USD / 0.02096 BTC |
| Riesgo en USD y % | 100 USD / 1% |
| Tipo de orden y comisión | Límite *post-only* / 0.02% |
| Salida (precio, fecha, motivo) | 87,123 / 2026-11-07 / objetivo |
| Resultado (USD y R) | +200 USD / +2R |
| ¿Se siguió el plan? | Sí |
| Emoción y notas | Ganas de cerrar antes en 1R; se mantuvo |

Una hoja de cálculo basta. Exporta también el historial del exchange para los impuestos.

## 7.4 Métricas (KPIs)

| Métrica | Fórmula | Referencia mínima |
|---|---|---|
| Expectativa por operación | `acierto × ganancia media − (1 − acierto) × pérdida media − costos` | > 0 después de costos |
| Expectativa en R | Media de los resultados en múltiplos de R | > 0.1R |
| Profit factor | Suma de ganancias / suma de pérdidas | > 1.2 |
| Caída máxima | Peor caída desde un máximo de tu capital | Dentro de tu límite (−15%) |
| Sharpe anualizado | Media / desviación de los retornos diarios × √365 | Comparar con comprar y mantener en el mismo periodo |
| Cumplimiento del plan | Operaciones según el plan / total | 100% en el último mes |
| Cobertura de bandas | Semanas dentro de la banda del 68% / total | 60–76% |

**Tamaño de muestra:** con 50 operaciones, un acierto medido del 55% tiene un error estándar de
±7 puntos (intervalo del 95%: 41–69%). No concluyas nada con menos operaciones.

## 7.5 Cuándo dejar de usar el sistema

- La expectativa en R es negativa después de 100 operaciones reales.
- Tu resultado queda por debajo de comprar y mantener durante 12 meses con más riesgo.
- Violas tu propio plan de forma repetida (es una señal de que el plan no es para ti).
- El acierto del modelo direccional en vivo cae por debajo del 50% en 500 predicciones o la
  cobertura de la banda del 68% sale de 55–80% durante 3 meses: el sistema dejó de describir el
  mercado; detente y revísalo.
