# 8. Automatización segura

El proyecto **no envía órdenes** a ningún exchange y no se recomienda automatizar la predicción
direccional (no tiene ventaja después de costos). Este documento existe para que, si decides
automatizar **reglas de gestión de riesgo** (por ejemplo, el rebalanceo por volatilidad o el
filtro de tendencia), lo hagas sin exponerte a errores catastróficos.

## 8.1 Requisitos previos

1. Las reglas ya funcionaron en simulación manual (fase 2 del [plan operativo](./07-plan-operativo-y-diario.md)).
2. Las reglas están escritas sin ambigüedad: entradas, salidas, tamaño, límites.
3. Aceptas que un bot ejecuta errores a la velocidad de una computadora.

## 8.2 Arquitectura mínima

```text
[API del proyecto] ──► [motor de decisión] ──► [gestor de riesgo] ──► [cliente del exchange]
                              │                       │                      │
                              └──────── registro y alertas ◄─────────────────┘
```

- **Motor de decisión:** lee `/api/v1/...` y calcula la posición objetivo. No envía órdenes.
- **Gestor de riesgo:** independiente; puede **vetar** cualquier orden. Verifica tamaño máximo,
  exposición total, límites diarios y estado de los datos.
- **Cliente del exchange:** el único componente con claves; ejecuta y concilia.

## 8.3 Controles obligatorios

| Control | Detalle |
|---|---|
| Entorno de pruebas | Empezar en el **testnet** del exchange y después con montos mínimos reales |
| Claves de API | Sin permiso de retiro; restringidas por IP; solo los permisos necesarios; guardadas en un gestor de secretos, nunca en el código ni en git |
| Idempotencia | `clientOrderId` determinista por decisión; antes de enviar, comprobar que no exista ya |
| Tamaño máximo por orden | Límite fijo en el código (por ejemplo, 5% del capital) aunque el cálculo diga más |
| Datos frescos | No operar si la última vela tiene más de 8h o si el pronóstico de volatilidad es de respaldo |
| Interruptor de apagado (*kill switch*) | Un archivo o variable que detiene todo y cancela órdenes abiertas |
| Cortacircuitos | Los mismos límites del [documento 6](./06-gestion-de-riesgo.md#62-cortacircuitos-límites-de-pérdida), aplicados por el gestor de riesgo |
| Conciliación | Cada ciclo compara la posición real del exchange con la esperada; si difieren, se detiene y alerta |
| Límites de la API | Respetar los límites de peticiones; reintentos con espera exponencial |
| Registro | Cada decisión, orden, ejecución y error, con hora UTC |
| Alertas | Aviso inmediato (correo, Telegram) ante errores, cortacircuitos o desconciliación |
| Stops en el exchange | Los stops viven en el exchange (OCO o stop-market), no solo en el bot: si el bot cae, la protección sigue |

## 8.4 Qué no automatizar

- La predicción direccional de 4h como señal de compra/venta.
- Cualquier estrategia con apalancamiento > 2x.
- Promediar a la baja o aumentar el tamaño tras pérdidas.
- Cambios de parámetros sin pasar de nuevo por simulación.

## 8.5 Pruebas antes de activar

1. Pruebas unitarias del cálculo de tamaño y de los límites con casos extremos (capital 0,
   stop 0, volatilidad 0 o ausente).
2. Repetición histórica (*replay*) de al menos 6 meses de datos con el código exacto del bot.
3. Testnet durante 2 semanas sin intervención manual.
4. Dinero real mínimo durante 1 mes, revisando la conciliación cada día.
