# Guía para usar el proyecto en trading real

> ⚠️ **Aviso importante.** Esta documentación es **educativa**. No es asesoría financiera,
> fiscal ni legal, y nada aquí garantiza ganancias. Operar criptomonedas puede hacerte perder
> **todo** el capital que arriesgues, y más si usas apalancamiento. Las cifras provienen de
> pruebas históricas (2022–2026) que no aseguran resultados futuros. Antes de arriesgar dinero
> real, sigue el [plan operativo](./07-plan-operativo-y-diario.md): estudio, simulación
> (*paper trading*) y capital pequeño.

## Qué es y qué no es este proyecto, en una frase

Es una herramienta que **mide el riesgo** de BTC (cuánto puede moverse) con evidencia sólida,
y que además muestra una **predicción de dirección experimental** que, en las pruebas, no
supera a reglas simples ni cubre comisiones. Úsalo para **decidir cuánto arriesgar y dónde
poner tus límites**, no para adivinar si BTC sube o baja.

## Orden de lectura

| # | Documento | Para qué sirve |
|---|---|---|
| 1 | [Qué ofrece y qué no](./01-que-ofrece-y-que-no.md) | Inventario honesto de lo validado y lo experimental |
| 2 | [Viabilidad en trading real](./02-viabilidad-en-trading-real.md) | Comisiones, acierto mínimo necesario y por qué la dirección sola no basta |
| 3 | [Cómo leer y usar los datos](./03-como-leer-y-usar-los-datos.md) | Cada panel, cada número, conversiones y la API |
| 4 | [Estrategias](./04-estrategias.md) | Formas de usar las señales, cada una con su nivel de evidencia |
| 5 | [Órdenes y tamaño de posición](./05-ordenes-y-tamano-de-posicion.md) | Tipos de orden, cuánto comprar, stops, apalancamiento |
| 6 | [Gestión de riesgo](./06-gestion-de-riesgo.md) | Reglas para limitar pérdidas y proteger el capital |
| 7 | [Plan operativo y diario](./07-plan-operativo-y-diario.md) | Rutina, simulación, paso a dinero real, métricas y cuándo parar |
| 8 | [Automatización segura](./08-automatizacion-segura.md) | Si algún día conectas un bot: requisitos mínimos |
| — | [Glosario](./glosario.md) | Términos usados |

## Estado de cada señal (actualizado 2026-11-02)

| Señal | Evidencia | Uso recomendado |
|---|---|---|
| Volatilidad pronosticada (HAR-X-live, sin interés abierto) | ✅ Validada fuera de muestra a 7 días (E-12, E-12b); a 1 día no se demuestra ventaja sobre HAR | Tamaño de posición, distancia de stops, alertas de régimen |
| Rango probable de precio | 🟡 Multiplicadores conformes con cobertura nominal fuera de muestra a 7 días (E-12b); **sombra prospectiva de 12 semanas en curso** (2026-11-03 a 2027-01-26) | Colocar stops y objetivos fuera del ruido normal |
| Filtro de tendencia de 180 días (regla, sin ML) | 🟡 Reduce caídas (−33% vs −53%) sin mejorar el retorno ajustado (E-11) | Estar o no estar en el mercado, si priorizas evitar caídas grandes |
| Predicción direccional a 4h (modelo v2) | 🔴 No supera a la regla simple; por debajo del acierto mínimo con comisiones | Solo observación; nunca como única razón para operar |

La evidencia completa está en [docs/research](../research/README.md). La volatilidad ya está
integrada en el backend y el dashboard (ver
[volatility-integration-plan.md](../plans/volatility-integration-plan.md) y el
[runbook](../runbooks/full-stack-runbook.md)). Hasta que la sombra prospectiva termine, el panel
se muestra como **«en validación»**: las bandas y la volatilidad son útiles para dimensionar
riesgo, pero su cobertura real en vivo aún no está confirmada. No hay todavía cifras de
producción; se publicarán en el registro de experimentos al cerrar la sombra.
