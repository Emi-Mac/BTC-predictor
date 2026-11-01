# 6. Gestión de riesgo

La gestión de riesgo no aumenta tu ventaja; **evita que una racha mala o un error te saquen del
juego**. Con señales sin ventaja demostrada, es la parte más importante.

## 6.1 Reglas de capital

| Regla | Valor sugerido | Por qué |
|---|---|---|
| Asignación máxima a BTC | 5–20% del patrimonio invertible | BTC cayó −53% en el periodo estudiado y más de −75% en ciclos anteriores |
| Riesgo por operación | 0.5% (máx. 1–2%) | 10 pérdidas seguidas con 1% ≈ −9.6%; con 5% ≈ −40% |
| Riesgo abierto total | ≤ 3% del capital de trading | Varias posiciones en BTC están correlacionadas: es un solo riesgo |
| Apalancamiento | Ninguno; máximo 2x en futuros | Ver tabla de liquidaciones del [documento 5](./05-ordenes-y-tamano-de-posicion.md#57-apalancamiento-y-futuros) |
| Fondo de emergencia | Fuera del exchange y nunca invertido | Evita vender en el peor momento por necesidad |

## 6.2 Cortacircuitos (límites de pérdida)

Decide estos límites **por escrito y antes** de operar:

| Nivel | Límite | Acción |
|---|---|---|
| Día | −2% del capital de trading | No abrir más operaciones hasta el día siguiente |
| Semana | −4% | Sin operaciones nuevas el resto de la semana; revisar el diario |
| Mes | −6% | Volver a simulación al menos 2 semanas |
| Caída desde máximo | −15% del capital de trading | Parar por completo y revisar todo el sistema con datos |

Los límites del bloque de largo plazo son distintos: ahí la regla es **no vender por pánico**
mientras la caída esté dentro de lo históricamente posible y tu horizonte siga siendo largo.

## 6.3 Riesgo de mercado específico de BTC

- **Colas gruesas:** movimientos de 3σ ocurren mucho más que en una distribución normal. Por
  eso la banda del 95% usa k = 2.52 y no 1.96.
- **Régimen de volatilidad:** la volatilidad se agrupa. Tras un día muy volátil, espera más
  días volátiles; reduce el tamaño en régimen alto o extremo.
- **Eventos macro:** decisiones del FOMC y datos de inflación de EE. UU. mueven el mercado.
  Evita abrir posiciones en las 72h alrededor del FOMC o reduce tamaño.
- **Fines de semana y madrugadas:** menos liquidez; los movimientos bruscos se amplifican.
- **Correlación con acciones:** en crisis, BTC suele caer junto con la bolsa; no es una
  cobertura confiable.

## 6.4 Riesgos fuera del precio

| Riesgo | Mitigación |
|---|---|
| Quiebra o congelamiento del exchange | Mantén en el exchange solo el capital de trading; el resto en autocustodia (hardware wallet) con respaldo seguro de la frase semilla |
| Robo de cuenta | 2FA con app (no SMS), código anti-phishing, contraseña única, correo dedicado |
| Claves de API | Sin permiso de retiro, restringidas por IP, rotación periódica (ver [documento 8](./08-automatizacion-segura.md)) |
| Phishing y estafas | Nunca compartas la frase semilla; desconfía de «señales garantizadas» y de soporte que te contacta primero |
| Errores operativos | Lista de verificación de la [sección 5.2](./05-ordenes-y-tamano-de-posicion.md#52-lista-de-verificación-antes-de-enviar-una-orden) en cada orden |
| Impuestos | Registro completo de operaciones; consulta a un contador |
| Riesgo regulatorio | Usa exchanges autorizados en tu país y revisa sus condiciones |

## 6.5 Riesgo del propio sistema

- **El modelo puede degradarse.** Vigila el acierto en vivo frente al ≈54% esperado y la
  cobertura de las bandas frente a su nivel nominal. Si la banda del 68% contiene el precio
  menos del 60% de las veces durante un mes, amplía los stops y revisa el pronóstico.
- **Fallas de datos:** si el dashboard muestra datos viejos (última vela de hace más de 8h),
  no operes con él.
- **Sobreajuste propio:** cambiar las reglas después de cada pérdida es la forma más rápida de
  ajustar el ruido. Cambia reglas solo con evidencia de al menos 3 meses o 50 operaciones.

## 6.6 Psicología: errores que cuestan más que cualquier comisión

| Sesgo | Cómo se ve | Antídoto |
|---|---|---|
| Revancha | Aumentar el tamaño después de perder | Cortacircuito diario y tamaño por fórmula |
| Aversión a la pérdida | Mover el stop para no aceptarla | El stop solo se mueve a favor |
| FOMO | Entrar tarde después de una subida grande | Si no estaba en el plan, no existe |
| Exceso de confianza | Subir el riesgo tras una racha ganadora | El riesgo por operación es fijo |
| Ver patrones en el ruido | «El modelo acertó 5 seguidas» | Con 54% de acierto, rachas de 5 ocurren solas |

## 6.7 Consejos para mejorar resultados sin aumentar el riesgo

1. **Opera menos.** Cada operación cuesta 0.04–0.20% ida y vuelta; con señales sin ventaja,
   menos operaciones significa menos pérdida.
2. **Usa órdenes maker** (límite, *post-only*): en futuros la comisión baja de 0.05% a 0.02%.
3. **Spot para mantener, futuros solo para coberturas puntuales** (el funding cuesta ≈11% anual).
4. **Horizontes más largos:** a 7 días el acierto necesario para cubrir costos es ≈51.5%,
   contra ≈61% a 4h.
5. **Deja correr las ganancias** con objetivos ≥ 1.5R y stops que solo se mueven a favor.
6. **Reduce el tamaño en régimen alto**, no la distancia del stop.
7. **Revisa tu diario cada mes:** tus peores errores suelen repetirse y son evitables.
