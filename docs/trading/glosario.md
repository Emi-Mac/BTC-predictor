# Glosario

| Término | Significado |
|---|---|
| **Acierto (accuracy)** | Porcentaje de predicciones de dirección correctas |
| **Apalancamiento** | Operar con más capital del que tienes, prestado por el exchange; multiplica ganancias y pérdidas |
| **Backtest** | Simulación de una estrategia sobre datos históricos |
| **Banda / rango probable** | Intervalo de precio donde históricamente cae el precio con cierta probabilidad (68, 90, 95%) |
| **Baseline / regla simple** | Estrategia trivial contra la que se compara un modelo (p. ej. «siempre sube») |
| **Buy & hold** | Comprar y mantener sin operar |
| **Caída máxima (drawdown)** | Mayor pérdida desde un máximo previo del capital |
| **Colas gruesas** | Movimientos extremos más frecuentes de lo que predice una distribución normal |
| **Cortacircuito** | Límite de pérdida que, al alcanzarse, detiene la operación |
| **DCA** | Compras periódicas de un monto fijo |
| **DVOL** | Índice de volatilidad implícita de BTC a 30 días de Deribit, derivado de precios de opciones |
| **Exposición** | Porcentaje del capital invertido en BTC |
| **FOMC** | Comité de la Reserva Federal de EE. UU. que decide las tasas de interés |
| **Funding** | Pago periódico entre largos y cortos en futuros perpetuos |
| **HAR / HAR-X** | Modelo de volatilidad que combina la volatilidad de 1, 7 y 30 días; la X indica variables externas (DVOL, VIX, funding…) |
| **HAR-X-live** | HAR-X sin interés abierto, el modelo de volatilidad que corre en producción |
| **Holdout** | Periodo reservado que no se usa para desarrollar un modelo y sirve como prueba independiente |
| **Kelly** | Fórmula que da la fracción óptima a arriesgar si se conoce la ventaja exacta |
| **Liquidación** | Cierre forzoso de una posición apalancada cuando el margen no cubre las pérdidas |
| **Maker / taker** | Maker deja una orden en el libro (comisión menor); taker ejecuta contra una existente (comisión mayor) |
| **Multiplicador conforme (k)** | Factor que ensancha la banda `P·exp(±k·σ)` hasta que su cobertura real sea la nominal; se calcula con los resultados ya resueltos, sin supuestos de distribución |
| **OCO** | Par de órdenes (objetivo y stop) donde la ejecución de una cancela la otra |
| **Paper trading** | Operar en simulación, sin dinero real |
| **QLIKE** | Medida del error de un pronóstico de volatilidad; menor es mejor |
| **R** | Riesgo de una operación (distancia entre entrada y stop); las ganancias se miden en múltiplos de R |
| **Régimen** | Clasificación del nivel de volatilidad: bajo, normal, alto o extremo |
| **Sharpe** | Retorno dividido entre su volatilidad; mide el retorno ajustado por riesgo |
| **Slippage (deslizamiento)** | Diferencia entre el precio esperado y el obtenido |
| **Sombra prospectiva** | Periodo de 12 semanas en que el sistema corre en vivo, sin mostrarse, para comprobar con datos futuros que sus bandas cubren lo prometido |
| **Spot** | Compra directa del activo, sin apalancamiento ni funding |
| **Stop** | Orden que cierra la posición si el precio va en tu contra hasta cierto nivel |
| **Volatilidad anualizada** | Desviación de los retornos escalada a un año; σ de `d` días = σ anual × √(d/365) |
| **Volatilidad objetivo** | Nivel de riesgo deseado para un bloque de inversión; la exposición se ajusta para acercarse a él |
| **Walk-forward** | Validación que entrena con el pasado y evalúa en el periodo siguiente, mes a mes |
