# Investigación del modelo de predicción BTC

Esta carpeta es la **memoria de investigación** del proyecto: qué se intentó, cómo se evaluó,
qué resultó y qué no conviene repetir. Los artefactos (`ml/artifacts/`) **no se versionan**
(`.gitignore`), por eso las cifras clave, las configuraciones y las conclusiones viven aquí.

## Cómo usar esta carpeta

1. Antes de proponer o ejecutar un experimento nuevo, leer el
   [registro de experimentos](./experiment-log.md). Si la idea ya está registrada, consultar
   su resultado y su sección «Qué no repetir / cuándo reabrir».
2. Seguir el [protocolo de evaluación](./methodology.md): sin él, los resultados no son
   comparables con los anteriores.
3. Consultar las [fuentes de datos](./data-sources.md) y sus supuestos de disponibilidad antes
   de añadir una feature.
4. Consultar el [catálogo de features](./feature-catalog.md) para ver qué existe y qué aportó.
5. Al terminar un experimento: añadir una entrada al registro (plantilla al final del
   registro), actualizar el estado en [model-improvement-plan.md](../plans/model-improvement-plan.md)
   y conservar el `report.json` del artefacto.

## Documentos

| Documento | Contenido |
|---|---|
| [experiment-log.md](./experiment-log.md) | Registro cronológico de los experimentos E-01…E-09 con hipótesis, método, resultados, artefactos y conclusiones |
| [methodology.md](./methodology.md) | Protocolo de evaluación: particiones, purga, baselines, inferencia, gates e integridad de holdouts |
| [data-sources.md](./data-sources.md) | Fuentes, rangos, supuestos de disponibilidad y limitaciones conocidas |
| [feature-catalog.md](./feature-catalog.md) | Todas las features con fórmula, disponibilidad y resultado observado |
| [../plans/model-improvement-plan.md](../plans/model-improvement-plan.md) | Plan por fases y estado de cada fase |
