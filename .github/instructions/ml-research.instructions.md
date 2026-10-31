---
description: "Use when working on the BTC prediction model, ML research, experiments, features, training, evaluation, backtests, holdouts or model accuracy improvements. Points to the research memory of what was already tried and the evaluation protocol."
applyTo: "ml/**"
---
# Investigación del modelo BTC

Antes de proponer o ejecutar un experimento, lee [docs/research](../../docs/research/README.md):

- [experiment-log.md](../../docs/research/experiment-log.md): E-01 a E-09, con resultados y
  la sección «No repetir / reabrir solo si». Estado al 2026-10-30: sin señal direccional
  explotable a 4h, 24h ni 3d; el modelo sigue inactivo.
- [methodology.md](../../docs/research/methodology.md): 48 folds walk-forward con purga
  `h+1`, tres baselines por horizonte, gate de límites simultáneos (≥ +3 pp) y economía neta
  con funding.
- [data-sources.md](../../docs/research/data-sources.md) y
  [feature-catalog.md](../../docs/research/feature-catalog.md).

Reglas:

- No modifiques ni formatees los archivos de `IMPLEMENTATION_FILES` en
  `ml/src/research/provenance.py`: invalidan el holdout reservado de noviembre de 2026.
  Formatea solo archivos nuevos y explícitos; nunca `ruff format src`.
- Las fuentes nuevas van a un snapshot inmutable (`ml/src/research/aux_sources.py`), no al
  dataset congelado. Cada feature declara su disponibilidad y necesita un test de
  no-anticipación.
- Registra cada experimento nuevo en `experiment-log.md` con la plantilla del final, incluidos
  los fallidos.
