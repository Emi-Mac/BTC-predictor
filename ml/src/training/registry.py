"""Registro de experimentos; los modelos nunca se activan automáticamente."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.training.walk_forward import ModelBundle


async def register_model(
    engine: AsyncEngine,
    bundle: ModelBundle,
    report: dict[str, Any],
    directory: Path,
) -> int:
    last = report["folds"][-1]
    metrics = last["models"][bundle.family]["calibrated"]
    passed = report["criteria"]["passed"]
    async with engine.begin() as conn:
        table = await conn.run_sync(
            lambda sync: sa.Table("model_registry", sa.MetaData(), autoload_with=sync)
        )
        statement = (
            table.insert()
            .values(
                model_type=bundle.family,
                hyperparameters={
                    "model": bundle.parameters,
                    "calibration": vars(bundle.calibration),
                    "quality_criteria": report["criteria"],
                    "training_cutoff": last["calibration_start"],
                    "test_start": last["validation_start"],
                    "test_end_exclusive": last["validation_end_exclusive"],
                    "seed": report["seed"],
                },
                val_accuracy=metrics["accuracy"],
                val_f1=metrics["f1_macro"],
                val_precision=metrics["precision_up"],
                val_recall=metrics["recall_up"],
                baseline_accuracy=report["final_fold_best_baseline"],
                artifact_path=str((directory / "model.onnx").resolve()),
                scaler_path=str((directory / "preprocessor.json").resolve()),
                feature_names=bundle.preprocessor.feature_names,
                is_active=False,
                notes="Criterios cumplidos; no activado"
                if passed
                else "Criterios no cumplidos; no activado",
            )
            .returning(table.c.version)
        )
        version = (await conn.execute(statement)).scalar_one()
    return int(version)
