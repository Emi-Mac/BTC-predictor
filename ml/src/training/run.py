"""Entrenamiento explícito: python -m src.training.run."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import structlog
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import get_settings
from src.export.onnx_model import export_bundle
from src.features.compute import FEATURE_COLUMNS
from src.ingestion.binance import interval_to_timedelta
from src.ingestion.run import configure_logging
from src.training.data import load_training_data
from src.training.registry import register_model
from src.training.walk_forward import TrainingOptions, run_walk_forward

log = structlog.get_logger(__name__)


async def execute(artifact_root: Path) -> int:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        dataset = await load_training_data(engine, settings)
        checksum = hashlib.sha256(
            pd.util.hash_pandas_object(
                dataset[["open_time", "target_label", "previous_direction", *FEATURE_COLUMNS]],
                index=False,
            )
            .to_numpy()
            .tobytes()
        ).hexdigest()
        log.info(
            "training_dataset_loaded",
            rows=len(dataset),
            start=dataset["open_time"].iloc[0].isoformat(),
            end=dataset["open_time"].iloc[-1].isoformat(),
            sha256=checksum,
        )
        outcome = run_walk_forward(
            dataset, interval_to_timedelta(settings.default_interval), TrainingOptions()
        )
        report = outcome.report
        report["dataset"] = {
            "rows": len(dataset),
            "start": dataset["open_time"].iloc[0].isoformat(),
            "end": dataset["open_time"].iloc[-1].isoformat(),
            "sha256": checksum,
            "symbol": settings.btc_symbol,
            "interval": settings.default_interval,
        }
        name = f"run-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        directory = artifact_root / name
        selected = outcome.candidates[report["selected_family"]]
        report["onnx_parity"] = export_bundle(selected, directory)
        (directory / "walk_forward.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        version = await register_model(engine, selected, report, directory)
        log.info(
            "training_complete",
            model_version=version,
            family=selected.family,
            artifact=str(directory / "model.onnx"),
            cv_accuracy=report["mean_accuracy"][selected.family],
            cv_baseline=report["best_baseline_mean_accuracy"],
            final_accuracy=report["final_fold_accuracy"],
            final_baseline=report["final_fold_best_baseline"],
            quality_passed=report["criteria"]["passed"],
            active=False,
        )
        if not report["criteria"]["passed"]:
            log.warning("quality_gate_failed", criteria=report["criteria"])
            return 2
        return 0
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrenamiento walk-forward (Fase 3)")
    parser.add_argument("--artifact-dir", type=Path, help="Directorio de artefactos")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    return asyncio.run(execute(args.artifact_dir or Path(settings.model_artifact_dir)))


if __name__ == "__main__":
    raise SystemExit(main())
