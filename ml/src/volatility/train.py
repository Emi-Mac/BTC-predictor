"""Entrenamiento de los modelos de volatilidad: `python -m src.volatility.train --horizon 6|42`.

Por horizonte ajusta el primario (`har_x_live`) y el respaldo (`har`) con todo el desarrollo y
guarda artefactos JSON con coeficientes, varianza residual, umbrales de régimen, métricas de
validación walk-forward mensual (como E-12) y la semilla de calibración (residuos
estandarizados fuera de muestra de los últimos dos años). Los modelos se registran **inactivos**;
`python -m src.volatility.activation` los activa de forma explícita.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy.ext.asyncio import create_async_engine
from threadpoolctl import threadpool_limits

from src.config import get_settings
from src.ingestion.run import configure_logging
from src.research import aux_sources
from src.research import evaluation_harness as harness
from src.research import volatility_study as e12
from src.research.dataset import verify_dataset
from src.training.holdouts import assert_no_reserved_training_data
from src.volatility import calibration
from src.volatility.features import EPSILON, HORIZONS, STEP
from src.volatility.model import VolatilityArtifact, fit_artifact, horizon_sigma, regime_thresholds
from src.volatility.registry import register
from src.volatility.study import Bundle, pool, predict

if TYPE_CHECKING:
    from collections.abc import Callable

log = structlog.get_logger(__name__)
ROLE_KIND = {"primary": "har_x_live", "fallback": "har"}
PRIMARY_NOTE = "primario: har_x_live (sin OI); E-12b"
FALLBACK_NOTE = "respaldo: HAR sin exógenas; E-12b"


def horizon_name(candles: int) -> str:
    for name, value in HORIZONS.items():
        if value == candles:
            return name
    raise ValueError(f"Horizonte no soportado: {candles} (usa {sorted(HORIZONS.values())})")


def calibration_seed(
    predictions: pd.DataFrame, model: str, horizon: str, close: pd.Series
) -> dict[str, list[Any]]:
    """Residuos estandarizados fuera de muestra de la última ventana de calibración."""
    candles = HORIZONS[horizon]
    times = pd.to_datetime(predictions["open_time"], utc=True).reset_index(drop=True)
    returns = calibration.horizon_log_returns(close, candles).reindex(times).to_numpy()
    sigma = horizon_sigma(predictions[f"{model}_{horizon}"].to_numpy(dtype=float), candles)
    z = calibration.standardized_residual(returns, sigma)
    keep = np.isfinite(z) & (times > times.iloc[-1] - calibration.WINDOW).to_numpy()
    issued = times[keep] + STEP
    resolved = issued + candles * STEP
    return {
        "issued_at": [t.isoformat() for t in issued],
        "resolved_at": [t.isoformat() for t in resolved],
        "z": [float(v) for v in z[keep]],
    }


def validation_summary(predictions: pd.DataFrame, model: str, horizon: str) -> dict[str, Any]:
    realized = predictions[f"target_var_{horizon}"].to_numpy(dtype=float)
    forecast = predictions[f"{model}_{horizon}"].to_numpy(dtype=float)
    naive = predictions[f"naive_{horizon}"].to_numpy(dtype=float)
    metrics = e12.forecast_metrics(realized, forecast)
    log_real = np.log(np.maximum(realized, EPSILON))
    log_error = (log_real - np.log(forecast)) ** 2
    naive_error = (log_real - np.log(np.maximum(naive, EPSILON))) ** 2
    return {
        "oos_samples": len(predictions),
        "first_validation_start": predictions["open_time"].iloc[0].isoformat(),
        "qlike": metrics["qlike"],
        "qlike_naive": float(e12.qlike(realized, naive).mean()),
        "mincer_zarnowitz_slope": metrics["mincer_zarnowitz_slope"],
        "oos_r2_log_vs_naive": float(1 - log_error.mean() / naive_error.mean()),
    }


def train_artifacts(
    bundle: Bundle,
    development: pd.DataFrame,
    candles: int,
    n_folds: int,
    provenance: dict[str, Any],
    predictor: Callable[..., Any] = predict,
) -> dict[str, VolatilityArtifact]:
    """Artefactos por rol (`primary`, `fallback`) para un horizonte."""
    horizon = horizon_name(candles)
    with threadpool_limits(limits=2):
        predictions = e12.run_walk_forward(development, n_folds, pool(), predictor)
    artifacts: dict[str, VolatilityArtifact] = {}
    for role, kind in ROLE_KIND.items():
        artifact = fit_artifact(kind, development, horizon)
        # Los umbrales de régimen salen de los pronósticos fuera de muestra, no del ajuste.
        oos_vol = np.sqrt(predictions[f"{kind}_{horizon}"].to_numpy(dtype=float))
        artifacts[role] = replace(
            artifact,
            regime_thresholds=regime_thresholds(oos_vol),
            calibration_seed=calibration_seed(predictions, kind, horizon, bundle.close),
            validation={**validation_summary(predictions, kind, horizon), **provenance},
        )
    return artifacts


def write_artifacts(
    artifacts: dict[str, VolatilityArtifact], root: Path, name: str
) -> dict[str, Path]:
    target = root / name
    target.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {}
    for role, artifact in artifacts.items():
        path = target / f"h{artifact.horizon_candles}-{role}-{artifact.kind}.json"
        artifact.save(path)
        paths[role] = path
    return paths


async def _register(paths: dict[str, Path]) -> dict[str, int]:
    engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
    notes = {"primary": PRIMARY_NOTE, "fallback": FALLBACK_NOTE}
    try:
        return {
            role: await register(engine, path, role, notes[role]) for role, path in paths.items()
        }
    finally:
        await engine.dispose()


def execute(
    dataset: Path,
    aux_directory: Path,
    cutoff: pd.Timestamp,
    artifact_root: Path,
    candles: int,
    n_folds: int,
    register_models: bool,
) -> dict[str, Any]:
    dataset_manifest = verify_dataset(dataset)
    aux_manifest = aux_sources.verify(aux_directory)
    bundle = Bundle(dataset, aux_directory)
    development = e12.development_rows(bundle.data, cutoff)
    assert_no_reserved_training_data(artifact_root, development["open_time"], STEP)
    provenance = {
        "study": "E-12b",
        "dataset_id": dataset_manifest["dataset_id"],
        "aux_id": aux_manifest["aux_id"],
        "development_end_exclusive": cutoff.isoformat(),
    }
    artifacts = train_artifacts(bundle, development, candles, n_folds, provenance)
    name = f"volatility-models-h{candles}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    paths = write_artifacts(artifacts, artifact_root, name)
    versions = asyncio.run(_register(paths)) if register_models else {}
    summary = {
        "horizon_candles": candles,
        "artifacts": {role: str(path) for role, path in paths.items()},
        "registered_versions": versions,
        "validation": {role: a.validation for role, a in artifacts.items()},
        "regime_thresholds": {role: a.regime_thresholds for role, a in artifacts.items()},
    }
    log.info("volatility_models_trained", **{k: v for k, v in summary.items() if k != "validation"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena los modelos de volatilidad")
    parser.add_argument("--horizon", type=int, required=True, choices=sorted(HORIZONS.values()))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--aux", type=Path, required=True)
    parser.add_argument("--development-end", required=True)
    parser.add_argument("--folds", type=int, default=harness.WALK_FORWARD_FOLDS)
    parser.add_argument("--register", action="store_true", help="registra los modelos (inactivos)")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    summary = execute(
        args.dataset,
        args.aux,
        pd.Timestamp(args.development_end),
        Path(settings.model_artifact_dir),
        args.horizon,
        args.folds,
        args.register,
    )
    print(json.dumps(summary, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
