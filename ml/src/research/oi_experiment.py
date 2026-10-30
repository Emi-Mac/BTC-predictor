"""Ablación temporal anidada para un dataset de investigación congelado."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from src.config import get_settings
from src.evaluation.metrics import evaluate, evaluate_baselines
from src.features.compute import FEATURE_COLUMNS
from src.features.funding_context import FUNDING_FEATURE_COLUMNS
from src.features.price_context import PRICE_CONTEXT_COLUMNS
from src.ingestion.run import configure_logging
from src.research.dataset import verify_dataset
from src.research.metrics_features import OI_FEATURE_COLUMNS, POSITION_FEATURE_COLUMNS
from src.research.provenance import implementation_signature
from src.research.source_snapshot import file_sha256
from src.training.data import Fold, build_folds
from src.training.holdouts import assert_no_reserved_training_data
from src.training.research_models import _fit_logistic, _fit_preprocessor

if TYPE_CHECKING:
    from src.training.types import FloatArray, IntArray

log = structlog.get_logger(__name__)
SEED = 42
FOLDS = 10
REGULARIZATION_GRID = (0.003, 0.01, 0.03, 0.1)


def paired_fold_comparison(
    first: list[float], second: list[float], seed: int = SEED, draws: int = 5000
) -> dict[str, Any]:
    if len(first) != len(second) or len(first) < 2 or draws < 1:
        raise ValueError("Se requieren al menos dos folds pareados y remuestreos positivos")
    values = np.asarray([first, second], dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Las accuracies de los folds deben estar entre 0 y 1")
    difference = values[0] - values[1]
    samples = np.random.default_rng(seed).integers(0, len(difference), (draws, len(difference)))
    lower, upper = np.quantile(difference[samples].mean(axis=1), [0.025, 0.975])
    return {
        "mean_accuracy_difference": float(difference.mean()),
        "bootstrap_95_interval": [float(lower), float(upper)],
        "folds": len(difference),
        "draws": draws,
        "seed": seed,
    }


def variant_features() -> dict[str, list[str]]:
    base = [*FEATURE_COLUMNS, *PRICE_CONTEXT_COLUMNS, *FUNDING_FEATURE_COLUMNS]
    return {
        "base": base,
        "base_oi": [*base, *OI_FEATURE_COLUMNS],
        "base_oi_position": [*base, *OI_FEATURE_COLUMNS, *POSITION_FEATURE_COLUMNS],
    }


def inner_splits(
    data: pd.DataFrame, fold: Fold, step: timedelta
) -> list[tuple[IntArray, IntArray]]:
    times = pd.to_datetime(data["open_time"], utc=True)
    known_at = times + 2 * pd.Timedelta(step)
    splits: list[tuple[IntArray, IntArray]] = []
    for months_before in (3, 2, 1):
        start = fold.start - pd.DateOffset(months=months_before)
        end = start + pd.DateOffset(months=1)
        train = np.flatnonzero((times < start) & (known_at <= start))
        valid = np.flatnonzero((times >= start) & (times < end) & (known_at <= fold.start))
        if not len(train) or len(valid) < 100:
            raise ValueError(f"Fold {fold.number}: validación interna insuficiente")
        if start - times.iloc[train[0]] < pd.Timedelta(days=365):
            raise ValueError(f"Fold {fold.number}: menos de un año de entrenamiento interno")
        splits.append((train, valid))
    return splits


def _fit_variant(
    data: pd.DataFrame, fold: Fold, names: list[str], step: timedelta
) -> tuple[FloatArray, dict[str, Any]]:
    labels = data["target_label"].to_numpy(dtype=int)
    scores: dict[float, list[float]] = {value: [] for value in REGULARIZATION_GRID}
    for train, valid in inner_splits(data, fold, step):
        pre = _fit_preprocessor(data.iloc[train], names)
        x_train = pre.transform(data.iloc[train])
        x_valid = pre.transform(data.iloc[valid])
        for value in REGULARIZATION_GRID:
            model = _fit_logistic(x_train, labels[train], value, SEED)
            p = model.predict_proba(x_valid)[:, 1]
            scores[value].append(float(log_loss(labels[valid], p, labels=[0, 1])))
    mean_scores = {value: float(np.mean(items)) for value, items in scores.items()}
    best_c = min(mean_scores, key=lambda value: (mean_scores[value], value))
    pre = _fit_preprocessor(data.iloc[fold.train], names)
    model = _fit_logistic(pre.transform(data.iloc[fold.train]), labels[fold.train], best_c, SEED)
    probability: FloatArray = model.predict_proba(pre.transform(data.iloc[fold.validation]))[:, 1]
    return probability, {
        "C": best_c,
        "inner_log_loss": {str(value): mean_scores[value] for value in REGULARIZATION_GRID},
        "train_samples": len(fold.train),
        "feature_names": names,
    }


def run_ablation(
    data: pd.DataFrame, cutoff: pd.Timestamp, step: timedelta
) -> tuple[dict[str, Any], pd.DataFrame]:
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("El corte de desarrollo debe incluir zona horaria")
    development = data.loc[
        (data["open_time"] < cutoff)
        & (data["open_time"] + 2 * pd.Timedelta(step) <= cutoff)
        & data["target_label"].notna()
    ].reset_index(drop=True)
    folds = build_folds(development, step, n_folds=FOLDS)
    labels = development["target_label"].to_numpy(dtype=int)
    previous = development["previous_direction"].to_numpy(dtype=int)
    variants = variant_features()
    results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    with threadpool_limits(limits=2):
        for fold in folds:
            baseline = evaluate_baselines(
                labels[fold.train],
                previous[fold.train],
                labels[fold.validation],
                previous[fold.validation],
                SEED + fold.number,
            )
            metrics: dict[str, Any] = {}
            tuning: dict[str, Any] = {}
            predictions = pd.DataFrame(
                {
                    "open_time": development["open_time"].iloc[fold.validation].to_numpy(),
                    "target_label": labels[fold.validation],
                    "previous_direction": previous[fold.validation],
                    "fold": fold.number,
                }
            )
            for name, names in variants.items():
                probability, tuning[name] = _fit_variant(development, fold, names, step)
                predictions[f"{name}_probability"] = probability
                metrics[name] = evaluate(
                    labels[fold.validation], (probability >= 0.5).astype(int), probability
                )
            results.append(
                {
                    "fold": fold.number,
                    "start": fold.start.isoformat(),
                    "end": fold.end.isoformat(),
                    "baselines": baseline,
                    "models": metrics,
                    "tuning": tuning,
                }
            )
            prediction_frames.append(predictions)
            log.info(
                "oi_ablation_fold",
                fold=fold.number,
                base=metrics["base"]["accuracy"],
                oi=metrics["base_oi"]["accuracy"],
                positioning=metrics["base_oi_position"]["accuracy"],
            )
    means = {
        name: {
            metric: float(np.mean([item["models"][name][metric] for item in results]))
            for metric in ("accuracy", "f1_macro", "log_loss", "brier")
        }
        for name in variants
    }
    baseline_means = {
        name: float(np.mean([item["baselines"][name]["accuracy"] for item in results]))
        for name in ("majority", "previous_direction", "conditional_direction", "weighted_random")
    }
    comparisons = {
        "base_oi_minus_base": paired_fold_comparison(
            [item["models"]["base_oi"]["accuracy"] for item in results],
            [item["models"]["base"]["accuracy"] for item in results],
        ),
        "base_oi_position_minus_base": paired_fold_comparison(
            [item["models"]["base_oi_position"]["accuracy"] for item in results],
            [item["models"]["base"]["accuracy"] for item in results],
        ),
        "base_minus_conditional": paired_fold_comparison(
            [item["models"]["base"]["accuracy"] for item in results],
            [item["baselines"]["conditional_direction"]["accuracy"] for item in results],
        ),
    }
    return {
        "evaluation": "exploratory_development_only",
        "independent_holdout_evaluated": False,
        "model_registered": False,
        "development_end_exclusive": cutoff.isoformat(),
        "seed": SEED,
        "grid_C": list(REGULARIZATION_GRID),
        "nested_inner_months": 3,
        "base_model_uses_all_labels_known_before_validation": True,
        "source_availability_point_in_time_verified": False,
        "models": means,
        "baseline_mean_accuracy": baseline_means,
        "paired_fold_comparisons": comparisons,
        "uncertainty_note": (
            "Bootstrap exploratorio de folds mensuales pareados; no corrige selección múltiple, "
            "dependencia temporal ni solapamiento de los conjuntos de entrenamiento."
        ),
        "best_baseline_per_fold_mean": float(
            np.mean([max(b["accuracy"] for b in item["baselines"].values()) for item in results])
        ),
        "folds": results,
    }, pd.concat(prediction_frames, ignore_index=True)


def execute(directory: Path, cutoff: pd.Timestamp, artifact_root: Path) -> Path:
    manifest = verify_dataset(directory)
    implementation = implementation_signature()
    data = pd.read_parquet(directory / manifest["files"]["features"]["path"])
    step = timedelta(hours=4)
    development = data.loc[
        (data["open_time"] < cutoff) & (data["open_time"] + 2 * pd.Timedelta(step) <= cutoff),
        "open_time",
    ]
    assert_no_reserved_training_data(artifact_root, development, step)
    report, predictions = run_ablation(data, cutoff, step)
    if implementation_signature() != implementation:
        raise RuntimeError("La implementación cambió durante la evaluación")
    report["dataset_id"] = manifest["dataset_id"]
    report["dataset_manifest_sha256"] = manifest["manifest_sha256"]
    report["dataset_quality"] = manifest["quality"]
    report["implementation"] = implementation
    name = f"oi-ablation-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    target = artifact_root / name
    target.mkdir(parents=True, exist_ok=False)
    predictions_path = target / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    report["predictions_sha256"] = file_sha256(predictions_path)
    path = target / "ablation.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    log.info("oi_ablation_complete", report=str(path), models=report["models"])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablación de OI con validación temporal anidada")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--development-end", required=True)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    execute(args.dataset, pd.Timestamp(args.development_end), Path(settings.model_artifact_dir))


if __name__ == "__main__":
    main()
