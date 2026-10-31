"""Compara hipótesis anteriores a un holdout explícito y evalúa ese holdout una sola vez."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pandas as pd
import structlog
from lightgbm import LGBMClassifier
from sqlalchemy.ext.asyncio import create_async_engine
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from src.config import get_settings
from src.evaluation.calibration import PlattCalibration
from src.evaluation.importance import xgb_importance
from src.evaluation.metrics import evaluate, evaluate_baselines
from src.export.onnx_model import export_bundle
from src.features.funding_context import add_funding_context
from src.features.price_context import add_price_context
from src.features.taker_flow import add_taker_flow
from src.ingestion.binance import interval_to_timedelta
from src.ingestion.funding_repository import FundingRepository
from src.ingestion.repository import IngestionRepository
from src.ingestion.run import configure_logging
from src.training.data import Fold, build_folds, load_training_data
from src.training.holdouts import assert_no_reserved_training_data, assert_unused_holdout
from src.training.registry import register_model
from src.training.research_models import candidates, fit_candidate
from src.training.snapshot import snapshot_hash
from src.training.walk_forward import ModelBundle

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.config import Settings
    from src.training.research_models import Candidate

log = structlog.get_logger(__name__)
SEED = 42
FOLDS = 10


def _candidate_pool(
    include_taker: bool, include_lightgbm: bool, include_funding: bool
) -> list[Candidate]:
    if not include_taker and not include_lightgbm and not include_funding:
        return candidates()
    if include_funding:
        return candidates(
            include_taker=include_taker,
            include_lightgbm=include_lightgbm,
            include_funding=True,
        )
    return candidates(include_taker=include_taker, include_lightgbm=include_lightgbm)


async def load_snapshot(
    engine: AsyncEngine,
    settings: Settings,
    start: pd.Timestamp,
    end: pd.Timestamp,
    include_taker: bool = False,
    include_funding: bool = False,
) -> pd.DataFrame:
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("El holdout requiere fechas válidas con zona horaria")
    base = await load_training_data(engine, settings)
    candles = await IngestionRepository(engine).load_candles(
        settings.btc_symbol, settings.default_interval
    )
    step = pd.Timedelta(interval_to_timedelta(settings.default_interval))
    data = add_price_context(base, candles, step.to_pytimedelta())
    if include_taker:
        data = add_taker_flow(data, candles, step.to_pytimedelta())
    if include_funding:
        data = add_funding_context(
            data,
            await FundingRepository(engine).load(settings.btc_symbol),
            step.to_pytimedelta(),
            timedelta(hours=settings.funding_feature_max_age_hours),
        )
    data = data.loc[data["open_time"] < end].reset_index(drop=True)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    if (start - epoch) % step or (end - epoch) % step:
        raise ValueError("Los límites del holdout deben estar alineados con las velas")
    expected = len(pd.date_range(start, end - step, freq=step))
    holdout = data.loc[(data["open_time"] >= start) & (data["open_time"] < end)]
    if expected == 0 or len(holdout) != expected:
        raise ValueError("El período reservado debe contener todas sus etiquetas conocidas")
    return data


def develop(
    data: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    step: timedelta,
    include_taker: bool = False,
    include_lightgbm: bool = False,
    include_funding: bool = False,
) -> dict[str, Any]:
    development = data.loc[data["open_time"] < start].reset_index(drop=True)
    folds = build_folds(development, step, n_folds=FOLDS)
    y = development["target_label"].to_numpy(dtype=int)
    previous = development["previous_direction"].to_numpy(dtype=int)
    fold_reports: list[dict[str, Any]] = []
    with threadpool_limits(limits=2):
        for fold in folds:
            log.info("research_fold_start", fold=fold.number, start=fold.start.isoformat())
            baseline = evaluate_baselines(
                y[fold.train],
                previous[fold.train],
                y[fold.validation],
                previous[fold.validation],
                SEED + fold.number,
            )
            metrics: dict[str, Any] = {}
            candidate_pool = _candidate_pool(include_taker, include_lightgbm, include_funding)
            for candidate in candidate_pool:
                fitted = fit_candidate(development, fold, step, candidate, SEED)
                for mode in ("raw", "platt"):
                    p = (
                        fitted.raw_validation
                        if mode == "raw"
                        else fitted.platt.transform(fitted.raw_validation)
                    )
                    metrics[f"{candidate.name}_{mode}"] = evaluate(
                        y[fold.validation], (p >= 0.5).astype(int), p
                    )
            fold_reports.append(
                {
                    "fold": fold.number,
                    "start": fold.start.isoformat(),
                    "end": fold.end.isoformat(),
                    "baselines": baseline,
                    "candidates": metrics,
                }
            )
            log.info("research_fold_done", fold=fold.number, candidates=len(metrics))

    keys = list(fold_reports[0]["candidates"])
    rankings = []
    for key in keys:
        rankings.append(
            {
                "candidate": key,
                "mean_accuracy": float(
                    np.mean([f["candidates"][key]["accuracy"] for f in fold_reports])
                ),
                "mean_log_loss": float(
                    np.mean([f["candidates"][key]["log_loss"] for f in fold_reports])
                ),
            }
        )
    rankings.sort(
        key=lambda result: (-result["mean_accuracy"], result["mean_log_loss"], result["candidate"])
    )
    strongest = float(
        np.mean(
            [max(item["accuracy"] for item in fold["baselines"].values()) for fold in fold_reports]
        )
    )
    return {
        "status": "selection_frozen",
        "seed": SEED,
        "holdout_start": start.isoformat(),
        "holdout_end_exclusive": end.isoformat(),
        "snapshot_sha256": snapshot_hash(
            data, include_taker=include_taker, include_funding=include_funding
        ),
        "include_taker": include_taker,
        "include_lightgbm": include_lightgbm,
        "include_funding": include_funding,
        "snapshot_rows": len(data),
        "holdout_rows": int(((data["open_time"] >= start) & (data["open_time"] < end)).sum()),
        "selection_rule": "max development mean accuracy; tie: lower log loss, then name",
        "candidate_count": len(rankings),
        "selected_candidate": rankings[0]["candidate"],
        "best_baseline_mean_accuracy": strongest,
        "rankings": rankings,
        "folds": fold_reports,
    }


def _holdout_fold(
    data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, step: timedelta
) -> Fold:
    times = data["open_time"]
    train = np.flatnonzero((times < start) & (times + 2 * pd.Timedelta(step) <= start))
    valid = np.flatnonzero((times >= start) & (times < end))
    if not len(train) or not len(valid):
        raise ValueError("Faltan datos de entrenamiento o evaluación")
    return Fold(FOLDS + 1, train, valid, start, end)


async def evaluate_holdout(
    engine: AsyncEngine,
    data: pd.DataFrame,
    development: dict[str, Any],
    directory: Path,
    step: timedelta,
) -> tuple[dict[str, Any], int]:
    if development["status"] != "selection_frozen":
        raise ValueError("La selección de candidatos no está congelada")
    include_taker = development.get("include_taker", False)
    include_lightgbm = development.get("include_lightgbm", False)
    include_funding = development.get("include_funding", False)
    if (
        snapshot_hash(data, include_taker=include_taker, include_funding=include_funding)
        != development["snapshot_sha256"]
    ):
        raise ValueError("El dataset cambió desde la selección; no se evaluará el holdout")
    start = pd.Timestamp(development["holdout_start"])
    end = pd.Timestamp(development["holdout_end_exclusive"])
    fold = _holdout_fold(data, start, end, step)
    selected = development["selected_candidate"]
    mode = "platt" if selected.endswith("_platt") else "raw"
    key = selected.removesuffix(f"_{mode}")
    candidate_pool = _candidate_pool(include_taker, include_lightgbm, include_funding)
    candidate = next((item for item in candidate_pool if item.name == key), None)
    if candidate is None:
        raise ValueError("El candidato congelado no existe")
    assert_no_reserved_training_data(directory.parent, data.iloc[fold.train]["open_time"], step)
    assert_unused_holdout(directory.parent, start, end, exclude=directory)
    lock = directory / "holdout.lock"
    with lock.open("x", encoding="utf-8") as file:
        file.write(f"Holdout iniciado {datetime.now(UTC).isoformat()}\n")

    y = data["target_label"].to_numpy(dtype=int)
    previous = data["previous_direction"].to_numpy(dtype=int)
    with threadpool_limits(limits=2):
        fitted = fit_candidate(data, fold, step, candidate, SEED)
    calibration = fitted.platt if mode == "platt" else PlattCalibration(1.0, 0.0)
    p = calibration.transform(fitted.raw_validation)
    prediction = (p >= 0.5).astype(int)
    metrics = evaluate(y[fold.validation], prediction, p)
    baseline = evaluate_baselines(
        y[fold.train],
        previous[fold.train],
        y[fold.validation],
        previous[fold.validation],
        SEED + 11,
    )
    best_baseline = max(item["accuracy"] for item in baseline.values())
    names = fitted.preprocessor.feature_names
    if isinstance(fitted.model, (XGBClassifier, LGBMClassifier)):
        importance = xgb_importance(fitted.model, fitted.scaled_validation, names)
    else:
        weights = np.abs(fitted.model.coef_[0])
        total = float(weights.sum())
        importance = {
            name: float(value / total) if total else 0.0
            for name, value in zip(names, weights, strict=True)
        }
    criteria = {
        "at_least_10_development_folds": len(development["folds"]) >= 10,
        "development_above_baseline_plus_3pp": (
            development["rankings"][0]["mean_accuracy"]
            > development["best_baseline_mean_accuracy"] + 0.03
        ),
        "holdout_above_baseline_plus_2pp": metrics["accuracy"] > best_baseline + 0.02,
        "predicts_both_directions": np.unique(prediction).size == 2,
        "no_dominant_feature": max(importance.values(), default=0.0) <= 0.5,
    }
    criteria["passed"] = all(criteria.values())
    bundle = ModelBundle(
        candidate.family,
        fitted.model,
        fitted.preprocessor,
        calibration,
        fitted.parameters,
        fitted.scaled_validation,
    )
    model_dir = directory / "model"
    parity = export_bundle(bundle, model_dir)
    result = {
        "selected_candidate": selected,
        "holdout_start": start.isoformat(),
        "holdout_end_exclusive": end.isoformat(),
        "snapshot_sha256": development["snapshot_sha256"],
        "development_accuracy": development["rankings"][0]["mean_accuracy"],
        "development_baseline": development["best_baseline_mean_accuracy"],
        "metrics": metrics,
        "baselines": baseline,
        "best_baseline_accuracy": best_baseline,
        "predicted_up": int(prediction.sum()),
        "predicted_down": int((1 - prediction).sum()),
        "parameters": fitted.parameters,
        "tuning_scores": fitted.tuning_scores,
        "calibration_mode": mode,
        "calibration": vars(calibration),
        "features": names,
        "importance": importance,
        "criteria": criteria,
        "onnx_parity": parity,
    }
    (directory / "holdout.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    registry_report = {
        "seed": SEED,
        "criteria": criteria,
        "final_fold_best_baseline": best_baseline,
        "folds": [
            {
                "models": {candidate.family: {"calibrated": metrics}},
                "calibration_start": (start - pd.DateOffset(months=1)).isoformat(),
                "validation_start": start.isoformat(),
                "validation_end_exclusive": end.isoformat(),
            }
        ],
    }
    version = await register_model(engine, bundle, registry_report, model_dir)
    log.info(
        "research_complete", version=version, candidate=selected, criteria=criteria, active=False
    )
    return result, version


async def execute(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    step = interval_to_timedelta(settings.default_interval)
    try:
        if args.command == "develop":
            start = pd.Timestamp(args.holdout_start)
            end = pd.Timestamp(args.holdout_end)
            assert_unused_holdout(Path(settings.model_artifact_dir), start, end)
            data = await load_snapshot(
                engine, settings, start, end, args.include_taker, args.include_funding
            )
            assert_no_reserved_training_data(
                Path(settings.model_artifact_dir),
                data.loc[data["open_time"] < start, "open_time"],
                step,
            )
            result = develop(
                data,
                start,
                end,
                step,
                args.include_taker,
                args.include_lightgbm,
                args.include_funding,
            )
            directory = Path(settings.model_artifact_dir) / (
                f"research-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
            )
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "development.json").write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
            )
            log.info(
                "research_selection_frozen",
                directory=str(directory),
                candidate=result["selected_candidate"],
                accuracy=result["rankings"][0]["mean_accuracy"],
                baseline=result["best_baseline_mean_accuracy"],
            )
            return 0
        directory = Path(args.directory)
        development = json.loads((directory / "development.json").read_text(encoding="utf-8"))
        start = pd.Timestamp(development["holdout_start"])
        end = pd.Timestamp(development["holdout_end_exclusive"])
        data = await load_snapshot(
            engine,
            settings,
            start,
            end,
            development.get("include_taker", False),
            development.get("include_funding", False),
        )
        if args.command == "compare":
            from src.training.compare_holdout import compare_fixed_models

            assert_unused_holdout(directory.parent, start, end, exclude=directory)
            result = compare_fixed_models(data, directory, Path(args.reference_directory))
            with (directory / "comparison.json").open("x", encoding="utf-8") as file:
                json.dump(result, file, indent=2, allow_nan=False)
            log.info(
                "fixed_models_compared",
                samples=result["samples"],
                accuracy_difference=result["accuracy_difference"],
                report=str(directory / "comparison.json"),
            )
            return 0
        result, _ = await evaluate_holdout(engine, data, development, directory, step)
        if not result["criteria"]["passed"]:
            log.warning("research_quality_gate_failed", criteria=result["criteria"])
            return 2
        return 0
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Experimentos con holdout reservado")
    subparsers = parser.add_subparsers(dest="command", required=True)
    develop_parser = subparsers.add_parser("develop")
    develop_parser.add_argument("--holdout-start", required=True)
    develop_parser.add_argument("--holdout-end", required=True)
    develop_parser.add_argument("--include-taker", action="store_true")
    develop_parser.add_argument("--include-lightgbm", action="store_true")
    develop_parser.add_argument("--include-funding", action="store_true")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--directory", required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--directory", required=True)
    compare_parser.add_argument("--reference-directory", required=True)
    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    return asyncio.run(execute(args))


if __name__ == "__main__":
    raise SystemExit(main())
