"""Compara el modelo congelado con y sin flujo agresor solo en desarrollo."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import structlog
from sqlalchemy.ext.asyncio import create_async_engine
from threadpoolctl import threadpool_limits

from src.config import get_settings
from src.evaluation.metrics import evaluate, evaluate_baselines
from src.features.funding_context import FUNDING_FEATURE_COLUMNS, add_funding_context
from src.features.price_context import add_price_context
from src.features.taker_flow import TAKER_FEATURE_COLUMNS, add_taker_flow
from src.ingestion.binance import interval_to_timedelta
from src.ingestion.funding_repository import FundingRepository
from src.ingestion.repository import IngestionRepository
from src.ingestion.run import configure_logging
from src.training.data import build_folds, load_training_data
from src.training.research_models import Candidate, fit_candidate
from src.training.snapshot import snapshot_hash

log = structlog.get_logger(__name__)
FOLDS = 10
SEED = 42


def run_ablation(
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    step: timedelta,
    include_lightgbm: bool = False,
    include_funding: bool = False,
) -> dict[str, Any]:
    if cutoff.tzinfo is None:
        raise ValueError("El corte de desarrollo debe incluir zona horaria")
    development = data.loc[data["open_time"] < cutoff].reset_index(drop=True)
    folds = build_folds(development, step, n_folds=FOLDS)
    labels = development["target_label"].to_numpy(dtype=int)
    previous = development["previous_direction"].to_numpy(dtype=int)
    variants = {
        "without_taker": Candidate("logistic", "expanded", None),
        "taker_ratio": Candidate("logistic", "expanded_taker_ratio", None),
        "taker_multiscale": Candidate("logistic", "expanded_taker_multiscale", None),
        "with_taker": Candidate("logistic", "expanded_taker", None),
    }
    if include_lightgbm:
        variants.update(
            {
                "lightgbm_without_taker": Candidate("lightgbm", "expanded", None),
                "lightgbm_with_taker": Candidate("lightgbm", "expanded_taker", None),
            }
        )
    if include_funding:
        variants.update(
            {
                "funding_without_taker": Candidate("logistic", "expanded_funding", None),
                "funding_with_taker": Candidate("logistic", "expanded_taker_funding", None),
            }
        )
        if include_lightgbm:
            variants.update(
                {
                    "lightgbm_funding_without_taker": Candidate(
                        "lightgbm", "expanded_funding", None
                    ),
                    "lightgbm_funding_with_taker": Candidate(
                        "lightgbm", "expanded_taker_funding", None
                    ),
                }
            )
    results: list[dict[str, Any]] = []
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
            for name, candidate in variants.items():
                fitted = fit_candidate(development, fold, step, candidate, SEED)
                probability = fitted.raw_validation
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
                }
            )
            log.info(
                "taker_ablation_fold",
                fold=fold.number,
                without_taker=metrics["without_taker"]["accuracy"],
                with_taker=metrics["with_taker"]["accuracy"],
            )
    means = {
        name: float(np.mean([item["models"][name]["accuracy"] for item in results]))
        for name in variants
    }
    without, with_taker = means["without_taker"], means["with_taker"]
    flow_variants = ["taker_ratio", "taker_multiscale", "with_taker"]
    if include_lightgbm:
        flow_variants.append("lightgbm_with_taker")
    best_flow = max(flow_variants, key=lambda name: (means[name], name))
    baseline_mean = float(
        np.mean([max(b["accuracy"] for b in item["baselines"].values()) for item in results])
    )
    return {
        "evaluation": "exploratory_development_only",
        "independent_holdout_evaluated": False,
        "development_end_exclusive": cutoff.isoformat(),
        "snapshot_sha256": snapshot_hash(
            development, include_taker=True, include_funding=include_funding
        ),
        "rows": len(development),
        "rows_with_missing_taker_features": int(
            development[TAKER_FEATURE_COLUMNS].isna().any(axis=1).sum()
        ),
        "mean_accuracy_without_taker": without,
        "mean_accuracy_with_taker": with_taker,
        "mean_accuracy_difference": with_taker - without,
        "variant_mean_accuracy": means,
        "best_flow_variant": best_flow,
        "best_flow_development_difference": means[best_flow] - without,
        "include_lightgbm": include_lightgbm,
        "include_funding": include_funding,
        "rows_with_missing_funding_features": (
            int(development[FUNDING_FEATURE_COLUMNS].isna().any(axis=1).sum())
            if include_funding
            else None
        ),
        "funding_development_difference": (
            means["funding_without_taker"] - without if include_funding else None
        ),
        "lightgbm_flow_difference": (
            means["lightgbm_with_taker"] - means["lightgbm_without_taker"]
            if include_lightgbm
            else None
        ),
        "note": "La comparación de subconjuntos se añadió tras observar la primera ablación.",
        "mean_strongest_baseline": baseline_mean,
        "folds": results,
    }


async def execute(
    cutoff: pd.Timestamp,
    artifact_root: Path,
    include_lightgbm: bool = False,
    include_funding: bool = False,
) -> Path:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        base = await load_training_data(engine, settings)
        candles = await IngestionRepository(engine).load_candles(
            settings.btc_symbol, settings.default_interval
        )
        step = interval_to_timedelta(settings.default_interval)
        data = add_price_context(base, candles, step)
        data = add_taker_flow(data, candles, step)
        if include_funding:
            data = add_funding_context(
                data,
                await FundingRepository(engine).load(settings.btc_symbol),
                step,
                timedelta(hours=settings.funding_feature_max_age_hours),
            )
        report = run_ablation(data, cutoff, step, include_lightgbm, include_funding)
        directory = (
            artifact_root / f"taker-ablation-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        )
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "ablation.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        log.info(
            "taker_ablation_complete",
            report=str(path),
            without_taker=report["mean_accuracy_without_taker"],
            with_taker=report["mean_accuracy_with_taker"],
            baseline=report["mean_strongest_baseline"],
        )
        return path
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablación temporal del volumen agresor")
    parser.add_argument(
        "--development-end", required=True, help="Corte exclusivo ISO 8601 con zona horaria"
    )
    parser.add_argument("--include-lightgbm", action="store_true")
    parser.add_argument("--include-funding", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(
        execute(
            pd.Timestamp(args.development_end),
            Path(settings.model_artifact_dir),
            args.include_lightgbm,
            args.include_funding,
        )
    )


if __name__ == "__main__":
    main()
