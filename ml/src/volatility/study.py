"""E-12b: `har_x_live` (sin interés abierto) con las reglas de producción, calibración y paridad.

Responde a los criterios 1–3 de la sección 9 del plan de integración de la volatilidad:

1. `har_x_live` mejora a HAR en QLIKE a 7d con IC por bloques > 0, y pierde menos de 0.005 de
   QLIKE frente a HAR-X (que sí usa OI con 48 h de demora simulada).
2. Bandas con multiplicadores conformes (ventana móvil, solo pronósticos ya resueltos): la
   cobertura de 68/90/95% en periodos posteriores al ajuste contiene el nivel nominal.
3. Paridad: las entradas calculadas con la ruta de producción (`live_inputs` sobre tablas
   crudas y ventanas finales) coinciden con las de la investigación.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pandas as pd
import structlog
from threadpoolctl import threadpool_limits

from src.config import get_settings
from src.evaluation.robustness import block_bootstrap_mean
from src.features.long_horizon import build_long_horizon_features, price_history
from src.ingestion.run import configure_logging
from src.research import aux_sources
from src.research import evaluation_harness as harness
from src.research import volatility_study as e12
from src.research.dataset import verify_dataset
from src.research.source_snapshot import file_sha256
from src.training.holdouts import assert_no_reserved_training_data
from src.volatility import calibration
from src.volatility.features import (
    HAR_COLUMNS,
    HAR_X_EXTRA,
    HAR_X_LIVE_EXTRA,
    HORIZONS,
    LIVE_WINDOW_CANDLES,
    STEP,
    live_inputs,
    model_inputs,
)
from src.volatility.model import fit_artifact, horizon_sigma, predict_variance

if TYPE_CHECKING:
    from src.training.types import FloatArray

log = structlog.get_logger(__name__)
SEED = harness.SEED
BLOCK = e12.BLOCK
PRIMARY_HORIZON = "7d"
QLIKE_TOLERANCE = 0.005
PARITY_INPUT_TOLERANCE = 1e-9
PARITY_FORECAST_TOLERANCE = 1e-6
PARITY_SAMPLES = 25
IMPLEMENTATION_FILES = (
    "volatility/study.py",
    "volatility/features.py",
    "volatility/model.py",
    "volatility/calibration.py",
    "research/volatility_study.py",
    "features/long_horizon.py",
    "research/aux_sources.py",
    "evaluation/robustness.py",
)
LIVE_COLUMNS = [*HAR_COLUMNS, *HAR_X_LIVE_EXTRA]
ARTIFACT_KINDS = {"har": "har", "har_x_live": "har_x_live"}


def pool() -> list[e12.VolModel]:
    return [
        e12.VolModel("naive", (), "naive"),
        e12.VolModel("har", tuple(HAR_COLUMNS), "linear"),
        e12.VolModel("har_x", (*HAR_COLUMNS, *HAR_X_EXTRA), "linear"),
        e12.VolModel("har_x_live", tuple(LIVE_COLUMNS), "linear"),
    ]


def predict(
    model: e12.VolModel, train: pd.DataFrame, test: pd.DataFrame, horizon: str
) -> FloatArray:
    """HAR y HAR-X-live con el artefacto de producción; HAR-X (con OI) con el código de E-12."""
    kind = ARTIFACT_KINDS.get(model.name)
    if kind is None:
        return e12.fit_predict_variance(model, train, test, horizon)
    return predict_variance(fit_artifact(kind, train, horizon), test)


# ─── Datos ───────────────────────────────────────────────────────────────────


class Bundle:
    """Entradas crudas y derivadas del dataset congelado para la investigación y la paridad."""

    def __init__(self, dataset: Path, aux_directory: Path) -> None:
        manifest = verify_dataset(dataset)
        self.features = pd.read_parquet(dataset / manifest["files"]["features"]["path"])
        self.metrics = pd.read_parquet(dataset / manifest["files"]["metrics"]["path"])
        source_manifest = json.loads((dataset / "source-manifest.json").read_text(encoding="utf-8"))
        self.sources = {
            name: pd.read_parquet(dataset / item["path"])
            for name, item in source_manifest["files"].items()
        }
        self.aux = aux_sources.load(aux_directory)
        self.enriched = build_long_horizon_features(
            self.features, self.sources, self.metrics, self.aux
        )
        candles = self.sources["candles"].copy()
        candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True)
        self.close = price_history(candles, self.aux["btc_spot_klines"])
        self.data = model_inputs(self.enriched, self.close)


# ─── Evaluación ──────────────────────────────────────────────────────────────


def accuracy_report(predictions: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        realized = predictions[f"target_var_{horizon}"].to_numpy(dtype=float)
        loss = {
            name: e12.qlike(realized, predictions[f"{name}_{horizon}"].to_numpy(dtype=float))
            for name in ("naive", "har", "har_x", "har_x_live")
        }
        metrics = {
            name: e12.forecast_metrics(
                realized, predictions[f"{name}_{horizon}"].to_numpy(dtype=float)
            )
            for name in loss
        }
        result[horizon] = {
            "models": metrics,
            "har_minus_naive": block_bootstrap_mean(loss["naive"] - loss["har"], BLOCK, seed=SEED),
            "har_x_live_minus_har": block_bootstrap_mean(
                loss["har"] - loss["har_x_live"], BLOCK, seed=SEED
            ),
            "har_x_live_minus_naive": block_bootstrap_mean(
                loss["naive"] - loss["har_x_live"], BLOCK, seed=SEED
            ),
            "qlike_loss_vs_har_x": float(loss["har_x_live"].mean() - loss["har_x"].mean()),
            "har_x_minus_har_x_live": block_bootstrap_mean(
                loss["har_x_live"] - loss["har_x"], BLOCK, seed=SEED
            ),
        }
    return result


def calibration_report(predictions: pd.DataFrame, close: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    times = pd.to_datetime(predictions["open_time"], utc=True).reset_index(drop=True)
    for horizon, candles in HORIZONS.items():
        returns = calibration.horizon_log_returns(close, candles).reindex(times).to_numpy()
        var = predictions[f"har_x_live_{horizon}"].to_numpy(dtype=float)
        sigma = horizon_sigma(var, candles)
        z = calibration.standardized_residual(returns, sigma)
        k = calibration.rolling_multipliers(times, z, candles)
        inside = calibration.coverage(z, k)
        evaluated = inside.notna().all(axis=1)
        if evaluated.sum() < BLOCK:  # sin datos suficientes para un IC por bloques
            result[horizon] = {
                "evaluated": int(evaluated.sum()),
                "all_levels_contain_nominal": False,
            }
            continue
        gaussian = pd.DataFrame(
            {
                f"inside_{round(level * 100)}": (z <= calibration.GAUSSIAN_K[level]).astype(float)
                for level in calibration.LEVELS
            }
        )[evaluated.to_numpy()]
        by_year = {}
        years = times.dt.year[evaluated.to_numpy()].to_numpy()
        for year in sorted(set(years)):
            rows = inside[evaluated].loc[years == year]
            by_year[str(year)] = {column: float(rows[column].mean()) for column in rows.columns} | {
                "n": int(len(rows))
            }
        summary = calibration.summarize_levels(inside[evaluated], BLOCK, SEED)
        result[horizon] = {
            "evaluated": int(evaluated.sum()),
            "first_evaluated": times[evaluated].iloc[0].isoformat(),
            "conformal": summary,
            "gaussian_coverage": {c: float(gaussian[c].mean()) for c in gaussian.columns},
            "mean_k": {c: float(k.loc[evaluated, c].mean()) for c in k.columns},
            "by_year": by_year,
            "all_levels_contain_nominal": all(
                entry["ci_contains_nominal"] for entry in summary.values()
            ),
        }
    return result


def trailing_inputs(bundle: Bundle, position: int) -> pd.Series:
    """Fila de entradas en `position` calculada solo con las ventanas finales (producción)."""
    base = bundle.features[["open_time", *("vix_close", "volume_sma_ratio")]].copy()
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    base = base.sort_values("open_time").reset_index(drop=True)
    window = base.iloc[max(0, position - LIVE_WINDOW_CANDLES + 1) : position + 1]
    decision = window["open_time"].iloc[-1] + STEP
    first = window["open_time"].iloc[0]
    close = bundle.close.loc[first : window["open_time"].iloc[-1]]
    funding = bundle.sources["funding"]
    funding = funding.loc[pd.to_datetime(funding["funding_time"], utc=True) <= decision].tail(120)
    dvol = bundle.aux["deribit_dvol"]
    stamps = pd.to_datetime(dvol["open_time"], utc=True)
    dvol = dvol.loc[(stamps <= decision) & (stamps > decision - pd.Timedelta(days=30))]
    return live_inputs(window, close, funding, dvol, bundle.aux["fomc"]).iloc[-1]


def parity_report(bundle: Bundle, development: pd.DataFrame, horizon: str) -> dict[str, Any]:
    """Compara la ruta de producción con la de investigación (entradas y pronóstico)."""
    full = live_inputs(
        bundle.features[["open_time", "vix_close", "volume_sma_ratio"]],
        bundle.close,
        bundle.sources["funding"],
        bundle.aux["deribit_dvol"],
        bundle.aux["fomc"],
    )
    research = bundle.data.set_index("open_time")[LIVE_COLUMNS]
    candidate = full.set_index("open_time")[LIVE_COLUMNS]
    full_diff = float(np.nanmax(np.abs(research.to_numpy() - candidate.to_numpy())))
    same_nan = bool((research.isna().to_numpy() == candidate.isna().to_numpy()).all())

    artifact = fit_artifact("har_x_live", development, horizon)
    ordered = bundle.features.sort_values("open_time").reset_index(drop=True)
    positions = [
        int(i)
        for i in np.linspace(1300, len(ordered) - 1, PARITY_SAMPLES).round()
        if pd.notna(bundle.data.loc[int(i), LIVE_COLUMNS]).all()
    ]
    trailing_diffs, forecast_diffs = [], []
    for position in positions:
        row = trailing_inputs(bundle, position)
        expected = bundle.data.loc[position, LIVE_COLUMNS].to_numpy(dtype=float)
        trailing_diffs.append(
            float(np.max(np.abs(row[LIVE_COLUMNS].to_numpy(dtype=float) - expected)))
        )
        production = predict_variance(artifact, pd.DataFrame([row]))[0]
        reference = predict_variance(artifact, bundle.data.iloc[[position]])[0]
        forecast_diffs.append(abs(production - reference) / reference)
    return {
        "full_history_max_abs_diff": full_diff,
        "full_history_same_missing_pattern": same_nan,
        "trailing_window_samples": len(positions),
        "trailing_window_max_abs_diff": max(trailing_diffs),
        "forecast_max_relative_diff": max(forecast_diffs),
    }


def build_report(
    bundle: Bundle, development: pd.DataFrame, n_folds: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    models = pool()
    with threadpool_limits(limits=2):
        predictions = e12.run_walk_forward(development, n_folds, models, predict)
    accuracy = accuracy_report(predictions)
    calibration_result = calibration_report(predictions, bundle.close)
    parity = parity_report(bundle, development, PRIMARY_HORIZON)
    primary = accuracy[PRIMARY_HORIZON]
    gate = {
        "har_x_live_beats_har_ci_95": primary["har_x_live_minus_har"]["ci_95"],
        "criterion_1a_beats_har": bool(primary["har_x_live_minus_har"]["ci_95"][0] > 0),
        "qlike_loss_vs_har_x": primary["qlike_loss_vs_har_x"],
        "criterion_1b_loss_vs_har_x_below_tolerance": bool(
            primary["qlike_loss_vs_har_x"] < QLIKE_TOLERANCE
        ),
        "calibration_7d_levels_contain_nominal": calibration_result[PRIMARY_HORIZON][
            "all_levels_contain_nominal"
        ],
        "calibration_1d_levels_contain_nominal": calibration_result["1d"][
            "all_levels_contain_nominal"
        ],
        "criterion_3_inputs_match": bool(
            parity["full_history_max_abs_diff"] < PARITY_INPUT_TOLERANCE
            and parity["full_history_same_missing_pattern"]
            and parity["trailing_window_max_abs_diff"] < PARITY_INPUT_TOLERANCE
        ),
        "criterion_3_forecast_match": bool(
            parity["forecast_max_relative_diff"] < PARITY_FORECAST_TOLERANCE
        ),
    }
    gate["har_x_live_accepted"] = bool(
        gate["criterion_1a_beats_har"] and gate["criterion_1b_loss_vs_har_x_below_tolerance"]
    )
    report = {
        "evaluation": "volatility_live_study",
        "model_registered": False,
        "seed": SEED,
        "horizons_candles": HORIZONS,
        "walk_forward": {
            "folds": int(predictions["fold"].nunique()),
            "oos_samples": len(predictions),
            "first_validation_start": predictions["open_time"].iloc[0].isoformat(),
        },
        "pre_specified": {
            "primary_horizon": PRIMARY_HORIZON,
            "live_features": LIVE_COLUMNS,
            "excluded": ["oi_change_7d"],
            "acceptance": (
                "har_x_live mejora a HAR en QLIKE a 7d con IC 95% por bloques de 180 velas > 0 "
                f"y su QLIKE medio supera al de HAR-X en menos de {QLIKE_TOLERANCE}"
            ),
            "calibration": {
                "levels": list(calibration.LEVELS),
                "window_days": calibration.WINDOW.days,
                "min_resolved": calibration.MIN_RESOLVED,
                "block": BLOCK,
            },
        },
        "accuracy": accuracy,
        "calibration": calibration_result,
        "parity": parity,
        "gate": gate,
        "caveats": [
            harness.POINT_IN_TIME_CAVEAT,
            "La calibración solo evalúa pronósticos con >= MIN_RESOLVED resueltos previos "
            "dentro de la muestra fuera de muestra; el resto del periodo queda sin evaluar.",
            "HAR y HAR-X-live usan float64 (artefacto); HAR-X conserva el código de E-12.",
        ],
    }
    return report, predictions


def execute(
    dataset: Path, aux_directory: Path, cutoff: pd.Timestamp, artifact_root: Path, n_folds: int
) -> Path:
    root = Path(__file__).resolve().parents[1]
    implementation = {name: file_sha256(root / name) for name in IMPLEMENTATION_FILES}
    aux_manifest = aux_sources.verify(aux_directory)
    dataset_manifest = verify_dataset(dataset)
    bundle = Bundle(dataset, aux_directory)
    development = e12.development_rows(bundle.data, cutoff)
    assert_no_reserved_training_data(artifact_root, development["open_time"], STEP)
    report, predictions = build_report(bundle, development, n_folds)
    if {name: file_sha256(root / name) for name in IMPLEMENTATION_FILES} != implementation:
        raise RuntimeError("La implementación cambió durante la evaluación")
    name = f"volatility-live-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    report.update(
        {
            "run_id": name,
            "development_end_exclusive": cutoff.isoformat(),
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
            "aux_id": aux_manifest["aux_id"],
            "aux_manifest_sha256": aux_manifest["manifest_sha256"],
            "implementation": implementation,
        }
    )
    target = artifact_root / name
    target.mkdir(parents=True, exist_ok=False)
    predictions_path = target / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    report["predictions_sha256"] = file_sha256(predictions_path)
    path = target / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    harness.append_registry(
        artifact_root,
        {
            "run_id": name,
            "kind": report["evaluation"],
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_id": dataset_manifest["dataset_id"],
            "aux_id": aux_manifest["aux_id"],
            "models": [m.name for m in pool()],
            "oos_samples": len(predictions),
            "gate": report["gate"],
            "report_sha256": file_sha256(path),
        },
    )
    log.info("volatility_live_study_complete", report=str(path), gate=report["gate"])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="E-12b: har_x_live, calibración y paridad")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--aux", type=Path, required=True)
    parser.add_argument("--development-end", required=True)
    parser.add_argument("--folds", type=int, default=harness.WALK_FORWARD_FOLDS)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    execute(
        args.dataset,
        args.aux,
        pd.Timestamp(args.development_end),
        Path(settings.model_artifact_dir),
        args.folds,
    )


if __name__ == "__main__":
    main()
