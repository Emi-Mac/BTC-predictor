"""Línea base de evaluación (Fase 0) sobre un dataset de investigación congelado.

Estimación causal: walk-forward mensual expansivo. CPCV y PBO son solo diagnósticos de
estabilidad y de sobreajuste por selección; no sustituyen al walk-forward.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pandas as pd
import structlog
from threadpoolctl import threadpool_limits

from src.config import get_settings
from src.evaluation.economic import (
    backtest,
    performance,
    positions_from_probabilities,
    strategy_returns,
)
from src.evaluation.metrics import evaluate
from src.evaluation.recalibration import METHODS, expanding_oos_calibration
from src.evaluation.robustness import (
    block_bootstrap_mean,
    cpcv_paths,
    cpcv_splits,
    deflated_sharpe_ratio,
    effective_trials,
    probability_of_backtest_overfitting,
    reality_check,
)
from src.evaluation.selective import coverage_curve, skill_scores
from src.features.compute import FEATURE_COLUMNS
from src.features.funding_context import FUNDING_FEATURE_COLUMNS
from src.features.price_context import PRICE_CONTEXT_COLUMNS
from src.ingestion.run import configure_logging
from src.research.dataset import verify_dataset
from src.research.metrics_features import OI_FEATURE_COLUMNS
from src.research.provenance import implementation_signature
from src.research.source_snapshot import file_sha256
from src.training.holdouts import assert_no_reserved_training_data, reservation_lock
from src.training.models import fit_lightgbm
from src.training.research_models import _fit_logistic, _fit_preprocessor

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.training.types import FloatArray, IntArray

log = structlog.get_logger(__name__)
SEED = 42
STEP = timedelta(hours=4)
HORIZON = 1
WALK_FORWARD_FOLDS = 48
MIN_TRAIN_DAYS = 365
CALIBRATION_MIN_SAMPLES = 1000
CALIBRATION_WINDOW = 6 * 365
PRIMARY_CANDIDATE = "logistic_base_C0.01"
PRIMARY_BASELINE = "conditional_direction"
PRIMARY_MARGIN = 0.0
PRIMARY_COST_BPS = 7.0
BOOTSTRAP_BLOCKS = (12, 42, 180)
PRIMARY_BLOCK = 42
CPCV_GROUPS = 12
CPCV_TEST_GROUPS = 2
CPCV_EMBARGO_FRACTION = 0.01
PBO_PARTITIONS = 16
LIGHTGBM_PARAMETERS = {
    "max_depth": 3,
    "num_leaves": 8,
    "learning_rate": 0.03,
    "n_estimators": 200,
    "subsample": 0.8,
}
EXTRA_IMPLEMENTATION_FILES = (
    "evaluation/calibration.py",
    "evaluation/economic.py",
    "evaluation/recalibration.py",
    "evaluation/robustness.py",
    "evaluation/selective.py",
    "research/evaluation_harness.py",
    "training/models.py",
)
POINT_IN_TIME_CAVEAT = (
    "El dataset no tiene verificación point-in-time (point_in_time_verified=false): las "
    "revisiones históricas de las fuentes son desconocidas y pueden sesgar los resultados."
)


def feature_sets() -> dict[str, list[str]]:
    base = [*FEATURE_COLUMNS, *PRICE_CONTEXT_COLUMNS, *FUNDING_FEATURE_COLUMNS]
    return {
        "existing": list(FEATURE_COLUMNS),
        "base": base,
        "base_oi": [*base, *OI_FEATURE_COLUMNS],
    }


@dataclass(frozen=True)
class Candidate:
    """Configuración totalmente especificada; la selección entre ellas la miden PBO/DSR."""

    name: str
    family: str
    features: str
    c: float | None = None


def candidates() -> list[Candidate]:
    result = [
        Candidate(f"logistic_{features}_C{c}", "logistic", features, c)
        for features in feature_sets()
        for c in (0.003, 0.01, 0.03)
    ]
    result.append(Candidate("lightgbm_base", "lightgbm", "base"))
    return result


def fit_predict(candidate: Candidate, train: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
    names = feature_sets()[candidate.features]
    pre = _fit_preprocessor(train, names)
    x_train, x_test = pre.transform(train), pre.transform(test)
    y = train["target_label"].to_numpy(dtype=int)
    if candidate.family == "logistic":
        if candidate.c is None:
            raise ValueError("La regresión logística requiere C")
        model: Any = _fit_logistic(x_train, y, candidate.c, SEED)
    elif candidate.family == "lightgbm":
        model = fit_lightgbm(x_train, y, LIGHTGBM_PARAMETERS, SEED)
    else:
        raise ValueError(f"Familia desconocida: {candidate.family}")
    result: FloatArray = np.asarray(model.predict_proba(x_test), dtype=float)[:, 1]
    return result


# ─── Datos y particiones ─────────────────────────────────────────────────────


def attach_forward_returns(features: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Añade el retorno de la siguiente vela y verifica que coincide con la etiqueta."""
    ordered = candles.sort_values("open_time").reset_index(drop=True)
    prices = ordered[["open_time"]].copy()
    prices["forward_return"] = ordered["close"].shift(-HORIZON) / ordered["close"] - 1
    data = features.merge(prices, on="open_time", how="left", validate="one_to_one")
    known = data["target_label"].notna()
    if data.loc[known, "forward_return"].isna().any():
        raise ValueError("Faltan cierres futuros para filas etiquetadas")
    expected = (data.loc[known, "forward_return"] > 0).astype(int)
    if not expected.eq(data.loc[known, "target_label"].astype(int)).all():
        raise ValueError("Los retornos futuros no coinciden con las etiquetas")
    return data


def development_rows(data: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("El corte de desarrollo debe incluir zona horaria")
    known_at = data["open_time"] + (HORIZON + 1) * pd.Timedelta(STEP)
    selected = data.loc[
        (data["open_time"] < cutoff) & (known_at <= cutoff) & data["target_label"].notna()
    ]
    return selected.sort_values("open_time").reset_index(drop=True)


@dataclass(frozen=True)
class MonthlyFold:
    number: int
    start: pd.Timestamp
    end: pd.Timestamp
    train: IntArray
    validation: IntArray


def monthly_walk_forward(
    times: pd.Series, n_folds: int, horizon: int = HORIZON, min_train_days: int = MIN_TRAIN_DAYS
) -> list[MonthlyFold]:
    """Meses completos consecutivos; el train solo contiene etiquetas conocidas al inicio."""
    stamps = pd.to_datetime(times, utc=True).reset_index(drop=True)
    step = pd.Timedelta(STEP)
    if n_folds < 1 or stamps.empty or (stamps.diff().dropna() != step).any():
        raise ValueError("Se requieren folds y observaciones continuas y ordenadas")
    last_end = stamps.iloc[-1].floor("D").replace(day=1)
    folds = []
    for number in range(1, n_folds + 1):
        end = last_end - pd.DateOffset(months=n_folds - number)
        start = end - pd.DateOffset(months=1)
        train = np.flatnonzero((stamps < start) & (stamps + (horizon + 1) * step <= start))
        validation = np.flatnonzero((stamps >= start) & (stamps < end))
        if not len(train) or start - stamps.iloc[train[0]] < pd.Timedelta(days=min_train_days):
            raise ValueError(f"El fold {number} requiere al menos {min_train_days} días de train")
        if len(validation) != len(pd.date_range(start, end - step, freq=step)):
            raise ValueError(f"El fold {number} no tiene un mes completo de validación")
        folds.append(MonthlyFold(number, start, end, train, validation))
    return folds


def baseline_probabilities(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, FloatArray]:
    y = train["target_label"].to_numpy(dtype=int)
    previous = train["previous_direction"].to_numpy(dtype=int)
    test_previous = test["previous_direction"].to_numpy(dtype=int)
    conditional = {d: (y[previous == d].sum() + 1) / ((previous == d).sum() + 2) for d in (0, 1)}
    return {
        "majority": np.full(len(test), float(y.mean())),
        "previous_direction": test_previous.astype(float),
        "conditional_direction": np.array([conditional[int(d)] for d in test_previous]),
    }


# ─── Walk-forward ────────────────────────────────────────────────────────────


def run_walk_forward(
    development: pd.DataFrame,
    folds: list[MonthlyFold],
    pool: list[Candidate],
    predictor: Callable[[Candidate, pd.DataFrame, pd.DataFrame], FloatArray] = fit_predict,
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, test = development.iloc[fold.train], development.iloc[fold.validation]
        frame = test[
            ["open_time", "target_label", "previous_direction", "forward_return"]
        ].reset_index(drop=True)
        frame["fold"] = fold.number
        for name, values in baseline_probabilities(train, test).items():
            frame[f"baseline_{name}"] = values
        for candidate in pool:
            frame[candidate.name] = predictor(candidate, train, test)
        frames.append(frame)
        log.info("evaluation_fold", fold=fold.number, start=fold.start.isoformat())
    predictions = pd.concat(frames, ignore_index=True)
    predictions["target_label"] = predictions["target_label"].astype(int)
    return predictions


def _summary(
    predictions: pd.DataFrame, column: str, hard: FloatArray | None = None
) -> dict[str, Any]:
    y = predictions["target_label"].to_numpy(dtype=int)
    p = predictions[column].to_numpy(dtype=float)
    decided = (p >= 0.5).astype(int) if hard is None else np.asarray(hard, dtype=int)
    metrics = evaluate(y, decided, np.clip(p, 0, 1))
    metrics.pop("calibration_bins")
    per_fold = predictions.assign(correct=(decided == y)).groupby("fold")["correct"].mean()
    by_year = (
        predictions.assign(correct=(decided == y))
        .groupby(predictions["open_time"].dt.year)["correct"]
        .agg(["mean", "size"])
    )
    metrics["fold_accuracy"] = {
        "mean": float(per_fold.mean()),
        "std": float(per_fold.std(ddof=1)) if len(per_fold) > 1 else 0.0,
        "min": float(per_fold.min()),
        "max": float(per_fold.max()),
    }
    metrics["accuracy_by_year"] = {
        str(year): {"accuracy": float(row["mean"]), "samples": int(row["size"])}
        for year, row in by_year.iterrows()
    }
    return metrics


def _correct(predictions: pd.DataFrame, column: str) -> FloatArray:
    p = predictions[column].to_numpy(dtype=float)
    y = predictions["target_label"].to_numpy(dtype=int)
    decided = p.astype(int) if column == "baseline_previous_direction" else (p >= 0.5).astype(int)
    result: FloatArray = (decided == y).astype(float)
    return result


def _net_returns(predictions: pd.DataFrame, column: str, margin: float, cost: float) -> FloatArray:
    p = predictions[column].to_numpy(dtype=float)
    return strategy_returns(
        positions_from_probabilities(p, margin),
        predictions["forward_return"].to_numpy(dtype=float),
        cost,
    )


def calibration_report(
    predictions: pd.DataFrame, folds: list[MonthlyFold], column: str
) -> dict[str, Any]:
    starts = {fold.number: fold.start for fold in folds}
    y = predictions["target_label"].to_numpy(dtype=int)
    raw = predictions[column].to_numpy(dtype=float)
    variants: dict[str, FloatArray] = {"raw": raw}
    for method in METHODS:
        for label, window in (("expanding", None), ("last_year", CALIBRATION_WINDOW)):
            variants[f"{method}_{label}"] = expanding_oos_calibration(
                predictions["open_time"],
                predictions["fold"].to_numpy(),
                starts,
                raw,
                y,
                STEP,
                method,
                HORIZON,
                CALIBRATION_MIN_SAMPLES,
                window,
            )
    common = np.logical_and.reduce([np.isfinite(v) for v in variants.values()])
    if not common.any():
        return {"evaluable_samples": 0, "methods": {}}
    result = {}
    for name, values in variants.items():
        metrics = evaluate(y[common], (values[common] >= 0.5).astype(int), values[common])
        result[name] = {
            key: metrics[key] for key in ("accuracy", "log_loss", "brier", "ece", "auc_roc")
        }
        result[name]["probability_std"] = float(values[common].std())
    return {
        "candidate": column,
        "evaluable_samples": int(common.sum()),
        "evaluable_folds": sorted(
            int(f) for f in np.unique(predictions["fold"].to_numpy()[common])
        ),
        "note": "Calibración online con predicciones OOS de folds previos; método no seleccionado.",
        "methods": result,
    }


def inference_report(predictions: pd.DataFrame, pool: list[Candidate]) -> dict[str, Any]:
    baseline_column = f"baseline_{PRIMARY_BASELINE}"
    baseline_correct = _correct(predictions, baseline_column)
    primary = _correct(predictions, PRIMARY_CANDIDATE) - baseline_correct
    primary_pair = {
        "candidate": PRIMARY_CANDIDATE,
        "baseline": PRIMARY_BASELINE,
        "accuracy_difference": {
            str(block): block_bootstrap_mean(primary, block, seed=SEED)
            for block in BOOTSTRAP_BLOCKS
        },
        "brier_difference_baseline_minus_candidate": block_bootstrap_mean(
            (predictions[baseline_column] - predictions["target_label"]) ** 2
            - (predictions[PRIMARY_CANDIDATE] - predictions["target_label"]) ** 2,
            PRIMARY_BLOCK,
            seed=SEED,
        ),
    }
    differences = np.column_stack([_correct(predictions, c.name) - baseline_correct for c in pool])
    check = reality_check(differences, PRIMARY_BLOCK, seed=SEED)
    check["best_candidate"] = pool[check.pop("best_column")].name
    return {
        "primary_pair": primary_pair,
        "best_of_candidates_reality_check": check,
        "note": (
            "El par primario se fijó antes de ver los resultados. El IC del mejor candidato "
            "elegido a posteriori no es válido; para él se informa la prueba de White."
        ),
    }


def overfitting_report(predictions: pd.DataFrame, pool: list[Candidate]) -> dict[str, Any]:
    correct = np.column_stack([_correct(predictions, c.name) for c in pool])
    returns = np.column_stack(
        [_net_returns(predictions, c.name, PRIMARY_MARGIN, PRIMARY_COST_BPS) for c in pool]
    )
    sharpes = returns.mean(axis=0) / np.where(
        returns.std(axis=0, ddof=1) > 0, returns.std(axis=0, ddof=1), 1
    )
    best = int(sharpes.argmax())
    n_effective = effective_trials(returns)
    dsr = deflated_sharpe_ratio(returns[:, best], sharpes, n_effective)
    dsr["candidate"] = pool[best].name
    dsr["n_trials_nominal"] = len(pool)
    return {
        "pbo_accuracy": probability_of_backtest_overfitting(correct, PBO_PARTITIONS, "mean"),
        "pbo_sharpe": probability_of_backtest_overfitting(returns, PBO_PARTITIONS, "sharpe"),
        "deflated_sharpe": dsr,
        "cost_bps": PRIMARY_COST_BPS,
        "note": (
            "PBO mide sobreajuste por selección solo entre estos candidatos y supone bloques "
            "intercambiables, algo dudoso con regímenes no estacionarios; no es un p-valor."
        ),
    }


# ─── CPCV (diagnóstico) ──────────────────────────────────────────────────────


def run_cpcv(
    development: pd.DataFrame,
    pool: list[Candidate],
    predictor: Callable[[Candidate, pd.DataFrame, pd.DataFrame], FloatArray] = fit_predict,
) -> dict[str, Any]:
    n = len(development)
    embargo = int(np.ceil(CPCV_EMBARGO_FRACTION * n))
    groups, splits = cpcv_splits(n, CPCV_GROUPS, CPCV_TEST_GROUPS, HORIZON + 1, embargo)
    paths = cpcv_paths(CPCV_GROUPS, CPCV_TEST_GROUPS)
    y = development["target_label"].to_numpy(dtype=int)
    forward = development["forward_return"].to_numpy(dtype=float)
    outputs: dict[str, list[FloatArray]] = {c.name: [] for c in pool}
    for index, split in enumerate(splits):
        train, test = development.iloc[split.train], development.iloc[split.test]
        for candidate in pool:
            outputs[candidate.name].append(predictor(candidate, train, test))
        log.info("evaluation_cpcv_split", split=index + 1, total=len(splits))
    group_position = {}
    for index, split in enumerate(splits):
        offset = 0
        for g in split.test_groups:
            group_position[(index, g)] = (offset, offset + len(groups[g]))
            offset += len(groups[g])
    result = {}
    for candidate in pool:
        accuracies, sharpes = [], []
        for path in paths:
            p = np.concatenate(
                [
                    outputs[candidate.name][path[g]][slice(*group_position[(path[g], g)])]
                    for g in range(CPCV_GROUPS)
                ]
            )
            accuracies.append(float(((p >= 0.5).astype(int) == y).mean()))
            net = strategy_returns(
                positions_from_probabilities(p, PRIMARY_MARGIN), forward, PRIMARY_COST_BPS
            )
            sharpes.append(float(net.mean() / net.std(ddof=1) * np.sqrt(6 * 365)))
        result[candidate.name] = {
            "path_accuracy": {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies, ddof=1)),
                "min": float(np.min(accuracies)),
                "max": float(np.max(accuracies)),
            },
            "path_sharpe_annualized": {
                "mean": float(np.mean(sharpes)),
                "std": float(np.std(sharpes, ddof=1)),
                "min": float(np.min(sharpes)),
                "max": float(np.max(sharpes)),
            },
        }
    return {
        "groups": CPCV_GROUPS,
        "test_groups": CPCV_TEST_GROUPS,
        "splits": len(splits),
        "paths": len(paths),
        "purge_rows": HORIZON + 1,
        "embargo_rows": embargo,
        "cost_bps": PRIMARY_COST_BPS,
        "note": (
            "Diagnóstico de estabilidad: el entrenamiento incluye datos posteriores al bloque "
            "de prueba, por lo que no es una estimación causal del desempeño."
        ),
        "candidates": result,
    }


# ─── Orquestación ────────────────────────────────────────────────────────────


def build_report(
    development: pd.DataFrame,
    n_folds: int = WALK_FORWARD_FOLDS,
    include_cpcv: bool = True,
    predictor: Callable[[Candidate, pd.DataFrame, pd.DataFrame], FloatArray] = fit_predict,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pool = candidates()
    if PRIMARY_CANDIDATE not in {c.name for c in pool}:
        raise ValueError("El candidato primario debe pertenecer al conjunto evaluado")
    folds = monthly_walk_forward(development["open_time"], n_folds)
    with threadpool_limits(limits=2):
        predictions = run_walk_forward(development, folds, pool, predictor)
        cpcv = run_cpcv(development, pool, predictor) if include_cpcv else None
    y = predictions["target_label"].to_numpy(dtype=int)
    climatology = predictions["baseline_majority"].to_numpy(dtype=float)
    columns = [c.name for c in pool] + [
        f"baseline_{name}" for name in ("majority", "previous_direction", "conditional_direction")
    ]
    models: dict[str, Any] = {}
    for column in columns:
        if column == "baseline_previous_direction":
            # Regla dura; sus métricas probabilísticas usan P(sube | dirección previa).
            hard = predictions[column].to_numpy(dtype=int)
            probability_column = "baseline_conditional_direction"
        else:
            hard, probability_column = None, column
        p = predictions[probability_column].to_numpy(dtype=float)
        entry = _summary(predictions, probability_column, hard)
        entry["skill"] = skill_scores(y, np.clip(p, 0, 1), climatology)
        if column != "baseline_previous_direction":
            entry["coverage"] = coverage_curve(y, p)
            entry["backtest"] = backtest(p, predictions["forward_return"].to_numpy(dtype=float))
        models[column] = entry
    buy_hold = strategy_returns(
        np.ones(len(predictions)), predictions["forward_return"].to_numpy(dtype=float), 0.0
    )
    inference = inference_report(predictions, pool)
    overfitting = overfitting_report(predictions, pool)
    primary = models[PRIMARY_CANDIDATE]
    baseline = models[f"baseline_{PRIMARY_BASELINE}"]
    difference = inference["primary_pair"]["accuracy_difference"][str(PRIMARY_BLOCK)]
    primary_backtest = next(
        row
        for row in primary["backtest"]
        if row["margin"] == PRIMARY_MARGIN and row["cost_bps"] == PRIMARY_COST_BPS
    )
    report = {
        "evaluation": "phase0_evaluation_baseline",
        "independent_holdout_evaluated": False,
        "model_registered": False,
        "seed": SEED,
        "interval": "4h",
        "label_horizon_candles": HORIZON,
        "walk_forward": {
            "folds": len(folds),
            "first_validation_start": folds[0].start.isoformat(),
            "last_validation_end_exclusive": folds[-1].end.isoformat(),
            "oos_samples": len(predictions),
            "purge_rows": HORIZON + 1,
            "min_train_days": MIN_TRAIN_DAYS,
        },
        "pre_specified": {
            "primary_candidate": PRIMARY_CANDIDATE,
            "primary_baseline": PRIMARY_BASELINE,
            "primary_margin": PRIMARY_MARGIN,
            "primary_cost_bps": PRIMARY_COST_BPS,
            "primary_block": PRIMARY_BLOCK,
        },
        "candidates": [asdict(c) for c in pool],
        "lightgbm_parameters": LIGHTGBM_PARAMETERS,
        "models": models,
        "buy_and_hold": performance(buy_hold),
        "calibration": calibration_report(predictions, folds, PRIMARY_CANDIDATE),
        "inference": inference,
        "overfitting": overfitting,
        "cpcv": cpcv,
        "gate": {
            "primary_accuracy": primary["accuracy"],
            "primary_baseline_accuracy": baseline["accuracy"],
            "accuracy_difference": difference["mean"],
            "accuracy_difference_ci_95": difference["ci_95"],
            "ci_excludes_zero": difference["ci_95"][0] > 0,
            "meets_plus_3pp": difference["mean"] >= 0.03,
            "brier_skill_score": primary["skill"]["brier_skill_score"],
            "reality_check_p_value": inference["best_of_candidates_reality_check"]["p_value"],
            "pbo_accuracy": overfitting["pbo_accuracy"]["pbo"],
            "deflated_sharpe_ratio": overfitting["deflated_sharpe"]["deflated_sharpe_ratio"],
            "primary_net_sharpe": primary_backtest["sharpe"],
            "primary_profit_factor": primary_backtest["profit_factor"],
        },
        "caveats": [
            POINT_IN_TIME_CAVEAT,
            "Wilson supone independencia; usar los IC por bloques para inferencia.",
            "El profit factor se calcula sobre retornos netos por periodo de 4h.",
        ],
    }
    return report, predictions


def evaluation_signature() -> dict[str, Any]:
    signature = implementation_signature()
    root = Path(__file__).resolve().parents[1]
    signature["extra_files"] = {
        name: file_sha256(root / name) for name in EXTRA_IMPLEMENTATION_FILES
    }
    return signature


def append_registry(artifact_root: Path, entry: dict[str, Any]) -> None:
    """Registro append-only de todos los experimentos, incluidos los fallidos."""
    with reservation_lock(artifact_root):
        with (artifact_root / "experiments.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")


def execute(
    directory: Path,
    cutoff: pd.Timestamp,
    artifact_root: Path,
    n_folds: int = WALK_FORWARD_FOLDS,
    include_cpcv: bool = True,
) -> Path:
    manifest = verify_dataset(directory)
    signature = evaluation_signature()
    features = pd.read_parquet(directory / manifest["files"]["features"]["path"])
    sources = json.loads((directory / "source-manifest.json").read_text(encoding="utf-8"))
    candles = pd.read_parquet(directory / sources["files"]["candles"]["path"])
    data = attach_forward_returns(features, candles)
    development = development_rows(data, cutoff)
    assert_no_reserved_training_data(artifact_root, development["open_time"], STEP)
    report, predictions = build_report(development, n_folds, include_cpcv)
    if evaluation_signature() != signature:
        raise RuntimeError("La implementación cambió durante la evaluación")
    name = f"eval-baseline-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    report.update(
        {
            "run_id": name,
            "development_end_exclusive": cutoff.isoformat(),
            "dataset_id": manifest["dataset_id"],
            "dataset_manifest_sha256": manifest["manifest_sha256"],
            "point_in_time_verified": manifest.get("point_in_time_verified", False),
            "implementation": signature,
        }
    )
    target = artifact_root / name
    target.mkdir(parents=True, exist_ok=False)
    predictions_path = target / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    report["predictions_sha256"] = file_sha256(predictions_path)
    path = target / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    append_registry(
        artifact_root,
        {
            "run_id": name,
            "kind": report["evaluation"],
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_id": manifest["dataset_id"],
            "development_end_exclusive": cutoff.isoformat(),
            "label_horizon_candles": HORIZON,
            "candidates": [c.name for c in candidates()],
            "oos_samples": len(predictions),
            "gate": report["gate"],
            "report_sha256": file_sha256(path),
        },
    )
    log.info("evaluation_baseline_complete", report=str(path), gate=report["gate"])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Línea base de evaluación sobre datos 2020+")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--development-end", required=True)
    parser.add_argument("--folds", type=int, default=WALK_FORWARD_FOLDS)
    parser.add_argument("--skip-cpcv", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    execute(
        args.dataset,
        pd.Timestamp(args.development_end),
        Path(settings.model_artifact_dir),
        args.folds,
        not args.skip_cpcv,
    )


if __name__ == "__main__":
    main()
