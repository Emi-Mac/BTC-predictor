"""Fase 1: rediseño del objetivo (horizonte y zona muerta) y abstención causal.

Compara horizontes de 4h, 12h, 24h y 3d y la exclusión del entrenamiento de movimientos
pequeños respecto a la volatilidad. La accuracy siempre se mide contra el signo del retorno a
h velas en *todas* las filas, porque en producción no se sabe de antemano si el movimiento será
pequeño. Los resultados económicos usan una cartera solapada (1/h del capital por señal) sobre
el mismo libro de retornos de 4h, lo que hace comparables todos los horizontes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

import numpy as np
import pandas as pd
import structlog
from threadpoolctl import threadpool_limits

from src.config import get_settings
from src.evaluation.economic import (
    overlapping_positions,
    performance,
    positions_from_probabilities,
    strategy_returns,
)
from src.evaluation.metrics import evaluate
from src.evaluation.robustness import (
    block_bootstrap_mean,
    deflated_sharpe_ratio,
    effective_trials,
    maximin_pbo,
    probability_of_backtest_overfitting,
    reality_check,
    simultaneous_lower_bounds,
)
from src.evaluation.selective import causal_margin_selection, coverage_curve, skill_scores
from src.ingestion.run import configure_logging
from src.research import evaluation_harness as harness
from src.research.dataset import verify_dataset
from src.research.source_snapshot import file_sha256
from src.training.holdouts import assert_no_reserved_training_data

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from src.training.types import FloatArray, IntArray

log = structlog.get_logger(__name__)
SEED = harness.SEED
STEP = harness.STEP
HORIZONS = (1, 3, 6, 18)
DEAD_ZONES = (0.0, 0.25, 0.5)
MODELS = ("logistic_base_C0.01", "lightgbm_base")
VOLATILITY_WINDOW = 42
LARGE_MOVE_K = 0.5
SELECTION_MARGINS = (0.0, 0.02, 0.05, 0.08, 0.12)
SELECTION_COST_BPS = 7.0
COSTS_BPS = (0.0, 7.0, 15.0)
MIN_SELECTION_HISTORY = 6 * 180
BASELINES = ("majority", "conditional_previous_1", "conditional_previous_h")
PRIMARY_CONFIG = "h6_k0.25_logistic_base_C0.01"
BOOTSTRAP_BLOCKS = (42, 180)
PRIMARY_BLOCK = 180
PBO_PARTITIONS = 16
GATE_MIN_DIFFERENCE = 0.03
GATE_CONFIDENCE = 0.95
# Funding de perpetuos se liquida cada 8h; una vela de 4h acumula la mitad de la tasa.
FUNDING_FRACTION_PER_CANDLE = 0.5


class NamedHorizon(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def horizon(self) -> int: ...


@dataclass(frozen=True)
class Config:
    horizon: int
    dead_zone: float
    model: str

    @property
    def name(self) -> str:
        return f"h{self.horizon}_k{self.dead_zone}_{self.model}"


def configs() -> list[Config]:
    return [Config(h, k, m) for h in HORIZONS for k in DEAD_ZONES for m in MODELS]


# ─── Datos ───────────────────────────────────────────────────────────────────


def attach_horizons(features: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Añade retornos futuros/pasados por horizonte y la volatilidad conocida en t."""
    ordered = candles.sort_values("open_time").reset_index(drop=True)
    close = ordered["close"].astype(float)
    extra = ordered[["open_time"]].copy()
    extra["sigma"] = (
        np.log(close).diff().rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std()
    )
    for h in sorted({1, *HORIZONS}):
        extra[f"forward_return_{h}"] = close.shift(-h) / close - 1
        extra[f"past_return_{h}"] = close / close.shift(h) - 1
    data = features.merge(extra, on="open_time", how="left", validate="one_to_one")
    known = data["target_label"].notna() & data["forward_return_1"].notna()
    if (
        not (data.loc[known, "forward_return_1"] > 0)
        .astype(int)
        .eq(data.loc[known, "target_label"].astype(int))
        .all()
    ):
        raise ValueError("Los retornos futuros no coinciden con las etiquetas")
    if (
        not (data["past_return_1"].dropna() > 0)
        .astype(int)
        .eq(data.loc[data["past_return_1"].notna(), "previous_direction"].astype(int))
        .all()
    ):
        raise ValueError("La dirección previa no coincide con las velas")
    return data


def development_rows(data: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Filas comunes a todos los horizontes: sus etiquetas a max(h) se conocen antes del corte."""
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("El corte de desarrollo debe incluir zona horaria")
    longest = max(HORIZONS)
    required = ["sigma", *(f"forward_return_{h}" for h in HORIZONS)]
    required += [f"past_return_{h}" for h in HORIZONS]
    mask = (
        (data["open_time"] < cutoff)
        & (data["open_time"] + (longest + 1) * pd.Timedelta(STEP) <= cutoff)
        & data["target_label"].notna()
        & data[required].notna().all(axis=1)
    )
    result = data.loc[mask].sort_values("open_time").reset_index(drop=True)
    if (result["open_time"].diff().dropna() != pd.Timedelta(STEP)).any():
        raise ValueError("Las filas de desarrollo deben ser continuas")
    return result


def horizon_label(frame: pd.DataFrame, horizon: int) -> IntArray:
    result: IntArray = (frame[f"forward_return_{horizon}"] > 0).to_numpy(dtype=int)
    return result


def dead_zone_mask(frame: pd.DataFrame, horizon: int, k: float) -> FloatArray:
    threshold = k * frame["sigma"].to_numpy(dtype=float) * np.sqrt(horizon)
    result: FloatArray = (
        np.abs(frame[f"forward_return_{horizon}"].to_numpy(dtype=float)) >= threshold
    )
    return result


def baseline_probabilities(
    train: pd.DataFrame, test: pd.DataFrame, horizon: int
) -> dict[str, FloatArray]:
    y = horizon_label(train, horizon)
    result: dict[str, FloatArray] = {"majority": np.full(len(test), float(y.mean()))}
    for name, lookback in (("conditional_previous_1", 1), ("conditional_previous_h", horizon)):
        past = (train[f"past_return_{lookback}"] > 0).to_numpy(dtype=int)
        test_past = (test[f"past_return_{lookback}"] > 0).to_numpy(dtype=int)
        table = {d: (y[past == d].sum() + 1) / ((past == d).sum() + 2) for d in (0, 1)}
        result[name] = np.array([table[int(d)] for d in test_past], dtype=float)
    return result


# ─── Walk-forward ────────────────────────────────────────────────────────────


def run_walk_forward(
    development: pd.DataFrame,
    n_folds: int,
    pool: list[Config],
    predictor: Callable[[harness.Candidate, pd.DataFrame, pd.DataFrame], FloatArray] = (
        harness.fit_predict
    ),
) -> pd.DataFrame:
    candidates = {c.name: c for c in harness.candidates()}
    keep = ["open_time", "sigma", "forward_return_1", "funding_rate"]
    keep += [f"forward_return_{h}" for h in HORIZONS if h != 1]
    predictions: pd.DataFrame | None = None
    for horizon in HORIZONS:
        folds = harness.monthly_walk_forward(development["open_time"], n_folds, horizon=horizon)
        validation = np.concatenate([fold.validation for fold in folds])
        if predictions is None:
            predictions = development.iloc[validation][keep].reset_index(drop=True)
            predictions["fold"] = np.concatenate(
                [np.full(len(fold.validation), fold.number) for fold in folds]
            )
            reference = validation
        elif not np.array_equal(validation, reference):
            raise ValueError("Todos los horizontes deben compartir las filas de validación")
        predictions[f"label_h{horizon}"] = horizon_label(development.iloc[validation], horizon)
        columns: dict[str, list[FloatArray]] = {}
        for fold in folds:
            train = development.iloc[fold.train].copy()
            test = development.iloc[fold.validation]
            for name, values in baseline_probabilities(train, test, horizon).items():
                columns.setdefault(f"baseline_{name}_h{horizon}", []).append(values)
            train["target_label"] = horizon_label(train, horizon)
            for k in DEAD_ZONES:
                subset = train.loc[dead_zone_mask(train, horizon, k)]
                for config in (c for c in pool if c.horizon == horizon and c.dead_zone == k):
                    columns.setdefault(config.name, []).append(
                        predictor(candidates[config.model], subset, test)
                    )
            log.info("target_study_fold", horizon=horizon, fold=fold.number)
        for name, parts in columns.items():
            predictions[name] = np.concatenate(parts)
    if predictions is None:
        raise ValueError("No hay horizontes configurados")
    return predictions


# ─── Evaluación ──────────────────────────────────────────────────────────────


def _correct(predictions: pd.DataFrame, column: str, horizon: int) -> FloatArray:
    p = predictions[column].to_numpy(dtype=float)
    result: FloatArray = ((p >= 0.5).astype(int) == predictions[f"label_h{horizon}"]).astype(float)
    return result


def selective_strategy(
    predictions: pd.DataFrame, column: str, horizon: int, allow_short: bool
) -> dict[str, Any]:
    p = predictions[column].to_numpy(dtype=float)
    r1 = predictions["forward_return_1"].to_numpy(dtype=float)
    margins, choices = causal_margin_selection(
        predictions["fold"].to_numpy(),
        p,
        r1,
        horizon,
        SELECTION_MARGINS,
        SELECTION_COST_BPS,
        MIN_SELECTION_HISTORY,
        allow_short,
        funding_carry(predictions),
    )
    active = np.isfinite(margins)
    signals = np.zeros(len(p))
    for margin in np.unique(margins[active]):
        rows = margins == margin
        signals[rows] = positions_from_probabilities(p[rows], float(margin), allow_short)
    position = overlapping_positions(signals, horizon)
    label = predictions[f"label_h{horizon}"].to_numpy(dtype=int)
    traded = active & (signals != 0)
    hits = (signals[traded] > 0).astype(int) == label[traded]
    return {
        "signals": signals,
        "position": position,
        "evaluable": active,
        "margins_by_fold": {str(fold): value for fold, value in choices.items()},
        "signal_coverage": float(traded.sum() / max(active.sum(), 1)),
        "signal_accuracy": float(hits.mean()) if traded.any() else None,
        "signal_count": int(traded.sum()),
    }


def summarize(
    predictions: pd.DataFrame, column: str, horizon: int, climatology: FloatArray
) -> dict[str, Any]:
    y = predictions[f"label_h{horizon}"].to_numpy(dtype=int)
    p = predictions[column].to_numpy(dtype=float)
    metrics = evaluate(y, (p >= 0.5).astype(int), p)
    metrics.pop("calibration_bins")
    large = dead_zone_mask(predictions, horizon, LARGE_MOVE_K)
    correct = _correct(predictions, column, horizon)
    by_fold = pd.Series(correct).groupby(predictions["fold"].to_numpy()).mean()
    by_year = pd.Series(correct).groupby(predictions["open_time"].dt.year.to_numpy()).mean()
    metrics.update(
        {
            "accuracy_large_moves": float(correct[large].mean()),
            "large_move_fraction": float(large.mean()),
            "fold_accuracy_std": float(by_fold.std(ddof=1)) if len(by_fold) > 1 else 0.0,
            "fold_accuracy_min": float(by_fold.min()),
            "accuracy_by_year": {str(year): float(value) for year, value in by_year.items()},
            "skill": skill_scores(y, p, climatology),
            "coverage": coverage_curve(y, p),
        }
    )
    return metrics


def funding_carry(predictions: pd.DataFrame) -> FloatArray:
    """Costo de financiamiento por vela para perpetuos con la última tasa conocida en t."""
    rate = predictions["funding_rate"].to_numpy(dtype=float)
    if not np.isfinite(rate).all():
        raise ValueError("Faltan tasas de funding en las filas evaluadas")
    result: FloatArray = rate * FUNDING_FRACTION_PER_CANDLE
    return result


def strategy_table(
    predictions: pd.DataFrame, pool: Sequence[NamedHorizon]
) -> tuple[dict[str, FloatArray], dict[str, Any], FloatArray]:
    """Estrategias selectivas causales en perpetuos: 7 pb por rotación más funding."""
    r1 = predictions["forward_return_1"].to_numpy(dtype=float)
    carry = funding_carry(predictions)
    returns: dict[str, FloatArray] = {}
    details: dict[str, Any] = {}
    evaluable: FloatArray | None = None
    for config in pool:
        for allow_short, side in ((True, "long_short"), (False, "long_flat")):
            strategy = selective_strategy(predictions, config.name, config.horizon, allow_short)
            mask = strategy["evaluable"]
            if evaluable is None:
                evaluable = mask
            elif not np.array_equal(evaluable, mask):
                raise ValueError("Las estrategias deben compartir la ventana evaluable")
            name = f"{config.name}_{side}"
            returns[name] = strategy_returns(strategy["position"], r1, SELECTION_COST_BPS, carry)
            details[name] = {
                "margins_by_fold": strategy["margins_by_fold"],
                "signal_coverage": strategy["signal_coverage"],
                "signal_accuracy": strategy["signal_accuracy"],
                "signal_count": strategy["signal_count"],
                "by_cost": {
                    str(cost): performance(
                        strategy_returns(strategy["position"], r1, cost, carry)[mask],
                        strategy["position"][mask],
                    )
                    for cost in COSTS_BPS
                },
            }
    if evaluable is None or not evaluable.any():
        raise ValueError("No hay ventana evaluable para las estrategias")
    return returns, details, evaluable


def build_report(
    development: pd.DataFrame,
    n_folds: int = harness.WALK_FORWARD_FOLDS,
    predictor: Callable[[harness.Candidate, pd.DataFrame, pd.DataFrame], FloatArray] = (
        harness.fit_predict
    ),
    pool: list[Config] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pool = pool or configs()
    if PRIMARY_CONFIG not in {c.name for c in pool}:
        raise ValueError("La configuración primaria debe pertenecer al conjunto evaluado")
    with threadpool_limits(limits=2):
        predictions = run_walk_forward(development, n_folds, pool, predictor)

    horizons = sorted({c.horizon for c in pool})
    baselines: dict[str, Any] = {}
    for h in horizons:
        climatology = predictions[f"baseline_majority_h{h}"].to_numpy(dtype=float)
        for name in BASELINES:
            column = f"baseline_{name}_h{h}"
            baselines[column] = summarize(predictions, column, h, climatology)
    models = {
        c.name: summarize(
            predictions, c.name, c.horizon, predictions[f"baseline_majority_h{c.horizon}"]
        )
        for c in pool
    }

    # Inferencia sobre accuracy: cada config contra los baselines de su propio horizonte.
    differences = {
        name: np.column_stack(
            [
                _correct(predictions, c.name, c.horizon)
                - _correct(predictions, f"baseline_{name}_h{c.horizon}", c.horizon)
                for c in pool
            ]
        )
        for name in BASELINES
    }
    checks = {}
    for name, matrix in differences.items():
        check = reality_check(matrix, PRIMARY_BLOCK, seed=SEED)
        check["best_config"] = pool[check.pop("best_column")].name
        checks[name] = check
    # Límites simultáneos sobre todos los pares (config, baseline): válidos tras seleccionar.
    joint = simultaneous_lower_bounds(
        np.hstack([differences[name] for name in BASELINES]),
        PRIMARY_BLOCK,
        seed=SEED,
        level=GATE_CONFIDENCE,
    )
    lower = joint["lower_bounds"].reshape(len(BASELINES), len(pool))
    means = joint["means"].reshape(len(BASELINES), len(pool))
    worst_case = means.min(axis=0)
    worst_lower = lower.min(axis=0)
    best_index = int(worst_lower.argmax())
    primary_index = [c.name for c in pool].index(PRIMARY_CONFIG)
    primary = {
        name: {
            str(block): block_bootstrap_mean(matrix[:, primary_index], block, seed=SEED)
            for block in BOOTSTRAP_BLOCKS
        }
        for name, matrix in differences.items()
    }
    pbo_accuracy = maximin_pbo(np.stack([differences[name] for name in BASELINES]), PBO_PARTITIONS)

    # Economía: estrategias causales sobre la misma ventana evaluable.
    returns, strategies, evaluable = strategy_table(predictions, pool)
    names = list(returns)
    matrix = np.column_stack([returns[n][evaluable] for n in names])
    r1 = predictions["forward_return_1"].to_numpy(dtype=float)[evaluable]
    buy_hold = performance(r1, np.ones(len(r1)))
    perpetual_long = strategy_returns(
        np.ones(len(predictions)),
        predictions["forward_return_1"].to_numpy(dtype=float),
        0.0,
        funding_carry(predictions),
    )[evaluable]
    buy_hold_perpetual = performance(perpetual_long, np.ones(len(r1)))
    economic_check = reality_check(matrix - r1[:, None], PRIMARY_BLOCK, seed=SEED)
    economic_check["best_strategy"] = names[economic_check.pop("best_column")]
    std = matrix.std(axis=0, ddof=1)
    sharpes = matrix.mean(axis=0) / np.where(std > 0, std, 1)
    best_strategy = int(sharpes.argmax())
    dsr = deflated_sharpe_ratio(matrix[:, best_strategy], sharpes, effective_trials(matrix))
    dsr["strategy"] = names[best_strategy]
    dsr["n_trials_nominal"] = len(names)
    dsr["inferential"] = False
    dsr["note"] = (
        "Supone retornos independientes; con posiciones solapadas y volatilidad agrupada "
        "sobreestima la significancia. Usar la prueba de White por bloques."
    )
    pbo_sharpe = probability_of_backtest_overfitting(matrix, PBO_PARTITIONS, "sharpe")

    best_name = pool[best_index].name
    gate = {
        "best_config_by_worst_case_lower_bound": best_name,
        "best_accuracy": models[best_name]["accuracy"],
        "best_worst_case_difference": float(worst_case[best_index]),
        "best_worst_case_simultaneous_lower_bound": float(worst_lower[best_index]),
        "simultaneous_confidence": GATE_CONFIDENCE,
        "reality_check_p_values": {name: check["p_value"] for name, check in checks.items()},
        "primary_config": PRIMARY_CONFIG,
        "primary_difference_ci_95": {
            name: values[str(PRIMARY_BLOCK)]["ci_95"] for name, values in primary.items()
        },
        "pbo_accuracy": pbo_accuracy["pbo"],
        "best_strategy": dsr["strategy"],
        "best_strategy_net_sharpe": strategies[dsr["strategy"]]["by_cost"][str(SELECTION_COST_BPS)][
            "sharpe"
        ],
        "buy_and_hold_spot_sharpe": buy_hold["sharpe"],
        "economic_reality_check_p_value": economic_check["p_value"],
        "deflated_sharpe_ratio": dsr["deflated_sharpe_ratio"],
        "pbo_sharpe": pbo_sharpe["pbo"],
    }
    gate["passed"] = bool(
        gate["best_worst_case_difference"] >= GATE_MIN_DIFFERENCE
        and gate["best_worst_case_simultaneous_lower_bound"] > 0
    )
    evaluable_times = predictions["open_time"][evaluable]
    report = {
        "evaluation": "phase1_target_study",
        "independent_holdout_evaluated": False,
        "model_registered": False,
        "seed": SEED,
        "interval": "4h",
        "horizons_candles": horizons,
        "dead_zones_sigma": list(DEAD_ZONES),
        "models": list(MODELS),
        "volatility_window": VOLATILITY_WINDOW,
        "walk_forward": {
            "folds": int(predictions["fold"].nunique()),
            "oos_samples": len(predictions),
            "first_validation_start": predictions["open_time"].iloc[0].isoformat(),
            "purge_rows": "horizon + 1",
        },
        "pre_specified": {
            "primary_config": PRIMARY_CONFIG,
            "baselines": list(BASELINES),
            "primary_block": PRIMARY_BLOCK,
            "gate": (
                "la config con mayor límite inferior simultáneo (95%, todos los pares "
                "config×baseline) debe superar a los tres baselines con límite > 0 y "
                f"diferencia media en el peor caso >= {GATE_MIN_DIFFERENCE}"
            ),
            "instrument": "perpetuo: 7 pb por rotación más funding (tasa/2 por vela de 4h)",
            "selection_margins": list(SELECTION_MARGINS),
            "selection_cost_bps": SELECTION_COST_BPS,
            "min_selection_history_rows": MIN_SELECTION_HISTORY,
        },
        "configs": {
            c.name: {
                **models[c.name],
                "difference_vs": {
                    name: float(differences[name][:, i].mean()) for name in BASELINES
                },
            }
            for i, c in enumerate(pool)
        },
        "baselines": baselines,
        "accuracy_inference": {
            "primary_pair_bootstrap": primary,
            "reality_check": checks,
            "simultaneous_bounds": {
                "quantile": joint["quantile"],
                "block": joint["block"],
                "lower_bounds": {
                    name: {c.name: float(lower[b, i]) for i, c in enumerate(pool)}
                    for b, name in enumerate(BASELINES)
                },
            },
            "pbo_maximin": pbo_accuracy,
        },
        "strategies": {
            "evaluable_start": evaluable_times.iloc[0].isoformat(),
            "evaluable_samples": int(evaluable.sum()),
            "buy_and_hold_spot": buy_hold,
            "buy_and_hold_perpetual": buy_hold_perpetual,
            "economic_reality_check_vs_buy_and_hold": economic_check,
            "deflated_sharpe": dsr,
            "pbo_sharpe": pbo_sharpe,
            "details": strategies,
        },
        "gate": gate,
        "caveats": [
            harness.POINT_IN_TIME_CAVEAT,
            "Con h>1 las etiquetas se solapan; la inferencia usa bloques de 180 velas.",
            "Los márgenes se eligen por fold con el Sharpe neto (costos y funding) del "
            "historial OOS previo.",
            "El Sharpe deflactado supone independencia y no es inferencial.",
        ],
    }
    return report, predictions


def execute(directory: Path, cutoff: pd.Timestamp, artifact_root: Path, n_folds: int) -> Path:
    manifest = verify_dataset(directory)
    signature = harness.evaluation_signature()
    own = file_sha256(Path(__file__))
    features = pd.read_parquet(directory / manifest["files"]["features"]["path"])
    sources = json.loads((directory / "source-manifest.json").read_text(encoding="utf-8"))
    candles = pd.read_parquet(directory / sources["files"]["candles"]["path"])
    development = development_rows(attach_horizons(features, candles), cutoff)
    assert_no_reserved_training_data(artifact_root, development["open_time"], STEP)
    report, predictions = build_report(development, n_folds)
    if harness.evaluation_signature() != signature or file_sha256(Path(__file__)) != own:
        raise RuntimeError("La implementación cambió durante la evaluación")
    signature["extra_files"]["research/target_study.py"] = own
    name = f"target-study-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
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
    harness.append_registry(
        artifact_root,
        {
            "run_id": name,
            "kind": report["evaluation"],
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_id": manifest["dataset_id"],
            "development_end_exclusive": cutoff.isoformat(),
            "configs": [c.name for c in configs()],
            "strategies": len(report["strategies"]["details"]),
            "oos_samples": len(predictions),
            "gate": report["gate"],
            "report_sha256": file_sha256(path),
        },
    )
    log.info("target_study_complete", report=str(path), gate=report["gate"])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1: horizonte, zona muerta y abstención")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--development-end", required=True)
    parser.add_argument("--folds", type=int, default=harness.WALK_FORWARD_FOLDS)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    execute(
        args.dataset,
        pd.Timestamp(args.development_end),
        Path(settings.model_artifact_dir),
        args.folds,
    )


if __name__ == "__main__":
    main()
