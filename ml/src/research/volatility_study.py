"""E-12: pronóstico de volatilidad realizada y exposición ajustada por riesgo.

Pregunta: aunque la dirección no sea predecible, ¿lo es la volatilidad? ¿Y sirve para
dimensionar la exposición? Se pronostica la varianza realizada futura (1 y 7 días) con modelos
de complejidad creciente y se evalúa con QLIKE en los mismos 48 folds mensuales. La aplicación
económica escala una posición larga en BTC por `objetivo / volatilidad pronosticada`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pandas as pd
import structlog
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from src.config import get_settings
from src.evaluation.economic import PERIODS_PER_YEAR_4H, performance, strategy_returns
from src.evaluation.robustness import (
    block_bootstrap_mean,
    reality_check,
    sharpe_difference_bootstrap,
    simultaneous_lower_bounds,
)
from src.features.long_horizon import (
    FAMILIES,
    build_long_horizon_features,
    price_history,
    stationary_feature_set,
)
from src.ingestion.run import configure_logging
from src.research import aux_sources
from src.research import evaluation_harness as harness
from src.research.dataset import verify_dataset
from src.research.source_snapshot import file_sha256
from src.training.holdouts import assert_no_reserved_training_data
from src.training.research_models import _fit_preprocessor

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.training.types import FloatArray

log = structlog.get_logger(__name__)
SEED = harness.SEED
STEP = pd.Timedelta(hours=4)
HORIZONS = {"1d": 6, "7d": 42}
PRIMARY_HORIZON = "7d"
PRIMARY_MODEL = "har_x"
EWMA_HALFLIFE = 42
HAR_COLUMNS = ["log_var_1d", "log_var_7d", "log_var_30d"]
HAR_X_EXTRA = [
    "log_dvol_var",
    "log_vix",
    "funding_sum_7d",
    "oi_change_7d",
    "drawdown_90d",
    "abs_return_1d",
    "volume_sma_ratio",
    "fomc_next_72h",
]
LIGHTGBM_PARAMETERS: dict[str, Any] = {
    "max_depth": 3,
    "num_leaves": 8,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
}
TARGET_VOLATILITY = 0.5
REBALANCE_BAND = 0.1
COST_BPS = 7.0
BLOCK = 180
EPSILON = 1e-12
IMPLEMENTATION_FILES = (
    "research/volatility_study.py",
    "features/long_horizon.py",
    "research/aux_sources.py",
    "evaluation/economic.py",
    "evaluation/robustness.py",
)


# ─── Variables de volatilidad ────────────────────────────────────────────────


def volatility_frame(close: pd.Series, open_times: pd.Series) -> pd.DataFrame:
    """Varianzas anualizadas pasadas (conocidas en t) y futuras (objetivo) por fila."""
    squared = np.log(close).diff() ** 2
    out = pd.DataFrame(index=close.index)
    for name, window in (("1d", 6), ("7d", 42), ("30d", 180)):
        out[f"var_{name}"] = (
            squared.rolling(window, min_periods=window).mean() * PERIODS_PER_YEAR_4H
        )
    out["var_ewma"] = (
        squared.ewm(halflife=EWMA_HALFLIFE, adjust=False, min_periods=EWMA_HALFLIFE).mean()
        * PERIODS_PER_YEAR_4H
    )
    for name, horizon in HORIZONS.items():
        future = squared[::-1].rolling(horizon, min_periods=horizon).mean()[::-1].shift(-1)
        out[f"target_var_{name}"] = future * PERIODS_PER_YEAR_4H
    out["abs_return_1d"] = np.log(close).diff(6).abs()
    out["forward_return"] = close.shift(-1) / close - 1
    times = pd.to_datetime(open_times, utc=True)
    return out.reindex(times).reset_index(drop=True)


def model_inputs(enriched: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    data = enriched.reset_index(drop=True).copy()
    vol = volatility_frame(close, data["open_time"])
    for column in vol:
        data[column] = vol[column].to_numpy()
    for name in ("1d", "7d", "30d"):
        data[f"log_var_{name}"] = np.log(data[f"var_{name}"].clip(lower=EPSILON))
    data["log_dvol_var"] = np.log((data["dvol"] ** 2).clip(lower=EPSILON))
    data["log_vix"] = np.log(data["vix_close"].clip(lower=EPSILON))
    for name in HORIZONS:
        data[f"log_target_{name}"] = np.log(data[f"target_var_{name}"].clip(lower=EPSILON))
    return data


def development_rows(data: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    if cutoff.tzinfo is None:
        raise ValueError("El corte debe incluir zona horaria")
    longest = max(HORIZONS.values())
    required = [*HAR_COLUMNS, "var_ewma", "forward_return"]
    required += [f"target_var_{name}" for name in HORIZONS]
    mask = (
        (data["open_time"] < cutoff)
        & (data["open_time"] + (longest + 1) * STEP <= cutoff)
        & data[required].notna().all(axis=1)
    )
    result = data.loc[mask].sort_values("open_time").reset_index(drop=True)
    if (result["open_time"].diff().dropna() != STEP).any():
        raise ValueError("Las filas de desarrollo deben ser continuas")
    return result


# ─── Modelos ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VolModel:
    name: str
    columns: tuple[str, ...]
    kind: str


def models() -> list[VolModel]:
    everything = tuple(
        dict.fromkeys(
            [
                *HAR_COLUMNS,
                *HAR_X_EXTRA,
                *stationary_feature_set(),
                *(c for family in FAMILIES.values() for c in family),
            ]
        )
    )
    return [
        VolModel("naive", (), "naive"),
        VolModel("ewma", (), "ewma"),
        VolModel("har", tuple(HAR_COLUMNS), "linear"),
        VolModel("har_x", (*HAR_COLUMNS, *HAR_X_EXTRA), "linear"),
        VolModel("lightgbm_all", everything, "lightgbm"),
    ]


def fit_predict_variance(
    model: VolModel, train: pd.DataFrame, test: pd.DataFrame, horizon: str
) -> FloatArray:
    """Pronóstico de varianza anualizada con corrección de sesgo lognormal."""
    if model.kind == "naive":
        result: FloatArray = test[f"var_{horizon}"].to_numpy(dtype=float)
        return result
    if model.kind == "ewma":
        result = test["var_ewma"].to_numpy(dtype=float)
        return result
    pre = _fit_preprocessor(train, list(model.columns))
    x_train, x_test = pre.transform(train), pre.transform(test)
    y = train[f"log_target_{horizon}"].to_numpy(dtype=float)
    estimator: Any
    if model.kind == "linear":
        estimator = LinearRegression().fit(x_train, y)
    elif model.kind == "lightgbm":
        estimator = LGBMRegressor(
            **LIGHTGBM_PARAMETERS,
            random_state=SEED,
            n_jobs=2,
            subsample_freq=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ).fit(x_train, y)
    else:
        raise ValueError(f"Modelo desconocido: {model.kind}")
    residual = y - estimator.predict(x_train)
    result = np.exp(estimator.predict(x_test) + 0.5 * float(np.var(residual)))
    return result


def run_walk_forward(
    development: pd.DataFrame,
    n_folds: int,
    pool: list[VolModel],
    predictor: Callable[[VolModel, pd.DataFrame, pd.DataFrame, str], FloatArray] = (
        fit_predict_variance
    ),
) -> pd.DataFrame:
    predictions: pd.DataFrame | None = None
    reference: Any = None
    for name, horizon in HORIZONS.items():
        folds = harness.monthly_walk_forward(development["open_time"], n_folds, horizon=horizon)
        validation = np.concatenate([fold.validation for fold in folds])
        if predictions is None:
            predictions = development.iloc[validation][
                ["open_time", "forward_return", "funding_rate"]
            ].reset_index(drop=True)
            predictions["fold"] = np.concatenate(
                [np.full(len(fold.validation), fold.number) for fold in folds]
            )
            reference = validation
        elif not np.array_equal(validation, reference):
            raise ValueError("Los horizontes deben compartir las filas de validación")
        predictions[f"target_var_{name}"] = development[f"target_var_{name}"].to_numpy()[validation]
        columns: dict[str, list[FloatArray]] = {}
        for fold in folds:
            train = development.iloc[fold.train]
            test = development.iloc[fold.validation]
            for model in pool:
                columns.setdefault(f"{model.name}_{name}", []).append(
                    predictor(model, train, test, name)
                )
            log.info("volatility_study_fold", horizon=name, fold=fold.number)
        for column, parts in columns.items():
            predictions[column] = np.concatenate(parts)
    if predictions is None:
        raise ValueError("No hay horizontes")
    return predictions


# ─── Evaluación del pronóstico ───────────────────────────────────────────────


def qlike(realized: FloatArray, forecast: FloatArray) -> FloatArray:
    """Pérdida QLIKE (robusta al ruido del proxy de varianza); 0 en pronóstico perfecto."""
    ratio = np.asarray(realized, dtype=float) / np.maximum(
        np.asarray(forecast, dtype=float), EPSILON
    )
    ratio = np.maximum(ratio, EPSILON)
    result: FloatArray = ratio - np.log(ratio) - 1
    return result


def forecast_metrics(realized: FloatArray, forecast: FloatArray) -> dict[str, float]:
    log_real = np.log(np.maximum(realized, EPSILON))
    log_fore = np.log(np.maximum(forecast, EPSILON))
    slope, intercept = np.polyfit(log_fore, log_real, 1)
    return {
        "qlike": float(qlike(realized, forecast).mean()),
        "mse_log_variance": float(np.mean((log_real - log_fore) ** 2)),
        "correlation_log": float(np.corrcoef(log_real, log_fore)[0, 1]),
        "mincer_zarnowitz_slope": float(slope),
        "mincer_zarnowitz_intercept": float(intercept),
        "mean_forecast_volatility": float(np.sqrt(forecast).mean()),
        "mean_realized_volatility": float(np.sqrt(realized).mean()),
    }


def forecast_report(predictions: pd.DataFrame, pool: list[VolModel]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        realized = predictions[f"target_var_{horizon}"].to_numpy(dtype=float)
        losses = {
            m.name: qlike(realized, predictions[f"{m.name}_{horizon}"].to_numpy(dtype=float))
            for m in pool
        }
        naive_log_error = (
            np.log(np.maximum(realized, EPSILON))
            - np.log(np.maximum(predictions[f"naive_{horizon}"].to_numpy(dtype=float), EPSILON))
        ) ** 2
        metrics = {}
        for m in pool:
            forecast = predictions[f"{m.name}_{horizon}"].to_numpy(dtype=float)
            entry = forecast_metrics(realized, forecast)
            log_error = (np.log(np.maximum(realized, EPSILON)) - np.log(forecast)) ** 2
            entry["oos_r2_log_vs_naive"] = float(1 - log_error.mean() / naive_log_error.mean())
            metrics[m.name] = entry
        names = [m.name for m in pool if m.name != "naive"]
        improvement = np.column_stack([losses["naive"] - losses[name] for name in names])
        bounds = simultaneous_lower_bounds(improvement, BLOCK, seed=SEED)
        pairs = {
            "har_minus_naive": ("naive", "har"),
            "har_x_minus_har": ("har", "har_x"),
            "lightgbm_minus_har": ("har", "lightgbm_all"),
            "lightgbm_minus_har_x": ("har_x", "lightgbm_all"),
        }
        result[horizon] = {
            "models": metrics,
            "qlike_improvement_vs_naive_simultaneous_lower": {
                name: float(bounds["lower_bounds"][index]) for index, name in enumerate(names)
            },
            "paired_qlike_improvement": {
                label: block_bootstrap_mean(losses[ref] - losses[cand], BLOCK, seed=SEED)
                for label, (ref, cand) in pairs.items()
            },
        }
    return result


# ─── Aplicación económica: exposición ajustada por volatilidad ───────────────


def target_exposure(forecast_variance: FloatArray) -> FloatArray:
    """Posición larga min(1, objetivo/σ̂) con banda de rebalanceo para limitar costos."""
    desired = np.minimum(1.0, TARGET_VOLATILITY / np.sqrt(np.maximum(forecast_variance, EPSILON)))
    position: FloatArray = np.empty_like(desired)
    current = desired[0]
    for index, value in enumerate(desired):
        if abs(value - current) > REBALANCE_BAND:
            current = value
        position[index] = current
    return position


def economic_report(predictions: pd.DataFrame, pool: list[VolModel]) -> dict[str, Any]:
    r1 = predictions["forward_return"].to_numpy(dtype=float)
    buy_hold = strategy_returns(np.ones(len(r1)), r1, 0.0)
    strategies: dict[str, Any] = {}
    returns: dict[str, FloatArray] = {}
    for horizon in HORIZONS:
        for model in pool:
            name = f"voltarget_{model.name}_{horizon}"
            position = target_exposure(predictions[f"{model.name}_{horizon}"].to_numpy(dtype=float))
            net = strategy_returns(position, r1, COST_BPS)
            constant = strategy_returns(np.full(len(r1), position.mean()), r1, COST_BPS)
            returns[name] = net
            strategies[name] = {
                **performance(net, position),
                "mean_exposure": float(position.mean()),
                "sharpe_minus_buy_hold": sharpe_difference_bootstrap(
                    net, buy_hold, BLOCK, PERIODS_PER_YEAR_4H, seed=SEED
                ),
                "constant_same_exposure": performance(constant, np.full(len(r1), position.mean())),
                "sharpe_minus_constant_exposure": sharpe_difference_bootstrap(
                    net, constant, BLOCK, PERIODS_PER_YEAR_4H, seed=SEED
                ),
            }
    names = list(returns)
    matrix = np.column_stack([returns[n] for n in names])
    check = reality_check(matrix - buy_hold[:, None], BLOCK, seed=SEED)
    check["best_strategy"] = names[check.pop("best_column")]
    return {
        "instrument": "spot largo, 7 pb por rotación, sin funding",
        "target_volatility": TARGET_VOLATILITY,
        "rebalance_band": REBALANCE_BAND,
        "buy_and_hold": performance(buy_hold, np.ones(len(r1))),
        "reality_check_returns_vs_buy_and_hold": check,
        "strategies": strategies,
    }


def build_report(
    development: pd.DataFrame,
    n_folds: int = harness.WALK_FORWARD_FOLDS,
    predictor: Callable[[VolModel, pd.DataFrame, pd.DataFrame, str], FloatArray] = (
        fit_predict_variance
    ),
) -> tuple[dict[str, Any], pd.DataFrame]:
    pool = models()
    with threadpool_limits(limits=2):
        predictions = run_walk_forward(development, n_folds, pool, predictor)
    forecasts = forecast_report(predictions, pool)
    economics = economic_report(predictions, pool)
    primary = forecasts[PRIMARY_HORIZON]
    har_x = primary["paired_qlike_improvement"]["har_x_minus_har"]
    har = primary["paired_qlike_improvement"]["har_minus_naive"]
    strategy = economics["strategies"][f"voltarget_{PRIMARY_MODEL}_{PRIMARY_HORIZON}"]
    gate = {
        "har_beats_naive_ci_95": har["ci_95"],
        "har_x_beats_har_ci_95": har_x["ci_95"],
        "volatility_predictable": bool(har["ci_95"][0] > 0),
        "har_x_adds_information": bool(har_x["ci_95"][0] > 0),
        "primary_qlike": primary["models"][PRIMARY_MODEL]["qlike"],
        "primary_oos_r2_log_vs_naive": primary["models"][PRIMARY_MODEL]["oos_r2_log_vs_naive"],
        "primary_strategy_sharpe": strategy["sharpe"],
        "primary_strategy_max_drawdown": strategy["max_drawdown"],
        "buy_and_hold_sharpe": economics["buy_and_hold"]["sharpe"],
        "buy_and_hold_max_drawdown": economics["buy_and_hold"]["max_drawdown"],
        "sharpe_minus_buy_hold_ci_95": strategy["sharpe_minus_buy_hold"]["ci_95"],
        "sharpe_minus_constant_exposure_ci_95": strategy["sharpe_minus_constant_exposure"]["ci_95"],
        "reality_check_returns_p_value": economics["reality_check_returns_vs_buy_and_hold"][
            "p_value"
        ],
    }
    report = {
        "evaluation": "volatility_forecasting_study",
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
            "primary_model": PRIMARY_MODEL,
            "loss": "QLIKE sobre varianza anualizada de retornos de 4h",
            "har_x_extra": HAR_X_EXTRA,
            "block": BLOCK,
            "gate": (
                "volatilidad predecible si HAR mejora QLIKE frente al ingenuo con IC > 0; "
                "HAR-X aporta si mejora a HAR con IC > 0; la aplicación económica se "
                "compara con buy & hold y con una exposición constante igual a la media"
            ),
        },
        "forecasts": forecasts,
        "economics": economics,
        "gate": gate,
        "caveats": [
            harness.POINT_IN_TIME_CAVEAT,
            "La varianza realizada usa retornos de 4h (proxy ruidoso); QLIKE es robusto a ello.",
            "DVOL no existe antes de 2021-03-24; se imputa la mediana del train en ese tramo.",
        ],
    }
    return report, predictions


def load_development(dataset: Path, aux_directory: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    manifest = verify_dataset(dataset)
    features = pd.read_parquet(dataset / manifest["files"]["features"]["path"])
    metrics = pd.read_parquet(dataset / manifest["files"]["metrics"]["path"])
    source_manifest = json.loads((dataset / "source-manifest.json").read_text(encoding="utf-8"))
    sources = {
        name: pd.read_parquet(dataset / item["path"])
        for name, item in source_manifest["files"].items()
    }
    aux = aux_sources.load(aux_directory)
    enriched = build_long_horizon_features(features, sources, metrics, aux)
    candles = sources["candles"].copy()
    candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True)
    close = price_history(candles, aux["btc_spot_klines"])
    return development_rows(model_inputs(enriched, close), cutoff)


def execute(
    dataset: Path, aux_directory: Path, cutoff: pd.Timestamp, artifact_root: Path, n_folds: int
) -> Path:
    root = Path(__file__).resolve().parents[1]
    implementation = {name: file_sha256(root / name) for name in IMPLEMENTATION_FILES}
    aux_manifest = aux_sources.verify(aux_directory)
    dataset_manifest = verify_dataset(dataset)
    development = load_development(dataset, aux_directory, cutoff)
    assert_no_reserved_training_data(artifact_root, development["open_time"], STEP)
    report, predictions = build_report(development, n_folds)
    if {name: file_sha256(root / name) for name in IMPLEMENTATION_FILES} != implementation:
        raise RuntimeError("La implementación cambió durante la evaluación")
    name = f"volatility-study-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
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
            "models": [m.name for m in models()],
            "oos_samples": len(predictions),
            "gate": report["gate"],
            "report_sha256": file_sha256(path),
        },
    )
    log.info("volatility_study_complete", report=str(path), gate=report["gate"])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pronóstico de volatilidad (E-12)")
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
