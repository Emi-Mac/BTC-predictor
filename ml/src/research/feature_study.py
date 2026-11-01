"""Fase 2: ablación de familias de features de horizonte largo a 24h y 3d.

Pregunta: ¿alguna familia (estacionariedad, régimen, derivados, on-chain, eventos o activos
relacionados) aporta información sobre la dirección a 24h o 3 días que las features técnicas
de corto plazo no tienen? Cada familia se añade sobre el conjunto estacionario y se compara de
forma pareada con él en los mismos 48 folds. El gate de la Fase 1 se mantiene sin cambios.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.config import get_settings
from src.evaluation.economic import performance
from src.evaluation.robustness import (
    block_bootstrap_mean,
    deflated_sharpe_ratio,
    effective_trials,
    maximin_pbo,
    probability_of_backtest_overfitting,
    reality_check,
    simultaneous_lower_bounds,
)
from src.features.long_horizon import (
    BASE_COLUMNS,
    FAMILIES,
    build_long_horizon_features,
    stationary_feature_set,
)
from src.ingestion.run import configure_logging
from src.research import aux_sources
from src.research import evaluation_harness as harness
from src.research import target_study as study
from src.research.dataset import verify_dataset
from src.research.source_snapshot import file_sha256
from src.training.holdouts import assert_no_reserved_training_data
from src.training.models import fit_lightgbm
from src.training.research_models import _fit_logistic, _fit_preprocessor

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.training.types import FloatArray

log = structlog.get_logger(__name__)
SEED = harness.SEED
STEP = harness.STEP
HORIZONS = (6, 18)
L2_C = 0.01
L1_C = 0.05
EXTENSIONS = ("regime", "derivatives", "onchain", "events", "cross_asset", "options")
REFERENCE_SET = "stationary"
PRIMARY_CONFIG = "h6_all_logistic_l2"
BLOCK = 180
CONFIDENCE = 0.95
GATE_MIN_DIFFERENCE = 0.03
PBO_PARTITIONS = 16
STABILITY_TOP = 25
IMPLEMENTATION_FILES = (
    "features/long_horizon.py",
    "research/aux_sources.py",
    "research/feature_study.py",
    "research/target_study.py",
)


def feature_sets() -> dict[str, list[str]]:
    stationary = stationary_feature_set()
    result = {"base": list(BASE_COLUMNS), "stationary": stationary}
    for family in EXTENSIONS:
        result[f"stationary_{family}"] = [*stationary, *FAMILIES[family]]
    result["all"] = [*stationary, *(c for family in EXTENSIONS for c in FAMILIES[family])]
    return result


@dataclass(frozen=True)
class FeatureConfig:
    horizon: int
    features: str
    model: str

    @property
    def name(self) -> str:
        return f"h{self.horizon}_{self.features}_{self.model}"


def configs() -> list[FeatureConfig]:
    result: list[FeatureConfig] = []
    for horizon in HORIZONS:
        result.extend(FeatureConfig(horizon, name, "logistic_l2") for name in feature_sets())
        result.append(FeatureConfig(horizon, "all", "logistic_l1"))
        result.append(FeatureConfig(horizon, "all", "lightgbm"))
    return result


def fit_config(
    config: FeatureConfig, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[FloatArray, dict[str, float] | None]:
    names = feature_sets()[config.features]
    pre = _fit_preprocessor(train, names)
    x_train, x_test = pre.transform(train), pre.transform(test)
    y = train["target_label"].to_numpy(dtype=int)
    model: Any
    if config.model == "logistic_l2":
        model = _fit_logistic(x_train, y, L2_C, SEED)
    elif config.model == "logistic_l1":
        model = LogisticRegression(
            l1_ratio=1.0, solver="liblinear", C=L1_C, random_state=SEED, max_iter=2000
        ).fit(x_train, y)
    elif config.model == "lightgbm":
        model = fit_lightgbm(x_train, y, harness.LIGHTGBM_PARAMETERS, SEED)
    else:
        raise ValueError(f"Modelo desconocido: {config.model}")
    probability: FloatArray = np.asarray(model.predict_proba(x_test), dtype=float)[:, 1]
    coefficients = (
        {name: float(value) for name, value in zip(names, model.coef_[0], strict=True)}
        if hasattr(model, "coef_")
        else None
    )
    return probability, coefficients


# ─── Walk-forward ────────────────────────────────────────────────────────────


def run_walk_forward(
    development: pd.DataFrame,
    n_folds: int,
    pool: list[FeatureConfig],
    fitter: Callable[
        [FeatureConfig, pd.DataFrame, pd.DataFrame], tuple[FloatArray, dict[str, float] | None]
    ] = fit_config,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, float]]]]:
    keep = ["open_time", "sigma", "funding_rate"]
    keep += [f"forward_return_{h}" for h in sorted({1, *HORIZONS})]
    predictions: pd.DataFrame | None = None
    reference: Any = None
    coefficients: dict[str, list[dict[str, float]]] = {}
    for horizon in sorted({c.horizon for c in pool}):
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
        predictions[f"label_h{horizon}"] = study.horizon_label(
            development.iloc[validation], horizon
        )
        columns: dict[str, list[FloatArray]] = {}
        for fold in folds:
            train = development.iloc[fold.train].copy()
            test = development.iloc[fold.validation]
            for name, values in study.baseline_probabilities(train, test, horizon).items():
                columns.setdefault(f"baseline_{name}_h{horizon}", []).append(values)
            train["target_label"] = study.horizon_label(train, horizon)
            for config in (c for c in pool if c.horizon == horizon):
                probability, coef = fitter(config, train, test)
                columns.setdefault(config.name, []).append(probability)
                if coef is not None:
                    coefficients.setdefault(config.name, []).append(coef)
            log.info("feature_study_fold", horizon=horizon, fold=fold.number)
        for name, parts in columns.items():
            predictions[name] = np.concatenate(parts)
    if predictions is None:
        raise ValueError("No hay configuraciones")
    return predictions, coefficients


# ─── Análisis ────────────────────────────────────────────────────────────────


def _log_loss_rows(predictions: pd.DataFrame, column: str, horizon: int) -> FloatArray:
    p = np.clip(predictions[column].to_numpy(dtype=float), 1e-7, 1 - 1e-7)
    y = predictions[f"label_h{horizon}"].to_numpy(dtype=int)
    result: FloatArray = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return result


def ablation_report(predictions: pd.DataFrame, pool: list[FeatureConfig]) -> dict[str, Any]:
    """Mejora pareada de cada conjunto frente al estacionario (y del estacionario frente a base).

    Una familia se conserva solo si los límites inferiores simultáneos de la mejora en log-loss
    y en accuracy son positivos (corregido por todas las comparaciones a la vez).
    """
    comparisons: list[tuple[str, str, int]] = []
    for horizon in HORIZONS:
        names = {c.features for c in pool if c.horizon == horizon and c.model == "logistic_l2"}
        if REFERENCE_SET not in names:
            continue
        reference = f"h{horizon}_{REFERENCE_SET}_logistic_l2"
        if "base" in names:
            comparisons.append((reference, f"h{horizon}_base_logistic_l2", horizon))
        for name in sorted(names - {"base", REFERENCE_SET}):
            comparisons.append((f"h{horizon}_{name}_logistic_l2", reference, horizon))
    if not comparisons:
        return {"comparisons": []}
    loss_gain = np.column_stack(
        [
            _log_loss_rows(predictions, ref, h) - _log_loss_rows(predictions, cand, h)
            for cand, ref, h in comparisons
        ]
    )
    accuracy_gain = np.column_stack(
        [
            study._correct(predictions, cand, h) - study._correct(predictions, ref, h)
            for cand, ref, h in comparisons
        ]
    )
    loss_bounds = simultaneous_lower_bounds(loss_gain, BLOCK, seed=SEED, level=CONFIDENCE)
    accuracy_bounds = simultaneous_lower_bounds(accuracy_gain, BLOCK, seed=SEED, level=CONFIDENCE)
    rows = []
    for index, (candidate, reference, horizon) in enumerate(comparisons):
        loss_lower = float(loss_bounds["lower_bounds"][index])
        accuracy_lower = float(accuracy_bounds["lower_bounds"][index])
        rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "horizon": horizon,
                "log_loss_improvement": float(loss_bounds["means"][index]),
                "log_loss_improvement_simultaneous_lower": loss_lower,
                "accuracy_improvement": float(accuracy_bounds["means"][index]),
                "accuracy_improvement_simultaneous_lower": accuracy_lower,
                "accuracy_improvement_block_ci_95": block_bootstrap_mean(
                    accuracy_gain[:, index], BLOCK, seed=SEED
                )["ci_95"],
                "kept": bool(loss_lower > 0 and accuracy_lower > 0),
            }
        )
    return {
        "rule": (
            "conservar si los límites inferiores simultáneos (95%) de la mejora en log-loss y "
            "accuracy son positivos"
        ),
        "block": BLOCK,
        "comparisons": rows,
    }


def coefficient_stability(coefficients: list[dict[str, float]]) -> list[dict[str, Any]]:
    """Signo y magnitud de cada coeficiente estandarizado a través de los folds (área D)."""
    frame = pd.DataFrame(coefficients)
    signs = np.sign(frame)
    summary = pd.DataFrame(
        {
            "mean_coefficient": frame.mean(),
            "mean_abs_coefficient": frame.abs().mean(),
            "sign_consistency": signs.mean().abs(),
            "nonzero_fraction": (frame != 0).mean(),
        }
    ).sort_values("mean_abs_coefficient", ascending=False)
    return [
        {"feature": str(name), **{k: float(v) for k, v in row.items()}}
        for name, row in summary.head(STABILITY_TOP).iterrows()
    ]


def coverage_report(development: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for family, columns in FAMILIES.items():
        frame = development[columns]
        result[family] = {
            name: {
                "missing_fraction": float(frame[name].isna().mean()),
                "first_available": (
                    development.loc[frame[name].notna(), "open_time"].min().isoformat()
                    if frame[name].notna().any()
                    else None
                ),
            }
            for name in columns
        }
    return result


def build_report(
    development: pd.DataFrame,
    n_folds: int = harness.WALK_FORWARD_FOLDS,
    fitter: Callable[
        [FeatureConfig, pd.DataFrame, pd.DataFrame], tuple[FloatArray, dict[str, float] | None]
    ] = fit_config,
    pool: list[FeatureConfig] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pool = pool or configs()
    if PRIMARY_CONFIG not in {c.name for c in pool}:
        raise ValueError("La configuración primaria debe pertenecer al conjunto evaluado")
    with threadpool_limits(limits=2):
        predictions, coefficients = run_walk_forward(development, n_folds, pool, fitter)

    horizons = sorted({c.horizon for c in pool})
    baselines = {}
    for h in horizons:
        climatology = predictions[f"baseline_majority_h{h}"].to_numpy(dtype=float)
        for name in study.BASELINES:
            column = f"baseline_{name}_h{h}"
            baselines[column] = study.summarize(predictions, column, h, climatology)
    models = {
        c.name: study.summarize(
            predictions, c.name, c.horizon, predictions[f"baseline_majority_h{c.horizon}"]
        )
        for c in pool
    }

    differences = {
        name: np.column_stack(
            [
                study._correct(predictions, c.name, c.horizon)
                - study._correct(predictions, f"baseline_{name}_h{c.horizon}", c.horizon)
                for c in pool
            ]
        )
        for name in study.BASELINES
    }
    joint = simultaneous_lower_bounds(
        np.hstack([differences[name] for name in study.BASELINES]),
        BLOCK,
        seed=SEED,
        level=CONFIDENCE,
    )
    lower = joint["lower_bounds"].reshape(len(study.BASELINES), len(pool))
    means = joint["means"].reshape(len(study.BASELINES), len(pool))
    worst_case, worst_lower = means.min(axis=0), lower.min(axis=0)
    best = int(worst_lower.argmax())
    checks = {}
    for name, matrix in differences.items():
        check = reality_check(matrix, BLOCK, seed=SEED)
        check["best_config"] = pool[check.pop("best_column")].name
        checks[name] = check
    pbo_accuracy = maximin_pbo(
        np.stack([differences[name] for name in study.BASELINES]), PBO_PARTITIONS
    )

    returns, strategies, evaluable = study.strategy_table(predictions, pool)
    names = list(returns)
    matrix = np.column_stack([returns[n][evaluable] for n in names])
    r1 = predictions["forward_return_1"].to_numpy(dtype=float)[evaluable]
    buy_hold = performance(r1, np.ones(len(r1)))
    economic = reality_check(matrix - r1[:, None], BLOCK, seed=SEED)
    economic["best_strategy"] = names[economic.pop("best_column")]
    std = matrix.std(axis=0, ddof=1)
    sharpes = matrix.mean(axis=0) / np.where(std > 0, std, 1)
    top = int(sharpes.argmax())
    dsr = deflated_sharpe_ratio(matrix[:, top], sharpes, effective_trials(matrix))
    dsr.update({"strategy": names[top], "n_trials_nominal": len(names), "inferential": False})
    pbo_sharpe = probability_of_backtest_overfitting(matrix, PBO_PARTITIONS, "sharpe")

    best_name = pool[best].name
    gate = {
        "best_config_by_worst_case_lower_bound": best_name,
        "best_accuracy": models[best_name]["accuracy"],
        "best_worst_case_difference": float(worst_case[best]),
        "best_worst_case_simultaneous_lower_bound": float(worst_lower[best]),
        "reality_check_p_values": {name: check["p_value"] for name, check in checks.items()},
        "primary_config": PRIMARY_CONFIG,
        "primary_accuracy": models[PRIMARY_CONFIG]["accuracy"],
        "primary_worst_case_difference": float(
            worst_case[[c.name for c in pool].index(PRIMARY_CONFIG)]
        ),
        "pbo_accuracy": pbo_accuracy["pbo"],
        "best_strategy": dsr["strategy"],
        "best_strategy_net_sharpe": strategies[dsr["strategy"]]["by_cost"][
            str(study.SELECTION_COST_BPS)
        ]["sharpe"],
        "buy_and_hold_spot_sharpe": buy_hold["sharpe"],
        "economic_reality_check_p_value": economic["p_value"],
        "pbo_sharpe": pbo_sharpe["pbo"],
    }
    gate["passed"] = bool(
        gate["best_worst_case_difference"] >= GATE_MIN_DIFFERENCE
        and gate["best_worst_case_simultaneous_lower_bound"] > 0
    )
    stability = {
        name: coefficient_stability(items)
        for name, items in coefficients.items()
        if name.split("_", 1)[1].startswith("all_")
    }
    report = {
        "evaluation": "phase2_feature_study",
        "independent_holdout_evaluated": False,
        "model_registered": False,
        "seed": SEED,
        "interval": "4h",
        "horizons_candles": horizons,
        "feature_sets": {name: len(columns) for name, columns in feature_sets().items()},
        "families": FAMILIES,
        "models": {"logistic_l2_C": L2_C, "logistic_l1_C": L1_C, **harness.LIGHTGBM_PARAMETERS},
        "walk_forward": {
            "folds": int(predictions["fold"].nunique()),
            "oos_samples": len(predictions),
            "first_validation_start": predictions["open_time"].iloc[0].isoformat(),
        },
        "pre_specified": {
            "primary_config": PRIMARY_CONFIG,
            "reference_set": REFERENCE_SET,
            "block": BLOCK,
            "gate": "igual que la Fase 1 (límites simultáneos frente a los tres baselines, ≥3 pp)",
            "ablation_rule": "mejora simultánea positiva en log-loss y accuracy",
        },
        "configs": {
            c.name: {
                **asdict(c),
                **models[c.name],
                "difference_vs": {
                    name: float(differences[name][:, i].mean()) for name in study.BASELINES
                },
            }
            for i, c in enumerate(pool)
        },
        "baselines": baselines,
        "ablation": ablation_report(predictions, pool),
        "coefficient_stability": stability,
        "feature_coverage": coverage_report(development),
        "accuracy_inference": {
            "simultaneous_quantile": joint["quantile"],
            "reality_check": checks,
            "pbo_maximin": pbo_accuracy,
        },
        "strategies": {
            "evaluable_start": predictions["open_time"][evaluable].iloc[0].isoformat(),
            "evaluable_samples": int(evaluable.sum()),
            "buy_and_hold_spot": buy_hold,
            "economic_reality_check_vs_buy_and_hold": economic,
            "deflated_sharpe": dsr,
            "pbo_sharpe": pbo_sharpe,
            "details": strategies,
        },
        "gate": gate,
        "caveats": [
            harness.POINT_IN_TIME_CAVEAT,
            "CoinMetrics puede revisar datos históricos; se asume disponibilidad en D+2.",
            "Las métricas de futuros (OI, posicionamiento) se usan con 48h de demora simulada.",
            "Faltan el calendario del CPI y los flujos de ETF (sin fuente accesible).",
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
    enriched = build_long_horizon_features(
        features, sources, metrics, aux_sources.load(aux_directory)
    )
    return study.development_rows(study.attach_horizons(enriched, sources["candles"]), cutoff)


def signature() -> dict[str, Any]:
    result = harness.evaluation_signature()
    root = Path(__file__).resolve().parents[1]
    result["extra_files"].update({name: file_sha256(root / name) for name in IMPLEMENTATION_FILES})
    return result


def execute(
    dataset: Path, aux_directory: Path, cutoff: pd.Timestamp, artifact_root: Path, n_folds: int
) -> Path:
    implementation = signature()
    aux_manifest = aux_sources.verify(aux_directory)
    dataset_manifest = verify_dataset(dataset)
    development = load_development(dataset, aux_directory, cutoff)
    assert_no_reserved_training_data(artifact_root, development["open_time"], STEP)
    report, predictions = build_report(development, n_folds)
    if signature() != implementation:
        raise RuntimeError("La implementación cambió durante la evaluación")
    name = f"feature-study-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    report.update(
        {
            "run_id": name,
            "development_end_exclusive": cutoff.isoformat(),
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
            "aux_id": aux_manifest["aux_id"],
            "aux_manifest_sha256": aux_manifest["manifest_sha256"],
            "point_in_time_verified": False,
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
            "development_end_exclusive": cutoff.isoformat(),
            "configs": [c.name for c in configs()],
            "strategies": len(report["strategies"]["details"]),
            "oos_samples": len(predictions),
            "gate": report["gate"],
            "report_sha256": file_sha256(path),
        },
    )
    log.info("feature_study_complete", report=str(path), gate=report["gate"])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2: ablación de familias de features")
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
