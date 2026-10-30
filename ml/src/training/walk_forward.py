"""Evaluación expanding-window con selección y calibración estrictamente temporales."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog
from xgboost import XGBClassifier

from src.evaluation.calibration import PlattCalibration, fit_platt
from src.evaluation.importance import lstm_importance, xgb_importance
from src.evaluation.metrics import evaluate, evaluate_baselines
from src.training.data import build_folds, inner_windows
from src.training.models import (
    DirectionLSTM,
    fit_lstm,
    fit_xgb,
    predict_lstm,
    sequences,
    tune_xgb,
)
from src.training.preprocess import Preprocessor, fit_preprocessor

if TYPE_CHECKING:
    from datetime import timedelta

    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression

    from src.training.types import FloatArray, IntArray

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TrainingOptions:
    n_folds: int = 10
    min_train_days: int = 365
    lstm_epochs: int = 6
    seed: int = 42
    grid: list[dict[str, Any]] | None = None


@dataclass
class ModelBundle:
    family: str
    model: XGBClassifier | DirectionLSTM | LogisticRegression | LGBMClassifier
    preprocessor: Preprocessor
    calibration: PlattCalibration
    parameters: dict[str, Any]
    sample: FloatArray


@dataclass
class WalkForwardResult:
    report: dict[str, Any]
    candidates: dict[str, ModelBundle]


def _model_metrics(
    labels: IntArray, raw: FloatArray, calibration: PlattCalibration
) -> dict[str, Any]:
    calibrated = calibration.transform(raw)
    return {
        "raw": evaluate(labels, (raw >= 0.5).astype(int), raw),
        "calibrated": evaluate(labels, (calibrated >= 0.5).astype(int), calibrated),
    }


def run_walk_forward(
    data: pd.DataFrame, step: timedelta, options: TrainingOptions | None = None
) -> WalkForwardResult:
    options = options or TrainingOptions()
    folds = build_folds(data, step, options.n_folds, options.min_train_days)
    y = data["target_label"].to_numpy(dtype=int)
    previous = data["previous_direction"].to_numpy(dtype=int)
    results: list[dict[str, Any]] = []
    candidates: dict[str, ModelBundle] = {}
    for fold in folds:
        log.info("training_fold_start", fold=fold.number, start=fold.start.isoformat())
        base, tuning, pre_cal, cal = inner_windows(data, fold, step)
        tuning_cutoff = fold.start - pd.DateOffset(months=2)
        calibration_cutoff = fold.start - pd.DateOffset(months=1)
        if calibration_cutoff - data["open_time"].iloc[pre_cal[0]] < pd.Timedelta(
            days=options.min_train_days
        ):
            raise ValueError(f"Fold {fold.number}: el ajuste requiere al menos un año de datos")
        tuning_pre = fit_preprocessor(data.iloc[base], tuning_cutoff.to_pydatetime(), step)
        tune_x = tuning_pre.transform(data)
        best_params, grid_scores = tune_xgb(
            tune_x[base], y[base], tune_x[tuning], y[tuning], options.seed, options.grid
        )

        pre = fit_preprocessor(data.iloc[pre_cal], calibration_cutoff.to_pydatetime(), step)
        scaled = pre.transform(data)
        xgb = fit_xgb(scaled[pre_cal], y[pre_cal], best_params, options.seed)
        xgb_cal_raw = xgb.predict_proba(scaled[cal])[:, 1]
        xgb_cal = fit_platt(xgb_cal_raw, y[cal])
        xgb_val_raw = xgb.predict_proba(scaled[fold.validation])[:, 1]

        x_train_seq, train_positions = sequences(scaled, pre_cal)
        x_cal_seq, cal_positions = sequences(scaled, cal)
        x_val_seq, val_positions = sequences(scaled, fold.validation)
        if not np.array_equal(cal_positions, cal) or not np.array_equal(
            val_positions, fold.validation
        ):
            raise ValueError("No hay contexto completo para calibración o validación")
        lstm = fit_lstm(x_train_seq, y[train_positions], options.seed, options.lstm_epochs)
        lstm_cal_raw = predict_lstm(lstm, x_cal_seq)
        lstm_cal = fit_platt(lstm_cal_raw, y[cal])
        lstm_val_raw = predict_lstm(lstm, x_val_seq)
        baselines = evaluate_baselines(
            y[fold.train],
            previous[fold.train],
            y[fold.validation],
            previous[fold.validation],
            options.seed + fold.number,
        )
        metrics = {
            "xgboost": _model_metrics(y[fold.validation], xgb_val_raw, xgb_cal),
            "lstm": _model_metrics(y[fold.validation], lstm_val_raw, lstm_cal),
        }
        xgb_train = xgb.predict_proba(scaled[pre_cal])[:, 1]
        lstm_train = predict_lstm(lstm, x_train_seq)
        train_accuracy = {
            "xgboost": float(np.mean((xgb_train >= 0.5) == y[pre_cal])),
            "lstm": float(np.mean((lstm_train >= 0.5) == y[train_positions])),
        }
        warnings = [
            f"{name}: train > 70% y validación < 52%"
            for name in ("xgboost", "lstm")
            if train_accuracy[name] > 0.70 and metrics[name]["calibrated"]["accuracy"] < 0.52
        ]
        result = {
            "fold": fold.number,
            "validation_start": fold.start.isoformat(),
            "validation_end_exclusive": fold.end.isoformat(),
            "training_start": data["open_time"].iloc[fold.train[0]].isoformat(),
            "training_end_known_at": fold.start.isoformat(),
            "tuning_start": tuning_cutoff.isoformat(),
            "calibration_start": calibration_cutoff.isoformat(),
            "training_samples": len(fold.train),
            "fitting_samples": len(pre_cal),
            "tuning_samples": len(tuning),
            "calibration_samples": len(cal),
            "training_up_fraction": float(y[fold.train].mean()),
            "selected_features": pre.feature_names,
            "xgboost_parameters": best_params,
            "xgboost_grid": grid_scores,
            "calibration": {
                "xgboost": vars(xgb_cal),
                "lstm": vars(lstm_cal),
            },
            "baselines": baselines,
            "models": metrics,
            "train_accuracy_raw": train_accuracy,
            "warnings": warnings,
        }
        results.append(result)
        log.info(
            "training_fold_done",
            fold=fold.number,
            xgboost=round(metrics["xgboost"]["calibrated"]["accuracy"], 4),
            lstm=round(metrics["lstm"]["calibrated"]["accuracy"], 4),
            baseline=round(max(item["accuracy"] for item in baselines.values()), 4),
        )
        if fold.number == len(folds):
            candidates = {
                "xgboost": ModelBundle(
                    "xgboost", xgb, pre, xgb_cal, best_params, scaled[fold.validation]
                ),
                "lstm": ModelBundle(
                    "lstm",
                    lstm,
                    pre,
                    lstm_cal,
                    {
                        "hidden_size": 64,
                        "num_layers": 2,
                        "dropout": 0.3,
                        "sequence_length": 20,
                        "epochs": options.lstm_epochs,
                    },
                    x_val_seq,
                ),
            }

    if len(results) < 2:
        raise ValueError("Se requieren folds de selección y un fold final de evaluación")
    selection_scores = {
        name: float(
            np.mean([fold["models"][name]["calibrated"]["accuracy"] for fold in results[:-1]])
        )
        for name in ("xgboost", "lstm")
    }
    winner = max(selection_scores, key=lambda name: (selection_scores[name], name == "xgboost"))
    averages = {
        name: float(np.mean([fold["models"][name]["calibrated"]["accuracy"] for fold in results]))
        for name in ("xgboost", "lstm")
    }
    baseline_averages = {
        name: float(np.mean([fold["baselines"][name]["accuracy"] for fold in results]))
        for name in ("majority", "previous_direction", "conditional_direction", "weighted_random")
    }
    last_baseline = max(item["accuracy"] for item in results[-1]["baselines"].values())
    last_accuracy = results[-1]["models"][winner]["calibrated"]["accuracy"]
    cv_baseline = float(
        np.mean(
            [
                max(baseline["accuracy"] for baseline in fold["baselines"].values())
                for fold in results
            ]
        )
    )
    xgb_model = candidates["xgboost"].model
    lstm_model = candidates["lstm"].model
    if not isinstance(xgb_model, XGBClassifier) or not isinstance(lstm_model, DirectionLSTM):
        raise TypeError("El tipo de modelo no coincide con su familia")
    importance = {
        "xgboost": xgb_importance(
            xgb_model,
            candidates["xgboost"].sample,
            candidates["xgboost"].preprocessor.feature_names,
        ),
        "lstm": lstm_importance(
            lstm_model,
            candidates["lstm"].sample,
            candidates["lstm"].preprocessor.feature_names,
        ),
    }
    dominant = {
        name: max(values.values(), default=0.0) > 0.5 for name, values in importance.items()
    }
    report = {
        "seed": options.seed,
        "folds": results,
        "selected_family": winner,
        "selection_mean_accuracy_without_final_fold": selection_scores,
        "mean_accuracy": averages,
        "baseline_mean_accuracy": baseline_averages,
        "best_baseline_mean_accuracy": cv_baseline,
        "final_fold_accuracy": last_accuracy,
        "final_fold_best_baseline": last_baseline,
        "criteria": {
            "at_least_10_folds": len(results) >= 10,
            "mean_accuracy_above_baseline_plus_3pp": averages[winner] > cv_baseline + 0.03,
            "final_accuracy_above_baseline_plus_2pp": last_accuracy > last_baseline + 0.02,
            "no_dominant_feature": not dominant[winner],
        },
        "importance": importance,
        "feature_dominance_warning": dominant,
    }
    report["criteria"]["passed"] = all(report["criteria"].values())
    return WalkForwardResult(report, candidates)
