"""Candidatos predefinidos para experimentos temporales comparables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from src.evaluation.calibration import PlattCalibration, fit_platt
from src.features.compute import FEATURE_COLUMNS
from src.features.funding_context import FUNDING_FEATURE_COLUMNS
from src.features.price_context import PRICE_CONTEXT_COLUMNS
from src.features.taker_flow import TAKER_FEATURE_COLUMNS
from src.training.data import inner_windows
from src.training.models import fit_lightgbm, fit_xgb, tune_lightgbm, tune_xgb, xgb_grid
from src.training.preprocess import Preprocessor, fit_preprocessor

if TYPE_CHECKING:
    from datetime import timedelta

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    from src.training.data import Fold
    from src.training.types import FloatArray, IntArray


@dataclass(frozen=True)
class Candidate:
    family: str
    features: str
    lookback_days: int | None

    @property
    def name(self) -> str:
        horizon = "expanding" if self.lookback_days is None else f"rolling{self.lookback_days}"
        return f"{self.family}_{self.features}_{horizon}"


def candidates(
    include_taker: bool = False,
    include_lightgbm: bool = False,
    include_funding: bool = False,
) -> list[Candidate]:
    result = [Candidate("xgboost", "legacy", None)] + [
        Candidate(family, features, lookback)
        for family in ("xgboost", "logistic")
        for features in ("existing", "expanded")
        for lookback in (None, 540)
    ]
    if include_taker:
        result.extend(
            Candidate(family, "expanded_taker", lookback)
            for family in ("xgboost", "logistic")
            for lookback in (None, 540)
        )
        result.extend(
            Candidate("logistic", name, None)
            for name in ("expanded_taker_ratio", "expanded_taker_multiscale")
        )
    if include_lightgbm:
        feature_sets = ["existing", "expanded"]
        if include_taker:
            feature_sets.append("expanded_taker")
        result.extend(
            Candidate("lightgbm", features, lookback)
            for features in feature_sets
            for lookback in (None, 540)
        )
    if include_funding:
        families = ["logistic", "xgboost"]
        if include_lightgbm:
            families.append("lightgbm")
        feature_sets = ["expanded_funding"]
        if include_taker:
            feature_sets.append("expanded_taker_funding")
        result.extend(
            Candidate(family, features, lookback)
            for family in families
            for features in feature_sets
            for lookback in (None, 540)
        )
    return result


@dataclass
class FittedCandidate:
    candidate: Candidate
    model: XGBClassifier | LogisticRegression | LGBMClassifier
    preprocessor: Preprocessor
    parameters: dict[str, Any]
    platt: PlattCalibration
    raw_validation: FloatArray
    scaled_validation: FloatArray
    tuning_scores: list[dict[str, Any]]
    fitting_samples: int
    calibration_samples: int


def _fit_preprocessor(data: pd.DataFrame, names: list[str]) -> Preprocessor:
    values = data[names].to_numpy(dtype=float)
    if not len(values) or np.isinf(values).any() or np.isnan(values).all(axis=0).any():
        raise ValueError("Train vacío, con infinitos o alguna feature totalmente ausente")
    imputer = SimpleImputer(strategy="median").fit(values)
    imputed = imputer.transform(values)
    scaler = StandardScaler().fit(imputed)
    return Preprocessor(
        feature_names=names,
        median=[float(value) for value in imputer.statistics_],
        mean=[float(value) for value in scaler.mean_],
        scale=[float(value) for value in scaler.scale_],
    )


def _fit_logistic(x: FloatArray, y: IntArray, c: float, seed: int) -> LogisticRegression:
    balance = float(np.mean(y))
    if not 0 < balance < 1:
        raise ValueError("El entrenamiento requiere ambas clases")
    model = LogisticRegression(
        C=c,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
        class_weight="balanced" if max(balance, 1 - balance) > 0.6 else None,
    )
    model.fit(x, y)
    return model


def _restrict(
    data: pd.DataFrame, positions: IntArray, cutoff: pd.Timestamp, lookback_days: int | None
) -> IntArray:
    limited = positions
    if lookback_days is not None:
        mask = data["open_time"].iloc[positions] >= cutoff - pd.Timedelta(days=lookback_days)
        limited = positions[mask.to_numpy()]
    if not len(limited) or cutoff - data["open_time"].iloc[limited[0]] < pd.Timedelta(days=365):
        raise ValueError("La ventana de entrenamiento requiere al menos un año")
    return limited


def fit_candidate(
    data: pd.DataFrame,
    fold: Fold,
    step: timedelta,
    candidate: Candidate,
    seed: int = 42,
) -> FittedCandidate:
    base, tuning, pre_cal, calibration = inner_windows(data, fold, step)
    tuning_start = fold.start - pd.DateOffset(months=2)
    calibration_start = fold.start - pd.DateOffset(months=1)
    base = _restrict(data, base, tuning_start, candidate.lookback_days)
    pre_cal = _restrict(data, pre_cal, calibration_start, candidate.lookback_days)
    y = data["target_label"].to_numpy(dtype=int)
    if candidate.features in (
        "expanded",
        "expanded_taker",
        "expanded_taker_ratio",
        "expanded_taker_multiscale",
        "expanded_funding",
        "expanded_taker_funding",
    ):
        names = [*FEATURE_COLUMNS, *PRICE_CONTEXT_COLUMNS]
        if candidate.features == "expanded_taker":
            names.extend(TAKER_FEATURE_COLUMNS)
        elif candidate.features == "expanded_taker_ratio":
            names.append("taker_buy_ratio")
        elif candidate.features == "expanded_taker_multiscale":
            names.extend(["taker_buy_ratio", "taker_buy_ratio_6", "taker_buy_ratio_24"])
        elif candidate.features == "expanded_taker_funding":
            names.extend(TAKER_FEATURE_COLUMNS)
            names.extend(FUNDING_FEATURE_COLUMNS)
        elif candidate.features == "expanded_funding":
            names.extend(FUNDING_FEATURE_COLUMNS)
    elif candidate.features in ("legacy", "existing"):
        names = FEATURE_COLUMNS
    else:
        raise ValueError(f"Conjunto de features desconocido: {candidate.features}")
    if candidate.features == "legacy":
        tuning_pre = fit_preprocessor(data.iloc[base], tuning_start.to_pydatetime(), step)
    else:
        tuning_pre = _fit_preprocessor(data.iloc[base], names)
    tuning_x = tuning_pre.transform(data)
    if candidate.family == "xgboost":
        grid = (
            xgb_grid()
            if candidate.features == "legacy"
            else [
                {"max_depth": depth, "learning_rate": rate, "n_estimators": 150, "subsample": 0.8}
                for depth in (2, 3)
                for rate in (0.03, 0.08)
            ]
        )
        parameters, scores = tune_xgb(
            tuning_x[base], y[base], tuning_x[tuning], y[tuning], seed, grid
        )
    elif candidate.family == "lightgbm":
        parameters, scores = tune_lightgbm(
            tuning_x[base], y[base], tuning_x[tuning], y[tuning], seed
        )
    elif candidate.family == "logistic":
        scores = []
        for c in (0.01, 0.1, 1.0):
            trial = _fit_logistic(tuning_x[base], y[base], c, seed)
            probability = trial.predict_proba(tuning_x[tuning])[:, 1]
            scores.append(
                {
                    "parameters": {"C": c},
                    "log_loss": float(log_loss(y[tuning], probability, labels=[0, 1])),
                }
            )
        parameters = min(scores, key=lambda item: float(item["log_loss"]))["parameters"]
    else:
        raise ValueError(f"Familia desconocida: {candidate.family}")

    if candidate.features == "legacy":
        preprocessor = fit_preprocessor(data.iloc[pre_cal], calibration_start.to_pydatetime(), step)
    else:
        preprocessor = _fit_preprocessor(data.iloc[pre_cal], names)
    scaled = preprocessor.transform(data)
    if candidate.family == "xgboost":
        model: XGBClassifier | LogisticRegression | LGBMClassifier = fit_xgb(
            scaled[pre_cal], y[pre_cal], parameters, seed
        )
    elif candidate.family == "lightgbm":
        model = fit_lightgbm(scaled[pre_cal], y[pre_cal], parameters, seed)
    else:
        model = _fit_logistic(scaled[pre_cal], y[pre_cal], float(parameters["C"]), seed)
    calibration_probabilities = np.asarray(model.predict_proba(scaled[calibration]), dtype=float)[
        :, 1
    ]
    platt = fit_platt(calibration_probabilities, y[calibration])
    return FittedCandidate(
        candidate=candidate,
        model=model,
        preprocessor=preprocessor,
        parameters=parameters,
        platt=platt,
        raw_validation=np.asarray(model.predict_proba(scaled[fold.validation]), dtype=float)[:, 1],
        scaled_validation=scaled[fold.validation],
        tuning_scores=scores,
        fitting_samples=len(pre_cal),
        calibration_samples=len(calibration),
    )
