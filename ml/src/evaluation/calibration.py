"""Calibración Platt entrenada en un período cronológico separado."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from src.training.types import FloatArray, IntArray


def logit(probabilities: FloatArray) -> FloatArray:
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilidades de calibración fuera de [0, 1]")
    values = np.clip(values, 1e-7, 1 - 1e-7)
    result: FloatArray = np.log(values / (1 - values))
    return result


@dataclass(frozen=True)
class PlattCalibration:
    slope: float
    intercept: float

    def transform(self, probabilities: FloatArray) -> FloatArray:
        z = np.clip(self.slope * logit(probabilities) + self.intercept, -700, 700)
        return 1 / (1 + np.exp(-z))


def fit_platt(probabilities: FloatArray, labels: IntArray) -> PlattCalibration:
    if len(probabilities) != len(labels) or np.unique(labels).size != 2:
        raise ValueError("La calibración requiere ambas clases y longitudes iguales")
    fitted = LogisticRegression(random_state=42, max_iter=500).fit(
        logit(probabilities).reshape(-1, 1), labels
    )
    return PlattCalibration(float(fitted.coef_[0, 0]), float(fitted.intercept_[0]))
