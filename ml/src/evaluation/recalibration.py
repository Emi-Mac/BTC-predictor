"""Recalibración ajustada solo con predicciones fuera de muestra de folds anteriores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.evaluation.calibration import fit_platt

if TYPE_CHECKING:
    from datetime import timedelta

    from src.training.types import FloatArray, IntArray

Method = Literal["platt", "isotonic"]
METHODS: tuple[Method, ...] = ("platt", "isotonic")
# Evita probabilidades extremas en bins isotónicos con pocas observaciones.
ISOTONIC_BOUNDS = (0.02, 0.98)


def _fit_transform(
    method: Method, train_p: FloatArray, train_y: IntArray, target_p: FloatArray
) -> FloatArray:
    if method == "platt":
        return fit_platt(train_p, train_y).transform(target_p)
    if method == "isotonic":
        model = IsotonicRegression(
            y_min=ISOTONIC_BOUNDS[0], y_max=ISOTONIC_BOUNDS[1], out_of_bounds="clip"
        ).fit(train_p, train_y)
        result: FloatArray = np.asarray(model.predict(target_p), dtype=float)
        return result
    raise ValueError(f"Método de calibración desconocido: {method}")


def expanding_oos_calibration(
    open_times: pd.Series,
    fold_ids: IntArray,
    fold_starts: dict[int, pd.Timestamp],
    raw_probabilities: FloatArray,
    labels: IntArray,
    step: timedelta,
    method: Method,
    horizon: int = 1,
    min_samples: int = 1000,
    window: int | None = None,
) -> FloatArray:
    """Calibración online: cada fold usa predicciones OOS previas cuya etiqueta ya se conocía.

    `window` limita el historial a las últimas observaciones (sensibilidad a la recencia, ya que
    los modelos base se reajustan cada mes). Devuelve NaN en los folds sin historial suficiente.
    """
    times = pd.to_datetime(pd.Series(open_times), utc=True).reset_index(drop=True)
    folds = np.asarray(fold_ids)
    p = np.asarray(raw_probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if not (len(times) == len(folds) == len(p) == len(y)) or not len(p):
        raise ValueError("Las entradas de calibración deben alinear y no estar vacías")
    if horizon < 1 or min_samples < 2:
        raise ValueError("Horizonte o tamaño mínimo inválidos")
    known_at = times + (horizon + 1) * pd.Timedelta(step)
    result = np.full(len(p), np.nan)
    for fold in np.unique(folds):
        start = pd.Timestamp(fold_starts[int(fold)])
        if start.tzinfo is None:
            raise ValueError("El inicio de cada fold debe incluir zona horaria")
        history = np.array((known_at <= start) & (times < start), dtype=bool)
        if window is not None:
            if window < min_samples:
                raise ValueError("La ventana debe ser al menos el tamaño mínimo")
            history[np.flatnonzero(history)[:-window]] = False
        target = folds == fold
        if history.sum() < min_samples or np.unique(y[history]).size < 2:
            continue
        result[target] = _fit_transform(method, p[history], y[history], p[target])
    return result
