"""Métricas comparables para modelos y baselines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

if TYPE_CHECKING:
    from src.training.types import FloatArray, IntArray


def evaluate(y: IntArray, predicted: IntArray, probabilities: FloatArray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if not len(y) or len(y) != len(predicted) or len(y) != len(p):
        raise ValueError("Las predicciones y etiquetas deben tener la misma longitud no nula")
    if (
        not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
        or not np.isin(y, [0, 1]).all()
        or not np.isin(predicted, [0, 1]).all()
    ):
        raise ValueError("Etiquetas o probabilidades inválidas")
    p = np.clip(p, 1e-7, 1 - 1e-7)
    precision, recall, _, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1], zero_division=0
    )
    bins: list[dict[str, Any]] = []
    membership = np.minimum((p * 10).astype(int), 9)
    for index in range(10):
        mask = membership == index
        if mask.any():
            bins.append(
                {
                    "lower": index / 10,
                    "count": int(mask.sum()),
                    "mean_probability": float(p[mask].mean()),
                    "fraction_positive": float(y[mask].mean()),
                }
            )
    ece = sum(
        item["count"] * abs(item["mean_probability"] - item["fraction_positive"]) for item in bins
    ) / len(y)
    return {
        "samples": len(y),
        "accuracy": float(accuracy_score(y, predicted)),
        "f1_macro": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "precision_down": float(precision[0]),
        "precision_up": float(precision[1]),
        "recall_down": float(recall[0]),
        "recall_up": float(recall[1]),
        "support_down": int(support[0]),
        "support_up": int(support[1]),
        "auc_roc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece),
        "calibration_bins": bins,
    }


def evaluate_baselines(
    train_y: IntArray,
    train_previous: IntArray,
    validation_y: IntArray,
    validation_previous: IntArray,
    seed: int,
) -> dict[str, dict[str, Any]]:
    y = np.asarray(train_y, dtype=int)
    previous = np.asarray(train_previous, dtype=int)
    if not len(y) or len(y) != len(previous) or not np.isin(y, [0, 1]).all():
        raise ValueError("El entrenamiento del baseline requiere etiquetas binarias")
    if not np.isin(previous, [0, 1]).all() or not np.isin(validation_previous, [0, 1]).all():
        raise ValueError("La dirección anterior debe ser binaria")
    p_up = float(y.mean())
    majority = int(p_up >= 0.5)
    n = len(validation_y)
    rng = np.random.default_rng(seed)
    random_predictions = rng.binomial(1, p_up, size=n)
    conditional = {
        direction: float((y[previous == direction].sum() + 1) / ((previous == direction).sum() + 2))
        for direction in (0, 1)
    }
    previous_prob = np.array([conditional[int(value)] for value in validation_previous])
    return {
        "majority": evaluate(validation_y, np.full(n, majority), np.full(n, p_up)),
        "previous_direction": evaluate(validation_y, validation_previous, previous_prob),
        "conditional_direction": evaluate(
            validation_y, (previous_prob >= 0.5).astype(int), previous_prob
        ),
        "weighted_random": evaluate(validation_y, random_predictions, np.full(n, p_up)),
    }
