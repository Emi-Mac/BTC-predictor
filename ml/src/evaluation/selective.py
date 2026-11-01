"""Métricas de habilidad y de predicción selectiva (abstención por baja confianza)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from src.evaluation.economic import (
    overlapping_positions,
    positions_from_probabilities,
    strategy_returns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.training.types import FloatArray, IntArray

CAUSAL_MARGINS = (0.0, 0.01, 0.02, 0.05)
DESCRIPTIVE_COVERAGES = (1.0, 0.5, 0.3, 0.2, 0.1)
EPSILON = 1e-7


def _validate(y: IntArray, p: FloatArray) -> tuple[IntArray, FloatArray]:
    labels = np.asarray(y, dtype=int)
    probabilities = np.asarray(p, dtype=float)
    if not len(labels) or labels.shape != probabilities.shape:
        raise ValueError("Etiquetas y probabilidades deben tener la misma longitud no nula")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Las etiquetas deben ser binarias")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilidades fuera de [0, 1]")
    return labels, probabilities


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Intervalo de Wilson; supone independencia, optimista en series autocorrelacionadas."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Conteos inválidos para el intervalo de Wilson")
    fraction = successes / total
    denominator = 1 + z**2 / total
    center = (fraction + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(fraction * (1 - fraction) / total + z**2 / (4 * total**2)) / denominator
    return float(center - half), float(center + half)


def _bucket(y: IntArray, p: FloatArray, mask: npt.NDArray[np.bool_]) -> dict[str, Any]:
    count = int(mask.sum())
    if not count:
        return {"samples": 0, "accuracy": None, "wilson_95": None, "predicted_up_fraction": None}
    correct = int(((p[mask] >= 0.5).astype(int) == y[mask]).sum())
    return {
        "samples": count,
        "accuracy": correct / count,
        "wilson_95": list(wilson_interval(correct, count)),
        "predicted_up_fraction": float((p[mask] >= 0.5).mean()),
    }


def coverage_curve(
    y: IntArray,
    p: FloatArray,
    margins: Sequence[float] = CAUSAL_MARGINS,
    coverages: Sequence[float] = DESCRIPTIVE_COVERAGES,
) -> dict[str, Any]:
    """Accuracy por cobertura.

    Los márgenes fijos son causales (el umbral no depende de los datos evaluados). Los cuantiles
    de cobertura se calculan sobre el propio conjunto evaluado y solo son descriptivos.
    """
    labels, probabilities = _validate(y, p)
    confidence = np.abs(probabilities - 0.5)
    causal = []
    for margin in margins:
        if not 0 <= margin < 0.5:
            raise ValueError("El margen debe estar en [0, 0.5)")
        mask = confidence >= margin if margin else np.ones(len(labels), dtype=bool)
        causal.append(
            {
                "margin": float(margin),
                "coverage": float(mask.mean()),
                **_bucket(labels, probabilities, mask),
            }
        )
    descriptive = []
    order = np.argsort(-confidence, kind="stable")
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("La cobertura debe estar en (0, 1]")
        mask = np.zeros(len(labels), dtype=bool)
        mask[order[: max(1, int(round(coverage * len(labels))))]] = True
        descriptive.append(
            {
                "coverage": float(coverage),
                "min_confidence": float(confidence[mask].min()),
                **_bucket(labels, probabilities, mask),
            }
        )
    return {"causal_margins": causal, "descriptive_top_confidence": descriptive}


def skill_scores(y: IntArray, p: FloatArray, reference: FloatArray) -> dict[str, float]:
    """Brier y log-loss relativos a una probabilidad de referencia (climatología del train).

    Valores > 0 indican mejora sobre la referencia; 0 equivale a no tener habilidad.
    """
    labels, probabilities = _validate(y, p)
    _, base = _validate(y, np.broadcast_to(np.asarray(reference, dtype=float), labels.shape))
    brier = float(np.mean((probabilities - labels) ** 2))
    brier_reference = float(np.mean((base - labels) ** 2))

    def logloss(values: FloatArray) -> float:
        clipped = np.clip(values, EPSILON, 1 - EPSILON)
        return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))

    loss, loss_reference = logloss(probabilities), logloss(base)
    return {
        "brier": brier,
        "brier_reference": brier_reference,
        "brier_skill_score": 1 - brier / brier_reference if brier_reference > 0 else 0.0,
        "log_loss": loss,
        "log_loss_reference": loss_reference,
        "log_loss_skill_score": 1 - loss / loss_reference if loss_reference > 0 else 0.0,
    }


def causal_margin_selection(
    fold_ids: IntArray,
    probabilities: FloatArray,
    returns: FloatArray,
    horizon: int,
    margins: Sequence[float],
    cost_bps: float,
    min_history: int,
    allow_short: bool = True,
    carry: FloatArray | None = None,
) -> tuple[FloatArray, dict[int, float | None]]:
    """Elige, al inicio de cada fold, el margen con mayor Sharpe neto en el historial OOS.

    Las filas deben ser cronológicas y contiguas; `returns[t]` es el retorno de t a t+1, por lo
    que el de la última fila previa al fold aún no se conoce y se excluye. Los folds sin
    historial suficiente quedan sin posición (margen NaN).
    """
    folds = np.asarray(fold_ids)
    p = np.asarray(probabilities, dtype=float)
    r = np.asarray(returns, dtype=float)
    if not (len(folds) == len(p) == len(r)) or not len(p) or not margins or min_history < 2:
        raise ValueError("Entradas inválidas para la selección de margen")
    signals = {m: positions_from_probabilities(p, m, allow_short) for m in margins}
    chosen = np.full(len(p), np.nan)
    choices: dict[int, float | None] = {}
    starts = np.flatnonzero(np.concatenate([[True], folds[1:] != folds[:-1]]))
    if len(np.unique(folds)) != len(starts):
        raise ValueError("Cada fold debe ocupar un bloque contiguo")
    for start in starts:
        fold = int(folds[start])
        history = start - 1
        if history < min_history:
            choices[fold] = None
            continue
        best: tuple[float, float] | None = None
        for margin in margins:
            position = overlapping_positions(signals[margin][:history], horizon)
            net = strategy_returns(
                position,
                r[:history],
                cost_bps,
                None if carry is None else np.asarray(carry, dtype=float)[:history],
            )
            std = float(net.std(ddof=1))
            sharpe = float(net.mean()) / std if std > 0 else 0.0
            if best is None or sharpe > best[0] + 1e-12:
                best = (sharpe, float(margin))
        if best is None:
            raise ValueError("No hay márgenes candidatos")
        choices[fold] = best[1]
        chosen[folds == fold] = best[1]
    return chosen, choices
