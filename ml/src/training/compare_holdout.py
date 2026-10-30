"""Compara dos artefactos congelados sin volver a ajustar ni seleccionar modelos."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate
from src.export.onnx_model import OnnxPredictor
from src.training.snapshot import snapshot_hash

if TYPE_CHECKING:
    from pathlib import Path

    from src.training.types import FloatArray, IntArray


def _probabilities(predictor: OnnxPredictor, data: pd.DataFrame, positions: IntArray) -> FloatArray:
    values = data[predictor.preprocessor.feature_names].to_numpy(dtype=float)
    if predictor.family == "lstm":
        if (positions < 19).any():
            raise ValueError("No hay contexto suficiente para el modelo anterior")
        raw = np.stack([values[index - 19 : index + 1] for index in positions])
    else:
        raw = values[positions]
    return predictor.predict_proba(raw)


def _wilson(successes: int, n: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n))
    half /= denominator
    return [center - half, center + half]


def compare_fixed_models(
    data: pd.DataFrame, directory: Path, reference_directory: Path
) -> dict[str, Any]:
    development = json.loads((directory / "development.json").read_text(encoding="utf-8"))
    holdout = json.loads((directory / "holdout.json").read_text(encoding="utf-8"))
    reference_report = json.loads(
        (reference_directory / "walk_forward.json").read_text(encoding="utf-8")
    )
    if (
        snapshot_hash(
            data,
            include_taker=development.get("include_taker", False),
            include_funding=development.get("include_funding", False),
        )
        != development["snapshot_sha256"]
    ):
        raise ValueError("El dataset cambió desde la selección")
    start = pd.Timestamp(holdout["holdout_start"])
    end = pd.Timestamp(holdout["holdout_end_exclusive"])
    if pd.Timestamp(reference_report["folds"][-1]["validation_end_exclusive"]) > start:
        raise ValueError("El modelo de referencia no precede al holdout")
    positions = np.flatnonzero((data["open_time"] >= start) & (data["open_time"] < end))
    if not len(positions) or len(positions) != holdout["metrics"]["samples"]:
        raise ValueError("El número de muestras del holdout cambió")
    labels = data["target_label"].to_numpy(dtype=int)[positions]
    current = OnnxPredictor(directory / "model")
    reference = OnnxPredictor(reference_directory)
    current_p = _probabilities(current, data, positions)
    reference_p = _probabilities(reference, data, positions)
    current_prediction = (current_p >= 0.5).astype(int)
    reference_prediction = (reference_p >= 0.5).astype(int)
    current_metrics = evaluate(labels, current_prediction, current_p)
    reference_metrics = evaluate(labels, reference_prediction, reference_p)
    if abs(current_metrics["accuracy"] - holdout["metrics"]["accuracy"]) > 1e-12:
        raise ValueError("El ONNX no reproduce la accuracy registrada")
    if abs(current_metrics["log_loss"] - holdout["metrics"]["log_loss"]) > 1e-5:
        raise ValueError("El ONNX no reproduce el log-loss registrado")
    current_correct = current_prediction == labels
    reference_correct = reference_prediction == labels
    return {
        "holdout_start": start.isoformat(),
        "holdout_end_exclusive": end.isoformat(),
        "samples": len(labels),
        "selected_model": current_metrics,
        "previous_model": reference_metrics,
        "selected_accuracy_wilson_95": _wilson(int(current_correct.sum()), len(labels)),
        "previous_accuracy_wilson_95": _wilson(int(reference_correct.sum()), len(labels)),
        "accuracy_difference": current_metrics["accuracy"] - reference_metrics["accuracy"],
        "selected_only_correct": int((current_correct & ~reference_correct).sum()),
        "previous_only_correct": int((reference_correct & ~current_correct).sum()),
        "always_up_accuracy": float(np.mean(labels == 1)),
        "always_down_accuracy": float(np.mean(labels == 0)),
        "note": (
            "Comparación posterior entre modelos ya congelados; los intervalos Wilson "
            "suponen observaciones independientes, lo que puede no cumplirse en series temporales."
        ),
    }
