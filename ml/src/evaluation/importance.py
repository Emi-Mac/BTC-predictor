"""Importancia por SHAP y gradientes, sin emplearla para ajustar el modelo."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shap
import torch

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    from src.training.models import DirectionLSTM
    from src.training.types import FloatArray


def _normalize(values: FloatArray, names: list[str]) -> dict[str, float]:
    scores = np.asarray(values, dtype=float)
    if scores.shape != (len(names),) or not np.isfinite(scores).all():
        raise ValueError("La importancia de features es inválida")
    total = float(scores.sum())
    return {
        name: float(score / total) if total else 0.0
        for name, score in zip(names, scores, strict=True)
    }


def xgb_importance(
    model: XGBClassifier | LGBMClassifier, x: FloatArray, names: list[str]
) -> dict[str, float]:
    values = np.asarray(shap.TreeExplainer(model).shap_values(x[:128]), dtype=float)
    return _normalize(np.abs(values).mean(axis=0), names)


def lstm_importance(model: DirectionLSTM, x: FloatArray, names: list[str]) -> dict[str, float]:
    model.eval()
    inputs = torch.tensor(x[:128], dtype=torch.float32, requires_grad=True)
    model.zero_grad()
    model(inputs).sum().backward()
    if inputs.grad is None:
        raise RuntimeError("No se pudieron calcular gradientes de entrada")
    scores = inputs.grad.detach().abs().mean(dim=(0, 1)).cpu().numpy()
    return _normalize(scores, names)
