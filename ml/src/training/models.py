"""Modelos XGBoost y LSTM con semillas y recursos controlados."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from torch import nn
from xgboost import XGBClassifier

if TYPE_CHECKING:
    from src.training.types import FloatArray, IntArray

SEQUENCE_LENGTH = 20


def class_weight(y: IntArray) -> float:
    fraction = float(np.mean(y))
    if not 0 < fraction < 1:
        raise ValueError("El entrenamiento requiere ambas clases")
    return (1 - fraction) / fraction if max(fraction, 1 - fraction) > 0.6 else 1.0


def xgb_grid() -> list[dict[str, Any]]:
    return [
        {
            "max_depth": depth,
            "learning_rate": rate,
            "n_estimators": count,
            "subsample": subsample,
        }
        for depth, rate, count, subsample in product((2, 3), (0.03, 0.08), (100, 200), (0.7, 1.0))
    ]


def fit_xgb(x: FloatArray, y: IntArray, parameters: dict[str, Any], seed: int) -> XGBClassifier:
    model = XGBClassifier(
        **parameters,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=2,
        random_state=seed,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        scale_pos_weight=class_weight(y),
    )
    model.fit(x, y)
    return model


def tune_xgb(
    x_train: FloatArray,
    y_train: IntArray,
    x_tuning: FloatArray,
    y_tuning: IntArray,
    seed: int,
    grid: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = grid if grid is not None else xgb_grid()
    if not candidates or not len(x_tuning):
        raise ValueError("La búsqueda requiere candidatos y datos de ajuste")
    scores: list[dict[str, Any]] = []
    for parameters in candidates:
        model = fit_xgb(x_train, y_train, parameters, seed)
        p = np.asarray(model.predict_proba(x_tuning), dtype=float)[:, 1]
        scores.append(
            {"parameters": parameters, "log_loss": float(log_loss(y_tuning, p, labels=[0, 1]))}
        )
    best = min(scores, key=lambda result: float(result["log_loss"]))
    return best["parameters"], scores


def fit_lightgbm(
    x: FloatArray, y: IntArray, parameters: dict[str, Any], seed: int
) -> LGBMClassifier:
    model = LGBMClassifier(
        **parameters,
        objective="binary",
        n_jobs=2,
        random_state=seed,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        scale_pos_weight=class_weight(y),
        subsample_freq=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def tune_lightgbm(
    x_train: FloatArray,
    y_train: IntArray,
    x_tuning: FloatArray,
    y_tuning: IntArray,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores: list[dict[str, Any]] = []
    for depth, rate in product((2, 3), (0.03, 0.08)):
        parameters = {
            "max_depth": depth,
            "num_leaves": 2**depth,
            "learning_rate": rate,
            "n_estimators": 150,
            "subsample": 0.8,
        }
        model = fit_lightgbm(x_train, y_train, parameters, seed)
        p = np.asarray(model.predict_proba(x_tuning), dtype=float)[:, 1]
        scores.append(
            {"parameters": parameters, "log_loss": float(log_loss(y_tuning, p, labels=[0, 1]))}
        )
    best = min(scores, key=lambda result: float(result["log_loss"]))
    return best["parameters"], scores


class DirectionLSTM(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_count, hidden_size=64, num_layers=2, dropout=0.3, batch_first=True
        )
        self.output = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        result: torch.Tensor = self.output(sequence[:, -1, :]).squeeze(-1)
        return result


def sequences(x: FloatArray, positions: IntArray) -> tuple[FloatArray, IntArray]:
    positions = np.asarray(positions, dtype=int)
    valid = positions[positions >= SEQUENCE_LENGTH - 1]
    if not len(valid):
        raise ValueError("No hay contexto suficiente para secuencias de 20 velas")
    windows = np.stack([x[pos - SEQUENCE_LENGTH + 1 : pos + 1] for pos in valid])
    return windows.astype(np.float32), valid


def fit_lstm(x: FloatArray, y: IntArray, seed: int, epochs: int = 6) -> DirectionLSTM:
    if epochs < 1 or len(x) != len(y):
        raise ValueError("Datos o número de épocas inválidos")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = DirectionLSTM(x.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weight(y), dtype=torch.float32))
    generator = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for batch in np.array_split(generator.permutation(len(y)), max(1, (len(y) + 255) // 256)):
            inputs = torch.from_numpy(x[batch])
            targets = torch.tensor(y[batch], dtype=torch.float32)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_lstm(model: DirectionLSTM, x: FloatArray) -> FloatArray:
    model.eval()
    with torch.no_grad():
        predictions: FloatArray = torch.sigmoid(model(torch.from_numpy(x))).cpu().numpy()
    return predictions
