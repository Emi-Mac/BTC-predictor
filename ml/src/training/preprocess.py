"""Imputación, selección y escalado ajustados exclusivamente al entrenamiento."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.features.compute import FEATURE_COLUMNS
from src.features.selection import select_features

if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from src.training.types import FloatArray


@dataclass(frozen=True)
class Preprocessor:
    feature_names: list[str]
    median: list[float]
    mean: list[float]
    scale: list[float]

    def transform(self, frame: pd.DataFrame) -> FloatArray:
        values = frame[self.feature_names].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("Las features contienen infinitos")
        imputed = np.where(np.isnan(values), np.asarray(self.median), values)
        scaled: FloatArray = ((imputed - np.asarray(self.mean)) / np.asarray(self.scale)).astype(
            np.float32
        )
        return scaled


def fit_preprocessor(training: pd.DataFrame, train_end: datetime, step: timedelta) -> Preprocessor:
    """Calcula estadísticas y selección con datos conocidos antes del corte indicado."""
    known = pd.to_datetime(training["open_time"], utc=True) + 2 * pd.Timedelta(step)
    training = training.loc[(known <= pd.Timestamp(train_end)) & training["target_label"].notna()]
    values = training[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not len(values) or np.isinf(values).any() or np.isnan(values).all(axis=0).any():
        raise ValueError("Train vacío, con infinitos o alguna feature totalmente ausente")
    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(
        imputer.fit_transform(values), columns=FEATURE_COLUMNS, index=training.index
    )
    selected_input = imputed.copy()
    selected_input["open_time"] = training["open_time"]
    selected_input["target_label"] = training["target_label"]
    selection = select_features(selected_input, train_end, step)
    names = selection.selected
    positions = [FEATURE_COLUMNS.index(name) for name in names]
    scaler = StandardScaler().fit(imputed[names].to_numpy())
    return Preprocessor(
        feature_names=names,
        median=[float(imputer.statistics_[index]) for index in positions],
        mean=[float(value) for value in scaler.mean_],
        scale=[float(value) for value in scaler.scale_],
    )
