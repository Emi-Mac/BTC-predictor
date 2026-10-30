"""Selección ajustada exclusivamente en la ventana temporal de entrenamiento."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.features.compute import FEATURE_COLUMNS

if TYPE_CHECKING:
    from datetime import datetime, timedelta


@dataclass(frozen=True)
class SelectionResult:
    selected: list[str]
    correlations: dict[str, float]
    mutual_information: dict[str, float]
    removed_collinear: dict[str, str]
    train_rows: int


def select_features(
    frame: pd.DataFrame,
    train_end: datetime,
    step: timedelta,
    min_correlation: float = 0.02,
    max_pair_correlation: float = 0.95,
) -> SelectionResult:
    """Incluye solo etiquetas conocidas a train_end y elimina colinealidad por MI descendente."""
    if train_end.tzinfo is None or train_end.utcoffset() is None:
        raise ValueError("train_end debe incluir zona horaria")
    known_at = pd.to_datetime(frame["open_time"], utc=True) + 2 * pd.Timedelta(step)
    train = frame.loc[(known_at <= pd.Timestamp(train_end)) & frame["target_label"].notna()].copy()
    if len(train) < 20 or train["target_label"].nunique() != 2:
        raise ValueError("Se requieren al menos 20 filas y ambas clases en entrenamiento")
    x = train[FEATURE_COLUMNS].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("El entrenamiento contiene features incompletas")
    y = train["target_label"].astype(int)
    corr = x.corrwith(y).fillna(0.0)
    eligible = corr[corr.abs() >= min_correlation].index.tolist()
    if not eligible:
        raise ValueError("Ninguna feature supera el umbral de correlación")
    discrete = [name.startswith("weekday_") or name == "is_weekend" for name in eligible]
    mi = mutual_info_classif(x[eligible], y, discrete_features=discrete, random_state=42)
    scores = dict(zip(eligible, (float(value) for value in mi), strict=True))
    ordered = sorted(eligible, key=lambda name: (-scores[name], name))
    pair_corr = x[ordered].corr().abs().fillna(0)
    selected: list[str] = []
    removed: dict[str, str] = {}
    for name in ordered:
        duplicate = next(
            (kept for kept in selected if pair_corr.loc[name, kept] > max_pair_correlation),
            None,
        )
        if duplicate is None:
            selected.append(name)
        else:
            removed[name] = duplicate
    return SelectionResult(
        selected=selected,
        correlations={str(name): float(value) for name, value in corr.items()},
        mutual_information=scores,
        removed_collinear=removed,
        train_rows=len(train),
    )
