"""Carga de datos y ventanas temporales sin usar etiquetas aún desconocidas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.features.compute import FEATURE_COLUMNS
from src.features.repository import FeatureRepository
from src.ingestion.repository import IngestionRepository

if TYPE_CHECKING:
    from datetime import timedelta

    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.config import Settings
    from src.training.types import IntArray


@dataclass(frozen=True)
class Fold:
    number: int
    train: IntArray
    validation: IntArray
    start: pd.Timestamp
    end: pd.Timestamp


async def load_training_data(engine: AsyncEngine, settings: Settings) -> pd.DataFrame:
    features = await FeatureRepository(engine).load_for_selection(
        settings.btc_symbol, settings.default_interval
    )
    candles = await IngestionRepository(engine).load_candles(
        settings.btc_symbol, settings.default_interval
    )
    if features.empty or candles.empty:
        raise ValueError("No hay features o velas para entrenar")
    candles = candles.sort_values("open_time").reset_index(drop=True)
    previous_close = candles["close"].shift(1)
    next_close = candles["close"].shift(-1)
    candles["previous_direction"] = (
        (candles["close"] > previous_close).astype("Int64").mask(previous_close.isna())
    )
    candles["expected_target"] = (
        (next_close > candles["close"]).astype("Int64").mask(next_close.isna())
    )
    data = (
        features.merge(
            candles[["open_time", "previous_direction", "expected_target"]],
            on="open_time",
            validate="one_to_one",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    known = data["target_label"].notna()
    if not data.loc[known, "target_label"].eq(data.loc[known, "expected_target"]).all():
        raise ValueError("Las etiquetas almacenadas no coinciden con el siguiente cierre")
    if (known & data["expected_target"].isna()).any():
        raise ValueError("Hay etiquetas para velas cuyo siguiente cierre aún no existe")
    data = data.loc[known].reset_index(drop=True)
    if data.empty or data["previous_direction"].isna().any():
        raise ValueError("No hay etiquetas o dirección previa suficiente")
    if not data["target_label"].isin([0, 1]).all():
        raise ValueError("El target debe ser binario")
    values = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Las features contienen infinitos")
    return data.drop(columns=["expected_target"])


def build_folds(
    data: pd.DataFrame,
    step: timedelta,
    n_folds: int = 10,
    min_train_days: int = 365,
) -> list[Fold]:
    """Usa los últimos meses completos; el último fold se reserva como test final."""
    if n_folds < 1 or data.empty:
        raise ValueError("Se requiere al menos un fold y datos no vacíos")
    times = pd.to_datetime(data["open_time"], utc=True)
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Las observaciones deben estar ordenadas y sin duplicados")
    step_ts = pd.Timedelta(step)
    if step_ts <= pd.Timedelta(0) or (times.diff().dropna() != step_ts).any():
        raise ValueError("Las features deben tener continuidad temporal")
    last_end = times.iloc[-1].floor("D").replace(day=1)
    folds: list[Fold] = []
    for number in range(1, n_folds + 1):
        end = last_end - pd.DateOffset(months=n_folds - number)
        start = end - pd.DateOffset(months=1)
        train = np.flatnonzero((times < start) & (times + 2 * step_ts <= start))
        validation = np.flatnonzero((times >= start) & (times < end))
        expected = len(pd.date_range(start, end - step_ts, freq=step_ts))
        if not len(train) or start - times.iloc[train[0]] < pd.Timedelta(days=min_train_days):
            raise ValueError(f"El fold {number} requiere al menos {min_train_days} días de train")
        if len(validation) != expected:
            raise ValueError(f"El fold {number} no tiene un mes completo de validación")
        folds.append(Fold(number, train, validation, start, end))
    return folds


def inner_windows(data: pd.DataFrame, fold: Fold, step: timedelta) -> tuple[IntArray, ...]:
    """Tune y calibración usan meses previos disjuntos, con embargo de etiqueta."""
    times = pd.to_datetime(data["open_time"], utc=True)
    known_at = times + 2 * pd.Timedelta(step)
    calibration_start = fold.start - pd.DateOffset(months=1)
    tuning_start = fold.start - pd.DateOffset(months=2)
    base = np.flatnonzero((times < tuning_start) & (known_at <= tuning_start))
    tuning = np.flatnonzero(
        (times >= tuning_start) & (times < calibration_start) & (known_at <= calibration_start)
    )
    pre_cal = np.flatnonzero((times < calibration_start) & (known_at <= calibration_start))
    calibration = np.flatnonzero(
        (times >= calibration_start) & (times < fold.start) & (known_at <= fold.start)
    )
    if min(map(len, (base, tuning, pre_cal, calibration))) == 0:
        raise ValueError(f"Fold {fold.number}: ventanas internas insuficientes")
    return base, tuning, pre_cal, calibration
