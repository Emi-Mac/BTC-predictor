"""Alineación causal de funding con las velas de BTC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import timedelta

FUNDING_FEATURE_COLUMNS = ["funding_rate", "funding_mean_3", "funding_change_1"]


def add_funding_context(
    features: pd.DataFrame,
    funding: pd.DataFrame,
    step: timedelta,
    max_age: timedelta,
) -> pd.DataFrame:
    """Usa cada evento solo tras una vela completa y limita su propagación hacia adelante."""
    if features.empty or funding.empty:
        raise ValueError("Se requieren features y eventos de funding")
    step_ts = pd.Timedelta(step)
    if step_ts <= pd.Timedelta(0) or pd.Timedelta(max_age) <= pd.Timedelta(0):
        raise ValueError("El intervalo y la vigencia del funding deben ser positivos")
    timestamps = pd.DatetimeIndex(funding["funding_time"])
    if timestamps.tz is None:
        raise ValueError("funding_time debe incluir zona horaria")
    events = funding.copy()
    events["funding_time"] = pd.to_datetime(events["funding_time"], utc=True).astype(
        pd.DatetimeTZDtype(unit="ns", tz="UTC")
    )
    events = events.sort_values("funding_time").reset_index(drop=True)
    if events["funding_time"].duplicated().any():
        raise ValueError("Hay eventos de funding duplicados")
    rate = pd.to_numeric(events["funding_rate"], errors="raise")
    if not np.isfinite(rate).all():
        raise ValueError("El funding contiene valores no finitos")
    events["funding_rate"] = rate
    gaps = events["funding_time"].diff()
    events["funding_mean_3"] = (
        rate.rolling(3, min_periods=3)
        .mean()
        .where(gaps.dt.total_seconds().rolling(2, min_periods=2).max() <= 12 * 3600)
    )
    events["funding_change_1"] = rate.diff().where(gaps <= pd.Timedelta(hours=12))
    events["funding_available_at"] = events["funding_time"] + step_ts

    left = features.copy()
    if left["open_time"].duplicated().any():
        raise ValueError("Hay velas duplicadas al alinear funding")
    left["_original_order"] = np.arange(len(left))
    left["_decision_time"] = (pd.to_datetime(left["open_time"], utc=True) + step_ts).astype(
        pd.DatetimeTZDtype(unit="ns", tz="UTC")
    )
    right = events[["funding_time", "funding_available_at", *FUNDING_FEATURE_COLUMNS]]
    out = pd.merge_asof(
        left.sort_values("_decision_time"),
        right.sort_values("funding_available_at"),
        left_on="_decision_time",
        right_on="funding_available_at",
        direction="backward",
        tolerance=pd.Timedelta(max_age),
    )
    return (
        out.sort_values("_original_order")
        .drop(columns=["_original_order", "_decision_time"])
        .reset_index(drop=True)
    )
