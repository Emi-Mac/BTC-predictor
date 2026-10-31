"""Features de futuros con tiempos de disponibilidad explícitos."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.research.metrics_archive import METRICS_COLUMNS

if TYPE_CHECKING:
    from datetime import timedelta

OI_FEATURE_COLUMNS = [
    "oi_log_change_4h",
    "oi_log_change_24h",
    "oi_value_log_change_4h",
    "oi_value_log_change_24h",
    "oi_price_interaction",
]
POSITION_FEATURE_COLUMNS = [
    "accounts_long_short_log",
    "top_accounts_long_short_log",
    "top_positions_long_short_log",
    "futures_taker_log_ratio",
]


def add_metrics_context(
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    step: timedelta,
    availability_delay: timedelta,
    max_staleness: timedelta,
) -> pd.DataFrame:
    """Usa eventos únicamente tras fin del intervalo y el retraso asumido."""
    if features.empty or metrics.empty:
        raise ValueError("Se requieren velas y métricas de futuros")
    if pd.Timedelta(step) <= pd.Timedelta(0):
        raise ValueError("El intervalo de velas debe ser positivo")
    if pd.Timedelta(availability_delay) < pd.Timedelta(0):
        raise ValueError("La demora de disponibilidad no puede ser negativa")
    if pd.Timedelta(max_staleness) <= pd.Timedelta(0):
        raise ValueError("La tolerancia de antigüedad debe ser positiva")
    if metrics["create_time"].duplicated().any() or features["open_time"].duplicated().any():
        raise ValueError("Hay timestamps duplicados en las métricas o velas")
    raw_times = pd.DatetimeIndex(metrics["create_time"])
    if raw_times.tz is None:
        raise ValueError("Las métricas deben incluir zona horaria")
    events = metrics.copy().sort_values("create_time")
    events["create_time"] = pd.to_datetime(events["create_time"], utc=True).astype(
        pd.DatetimeTZDtype(unit="ns", tz="UTC")
    )
    if (events["create_time"].astype("int64") % pd.Timedelta(minutes=5).value != 0).any():
        raise ValueError("Las métricas no están alineadas a intervalos de cinco minutos")
    expected = pd.date_range(
        events["create_time"].iloc[0], events["create_time"].iloc[-1], freq="5min", tz="UTC"
    )
    events = events.set_index("create_time").reindex(expected)
    for column in METRICS_COLUMNS:
        values = pd.to_numeric(events[column], errors="raise")
        if not np.isfinite(values.dropna()).all() or (values.dropna() < 0).any():
            raise ValueError(f"Métricas inválidas en {column}")
        events[column] = values

    oi = np.log(events["sum_open_interest"].where(events["sum_open_interest"] > 0))
    oi_value = np.log(
        events["sum_open_interest_value"].where(events["sum_open_interest_value"] > 0)
    )
    events["oi_log_change_4h"] = oi - oi.shift(48)
    events["oi_log_change_24h"] = oi - oi.shift(288)
    events["oi_value_log_change_4h"] = oi_value - oi_value.shift(48)
    events["oi_value_log_change_24h"] = oi_value - oi_value.shift(288)
    for target, source in (
        ("accounts_long_short_log", "count_long_short_ratio"),
        ("top_accounts_long_short_log", "count_toptrader_long_short_ratio"),
        ("top_positions_long_short_log", "sum_toptrader_long_short_ratio"),
        ("futures_taker_log_ratio", "sum_taker_long_short_vol_ratio"),
    ):
        events[target] = np.log(events[source].where(events[source] > 0))
    events["oi_source_time"] = pd.DatetimeIndex(events.index) + pd.Timedelta(minutes=5)
    events["oi_available_at"] = events["oi_source_time"] + pd.Timedelta(availability_delay)
    events = events.loc[events["sum_open_interest"].notna()].reset_index(drop=True)
    if events.empty:
        raise ValueError("No hay observaciones utilizables de interés abierto")

    left = features.copy()
    left["_original_order"] = np.arange(len(left))
    left["_decision_time"] = (
        pd.to_datetime(left["open_time"], utc=True) + pd.Timedelta(step)
    ).astype(pd.DatetimeTZDtype(unit="ns", tz="UTC"))
    right_names = [
        "oi_source_time",
        "oi_available_at",
        *[name for name in OI_FEATURE_COLUMNS if name != "oi_price_interaction"],
        *POSITION_FEATURE_COLUMNS,
    ]
    result = pd.merge_asof(
        left.sort_values("_decision_time"),
        events[right_names].sort_values("oi_available_at"),
        left_on="_decision_time",
        right_on="oi_available_at",
        direction="backward",
        tolerance=pd.Timedelta(max_staleness),
    )
    result["oi_price_interaction"] = result["oi_log_change_4h"] * result["return_1"]
    return (
        result.sort_values("_original_order")
        .drop(columns=["_original_order", "_decision_time"])
        .reset_index(drop=True)
    )
