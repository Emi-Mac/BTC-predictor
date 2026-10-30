"""Señales causales de compra agresora calculadas a partir de velas cerradas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.ingestion.cleaning import invalid_ohlcv_mask, invalid_taker_volume_mask

if TYPE_CHECKING:
    from datetime import timedelta

TAKER_FEATURE_COLUMNS = [
    "taker_buy_ratio",
    "taker_buy_quote_ratio",
    "taker_buy_ratio_6",
    "taker_buy_ratio_24",
]


def add_taker_flow(features: pd.DataFrame, candles: pd.DataFrame, step: timedelta) -> pd.DataFrame:
    """Conserva nulos cuando el volumen agresor no se conoce o el denominador es cero."""
    candles = candles.sort_values("open_time").reset_index(drop=True).copy()
    required = {"taker_buy_base_volume", "taker_buy_quote_volume", "quote_volume"}
    if features.empty or candles.empty or candles["open_time"].duplicated().any():
        raise ValueError("Se requieren velas únicas para calcular el flujo agresor")
    if not required.issubset(candles.columns):
        raise ValueError("Faltan columnas de volumen agresor en las velas")
    step_ts = pd.Timedelta(step)
    if step_ts <= pd.Timedelta(0) or (candles["open_time"].diff().dropna() != step_ts).any():
        raise ValueError("Las velas deben ser consecutivas y equidistantes")
    values = candles[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or invalid_ohlcv_mask(candles).any()
        or invalid_taker_volume_mask(candles).any()
    ):
        raise ValueError("Las velas contienen volúmenes agresores inválidos")
    buy = pd.to_numeric(candles["taker_buy_base_volume"], errors="raise")
    buy_quote = pd.to_numeric(candles["taker_buy_quote_volume"], errors="raise")
    volume = pd.to_numeric(candles["volume"], errors="raise")
    quote_volume = pd.to_numeric(candles["quote_volume"], errors="raise")
    context = pd.DataFrame({"open_time": candles["open_time"]})
    context["taker_buy_ratio"] = buy / volume.replace(0, np.nan)
    context["taker_buy_quote_ratio"] = buy_quote / quote_volume.replace(0, np.nan)
    for window in (6, 24):
        total = volume.rolling(window, min_periods=window).sum().replace(0, np.nan)
        context[f"taker_buy_ratio_{window}"] = buy.rolling(window, min_periods=window).sum() / total
    context["taker_buy_imbalance"] = 2 * context["taker_buy_ratio"] - 1
    out = features.merge(context, on="open_time", how="left", validate="one_to_one", indicator=True)
    if not out["_merge"].eq("both").all():
        raise ValueError("Faltan velas para algunas features")
    return out.drop(columns="_merge")
