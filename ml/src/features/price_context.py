"""Variables de precio y volumen calculadas solo con velas ya cerradas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.ingestion.cleaning import invalid_ohlcv_mask

if TYPE_CHECKING:
    from datetime import timedelta

PRICE_CONTEXT_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "return_1_lag1",
    "return_1_lag2",
    "return_1_lag3",
    "return_1_lag6",
    "candle_body",
    "close_position",
    "upper_wick",
    "lower_wick",
    "volume_change",
    "volatility_ratio_6_24",
    "distance_vwap",
    "distance_ema9",
    "distance_ema21",
    "relative_macd",
    "relative_atr",
    "bollinger_position",
]


def add_price_context(
    features: pd.DataFrame, candles: pd.DataFrame, step: timedelta
) -> pd.DataFrame:
    """Añade señales estacionarias sin incorporar ninguna vela futura."""
    candles = candles.sort_values("open_time").reset_index(drop=True).copy()
    if features.empty or candles.empty or candles["open_time"].duplicated().any():
        raise ValueError("Se requieren velas únicas para calcular el contexto")
    step_ts = pd.Timedelta(step)
    if step_ts <= pd.Timedelta(0) or (candles["open_time"].diff().dropna() != step_ts).any():
        raise ValueError("Las velas deben ser consecutivas y equidistantes")
    values = candles[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or invalid_ohlcv_mask(candles).any():
        raise ValueError("Las velas contienen precios o volúmenes inválidos")
    close, high, low, opening = (
        candles["close"],
        candles["high"],
        candles["low"],
        candles["open"],
    )
    one_period = close.pct_change(fill_method=None)
    context = pd.DataFrame({"open_time": candles["open_time"]})
    for period in (1, 3, 6, 12, 24):
        context[f"return_{period}"] = close.pct_change(periods=period, fill_method=None)
    for lag in (1, 2, 3, 6):
        context[f"return_1_lag{lag}"] = one_period.shift(lag)
    context["candle_body"] = (close - opening) / opening
    width = high - low
    context["close_position"] = ((close - low) / width.replace(0, np.nan)).mask(width.eq(0), 0.5)
    context["upper_wick"] = (high - np.maximum(opening, close)) / close
    context["lower_wick"] = (np.minimum(opening, close) - low) / close
    context["volume_change"] = np.log1p(candles["volume"]).diff()
    short_vol = one_period.rolling(6).std(ddof=0)
    long_vol = one_period.rolling(24).std(ddof=0)
    context["volatility_ratio_6_24"] = short_vol / long_vol.replace(0, np.nan)

    merged = features.merge(
        context, on="open_time", how="left", validate="one_to_one", indicator=True
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Faltan velas para algunas features")
    merged = merged.drop(columns="_merge")
    prices = candles[["open_time", "close"]]
    merged = merged.merge(prices, on="open_time", how="left", validate="one_to_one")
    merged["distance_vwap"] = merged["close"] / merged["vwap_daily"] - 1
    merged["distance_ema9"] = merged["close"] / merged["ema_9"] - 1
    merged["distance_ema21"] = merged["close"] / merged["ema_21"] - 1
    merged["relative_macd"] = merged["macd"] / merged["close"]
    merged["relative_atr"] = merged["atr_14"] / merged["close"]
    band_width = merged["bb_upper"] - merged["bb_lower"]
    merged["bollinger_position"] = (merged["close"] - merged["bb_middle"]) / band_width.replace(
        0, np.nan
    )
    return merged.drop(columns="close").replace([np.inf, -np.inf], np.nan)
