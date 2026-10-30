"""Cálculo retrospectivo de features para velas cerradas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.features.labels import make_target_label
from src.ingestion.alignment import _asof, _daily_available_series, align_external_to_candles
from src.ingestion.cleaning import detect_gaps, invalid_ohlcv_mask

if TYPE_CHECKING:
    from datetime import timedelta

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width",
    "ema_9",
    "ema_21",
    "ema_50",
    "sma_200",
    "atr_14",
    "hist_volatility_20",
    "hl_range_pct",
    "obv",
    "volume_sma_ratio",
    "vwap_daily",
    "accumulation_distribution",
    "roc_5",
    "roc_10",
    "williams_r",
    "stoch_rsi",
    "price_ema50_distance",
    "ema21_slope_5",
    "higher_highs_20",
    "lower_lows_20",
    "dxy_close",
    "dxy_change_pct",
    "sp500_close",
    "sp500_change_pct",
    "vix_close",
    "gold_change_pct",
    "btc_sp500_corr_20",
    "fear_greed_value",
    "fear_greed_sma_7",
    "fear_greed_momentum_7",
    "hour_of_day",
    "is_weekend",
    *(f"weekday_{day}" for day in WEEKDAYS),
]
STORED_COLUMNS = [*FEATURE_COLUMNS, "day_of_week", "target_label"]
REQUIRED_MACRO = {"DX-Y.NYB", "^GSPC", "^VIX", "GC=F"}


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _external_features(
    candles: pd.DataFrame,
    step: timedelta,
    macro: pd.DataFrame,
    fear_greed: pd.DataFrame,
    max_ffill_days: int,
) -> pd.DataFrame:
    """Deriva estadísticas diarias antes de alinearlas al cierre de cada vela."""
    missing = REQUIRED_MACRO - set(macro["symbol"])
    if missing or fear_greed.empty:
        raise ValueError(f"Faltan series externas: macro={sorted(missing)}, FGI={fear_greed.empty}")

    aligned = align_external_to_candles(
        candles, step, macro=macro, fear_greed=fear_greed, max_ffill_days=max_ffill_days
    )
    decision = pd.DataFrame(
        {
            "open_time": candles["open_time"],
            "decision_time": (candles["open_time"] + pd.Timedelta(step)).astype(
                pd.DatetimeTZDtype(unit="ns", tz="UTC")
            ),
        }
    )
    horizon = decision["decision_time"].max().floor("D")

    def align_daily(dates: pd.Series, values: pd.Series, name: str, delay: int) -> None:
        available = pd.to_datetime(dates, utc=True) + pd.Timedelta(days=delay)
        valid = values.notna()
        if not valid.any():
            aligned[name] = np.nan
            return
        daily = _daily_available_series(
            values[valid].reset_index(drop=True),
            available[valid].reset_index(drop=True),
            horizon,
            max_ffill_days,
        )
        aligned[name] = aligned["open_time"].map(
            _asof(decision, daily, name, pd.Timedelta(hours=24))
        )

    ordered = macro.sort_values(["symbol", "trade_date"]).copy()
    for symbol, name in (
        ("DX-Y.NYB", "dxy_change_pct"),
        ("^GSPC", "sp500_change_pct"),
        ("GC=F", "gold_change_pct"),
    ):
        series = ordered.loc[ordered["symbol"] == symbol].reset_index(drop=True)
        align_daily(series["trade_date"], series["close"].pct_change(fill_method=None), name, 1)

    decision_time = candles["open_time"] + pd.Timedelta(step)
    daily_btc = candles.loc[decision_time.dt.hour.eq(0) & decision_time.dt.minute.eq(0)].copy()
    daily_btc["trade_date"] = (decision_time.loc[daily_btc.index] - pd.Timedelta(days=1)).dt.date
    sp = ordered.loc[ordered["symbol"] == "^GSPC", ["trade_date", "close"]]
    paired = daily_btc[["trade_date", "close"]].merge(
        sp, on="trade_date", suffixes=("_btc", "_sp"), validate="one_to_one"
    )
    btc_returns = paired["close_btc"].pct_change(fill_method=None)
    sp_returns = paired["close_sp"].pct_change(fill_method=None)
    corr = btc_returns.rolling(20, min_periods=20).corr(sp_returns)
    align_daily(paired["trade_date"], corr, "btc_sp500_corr_20", 1)

    fgi = fear_greed.sort_values("obs_date").set_index("obs_date")["value"]
    fgi.index = pd.to_datetime(fgi.index, utc=True)
    calendar = pd.date_range(fgi.index.min(), fgi.index.max(), freq="D", tz="UTC")
    fgi = fgi.reindex(calendar).ffill(limit=max_ffill_days)
    dates = pd.Series(calendar)
    fgi = fgi.reset_index(drop=True)
    align_daily(dates, fgi.rolling(7, min_periods=7).mean(), "fear_greed_sma_7", 0)
    align_daily(dates, fgi - fgi.shift(7), "fear_greed_momentum_7", 0)
    return aligned


def compute_features(
    candles: pd.DataFrame,
    macro: pd.DataFrame,
    fear_greed: pd.DataFrame,
    step: timedelta,
    max_ffill_days: int = 5,
) -> pd.DataFrame:
    """Una fila por vela; las ventanas incompletas permanecen como NaN."""
    if candles.empty:
        raise ValueError("No hay velas para calcular features")
    step_ts = pd.Timedelta(step)
    if step_ts <= pd.Timedelta(0) or pd.Timedelta(days=1) % step_ts != pd.Timedelta(0):
        raise ValueError("El intervalo debe dividir un día UTC")
    df = candles.sort_values("open_time").reset_index(drop=True).copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    numeric = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if (
        df["open_time"].isna().any()
        or df["open_time"].duplicated().any()
        or not np.isfinite(numeric).all()
        or invalid_ohlcv_mask(df).any()
    ):
        raise ValueError("Velas duplicadas o inválidas")
    if detect_gaps(df["open_time"], step) or (df["open_time"].diff().dropna() != step_ts).any():
        raise ValueError("Las velas deben ser consecutivas y equidistantes")
    if (df["open_time"].dt.as_unit("ns").astype("int64") % step_ts.value != 0).any():
        raise ValueError("Las velas no están alineadas al intervalo UTC")

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    out = df[["id", "open_time"]].copy()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["rsi_14"] = (100 * gain / (gain + loss)).mask((gain + loss).eq(0), 50)
    out["macd"] = _ema(close, 12) - _ema(close, 26)
    out["macd_signal"] = _ema(out["macd"], 9)
    out["macd_histogram"] = out["macd"] - out["macd_signal"]
    out["bb_middle"] = close.rolling(20).mean()
    deviation = close.rolling(20).std(ddof=0)
    out["bb_upper"] = out["bb_middle"] + 2 * deviation
    out["bb_lower"] = out["bb_middle"] - 2 * deviation
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]
    for period in (9, 21, 50):
        out[f"ema_{period}"] = _ema(close, period)
    out["sma_200"] = close.rolling(200).mean()
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    out["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    periods_per_year = pd.Timedelta(days=365) / step_ts
    out["hist_volatility_20"] = close.pct_change(fill_method=None).rolling(20).std(
        ddof=0
    ) * np.sqrt(periods_per_year)
    out["hl_range_pct"] = (high - low) / close
    out["obv"] = (np.sign(delta).fillna(0) * volume).cumsum()
    out["volume_sma_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    day = df["open_time"].dt.floor("D")
    typical = (high + low + close) / 3
    out["vwap_daily"] = (typical * volume).groupby(day).cumsum() / volume.groupby(
        day
    ).cumsum().replace(0, np.nan)
    money_flow = ((2 * close - high - low) / (high - low).replace(0, np.nan)).fillna(0)
    out["accumulation_distribution"] = (money_flow * volume).cumsum()
    for period in (5, 10):
        out[f"roc_{period}"] = close.pct_change(periods=period, fill_method=None)
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    out["williams_r"] = (-100 * (highest - close) / (highest - lowest)).mask(
        (highest - lowest).eq(0), -50
    )
    rsi_low = out["rsi_14"].rolling(14).min()
    rsi_high = out["rsi_14"].rolling(14).max()
    out["stoch_rsi"] = (100 * (out["rsi_14"] - rsi_low) / (rsi_high - rsi_low)).mask(
        (rsi_high - rsi_low).eq(0), 50
    )
    out["price_ema50_distance"] = close / out["ema_50"] - 1
    out["ema21_slope_5"] = out["ema_21"].pct_change(periods=5, fill_method=None)
    out["higher_highs_20"] = high.gt(high.shift()).astype(float).rolling(20).sum()
    out["lower_lows_20"] = low.lt(low.shift()).astype(float).rolling(20).sum()

    external = _external_features(df, step, macro, fear_greed, max_ffill_days)
    out = out.merge(external, on="open_time", validate="one_to_one")
    decision_time = df["open_time"] + step_ts
    out["hour_of_day"] = decision_time.dt.hour
    out["day_of_week"] = decision_time.dt.day_of_week
    out["is_weekend"] = out["day_of_week"] >= 5
    for index, name in enumerate(WEEKDAYS):
        out[f"weekday_{name}"] = out["day_of_week"] == index
    out["target_label"] = make_target_label(close)
    return out[["id", "open_time", *STORED_COLUMNS]].replace([np.inf, -np.inf], np.nan)
