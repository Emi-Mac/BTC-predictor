"""Familias de features de horizonte largo (Fase 2), todas conocidas al cierre de la vela t.

Cada familia declara cuándo está disponible cada dato y se alinea al instante de decisión
(`open_time + 4h`) con `merge_asof` hacia atrás y una antigüedad máxima.

- `stationary` (C1): sustitutos estacionarios de los niveles absolutos del conjunto base.
- `regime` (C2): tendencia, volatilidad y drawdown en ventanas diarias/semanales.
- `derivatives` (C3): funding acumulado, OI multi-día, posicionamiento y basis spot-perpetuo.
- `onchain` (C4): actividad, valoración, flujos a exchanges y oferta de stablecoins.
- `events` (C5): calendario programado del FOMC y vencimientos mensuales de opciones.
- `cross_asset` (C6): ETH/BTC y transformaciones macro de 20–60 días.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.features.compute import FEATURE_COLUMNS
from src.features.funding_context import FUNDING_FEATURE_COLUMNS
from src.features.price_context import PRICE_CONTEXT_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Callable

STEP = pd.Timedelta(hours=4)
PERIODS_PER_YEAR = 6 * 365
NS_UTC = pd.DatetimeTZDtype(unit="ns", tz="UTC")

BASE_COLUMNS = [*FEATURE_COLUMNS, *PRICE_CONTEXT_COLUMNS, *FUNDING_FEATURE_COLUMNS]
NONSTATIONARY_COLUMNS = [
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "ema_9",
    "ema_21",
    "ema_50",
    "sma_200",
    "atr_14",
    "obv",
    "vwap_daily",
    "accumulation_distribution",
    "macd",
    "macd_signal",
    "macd_histogram",
    "dxy_close",
    "sp500_close",
]
FAMILIES: dict[str, list[str]] = {
    "stationary": [
        "distance_sma200",
        "obv_flow_42",
        "ad_flow_42",
        "relative_macd_signal",
        "relative_macd_histogram",
    ],
    "regime": [
        "regime_return_3d",
        "regime_return_7d",
        "regime_return_14d",
        "regime_return_30d",
        "regime_return_90d",
        "distance_ma_50d",
        "distance_ma_200d",
        "ma_50d_200d_spread",
        "realized_vol_7d",
        "realized_vol_30d",
        "realized_vol_ratio_7d_30d",
        "realized_vol_percentile_1y",
        "drawdown_90d",
        "drawdown_1y",
        "efficiency_ratio_30d",
        "range_position_30d",
    ],
    "derivatives": [
        "funding_sum_3d",
        "funding_sum_7d",
        "funding_sum_30d",
        "funding_zscore_90d",
        "oi_change_3d",
        "oi_change_7d",
        "oi_change_30d",
        "top_trader_ls_zscore_30d",
        "accounts_ls_zscore_30d",
        "taker_ls_mean_7d",
        "basis",
        "basis_mean_7d",
        "basis_zscore_30d",
        "premium_mean_7d",
    ],
    "onchain": [
        "mvrv",
        "mvrv_change_30d",
        "active_addresses_change_30d",
        "transactions_change_30d",
        "hashrate_change_30d",
        "fees_change_30d",
        "exchange_netflow_ratio_7d",
        "exchange_supply_change_30d",
        "stablecoin_supply_change_7d",
        "stablecoin_supply_change_30d",
    ],
    "events": [
        "days_to_next_fomc",
        "days_since_last_fomc",
        "fomc_next_72h",
        "fomc_last_72h",
        "days_to_monthly_expiry",
        "quarterly_expiry_week",
    ],
    "cross_asset": [
        "ethbtc_return_7d",
        "ethbtc_return_30d",
        "sp500_return_20d",
        "dxy_change_20d",
        "gold_return_20d",
        "vix_zscore_60d",
    ],
}
STATIONARY_BASE = [c for c in BASE_COLUMNS if c not in NONSTATIONARY_COLUMNS]
# Supuestos de disponibilidad (documentados también en docs/research/data-sources.md).
FUNDING_DELAY = STEP
FUNDING_MAX_AGE = timedelta(hours=36)
METRICS_DELAY = timedelta(hours=48) + timedelta(minutes=5)
METRICS_MAX_AGE = timedelta(hours=12)
ONCHAIN_DELAY = timedelta(days=2)
ONCHAIN_MAX_AGE = timedelta(days=5)
MACRO_DELAY = timedelta(days=1)
MACRO_MAX_AGE = timedelta(days=5)
MAX_EVENT_DAYS = 60.0


def stationary_feature_set() -> list[str]:
    return [*STATIONARY_BASE, *FAMILIES["stationary"]]


# ─── Utilidades ──────────────────────────────────────────────────────────────


def _decision_times(open_times: pd.Series) -> pd.Series:
    return (pd.to_datetime(open_times, utc=True) + STEP).astype(NS_UTC)


def asof_align(
    open_times: pd.Series,
    values: pd.DataFrame,
    available_at: pd.Series,
    max_age: timedelta,
) -> pd.DataFrame:
    """Último valor disponible a la hora de decisión de cada vela, con antigüedad acotada."""
    if len(values) != len(available_at):
        raise ValueError("Valores y tiempos de disponibilidad deben alinear")
    right = values.copy()
    right["_available_at"] = pd.to_datetime(available_at, utc=True).astype(NS_UTC).to_numpy()
    right = right.dropna(subset=["_available_at"]).sort_values("_available_at")
    right = right.drop_duplicates("_available_at", keep="last")
    left = pd.DataFrame(
        {"_decision": _decision_times(open_times), "_order": np.arange(len(open_times))}
    )
    merged = pd.merge_asof(
        left.sort_values("_decision"),
        right,
        left_on="_decision",
        right_on="_available_at",
        direction="backward",
        tolerance=pd.Timedelta(max_age),
    )
    return merged.sort_values("_order")[list(values.columns)].reset_index(drop=True)


def _complete_grid(frame: pd.DataFrame, column: str, freq: pd.Timedelta) -> pd.DataFrame:
    times = pd.to_datetime(frame[column], utc=True).astype(NS_UTC)
    if times.duplicated().any():
        raise ValueError(f"Timestamps duplicados en {column}")
    if ((times.astype("int64") % freq.value) != 0).any():
        raise ValueError(f"{column} no está alineado a {freq}")
    grid = pd.date_range(times.min(), times.max(), freq=freq, tz="UTC")
    return frame.assign(**{column: times}).set_index(column).reindex(grid)


def _zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    rolling = series.rolling(window, min_periods=min_periods or window)
    std = rolling.std()
    return (series - rolling.mean()) / std.where(std > 0)


def _log_change(series: pd.Series, periods: int) -> pd.Series:
    positive = series.where(series > 0)
    return np.log(positive) - np.log(positive.shift(periods))


# ─── C1: estacionariedad ─────────────────────────────────────────────────────


def stationary_features(features: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Normaliza niveles por precio o volumen; sin información posterior a t."""
    prices = candles.set_index("open_time")[["close", "volume"]]
    close = prices["close"].reindex(features["open_time"]).to_numpy(dtype=float)
    volume_sum = (
        prices["volume"].rolling(42, min_periods=42).sum().reindex(features["open_time"]).to_numpy()
    )
    out = pd.DataFrame(index=features.index)
    out["distance_sma200"] = close / features["sma_200"].to_numpy(dtype=float) - 1
    for name, source in (("obv_flow_42", "obv"), ("ad_flow_42", "accumulation_distribution")):
        values = features[source].astype(float)
        if (features["open_time"].diff().dropna() != STEP).any():
            raise ValueError("Las features deben ser continuas para diferencias de 42 velas")
        out[name] = (values - values.shift(42)).to_numpy() / np.where(
            volume_sum > 0, volume_sum, np.nan
        )
    out["relative_macd_signal"] = features["macd_signal"].to_numpy(dtype=float) / close
    out["relative_macd_histogram"] = features["macd_histogram"].to_numpy(dtype=float) / close
    return out


# ─── C2: régimen ─────────────────────────────────────────────────────────────


def price_history(candles: pd.DataFrame, history: pd.DataFrame | None) -> pd.Series:
    """Cierre spot 4h en una rejilla completa; antepone historia auxiliar si coincide."""
    current = candles[["open_time", "close"]].copy()
    current["open_time"] = pd.to_datetime(current["open_time"], utc=True)
    if history is not None and not history.empty:
        older = history[["open_time", "close"]].copy()
        older["open_time"] = pd.to_datetime(older["open_time"], utc=True)
        overlap = older.merge(current, on="open_time", suffixes=("_aux", ""))
        if overlap.empty or not np.allclose(overlap["close_aux"], overlap["close"], rtol=1e-9):
            raise ValueError("La historia auxiliar no coincide con las velas del dataset")
        older = older.loc[older["open_time"] < current["open_time"].min()]
        current = pd.concat([older, current], ignore_index=True)
    grid = _complete_grid(current, "open_time", STEP)
    return grid["close"].astype(float)


def regime_features(open_times: pd.Series, close: pd.Series) -> pd.DataFrame:
    log_price = np.log(close)
    returns = log_price.diff()
    out = pd.DataFrame(index=close.index)
    for name, periods in (("3d", 18), ("7d", 42), ("14d", 84), ("30d", 180), ("90d", 540)):
        out[f"regime_return_{name}"] = log_price - log_price.shift(periods)
    ma50 = close.rolling(300, min_periods=300).mean()
    ma200 = close.rolling(1200, min_periods=1200).mean()
    out["distance_ma_50d"] = log_price - np.log(ma50)
    out["distance_ma_200d"] = log_price - np.log(ma200)
    out["ma_50d_200d_spread"] = np.log(ma50) - np.log(ma200)
    out["realized_vol_7d"] = returns.rolling(42, min_periods=42).std() * np.sqrt(PERIODS_PER_YEAR)
    out["realized_vol_30d"] = returns.rolling(180, min_periods=180).std() * np.sqrt(
        PERIODS_PER_YEAR
    )
    out["realized_vol_ratio_7d_30d"] = out["realized_vol_7d"] / out["realized_vol_30d"]
    out["realized_vol_percentile_1y"] = (
        out["realized_vol_30d"].rolling(2190, min_periods=540).rank(pct=True)
    )
    out["drawdown_90d"] = close / close.rolling(540, min_periods=540).max() - 1
    out["drawdown_1y"] = close / close.rolling(2190, min_periods=1080).max() - 1
    path = returns.abs().rolling(180, min_periods=180).sum()
    out["efficiency_ratio_30d"] = (log_price - log_price.shift(180)).abs() / path.where(path > 0)
    low = close.rolling(180, min_periods=180).min()
    high = close.rolling(180, min_periods=180).max()
    out["range_position_30d"] = (close - low) / (high - low).where(high > low)
    times = pd.to_datetime(open_times, utc=True)
    return out.reindex(times).reset_index(drop=True)


# ─── C3: derivados ───────────────────────────────────────────────────────────


def funding_features(open_times: pd.Series, funding: pd.DataFrame) -> pd.DataFrame:
    events = funding.sort_values("funding_time").reset_index(drop=True)
    if events["funding_time"].duplicated().any():
        raise ValueError("Funding duplicado")
    rate = events["funding_rate"].astype(float)
    values = pd.DataFrame(
        {
            "funding_sum_3d": rate.rolling(9, min_periods=9).sum(),
            "funding_sum_7d": rate.rolling(21, min_periods=21).sum(),
            "funding_sum_30d": rate.rolling(90, min_periods=90).sum(),
            "funding_zscore_90d": _zscore(rate, 270, 90),
        }
    )
    available = pd.to_datetime(events["funding_time"], utc=True) + FUNDING_DELAY
    return asof_align(open_times, values, available, FUNDING_MAX_AGE)


def metrics_features(open_times: pd.Series, metrics: pd.DataFrame) -> pd.DataFrame:
    """OI y posicionamiento en barras de 4h (último dato de cada barra), con 48h de demora."""
    frame = metrics.copy()
    frame["create_time"] = pd.to_datetime(frame["create_time"], utc=True)
    frame = frame.sort_values("create_time")
    frame["_bar"] = frame["create_time"].dt.floor("4h")
    bars = frame.groupby("_bar").last()
    bars = bars.reindex(pd.date_range(bars.index.min(), bars.index.max(), freq=STEP, tz="UTC"))
    oi = bars["sum_open_interest"].astype(float)
    values = pd.DataFrame(index=bars.index)
    for name, periods in (("3d", 18), ("7d", 42), ("30d", 180)):
        values[f"oi_change_{name}"] = _log_change(oi, periods)
    for name, source in (
        ("top_trader_ls_zscore_30d", "sum_toptrader_long_short_ratio"),
        ("accounts_ls_zscore_30d", "count_long_short_ratio"),
    ):
        ratio = bars[source].astype(float)
        values[name] = _zscore(np.log(ratio.where(ratio > 0)), 180, 90)
    taker = bars["sum_taker_long_short_vol_ratio"].astype(float)
    values["taker_ls_mean_7d"] = np.log(taker.where(taker > 0)).rolling(42, min_periods=21).mean()
    source_time = bars["create_time"] + pd.Timedelta(minutes=5)
    available = (source_time + pd.Timedelta(hours=48)).reset_index(drop=True)
    return asof_align(open_times, values.reset_index(drop=True), available, METRICS_MAX_AGE)


def basis_features(
    open_times: pd.Series, spot_close: pd.Series, perp: pd.DataFrame, premium: pd.DataFrame
) -> pd.DataFrame:
    """Basis spot-perpetuo y prima del perpetuo al cierre de cada vela."""
    perp_close = _complete_grid(perp, "open_time", STEP)["close"].astype(float)
    spot = spot_close.reindex(perp_close.index)
    basis = np.log(perp_close) - np.log(spot)
    premium_close = _complete_grid(premium, "open_time", STEP)["close"].astype(float)
    values = pd.DataFrame(index=perp_close.index)
    values["basis"] = basis
    values["basis_mean_7d"] = basis.rolling(42, min_periods=21).mean()
    values["basis_zscore_30d"] = _zscore(basis, 180, 90)
    values["premium_mean_7d"] = (
        premium_close.reindex(perp_close.index).rolling(42, min_periods=21).mean()
    )
    times = pd.to_datetime(open_times, utc=True)
    return values.reindex(times).reset_index(drop=True)


# ─── C4: on-chain ────────────────────────────────────────────────────────────


def onchain_features(
    open_times: pd.Series, btc: pd.DataFrame, stablecoins: pd.DataFrame
) -> pd.DataFrame:
    daily = btc.copy()
    daily["time"] = pd.to_datetime(daily["time"], utc=True)
    daily = _complete_grid(daily, "time", pd.Timedelta(days=1))
    values = pd.DataFrame(index=daily.index)

    def smooth(name: str) -> pd.Series:
        return daily[name].astype(float).rolling(7, min_periods=5).mean()

    values["mvrv"] = daily["CapMVRVCur"].astype(float)
    values["mvrv_change_30d"] = _log_change(values["mvrv"], 30)
    values["active_addresses_change_30d"] = _log_change(smooth("AdrActCnt"), 30)
    values["transactions_change_30d"] = _log_change(smooth("TxCnt"), 30)
    values["hashrate_change_30d"] = _log_change(smooth("HashRate"), 30)
    values["fees_change_30d"] = _log_change(smooth("FeeTotNtv"), 30)
    inflow = daily["FlowInExUSD"].astype(float).rolling(7, min_periods=5).sum()
    outflow = daily["FlowOutExUSD"].astype(float).rolling(7, min_periods=5).sum()
    total = inflow + outflow
    values["exchange_netflow_ratio_7d"] = (inflow - outflow) / total.where(total > 0)
    values["exchange_supply_change_30d"] = _log_change(daily["SplyExNtv"].astype(float), 30)
    supply = stablecoins.copy()
    supply["time"] = pd.to_datetime(supply["time"], utc=True)
    total_supply = supply.pivot_table(index="time", columns="asset", values="SplyCur").sum(
        axis=1, min_count=supply["asset"].nunique()
    )
    total_supply = total_supply.reindex(values.index)
    values["stablecoin_supply_change_7d"] = _log_change(total_supply, 7)
    values["stablecoin_supply_change_30d"] = _log_change(total_supply, 30)
    available = pd.Series(values.index + pd.Timedelta(ONCHAIN_DELAY))
    return asof_align(open_times, values.reset_index(drop=True), available, ONCHAIN_MAX_AGE)


# ─── C5: eventos ─────────────────────────────────────────────────────────────


def _last_friday(year: int, month: int) -> pd.Timestamp:
    last = pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.offsets.MonthEnd(0)
    return last - pd.Timedelta(days=(last.day_of_week - 4) % 7)


def monthly_expiries(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Vencimientos mensuales de opciones de BTC: último viernes del mes, 08:00 UTC."""
    months = pd.period_range(
        start - pd.DateOffset(months=1), end + pd.DateOffset(months=2), freq="M"
    )
    return pd.DatetimeIndex([_last_friday(p.year, p.month) + pd.Timedelta(hours=8) for p in months])


def _nanoseconds(values: pd.Series | pd.DatetimeIndex) -> npt.NDArray[np.int64]:
    stamps = pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns")
    result: npt.NDArray[np.int64] = stamps.asi8
    return result


def event_features(open_times: pd.Series, fomc: pd.DataFrame) -> pd.DataFrame:
    decision = _decision_times(open_times)
    meetings = np.sort(_nanoseconds(fomc["decision_time"]))
    now = _nanoseconds(decision)
    if not len(meetings) or meetings[0] > now.min() or meetings[-1] <= now.max():
        raise ValueError("El calendario del FOMC no cubre todo el período")
    following = np.searchsorted(meetings, now, side="right")
    day = float(pd.Timedelta(days=1).value)
    to_next = (meetings[following] - now) / day
    since_last = (now - meetings[following - 1]) / day
    out = pd.DataFrame(
        {
            "days_to_next_fomc": np.minimum(to_next, MAX_EVENT_DAYS),
            "days_since_last_fomc": np.minimum(since_last, MAX_EVENT_DAYS),
            "fomc_next_72h": (to_next <= 3).astype(float),
            "fomc_last_72h": (since_last <= 3).astype(float),
        }
    )
    expiries = _nanoseconds(monthly_expiries(decision.min(), decision.max()))
    next_expiry = expiries[np.searchsorted(expiries, now, side="right")]
    out["days_to_monthly_expiry"] = (next_expiry - now) / day
    quarterly = pd.DatetimeIndex(pd.to_datetime(next_expiry, utc=True)).month.isin([3, 6, 9, 12])
    out["quarterly_expiry_week"] = (quarterly & (out["days_to_monthly_expiry"] <= 7)).astype(float)
    return out


# ─── C6: activos relacionados ────────────────────────────────────────────────


def cross_asset_features(
    open_times: pd.Series, ethbtc: pd.DataFrame, macro: pd.DataFrame
) -> pd.DataFrame:
    eth = np.log(_complete_grid(ethbtc, "open_time", STEP)["close"].astype(float))
    values = pd.DataFrame(index=eth.index)
    values["ethbtc_return_7d"] = eth - eth.shift(42)
    values["ethbtc_return_30d"] = eth - eth.shift(180)
    times = pd.to_datetime(open_times, utc=True)
    out = values.reindex(times).reset_index(drop=True)

    transforms: dict[str, tuple[str, Callable[[pd.Series], pd.Series]]] = {
        "sp500_return_20d": ("^GSPC", lambda s: _log_change(s, 20)),
        "dxy_change_20d": ("DX-Y.NYB", lambda s: _log_change(s, 20)),
        "gold_return_20d": ("GC=F", lambda s: _log_change(s, 20)),
        "vix_zscore_60d": ("^VIX", lambda s: _zscore(s, 60, 40)),
    }
    for name, (symbol, transform) in transforms.items():
        series = (
            macro.loc[macro["symbol"] == symbol, ["trade_date", "close"]]
            .assign(trade_date=lambda f: pd.to_datetime(f["trade_date"], utc=True))
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        if series.empty:
            raise ValueError(f"Falta la serie macro {symbol}")
        derived = pd.DataFrame({name: transform(series["close"].astype(float))})
        available = series["trade_date"] + pd.Timedelta(MACRO_DELAY)
        out[name] = asof_align(open_times, derived, available, MACRO_MAX_AGE)[name].to_numpy()
    return out


# ─── Ensamblado ──────────────────────────────────────────────────────────────


def build_long_horizon_features(
    features: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    aux: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Añade todas las familias a las features del dataset congelado (misma fila por vela)."""
    ordered = features.sort_values("open_time").reset_index(drop=True)
    times = ordered["open_time"]
    candles = sources["candles"].sort_values("open_time").reset_index(drop=True)
    candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True)
    close = price_history(candles, aux.get("btc_spot_klines"))
    parts = [
        stationary_features(ordered, candles),
        regime_features(times, close),
        funding_features(times, sources["funding"]),
        metrics_features(times, metrics),
        basis_features(times, close, aux["perp_klines"], aux["premium_index"]),
        onchain_features(times, aux["coinmetrics_btc"], aux["coinmetrics_stablecoins"]),
        event_features(times, aux["fomc"]),
        cross_asset_features(times, aux["ethbtc_klines"], sources["macro"]),
    ]
    added = pd.concat([part.reset_index(drop=True) for part in parts], axis=1)
    expected = [name for family in FAMILIES.values() for name in family]
    if sorted(added.columns) != sorted(expected):
        raise ValueError("Las familias no producen exactamente las columnas declaradas")
    clash = set(added.columns) & set(ordered.columns)
    if clash:
        raise ValueError(f"Columnas repetidas con el dataset: {sorted(clash)}")
    result = pd.concat([ordered, added[expected]], axis=1)
    return result.replace([np.inf, -np.inf], np.nan)
