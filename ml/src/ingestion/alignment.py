"""
Alineación temporal multi-fuente (velas 4h + macro diario + Fear & Greed).

Reglas anti-leakage:
- El feature de una vela se evalúa en su CIERRE (open_time + intervalo), nunca en su apertura.
- Un cierre macro diario del día D solo es utilizable desde D+1 00:00 UTC (conservador: el
  mercado de EE.UU. cierra durante el día D). Un fin de semana se propaga hacia adelante.
- El Fear & Greed del día D se publica a las 00:00 UTC de D y es utilizable desde ese instante.
- Retraso intrínseco de las features macro: hasta ~24h (más en fines de semana/festivos).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datetime import timedelta

MACRO_COLUMN_NAMES = {
    "DX-Y.NYB": "dxy_close",
    "^GSPC": "sp500_close",
    "^VIX": "vix_close",
    "GC=F": "gold_close",
}
ASOF_TOLERANCE = pd.Timedelta(hours=24)
_NS_UTC = pd.DatetimeTZDtype(unit="ns", tz="UTC")


def macro_column_name(symbol: str) -> str:
    known = MACRO_COLUMN_NAMES.get(symbol)
    if known:
        return known
    return re.sub(r"[^a-z0-9]+", "_", symbol.lower()).strip("_") + "_close"


def _daily_available_series(
    values: pd.Series, available_at: pd.Series, horizon: pd.Timestamp, max_ffill_days: int
) -> pd.DataFrame:
    """Serie con un punto por día calendario (00:00 UTC), propagada hacia adelante (acotada)."""
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(available_at, tz="UTC"))
    series = series[~series.index.duplicated(keep="last")].sort_index()
    last_day = max(series.index.max(), horizon)
    calendar = pd.date_range(series.index.min(), last_day, freq="D", tz="UTC")
    daily = series.reindex(calendar).ffill(limit=max_ffill_days)
    return daily.rename("value").rename_axis("available_at").reset_index().dropna()


def _asof(
    decision: pd.DataFrame, right: pd.DataFrame, column: str, tolerance: pd.Timedelta
) -> pd.Series:
    right = right.rename(columns={"value": column}).copy()
    right["available_at"] = right["available_at"].astype(_NS_UTC)
    merged = pd.merge_asof(
        decision.sort_values("decision_time"),
        right.sort_values("available_at"),
        left_on="decision_time",
        right_on="available_at",
        direction="backward",
        tolerance=tolerance,
    )
    return merged.set_index("open_time")[column]


def align_external_to_candles(
    candles: pd.DataFrame,
    step: timedelta,
    macro: pd.DataFrame | None = None,
    fear_greed: pd.DataFrame | None = None,
    max_ffill_days: int = 5,
) -> pd.DataFrame:
    """
    Devuelve `candles[['open_time']]` + columnas macro y `fear_greed_value`, alineadas con
    `pd.merge_asof` (backward, tolerancia 24h sobre una serie diaria con forward-fill acotado).
    """
    if candles.empty:
        return pd.DataFrame({"open_time": candles["open_time"]})

    step_ts = pd.Timedelta(step)
    decision = pd.DataFrame(
        {
            "open_time": candles["open_time"].to_numpy(),
            "decision_time": (candles["open_time"] + step_ts).astype(_NS_UTC).to_numpy(),
        }
    )
    decision["decision_time"] = decision["decision_time"].astype(_NS_UTC)
    horizon = decision["decision_time"].max().floor("D")
    out = pd.DataFrame({"open_time": candles["open_time"].to_numpy()}).set_index("open_time")

    if macro is not None and not macro.empty:
        for symbol, group in macro.groupby("symbol"):
            trade_date = pd.to_datetime(group["trade_date"]).dt.tz_localize("UTC")
            available_at = trade_date + pd.Timedelta(days=1)
            daily = _daily_available_series(
                group["close"].reset_index(drop=True),
                available_at.reset_index(drop=True),
                horizon,
                max_ffill_days,
            )
            column = macro_column_name(str(symbol))
            out[column] = _asof(decision, daily, column, ASOF_TOLERANCE)

    if fear_greed is not None and not fear_greed.empty:
        obs = pd.to_datetime(fear_greed["obs_date"]).dt.tz_localize("UTC")
        daily = _daily_available_series(
            fear_greed["value"].reset_index(drop=True),
            obs.reset_index(drop=True),
            horizon,
            max_ffill_days,
        )
        out["fear_greed_value"] = _asof(decision, daily, "fear_greed_value", ASOF_TOLERANCE)

    return out.reset_index()
