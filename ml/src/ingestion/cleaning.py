"""Pipeline de limpieza de velas OHLCV: validación, duplicados, gaps y relleno acotado."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog

if TYPE_CHECKING:
    from datetime import timedelta

log = structlog.get_logger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class Gap:
    """Hueco temporal: `missing` velas consecutivas ausentes a partir de `start`."""

    start: pd.Timestamp
    missing: int


@dataclass
class CleaningReport:
    rows_in: int = 0
    duplicates_removed: int = 0
    invalid_removed: int = 0
    filled: int = 0
    rows_out: int = 0
    unfilled_gaps: list[Gap] = field(default_factory=list)

    @property
    def completeness(self) -> float:
        """Fracción de velas reales (no rellenadas) sobre las esperadas en el rango."""
        missing = sum(g.missing for g in self.unfilled_gaps)
        expected = self.rows_out + missing
        if expected == 0:
            return 1.0
        return (self.rows_out - self.filled) / expected


def invalid_ohlcv_mask(df: pd.DataFrame) -> pd.Series:
    """True en filas inválidas: NaN, precios <= 0, volumen < 0 o high/low incoherentes."""
    nan_rows = df[[*PRICE_COLUMNS, "volume"]].isna().any(axis=1)
    non_positive = (df[PRICE_COLUMNS] <= 0).any(axis=1)
    negative_volume = df["volume"] < 0
    high_ok = df["high"] >= df[["open", "close", "low"]].max(axis=1)
    low_ok = df["low"] <= df[["open", "close", "high"]].min(axis=1)
    return nan_rows | non_positive | negative_volume | ~high_ok | ~low_ok


def invalid_taker_volume_mask(df: pd.DataFrame) -> pd.Series:
    """Detecta pares incompletos, valores no finitos o volúmenes agresores fuera de rango."""
    names = ("taker_buy_base_volume", "taker_buy_quote_volume")
    present = [name in df.columns for name in names]
    if not any(present):
        return pd.Series(False, index=df.index)
    if not all(present):
        raise ValueError("Los volúmenes agresores base y cotizado deben aparecer juntos")
    base = pd.to_numeric(df[names[0]], errors="raise")
    quote = pd.to_numeric(df[names[1]], errors="raise")
    volume = pd.to_numeric(df["volume"], errors="raise")
    quote_volume = pd.to_numeric(df["quote_volume"], errors="raise")
    supplied = base.notna() | quote.notna()
    invalid = (
        base.isna()
        | quote.isna()
        | ~np.isfinite(base)
        | ~np.isfinite(quote)
        | ~np.isfinite(volume)
        | ~np.isfinite(quote_volume)
        | (base < 0)
        | (base > volume)
        | (quote < 0)
        | (quote > quote_volume)
    )
    result: pd.Series = supplied & invalid
    return result


def detect_gaps(open_times: pd.Series, step: timedelta) -> list[Gap]:
    """Detecta huecos entre velas consecutivas (se asume `open_times` ordenado y sin duplicados)."""
    if len(open_times) < 2:
        return []
    step_ts = pd.Timedelta(step)
    deltas = open_times.diff()
    gaps: list[Gap] = []
    for idx in deltas.index[deltas > step_ts]:
        previous = open_times.loc[idx] - deltas.loc[idx]
        missing = int(deltas.loc[idx] / step_ts) - 1
        gaps.append(Gap(start=previous + step_ts, missing=missing))
    return gaps


def fill_small_gaps(
    df: pd.DataFrame, step: timedelta, max_fill: int = 1
) -> tuple[pd.DataFrame, int]:
    """
    Rellena huecos de hasta `max_fill` velas consecutivas con el último close
    (O=H=L=C=close previo, volumen 0). Huecos mayores se dejan intactos para no inventar datos.
    """
    step_ts = pd.Timedelta(step)
    fillable = [g for g in detect_gaps(df["open_time"], step) if g.missing <= max_fill]
    if not fillable:
        return df, 0

    close_by_open = df.set_index("open_time")["close"]
    synthetic = []
    for gap in fillable:
        prev_close = close_by_open.loc[gap.start - step_ts]
        for k in range(gap.missing):
            open_time = gap.start + k * step_ts
            synthetic.append(
                {
                    "open_time": open_time,
                    "close_time": open_time + step_ts - pd.Timedelta(milliseconds=1),
                    "open": prev_close,
                    "high": prev_close,
                    "low": prev_close,
                    "close": prev_close,
                    "volume": 0.0,
                    "quote_volume": 0.0,
                    "num_trades": 0,
                }
            )
    out = pd.concat([df, pd.DataFrame(synthetic)], ignore_index=True)
    out = out.sort_values("open_time").reset_index(drop=True)
    return out, len(synthetic)


def clean_candles(
    df: pd.DataFrame, step: timedelta, max_fill: int = 1
) -> tuple[pd.DataFrame, CleaningReport]:
    """Valida, deduplica y rellena huecos pequeños. Devuelve datos limpios y un reporte."""
    report = CleaningReport(rows_in=len(df))
    if df.empty:
        return df, report

    out = df.sort_values("open_time")
    before = len(out)
    out = out.drop_duplicates(subset="open_time", keep="last")
    report.duplicates_removed = before - len(out)

    invalid = invalid_ohlcv_mask(out) | invalid_taker_volume_mask(out)
    report.invalid_removed = int(invalid.sum())
    if report.invalid_removed:
        log.warning("invalid_candles_dropped", count=report.invalid_removed)
    out = out[~invalid].reset_index(drop=True)

    out, report.filled = fill_small_gaps(out, step, max_fill)
    report.unfilled_gaps = detect_gaps(out["open_time"], step)
    report.rows_out = len(out)
    if report.unfilled_gaps:
        log.warning("unfilled_gaps", gaps=len(report.unfilled_gaps))
    return out, report
