"""Conectores de datos diarios: macro (yfinance) y Fear & Greed Index (alternative.me)."""
from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pandas as pd
import structlog
import yfinance as yf
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

MACRO_COLUMNS = ["symbol", "trade_date", "close"]
FEAR_GREED_COLUMNS = ["obs_date", "value", "classification"]


def _download_one(symbol: str, start: date, end: date | None) -> pd.DataFrame:
    raw = yf.download(
        symbol,
        start=start.isoformat(),
        end=end.isoformat() if end else None,
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    return normalize_macro_frame(raw, symbol)


def normalize_macro_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normaliza la salida de yfinance a (symbol, trade_date, close), sin NaN."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=MACRO_COLUMNS)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.get_level_values(0)
    close = raw["Close"].dropna()
    index = pd.DatetimeIndex(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    df = pd.DataFrame(
        {"symbol": symbol, "trade_date": index.normalize().date, "close": close.to_numpy(float)}
    )
    return df.drop_duplicates(subset="trade_date", keep="last").reset_index(drop=True)


async def fetch_macro_series(
    symbols: list[str],
    start: date,
    end: date | None = None,
    max_attempts: int = 4,
) -> pd.DataFrame:
    """Descarga cierres diarios de cada símbolo. Un símbolo fallido no tumba a los demás."""
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        try:
            retrying = AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=2, max=30),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            )
            frame: pd.DataFrame = await retrying(
                asyncio.to_thread, _download_one, symbol, start, end
            )
        except Exception as exc:
            log.error("macro_download_failed", symbol=symbol, error=str(exc))
            continue
        if frame.empty:
            log.warning("macro_symbol_empty", symbol=symbol)
            continue
        log.info("macro_fetched", symbol=symbol, rows=len(frame))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=MACRO_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def parse_fear_greed(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Mapea la respuesta de alternative.me a (obs_date, value, classification).
    El timestamp de la API es 00:00 UTC del día de la observación.
    """
    data = payload.get("data") or []
    rows = []
    for item in data:
        try:
            ts = datetime.fromtimestamp(int(item["timestamp"]), tz=UTC)
            value = int(item["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= value <= 100:
            continue
        rows.append(
            {
                "obs_date": ts.date(),
                "value": value,
                "classification": item.get("value_classification"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=FEAR_GREED_COLUMNS)
    df = pd.DataFrame(rows).drop_duplicates(subset="obs_date", keep="last")
    return df.sort_values("obs_date").reset_index(drop=True)


async def fetch_fear_greed(
    base_url: str,
    limit: int = 0,
    max_attempts: int = 4,
    client: httpx.AsyncClient | None = None,
    backoff_multiplier: float = 2.0,
) -> pd.DataFrame:
    """`limit=0` descarga todo el histórico; usa un número pequeño para actualizaciones."""
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=30.0)
    try:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=backoff_multiplier, max=30),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                response = await http.get(base_url, params={"limit": limit, "format": "json"})
                response.raise_for_status()
                payload = response.json()
        df = parse_fear_greed(payload)
        log.info("fear_greed_fetched", rows=len(df))
        return df
    finally:
        if owns_client:
            await http.aclose()
