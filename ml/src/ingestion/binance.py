"""Conector asíncrono a la API pública de Binance (klines) con paginación y retry."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.ingestion.http_utils import RETRYABLE_STATUS
from src.ingestion.http_utils import MinIntervalLimiter as _MinIntervalLimiter
from src.ingestion.http_utils import is_retryable as _is_retryable
from src.ingestion.http_utils import wait_with_retry_after as _wait_with_retry_after

if TYPE_CHECKING:
    from types import TracebackType

log = structlog.get_logger(__name__)

KLINES_PATH = "/api/v3/klines"
KLINES_MAX_LIMIT = 1000
KLINES_WEIGHT = 2  # peso por request con limit > 100 (límite público: 1200 de peso/min)

CANDLE_COLUMNS = [
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]

_INTERVAL_RE = re.compile(r"^(\d+)([mhdw])$")
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def interval_to_timedelta(interval: str) -> timedelta:
    """Convierte un intervalo de Binance ('4h', '15m', '1d') a timedelta."""
    match = _INTERVAL_RE.match(interval)
    if not match:
        msg = f"Intervalo no soportado: {interval!r}"
        raise ValueError(msg)
    return timedelta(seconds=int(match.group(1)) * _UNIT_SECONDS[match.group(2)])


class BinanceKlinesClient:
    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        rate_limit_per_min: int = 1200,
        max_attempts: int = 6,
        backoff_max_seconds: float = 60.0,
        backoff_multiplier: float = 1.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=30.0)
        self._owns_client = client is None
        self._limiter = _MinIntervalLimiter(60.0 * KLINES_WEIGHT / rate_limit_per_min)
        self._max_attempts = max_attempts
        self._backoff_max = backoff_max_seconds
        self._backoff_multiplier = backoff_multiplier

    async def __aenter__(self) -> BinanceKlinesClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get_klines(self, params: dict[str, Any]) -> list[list[Any]]:
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._max_attempts),
            wait=_wait_with_retry_after(
                wait_exponential(multiplier=self._backoff_multiplier, max=self._backoff_max)
            ),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                await self._limiter.acquire()
                response = await self._client.get(KLINES_PATH, params=params)
                if response.status_code in RETRYABLE_STATUS:
                    log.warning(
                        "binance_retryable_status",
                        status=response.status_code,
                        attempt=attempt.retry_state.attempt_number,
                    )
                response.raise_for_status()
                data: list[list[Any]] = response.json()
                return data
        msg = "unreachable"  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Pagina el histórico desde `start` hasta `end` (o ahora) y devuelve solo velas CERRADAS.
        La vela en curso se descarta: su close cambiaría y contaminaría el dataset.
        """
        step = interval_to_timedelta(interval)
        now = datetime.now(UTC)
        end = min(end, now) if end else now
        cursor_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rows: list[list[Any]] = []

        while cursor_ms < end_ms:
            batch = await self._get_klines(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor_ms,
                    "endTime": end_ms,
                    "limit": KLINES_MAX_LIMIT,
                }
            )
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + int(step.total_seconds() * 1000)
            if next_cursor <= cursor_ms:  # protección ante respuestas inconsistentes
                break
            cursor_ms = next_cursor

        log.info("binance_klines_fetched", symbol=symbol, interval=interval, rows=len(rows))
        return parse_klines(rows, step, closed_before=now)


def parse_klines(
    rows: list[list[Any]],
    step: timedelta,
    closed_before: datetime | None = None,
) -> pd.DataFrame:
    """Convierte la respuesta cruda de klines en un DataFrame tipado y ordenado por tiempo."""
    if not rows:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    if any(len(row) < 11 for row in rows):
        raise ValueError("Binance devolvió una vela sin los campos de volumen agresor")

    df = pd.DataFrame(
        {
            "open_time": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
            "quote_volume": [float(r[7]) for r in rows],
            "num_trades": [int(r[8]) for r in rows],
            "taker_buy_base_volume": [float(r[9]) for r in rows],
            "taker_buy_quote_volume": [float(r[10]) for r in rows],
        }
    )
    # Convención Binance: close_time = open_time + intervalo - 1 ms
    df["close_time"] = df["open_time"] + step - pd.Timedelta(milliseconds=1)
    df = df.drop_duplicates(subset="open_time", keep="last").sort_values("open_time")
    if closed_before is not None:
        df = df[df["close_time"] < pd.Timestamp(closed_before)]
    return df[CANDLE_COLUMNS].reset_index(drop=True)
