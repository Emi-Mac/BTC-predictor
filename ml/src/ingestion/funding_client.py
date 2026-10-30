"""Histórico público de funding para futuros perpetuos de Binance."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd
import structlog
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from src.ingestion.http_utils import (
    RETRYABLE_STATUS,
    MinIntervalLimiter,
    is_retryable,
    wait_with_retry_after,
)

if TYPE_CHECKING:
    from types import TracebackType

log = structlog.get_logger(__name__)
FUNDING_PATH = "/fapi/v1/fundingRate"
MAX_LIMIT = 1000
FUNDING_COLUMNS = ["symbol", "funding_time", "funding_rate", "mark_price"]


def parse_decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, (str, int, float, Decimal)) or isinstance(value, bool):
        raise ValueError(f"{name} inválido")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} inválido") from exc
    if not result.is_finite():
        raise ValueError(f"{name} debe ser finito")
    return result


class BinanceFundingClient:
    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        rate_limit_per_min: int = 90,
        max_attempts: int = 6,
        backoff_multiplier: float = 1.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if rate_limit_per_min <= 0 or rate_limit_per_min > 100:
            raise ValueError("El límite de funding debe estar entre 1 y 100 solicitudes/minuto")
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=30.0)
        self._owns_client = client is None
        self._limiter = MinIntervalLimiter(60.0 / rate_limit_per_min)
        self._max_attempts = max_attempts
        self._backoff_multiplier = backoff_multiplier

    async def __aenter__(self) -> BinanceFundingClient:
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

    async def _get_page(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        retrying = AsyncRetrying(
            retry=retry_if_exception(is_retryable),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_with_retry_after(
                wait_exponential(multiplier=self._backoff_multiplier, max=60)
            ),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                await self._limiter.acquire()
                response = await self._client.get(FUNDING_PATH, params=params)
                if response.status_code in RETRYABLE_STATUS:
                    log.warning(
                        "funding_retryable_status",
                        status=response.status_code,
                        attempt=attempt.retry_state.attempt_number,
                    )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                    raise ValueError("La API de funding devolvió un formato inválido")
                return data
        raise RuntimeError("No se pudo completar la solicitud de funding")  # pragma: no cover

    async def fetch(
        self, symbol: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("El inicio del funding debe incluir zona horaria")
        now = datetime.now(UTC)
        if end is not None and (end.tzinfo is None or end.utcoffset() is None):
            raise ValueError("El fin del funding debe incluir zona horaria")
        end = min(end, now) if end is not None else now
        if start > end:
            raise ValueError("El inicio del funding es posterior al fin")
        cursor = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rows: list[dict[str, Any]] = []
        while cursor <= end_ms:
            page = await self._get_page(
                {
                    "symbol": symbol,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": MAX_LIMIT,
                }
            )
            if not page:
                break
            previous = cursor - 1
            for item in page:
                timestamp = item["fundingTime"]
                if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                    raise ValueError("fundingTime debe ser un entero en milisegundos")
                if item["symbol"] != symbol or timestamp <= previous or timestamp > end_ms:
                    raise ValueError("La API de funding devolvió eventos fuera de orden o rango")
                rate = parse_decimal(item["fundingRate"], "fundingRate")
                mark_raw = item.get("markPrice")
                mark = parse_decimal(mark_raw, "markPrice") if mark_raw not in (None, "") else None
                if mark is not None and mark <= 0:
                    raise ValueError("markPrice debe ser positivo")
                rows.append(
                    {
                        "symbol": symbol,
                        "funding_time": datetime.fromtimestamp(timestamp / 1000, tz=UTC),
                        "funding_rate": rate,
                        "mark_price": mark,
                    }
                )
                previous = timestamp
            cursor = previous + 1
        log.info("funding_fetched", symbol=symbol, rows=len(rows))
        return pd.DataFrame(rows, columns=FUNDING_COLUMNS)
