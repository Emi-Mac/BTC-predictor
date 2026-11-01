"""Historial de interés abierto (5 min) de futuros perpetuos de Binance."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import structlog

from src.ingestion.funding_client import parse_decimal
from src.ingestion.public_api import PublicApiClient

log = structlog.get_logger(__name__)
OI_PATH = "/futures/data/openInterestHist"
OI_STEP = timedelta(minutes=5)
MAX_LIMIT = 500
# Binance solo conserva 30 días y rechaza `startTime` más antiguo; se deja margen.
MAX_HISTORY = timedelta(days=29)
OI_COLUMNS = ["symbol", "create_time", "sum_open_interest", "sum_open_interest_value"]


class BinanceOpenInterestClient:
    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        rate_limit_per_min: int = 30,
        max_attempts: int = 6,
        backoff_multiplier: float = 1.0,
        client: PublicApiClient | None = None,
    ) -> None:
        self._api = client or PublicApiClient(
            base_url,
            "binance_open_interest",
            rate_limit_per_min=rate_limit_per_min,
            max_attempts=max_attempts,
            backoff_multiplier=backoff_multiplier,
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._api.aclose()

    async def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None = None,
        now: datetime | None = None,
    ) -> pd.DataFrame:
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("El inicio del interés abierto debe incluir zona horaria")
        if end is not None and (end.tzinfo is None or end.utcoffset() is None):
            raise ValueError("El fin del interés abierto debe incluir zona horaria")
        now = now or datetime.now(UTC)
        end = min(end, now) if end is not None else now
        start = max(start, now - MAX_HISTORY)
        if start > end:
            raise ValueError("El inicio del interés abierto es posterior al fin")
        step_ms = int(OI_STEP.total_seconds() * 1000)
        end_ms = int(end.timestamp() * 1000)
        cursor = int(start.timestamp() * 1000)
        rows: list[dict[str, Any]] = []
        previous = cursor - 1
        # Con `startTime` y `endTime` Binance devuelve las últimas `limit` filas del rango,
        # por lo que se pide en ventanas que contienen como máximo `MAX_LIMIT` pasos.
        while cursor <= end_ms:
            window_end = min(end_ms, cursor + (MAX_LIMIT - 1) * step_ms)
            response = await self._api.get(
                OI_PATH,
                {
                    "symbol": symbol,
                    "period": "5m",
                    "startTime": cursor,
                    "endTime": window_end,
                    "limit": MAX_LIMIT,
                },
            )
            page = response.json()
            if not isinstance(page, list) or not all(isinstance(item, dict) for item in page):
                raise ValueError("La API de interés abierto devolvió un formato inválido")
            for item in page:
                timestamp = item["timestamp"]
                if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                    raise ValueError("timestamp debe ser un entero en milisegundos")
                if (
                    item["symbol"] != symbol
                    or timestamp <= previous
                    or timestamp < cursor
                    or timestamp > window_end
                    or timestamp % step_ms != 0
                ):
                    raise ValueError(
                        "La API de interés abierto devolvió filas fuera de orden o rango"
                    )
                amount = parse_decimal(item["sumOpenInterest"], "sumOpenInterest")
                value = parse_decimal(item["sumOpenInterestValue"], "sumOpenInterestValue")
                if amount < 0 or value < 0:
                    raise ValueError("El interés abierto no puede ser negativo")
                rows.append(
                    {
                        "symbol": symbol,
                        "create_time": datetime.fromtimestamp(timestamp / 1000, tz=UTC),
                        "sum_open_interest": amount,
                        "sum_open_interest_value": value,
                    }
                )
                previous = timestamp
            cursor = window_end + 1
        log.info("open_interest_fetched", symbol=symbol, rows=len(rows))
        return pd.DataFrame(rows, columns=OI_COLUMNS)
