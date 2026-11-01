"""Índice de volatilidad implícita DVOL de Deribit (velas horarias, solo velas cerradas)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import structlog

from src.ingestion.public_api import PublicApiClient

log = structlog.get_logger(__name__)
DVOL_PATH = "/api/v2/public/get_volatility_index_data"
DVOL_STEP = timedelta(hours=1)
DVOL_COLUMNS = ["currency", "open_time", "close"]
MAX_PAGES = 200


def parse_dvol_page(page: object, cursor_ms: int) -> tuple[list[list[Any]], int | None]:
    """Valida una página `[ms, open, high, low, close]` y devuelve (filas, continuación)."""
    if not isinstance(page, dict):
        raise ValueError("Deribit devolvió un formato inválido")
    if "error" in page:
        raise ValueError(f"Deribit devolvió un error: {page['error']}")
    result = page.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("data"), list):
        raise ValueError("Deribit devolvió un formato inválido")
    continuation = result.get("continuation")
    if continuation is not None:
        if isinstance(continuation, bool) or not isinstance(continuation, int):
            raise ValueError("La continuación de Deribit debe ser un entero")
        if continuation >= cursor_ms:
            raise ValueError("La paginación de Deribit no retrocede")
    return result["data"], continuation


def to_candle(row: object, currency: str) -> dict[str, Any]:
    if not isinstance(row, list) or len(row) != 5:
        raise ValueError("Vela de DVOL inválida")
    timestamp, close = row[0], row[4]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise ValueError("La marca de tiempo de DVOL debe ser un entero en milisegundos")
    if timestamp % int(DVOL_STEP.total_seconds() * 1000) != 0:
        raise ValueError("La vela de DVOL no está alineada a la hora")
    if isinstance(close, bool) or not isinstance(close, (int, float)) or not math.isfinite(close):
        raise ValueError("El cierre de DVOL debe ser numérico y finito")
    if close <= 0:
        raise ValueError("El cierre de DVOL debe ser positivo")
    return {
        "currency": currency,
        "open_time": datetime.fromtimestamp(timestamp / 1000, tz=UTC),
        "close": Decimal(str(close)),
    }


class DeribitDvolClient:
    def __init__(
        self,
        base_url: str = "https://www.deribit.com",
        rate_limit_per_min: int = 30,
        max_attempts: int = 6,
        backoff_multiplier: float = 1.0,
        client: PublicApiClient | None = None,
    ) -> None:
        self._api = client or PublicApiClient(
            base_url,
            "deribit",
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
        currency: str,
        start: datetime,
        end: datetime | None = None,
        now: datetime | None = None,
    ) -> pd.DataFrame:
        """Velas horarias cerradas en [start, end). La hora en curso se descarta: aún cambia."""
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("El inicio de DVOL debe incluir zona horaria")
        if end is not None and (end.tzinfo is None or end.utcoffset() is None):
            raise ValueError("El fin de DVOL debe incluir zona horaria")
        now = now or datetime.now(UTC)
        end = min(end, now) if end is not None else now
        if start > end:
            raise ValueError("El inicio de DVOL es posterior al fin")
        start_ms = int(start.timestamp() * 1000)
        cursor: int | None = int(end.timestamp() * 1000)
        rows: dict[datetime, dict[str, Any]] = {}
        pages = 0
        while cursor is not None and cursor >= start_ms:
            pages += 1
            if pages > MAX_PAGES:
                raise ValueError("Deribit excedió el máximo de páginas")
            response = await self._api.get(
                DVOL_PATH,
                {
                    "currency": currency,
                    "start_timestamp": start_ms,
                    "end_timestamp": cursor,
                    "resolution": int(DVOL_STEP.total_seconds()),
                },
            )
            data, cursor = parse_dvol_page(response.json(), cursor)
            for item in data:
                candle = to_candle(item, currency)
                rows[candle["open_time"]] = candle
        closed = [
            row for time, row in sorted(rows.items()) if time >= start and time + DVOL_STEP <= end
        ]
        log.info("dvol_fetched", currency=currency, rows=len(closed))
        return pd.DataFrame(closed, columns=DVOL_COLUMNS)
