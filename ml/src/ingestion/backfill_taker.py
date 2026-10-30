"""Recupera los volúmenes agresores históricos sin modificar OHLCV ya almacenado."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import get_settings
from src.ingestion.binance import BinanceKlinesClient, interval_to_timedelta
from src.ingestion.cleaning import invalid_ohlcv_mask, invalid_taker_volume_mask
from src.ingestion.repository import IngestionRepository
from src.ingestion.run import configure_logging

if TYPE_CHECKING:
    from src.config import Settings

log = structlog.get_logger(__name__)


async def backfill_taker(
    repo: IngestionRepository, settings: Settings, client: BinanceKlinesClient
) -> int:
    symbol, interval = settings.btc_symbol, settings.default_interval
    first, last, count = await repo.missing_taker_bounds(symbol, interval)
    if not count:
        if await repo.last_open_time(symbol, interval) is None:
            raise ValueError("No hay velas en la base para recuperar volumen agresor")
        log.info("taker_backfill_complete", missing=0, updated=0)
        return 0
    if first is None or last is None:
        raise RuntimeError("La base devolvió límites incompletos para el backfill")
    raw = await client.fetch_klines(symbol, interval, first, last + interval_to_timedelta(interval))
    if raw.empty:
        raise ValueError("Binance no devolvió velas para el backfill")
    invalid = invalid_ohlcv_mask(raw) | invalid_taker_volume_mask(raw)
    if invalid.any():
        raise ValueError(f"Binance devolvió {int(invalid.sum())} velas inválidas")
    updated = await repo.backfill_taker_volumes(symbol, interval, raw)
    _, _, remaining = await repo.missing_taker_bounds(symbol, interval)
    if remaining:
        raise RuntimeError(f"Quedan {remaining} velas sin volumen agresor")
    log.info("taker_backfill_complete", missing=count, updated=updated)
    return updated


async def _run() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        async with BinanceKlinesClient(
            base_url=settings.binance_base_url,
            rate_limit_per_min=settings.binance_rate_limit_per_min,
        ) as client:
            await backfill_taker(IngestionRepository(engine), settings, client)
    finally:
        await engine.dispose()


def main() -> None:
    configure_logging(get_settings().log_level)
    asyncio.run(_run())


if __name__ == "__main__":
    main()
