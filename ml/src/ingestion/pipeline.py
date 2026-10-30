"""Orquestación de la ingesta: velas Binance + macro yfinance + Fear & Greed. Idempotente."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from src.ingestion.binance import BinanceKlinesClient, interval_to_timedelta
from src.ingestion.cleaning import clean_candles, detect_gaps
from src.ingestion.funding_client import BinanceFundingClient
from src.ingestion.funding_repository import FundingRepository
from src.ingestion.market_data import fetch_fear_greed, fetch_macro_series
from src.ingestion.time_utils import floor_to_step

if TYPE_CHECKING:
    from src.config import Settings
    from src.ingestion.repository import IngestionRepository

log = structlog.get_logger(__name__)

MACRO_REFETCH_DAYS = 7  # ventana de solape para absorber correcciones tardías
FEAR_GREED_UPDATE_LIMIT = 30


@dataclass
class IngestionSummary:
    candles_fetched: int = 0
    candles_inserted: int = 0
    macro_rows: int = 0
    fear_greed_rows: int = 0
    funding_rows: int = 0


async def ingest_candles(
    repo: IngestionRepository,
    client: BinanceKlinesClient,
    settings: Settings,
    summary: IngestionSummary,
) -> None:
    symbol, interval = settings.btc_symbol, settings.default_interval
    step = interval_to_timedelta(interval)
    last = await repo.last_open_time(symbol, interval)

    if last is None:
        start = floor_to_step(datetime.now(UTC) - timedelta(days=settings.history_days), step)
        log.info("candles_backfill_start", symbol=symbol, interval=interval, start=str(start))
    else:
        start = last - settings.update_overlap_candles * step

    windows: list[tuple[datetime, datetime | None]] = [(start, None)]

    # Reparar huecos históricos que hayan quedado en la DB (p. ej. por caídas del worker)
    stored = await repo.load_candles(symbol, interval)
    if not stored.empty:
        for gap in detect_gaps(stored["open_time"], step):
            gap_start = gap.start.to_pydatetime()
            if gap_start < start:
                windows.append((gap_start, gap_start + gap.missing * step))

    for window_start, window_end in windows:
        raw = await client.fetch_klines(symbol, interval, window_start, window_end)
        if raw.empty:
            continue
        clean, report = clean_candles(raw, step, max_fill=1)
        summary.candles_fetched += len(clean)
        summary.candles_inserted += await repo.upsert_candles(symbol, interval, clean)
        log.info(
            "candles_cleaned",
            filled=report.filled,
            dropped_invalid=report.invalid_removed,
            unfilled_gaps=len(report.unfilled_gaps),
            completeness=round(report.completeness, 5),
        )


async def ingest_macro(
    repo: IngestionRepository, settings: Settings, summary: IngestionSummary
) -> None:
    today = datetime.now(UTC).date()
    last = await repo.last_macro_date()
    if last is None:
        start = (datetime.now(UTC) - timedelta(days=settings.history_days + 30)).date()
    else:
        start = last - timedelta(days=MACRO_REFETCH_DAYS)

    df = await fetch_macro_series(settings.macro_symbols, start)
    # La barra del día en curso es parcial: se descarta para no almacenar valores no finales.
    df = df[df["trade_date"] < today]
    summary.macro_rows += await repo.upsert_macro(df)


async def ingest_fear_greed(
    repo: IngestionRepository, settings: Settings, summary: IngestionSummary
) -> None:
    today: date = datetime.now(UTC).date()
    last = await repo.last_fear_greed_date()
    if last is not None and last >= today:  # caché diario: ya está el valor de hoy
        log.info("fear_greed_cached", last=last.isoformat())
        return
    limit = 0 if last is None else FEAR_GREED_UPDATE_LIMIT
    df = await fetch_fear_greed(settings.fear_greed_base_url, limit=limit)
    summary.fear_greed_rows += await repo.upsert_fear_greed(df)


async def ingest_funding(
    repo: IngestionRepository,
    settings: Settings,
    summary: IngestionSummary,
    client: BinanceFundingClient | None = None,
) -> None:
    if not settings.funding_enabled:
        log.info("funding_ingestion_disabled")
        return
    first = await repo.first_open_time(settings.btc_symbol, settings.default_interval)
    if first is None:
        raise ValueError("No hay velas para determinar el inicio del histórico de funding")
    funding = FundingRepository(repo.engine)
    last = await funding.last_funding_time(settings.btc_symbol)
    start = first - timedelta(days=7) if last is None else last - timedelta(days=1)
    owns_client = client is None
    adapter = client or BinanceFundingClient(
        base_url=settings.binance_futures_base_url,
        rate_limit_per_min=settings.funding_rate_limit_per_min,
    )
    try:
        data = await adapter.fetch(settings.btc_symbol, start)
        if data.empty:
            raise ValueError("La API de Binance no devolvió eventos de funding")
        summary.funding_rows += await funding.upsert(data)
    finally:
        if owns_client:
            await adapter.aclose()


async def run_ingestion(
    repo: IngestionRepository,
    settings: Settings,
    client: BinanceKlinesClient | None = None,
) -> IngestionSummary:
    """Ejecuta backfill (si la DB está vacía) o actualización incremental. Seguro de repetir."""
    summary = IngestionSummary()
    owns_client = client is None
    binance = client or BinanceKlinesClient(
        base_url=settings.binance_base_url,
        rate_limit_per_min=settings.binance_rate_limit_per_min,
    )
    try:
        await ingest_candles(repo, binance, settings, summary)
    finally:
        if owns_client:
            await binance.aclose()

    # Las fuentes secundarias no deben abortar la ingesta de velas si fallan
    for name, step in (
        ("macro", ingest_macro),
        ("fear_greed", ingest_fear_greed),
        ("funding", ingest_funding),
    ):
        try:
            await step(repo, settings, summary)
        except Exception:
            log.exception("secondary_source_failed", source=name)

    log.info("ingestion_done", **summary.__dict__)
    return summary
