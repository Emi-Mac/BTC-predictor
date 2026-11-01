"""Persistencia, ingesta incremental y comprobaciones de salud de las fuentes de volatilidad."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import partial
from typing import TYPE_CHECKING, Any

import pandas as pd
import sqlalchemy as sa
import structlog
from sqlalchemy.dialects.postgresql import insert

from src.ingestion.deribit_client import DVOL_COLUMNS, DeribitDvolClient
from src.ingestion.fomc_calendar import EVENT_COLUMNS, EVENT_TYPE, FomcCalendarClient
from src.ingestion.freshness import funding_is_stale
from src.ingestion.open_interest_client import OI_COLUMNS, BinanceOpenInterestClient
from src.ingestion.tables import event_calendar, implied_volatility, open_interest
from src.ingestion.time_utils import floor_to_step

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.config import Settings
    from src.ingestion.repository import IngestionRepository
    from src.volatility_config import VolatilitySettings

log = structlog.get_logger(__name__)
CHUNK_SIZE = 1000
DVOL_OVERLAP = timedelta(hours=24)
OI_OVERLAP = timedelta(hours=1)
DVOL_BACKFILL_MARGIN = timedelta(days=7)
# Primera vela horaria del DVOL publicada por Deribit.
DVOL_FIRST_AVAILABLE = datetime(2021, 3, 24, tzinfo=UTC)


@dataclass
class LiveSourcesSummary:
    dvol_rows: int = 0
    open_interest_rows: int = 0
    fomc_rows: int = 0


def _aware(value: object, name: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} debe incluir zona horaria")
    moment: datetime = timestamp.to_pydatetime()
    return moment


def _decimal(value: object, name: str) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{name} debe ser finito")
    return result


class LiveSourcesRepository:
    """Upserts idempotentes (`ON CONFLICT DO NOTHING`): las fuentes no se revisan."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def _insert(self, table: sa.Table, rows: list[dict[str, Any]], keys: list[str]) -> int:
        inserted = 0
        async with self._engine.begin() as conn:
            for offset in range(0, len(rows), CHUNK_SIZE):
                stmt = insert(table).values(rows[offset : offset + CHUNK_SIZE])
                result = await conn.execute(stmt.on_conflict_do_nothing(index_elements=keys))
                inserted += result.rowcount or 0
        return inserted

    async def upsert_dvol(self, data: pd.DataFrame) -> int:
        rows = [
            {
                "currency": item["currency"],
                "open_time": _aware(item["open_time"], "open_time"),
                "close": _decimal(item["close"], "close"),
            }
            for item in data[DVOL_COLUMNS].to_dict("records")
        ]
        return await self._insert(implied_volatility, rows, ["currency", "open_time"])

    async def upsert_open_interest(self, data: pd.DataFrame) -> int:
        rows = [
            {
                "symbol": item["symbol"],
                "create_time": _aware(item["create_time"], "create_time"),
                "sum_open_interest": _decimal(item["sum_open_interest"], "sum_open_interest"),
                "sum_open_interest_value": _decimal(
                    item["sum_open_interest_value"], "sum_open_interest_value"
                ),
            }
            for item in data[OI_COLUMNS].to_dict("records")
        ]
        return await self._insert(open_interest, rows, ["symbol", "create_time"])

    async def upsert_events(self, data: pd.DataFrame) -> int:
        rows = [
            {
                "event_type": item["event_type"],
                "event_time": _aware(item["event_time"], "event_time"),
                "source_url": item["source_url"],
            }
            for item in data[EVENT_COLUMNS].to_dict("records")
        ]
        return await self._insert(event_calendar, rows, ["event_type", "event_time"])

    async def _latest(self, query: sa.Select[Any]) -> datetime | None:
        async with self._engine.connect() as conn:
            latest: datetime | None = (await conn.execute(query)).scalar()
        return latest

    async def last_dvol_open_time(self, currency: str) -> datetime | None:
        query = sa.select(sa.func.max(implied_volatility.c.open_time)).where(
            implied_volatility.c.currency == currency
        )
        return await self._latest(query)

    async def last_open_interest_time(self, symbol: str) -> datetime | None:
        query = sa.select(sa.func.max(open_interest.c.create_time)).where(
            open_interest.c.symbol == symbol
        )
        return await self._latest(query)

    async def last_event_capture(self, event_type: str) -> datetime | None:
        query = sa.select(sa.func.max(event_calendar.c.captured_at)).where(
            event_calendar.c.event_type == event_type
        )
        return await self._latest(query)

    async def next_event_time(self, event_type: str, now: datetime) -> datetime | None:
        query = sa.select(sa.func.min(event_calendar.c.event_time)).where(
            event_calendar.c.event_type == event_type, event_calendar.c.event_time > now
        )
        return await self._latest(query)


async def ingest_dvol(
    repo: IngestionRepository,
    settings: Settings,
    live: VolatilitySettings,
    summary: LiveSourcesSummary,
    client: DeribitDvolClient | None = None,
    now: datetime | None = None,
) -> None:
    now = now or datetime.now(UTC)
    store = LiveSourcesRepository(repo.engine)
    last = await store.last_dvol_open_time(live.dvol_currency)
    if last is None:
        first = await repo.first_open_time(settings.btc_symbol, settings.default_interval)
        if first is None:
            raise ValueError("No hay velas para determinar el inicio del histórico de DVOL")
        start = max(
            floor_to_step(first - DVOL_BACKFILL_MARGIN, timedelta(hours=1)), DVOL_FIRST_AVAILABLE
        )
    else:
        start = last - DVOL_OVERLAP
    owns_client = client is None
    adapter = client or DeribitDvolClient(
        base_url=live.deribit_base_url, rate_limit_per_min=live.dvol_rate_limit_per_min
    )
    try:
        data = await adapter.fetch(live.dvol_currency, start, now=now)
        if data.empty:
            raise ValueError("Deribit no devolvió velas cerradas de DVOL")
        summary.dvol_rows += await store.upsert_dvol(data)
    finally:
        if owns_client:
            await adapter.aclose()


async def ingest_open_interest(
    repo: IngestionRepository,
    settings: Settings,
    live: VolatilitySettings,
    summary: LiveSourcesSummary,
    client: BinanceOpenInterestClient | None = None,
    now: datetime | None = None,
) -> None:
    now = now or datetime.now(UTC)
    store = LiveSourcesRepository(repo.engine)
    last = await store.last_open_interest_time(settings.btc_symbol)
    start = now - timedelta(days=29) if last is None else last - OI_OVERLAP
    owns_client = client is None
    adapter = client or BinanceOpenInterestClient(
        base_url=settings.binance_futures_base_url, rate_limit_per_min=live.oi_rate_limit_per_min
    )
    try:
        data = await adapter.fetch(settings.btc_symbol, start, now=now)
        if data.empty:
            raise ValueError("Binance no devolvió filas de interés abierto")
        summary.open_interest_rows += await store.upsert_open_interest(data)
    finally:
        if owns_client:
            await adapter.aclose()


async def ingest_fomc_calendar(
    repo: IngestionRepository,
    live: VolatilitySettings,
    summary: LiveSourcesSummary,
    client: FomcCalendarClient | None = None,
    now: datetime | None = None,
) -> None:
    now = now or datetime.now(UTC)
    store = LiveSourcesRepository(repo.engine)
    captured = await store.last_event_capture(EVENT_TYPE)
    upcoming = await store.next_event_time(EVENT_TYPE, now)
    # Se refresca cada `fomc_refresh_days`, o ya si no queda ninguna reunión futura.
    if (
        captured is not None
        and upcoming is not None
        and now - captured < timedelta(days=live.fomc_refresh_days)
    ):
        log.info("fomc_calendar_cached", captured=str(captured))
        return
    owns_client = client is None
    adapter = client or FomcCalendarClient()
    try:
        events = await adapter.fetch()
        summary.fomc_rows += await store.upsert_events(events)
    finally:
        if owns_client:
            await adapter.aclose()


async def ingest_live_sources(
    repo: IngestionRepository, settings: Settings, live: VolatilitySettings
) -> LiveSourcesSummary:
    """Cada fuente es independiente: una falla se registra y no detiene a las demás."""
    summary = LiveSourcesSummary()
    steps = (
        ("dvol", partial(ingest_dvol, repo, settings, live, summary)),
        ("open_interest", partial(ingest_open_interest, repo, settings, live, summary)),
        ("fomc_calendar", partial(ingest_fomc_calendar, repo, live, summary)),
    )
    for name, step in steps:
        try:
            await step()
        except Exception:
            log.exception("live_source_failed", source=name)
    log.info("live_sources_done", **summary.__dict__)
    return summary


async def live_source_problems(
    engine: AsyncEngine, settings: Settings, live: VolatilitySettings, now: datetime
) -> list[str]:
    """Problemas de frescura de las fuentes en vivo; vacío si todo está al día."""
    store = LiveSourcesRepository(engine)
    problems: list[str] = []
    checks = (
        (
            "DVOL",
            await store.last_dvol_open_time(live.dvol_currency),
            timedelta(hours=live.dvol_max_lag_hours),
        ),
        (
            "el interés abierto",
            await store.last_open_interest_time(settings.btc_symbol),
            timedelta(hours=live.oi_max_lag_hours),
        ),
    )
    for name, latest, max_lag in checks:
        if funding_is_stale(latest, now, max_lag):
            problems.append(f"{name} no está actualizado")
    if await store.next_event_time(EVENT_TYPE, now) is None:
        problems.append("el calendario del FOMC no tiene reuniones futuras")
    return problems
