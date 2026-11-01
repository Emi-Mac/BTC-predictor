"""Fuentes en vivo de volatilidad contra PostgreSQL real (requiere TEST_DATABASE_URL)."""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import Settings
from src.ingestion import live_sources, pipeline, scheduler
from src.ingestion.live_sources import (
    LiveSourcesRepository,
    LiveSourcesSummary,
    ingest_dvol,
    ingest_fomc_calendar,
    ingest_live_sources,
    ingest_open_interest,
    live_source_problems,
)
from src.ingestion.repository import IngestionRepository
from src.ingestion.tables import metadata
from src.volatility_config import VolatilitySettings
from tests.conftest import make_candles

NOW = datetime(2026, 11, 1, 12, 30, tzinfo=UTC)


@pytest.fixture
async def engine(test_database_url):
    url = test_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(metadata.drop_all)
        await conn.run_sync(metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(metadata.drop_all)
    await eng.dispose()


@pytest.fixture
def settings():
    return Settings(_env_file=None, database_url="postgresql://x/x", funding_enabled=False)


@pytest.fixture
def live():
    return VolatilitySettings(_env_file=None, volatility_sources_enabled=True)


def dvol_frame(start, hours, close=40.0):
    times = [start + timedelta(hours=h) for h in range(hours)]
    return pd.DataFrame(
        {
            "currency": "BTC",
            "open_time": times,
            "close": [Decimal(str(close + h)) for h in range(hours)],
        }
    )


def oi_frame(start, steps):
    times = [start + timedelta(minutes=5 * n) for n in range(steps)]
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "create_time": times,
            "sum_open_interest": [Decimal("100.5")] * steps,
            "sum_open_interest_value": [Decimal("5000000.25")] * steps,
        }
    )


def events_frame(*times):
    return pd.DataFrame(
        {"event_type": "fomc", "event_time": list(times), "source_url": "https://fed/current"}
    )


class FakeDvol:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    async def fetch(self, currency, start, end=None, now=None):
        self.calls.append(start)
        return self.frame[self.frame["open_time"] >= start].reset_index(drop=True)

    async def aclose(self):
        raise AssertionError("un cliente inyectado no debe cerrarse")


class FakeOi(FakeDvol):
    async def fetch(self, symbol, start, end=None, now=None):
        self.calls.append(start)
        return self.frame[self.frame["create_time"] >= start].reset_index(drop=True)


class FakeFomc:
    def __init__(self, frame):
        self.frame = frame
        self.calls = 0

    async def fetch(self):
        self.calls += 1
        return self.frame


async def test_upserts_are_idempotent_and_keep_the_first_value(engine):
    store = LiveSourcesRepository(engine)
    start = NOW - timedelta(hours=10)
    assert await store.upsert_dvol(dvol_frame(start, 5)) == 5
    assert await store.upsert_dvol(dvol_frame(start, 5, close=99.0)) == 0
    assert await store.upsert_open_interest(oi_frame(start, 5)) == 5
    assert await store.upsert_open_interest(oi_frame(start, 5)) == 0
    assert await store.upsert_events(events_frame(NOW)) == 1
    assert await store.upsert_events(events_frame(NOW)) == 0
    async with engine.connect() as conn:
        close = (
            await conn.execute(
                sa.text("select close from implied_volatility order by open_time limit 1")
            )
        ).scalar()
        recorded = (
            await conn.execute(
                sa.text("select count(*) from open_interest where recorded_at is not null")
            )
        ).scalar()
    assert close == Decimal("40.000000")
    assert recorded == 5
    assert await store.last_dvol_open_time("BTC") == start + timedelta(hours=4)
    assert await store.last_open_interest_time("BTCUSDT") == start + timedelta(minutes=20)
    assert await store.last_dvol_open_time("ETH") is None


async def test_upsert_rejects_naive_timestamps(engine):
    store = LiveSourcesRepository(engine)
    naive = dvol_frame(datetime(2026, 1, 1, tzinfo=UTC), 1)
    naive["open_time"] = naive["open_time"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="zona horaria"):
        await store.upsert_dvol(naive)
    bad = dvol_frame(datetime(2026, 1, 1, tzinfo=UTC), 1)
    bad["close"] = float("nan")
    with pytest.raises(ValueError, match="finito"):
        await store.upsert_dvol(bad)


async def test_dvol_backfills_from_first_candle_then_updates_with_overlap(engine, settings, live):
    repo = IngestionRepository(engine)
    await repo.upsert_candles("BTCUSDT", "4h", make_candles(30, start="2026-10-01"))
    summary = LiveSourcesSummary()
    fake = FakeDvol(dvol_frame(datetime(2026, 9, 20, tzinfo=UTC), 24 * 40))
    await ingest_dvol(repo, settings, live, summary, client=fake, now=NOW)
    assert fake.calls[0] == datetime(2026, 9, 24, tzinfo=UTC)  # primera vela - 7 días
    assert summary.dvol_rows == 24 * 36
    summary = LiveSourcesSummary()
    await ingest_dvol(repo, settings, live, summary, client=fake, now=NOW)
    last = await LiveSourcesRepository(engine).last_dvol_open_time("BTC")
    assert fake.calls[1] == last - timedelta(hours=24)
    assert summary.dvol_rows == 0


async def test_dvol_without_candles_or_data_fails_loudly(engine, settings, live):
    repo = IngestionRepository(engine)
    with pytest.raises(ValueError, match="No hay velas"):
        await ingest_dvol(
            repo, settings, live, LiveSourcesSummary(), client=FakeDvol(dvol_frame(NOW, 1))
        )
    await repo.upsert_candles("BTCUSDT", "4h", make_candles(3, start="2026-10-01"))
    empty = FakeDvol(dvol_frame(NOW, 1).iloc[0:0])
    with pytest.raises(ValueError, match="no devolvió"):
        await ingest_dvol(repo, settings, live, LiveSourcesSummary(), client=empty, now=NOW)


async def test_open_interest_backfills_29_days_then_overlaps_one_hour(engine, settings, live):
    repo = IngestionRepository(engine)
    start = NOW - timedelta(days=29)
    fake = FakeOi(oi_frame(start - timedelta(days=1), 12 * 24 * 31))
    summary = LiveSourcesSummary()
    await ingest_open_interest(repo, settings, live, summary, client=fake, now=NOW)
    assert fake.calls[0] == start
    assert summary.open_interest_rows > 12 * 24 * 29 - 2
    summary = LiveSourcesSummary()
    await ingest_open_interest(repo, settings, live, summary, client=fake, now=NOW)
    last = await LiveSourcesRepository(engine).last_open_interest_time("BTCUSDT")
    assert fake.calls[1] == last - timedelta(hours=1)
    assert summary.open_interest_rows == 0
    with pytest.raises(ValueError, match="no devolvió"):
        await ingest_open_interest(
            repo, settings, live, summary, client=FakeOi(oi_frame(NOW, 1).iloc[0:0]), now=NOW
        )


async def test_fomc_calendar_refreshes_weekly_or_when_no_future_meeting(engine, live):
    repo = IngestionRepository(engine)
    fake = FakeFomc(events_frame(NOW + timedelta(days=30)))
    await ingest_fomc_calendar(repo, live, LiveSourcesSummary(), client=fake, now=NOW)
    assert fake.calls == 1
    await ingest_fomc_calendar(
        repo, live, LiveSourcesSummary(), client=fake, now=NOW + timedelta(days=6)
    )
    assert fake.calls == 1
    await ingest_fomc_calendar(
        repo, live, LiveSourcesSummary(), client=fake, now=NOW + timedelta(days=8)
    )
    assert fake.calls == 2
    # Tras la última reunión conocida no queda calendario futuro: se refresca aunque sea reciente
    await ingest_fomc_calendar(
        repo, live, LiveSourcesSummary(), client=fake, now=NOW + timedelta(days=31)
    )
    assert fake.calls == 3


async def test_health_reports_each_stale_source(engine, settings, live):
    problems = await live_source_problems(engine, settings, live, NOW)
    assert len(problems) == 3
    store = LiveSourcesRepository(engine)
    await store.upsert_dvol(dvol_frame(NOW - timedelta(hours=2), 1))
    await store.upsert_open_interest(oi_frame(NOW - timedelta(minutes=10), 1))
    await store.upsert_events(events_frame(NOW + timedelta(days=10)))
    assert await live_source_problems(engine, settings, live, NOW) == []
    later = NOW + timedelta(hours=4)
    assert len(await live_source_problems(engine, settings, live, later)) == 2


async def test_one_failing_source_does_not_stop_the_others(engine, settings, live, monkeypatch):
    repo = IngestionRepository(engine)

    async def boom(*_args, **_kwargs):
        raise RuntimeError("Deribit caído")

    async def fomc(repo_, live_, summary, **_kwargs):
        summary.fomc_rows += 7

    monkeypatch.setattr(live_sources, "ingest_dvol", boom)
    monkeypatch.setattr(live_sources, "ingest_open_interest", boom)
    monkeypatch.setattr(live_sources, "ingest_fomc_calendar", fomc)
    summary = await ingest_live_sources(repo, settings, live)
    assert summary.fomc_rows == 7
    assert summary.dvol_rows == 0


async def test_run_ingestion_only_touches_live_sources_when_enabled(engine, settings, monkeypatch):
    repo = IngestionRepository(engine)
    calls = []

    async def fake_live(*_args):
        calls.append(1)
        return LiveSourcesSummary(dvol_rows=3)

    async def nothing(*_args, **_kwargs):
        return None

    class Candles:
        async def fetch_klines(self, *_args):
            return pd.DataFrame()

        async def aclose(self):
            return None

    monkeypatch.setattr(pipeline, "ingest_live_sources", fake_live)
    for name in ("ingest_macro", "ingest_fear_greed", "ingest_funding", "ingest_candles"):
        monkeypatch.setattr(pipeline, name, nothing)
    for enabled in (False, True):
        monkeypatch.setattr(
            pipeline,
            "get_volatility_settings",
            lambda enabled=enabled: VolatilitySettings(
                _env_file=None, volatility_sources_enabled=enabled
            ),
        )
        summary = await pipeline.run_ingestion(repo, settings, Candles())
    assert calls == [1]
    assert summary.live_sources == {"dvol_rows": 3, "open_interest_rows": 0, "fomc_rows": 0}


async def test_watchdog_recovers_when_a_live_source_is_stale(monkeypatch):
    settings = SimpleNamespace(btc_symbol="BTCUSDT", default_interval="4h", funding_enabled=False)
    now = datetime(2024, 1, 2, 8, 2, tzinfo=UTC)
    latest = datetime(2024, 1, 2, 4, tzinfo=UTC)

    class Repo:
        engine = object()

        async def last_open_time(self, symbol, interval):
            return latest

    class Features:
        async def last_feature_open_time(self, symbol, interval):
            return latest

    calls = []
    problems = []

    async def run(repo, config):
        calls.append(1)

    async def live_problems(engine, config, live, moment):
        return problems

    monkeypatch.setattr(scheduler, "run_cycle", run)
    monkeypatch.setattr(scheduler, "live_source_problems", live_problems)
    for enabled, found, expected in ((False, ["DVOL"], 0), (True, [], 0), (True, ["DVOL"], 1)):
        problems[:] = found
        monkeypatch.setattr(
            scheduler,
            "get_volatility_settings",
            lambda enabled=enabled: VolatilitySettings(
                _env_file=None, volatility_sources_enabled=enabled
            ),
        )
        calls.clear()
        await scheduler._watchdog_tick(Repo(), Features(), settings, asyncio.Lock(), now)
        assert len(calls) == expected
