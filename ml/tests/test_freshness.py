import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from src.ingestion import scheduler
from src.ingestion.freshness import check_freshness, funding_is_stale
from src.ingestion.integrity import check_candle_integrity
from tests.conftest import make_candles

STEP = timedelta(hours=4)


def test_stale_data_is_not_reported_as_complete():
    candles = make_candles(6)
    now = datetime(2024, 1, 2, 8, 2, tzinfo=UTC)
    report = check_candle_integrity(candles, STEP, as_of=now)
    assert not report.ok
    assert report.missing_recent == 2
    assert report.expected_open_time == datetime(2024, 1, 2, 4, tzinfo=UTC)
    assert "velas recientes faltantes" in " ".join(report.problems)


def test_publication_grace_and_open_candle_detection():
    last = datetime(2024, 1, 1, 16, tzinfo=UTC)
    assert not check_freshness(last, datetime(2024, 1, 2, 0, 0, 30, tzinfo=UTC), STEP).stale
    assert check_freshness(last, datetime(2024, 1, 2, 0, 1, tzinfo=UTC), STEP).missing_recent == 1
    premature = check_freshness(
        datetime(2024, 1, 2, 0, tzinfo=UTC), datetime(2024, 1, 2, 0, 2, tzinfo=UTC), STEP
    )
    assert premature.premature_candle and premature.stale
    with pytest.raises(ValueError, match="zona horaria"):
        check_freshness(last, datetime(2024, 1, 2), STEP)


def test_minute_interval_uses_a_shorter_publication_grace():
    result = check_freshness(
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 2, 31, tzinfo=UTC),
        timedelta(minutes=1),
    )
    assert result.expected_open_time == datetime(2024, 1, 1, 0, 1, tzinfo=UTC)
    assert result.missing_recent == 1


def test_funding_freshness_rejects_missing_stale_and_future_data():
    now = datetime(2024, 1, 2, tzinfo=UTC)
    limit = timedelta(hours=24)
    assert funding_is_stale(None, now, limit)
    assert not funding_is_stale(now - limit, now, limit)
    assert funding_is_stale(now - limit - timedelta(seconds=1), now, limit)
    with pytest.raises(ValueError, match="futuro"):
        funding_is_stale(now + timedelta(seconds=1), now, limit)


async def test_watchdog_recovers_after_missed_jobs(monkeypatch):
    settings = SimpleNamespace(btc_symbol="BTCUSDT", default_interval="4h", funding_enabled=False)
    latest = datetime(2024, 1, 1, 20, tzinfo=UTC)

    class Repo:
        async def last_open_time(self, symbol, interval):
            return latest

    class Features:
        async def last_feature_open_time(self, symbol, interval):
            return latest

    calls = []

    async def run(repo, config):
        calls.append((repo, config))

    repo = Repo()
    monkeypatch.setattr(scheduler, "run_cycle", run)
    await scheduler._watchdog_tick(
        repo, Features(), settings, asyncio.Lock(), datetime(2024, 1, 2, 8, 2, tzinfo=UTC)
    )
    assert calls == [(repo, settings)]


async def test_watchdog_recovers_stale_features_but_skips_premature_candle(monkeypatch):
    settings = SimpleNamespace(btc_symbol="BTCUSDT", default_interval="4h", funding_enabled=False)
    now = datetime(2024, 1, 2, 8, 2, tzinfo=UTC)

    class Repo:
        latest = datetime(2024, 1, 2, 4, tzinfo=UTC)

        async def last_open_time(self, symbol, interval):
            return self.latest

    class Features:
        async def last_feature_open_time(self, symbol, interval):
            return datetime(2024, 1, 2, 0, tzinfo=UTC)

    calls = []

    async def run(repo, config):
        calls.append(1)

    repo = Repo()
    monkeypatch.setattr(scheduler, "run_cycle", run)
    await scheduler._watchdog_tick(repo, Features(), settings, asyncio.Lock(), now)
    assert calls == [1]
    repo.latest = datetime(2024, 1, 2, 8, tzinfo=UTC)
    await scheduler._watchdog_tick(repo, Features(), settings, asyncio.Lock(), now)
    assert calls == [1]


async def test_watchdog_retries_stale_funding_with_fresh_candles(monkeypatch):
    settings = SimpleNamespace(
        btc_symbol="BTCUSDT", default_interval="4h", funding_enabled=True, funding_max_lag_hours=24
    )
    now = datetime(2024, 1, 2, 8, 2, tzinfo=UTC)
    latest = datetime(2024, 1, 2, 4, tzinfo=UTC)

    class Repo:
        engine = object()

        async def last_open_time(self, symbol, interval):
            return latest

    class Features:
        async def last_feature_open_time(self, symbol, interval):
            return latest

    class Funding:
        def __init__(self, engine):
            pass

        async def last_funding_time(self, symbol):
            return now - timedelta(hours=25)

    calls = []

    async def run(repo, config):
        calls.append(1)

    monkeypatch.setattr(scheduler, "FundingRepository", Funding)
    monkeypatch.setattr(scheduler, "run_cycle", run)
    await scheduler._watchdog_tick(Repo(), Features(), settings, asyncio.Lock(), now)
    assert calls == [1]


async def test_concurrent_scheduler_and_watchdog_do_not_duplicate_runs(monkeypatch):
    started = asyncio.Event()
    finish = asyncio.Event()
    calls = []

    async def run(repo, config):
        calls.append(1)
        started.set()
        await finish.wait()

    monkeypatch.setattr(scheduler, "run_cycle", run)
    repo, settings, lock = object(), object(), asyncio.Lock()
    task = asyncio.create_task(scheduler._safe_run(repo, settings, lock))
    await started.wait()
    await scheduler._safe_run(repo, settings, lock)
    finish.set()
    await task
    assert calls == [1]


async def test_watchdog_rearms_scheduler_and_retries_after_failure(monkeypatch):
    stop = asyncio.Event()
    calls = []

    class FakeScheduler:
        def wakeup(self):
            calls.append("wakeup")

    async def tick(*args):
        calls.append("tick")
        if calls.count("tick") == 1:
            raise RuntimeError("temporary failure")
        stop.set()

    repo = SimpleNamespace(engine=object())
    settings = SimpleNamespace(ingestion_watchdog_seconds=0.01)
    monkeypatch.setattr(scheduler, "_watchdog_tick", tick)
    await asyncio.wait_for(
        scheduler._watchdog(repo, settings, asyncio.Lock(), FakeScheduler(), stop), 1
    )
    assert calls == ["wakeup", "tick", "wakeup", "tick"]
