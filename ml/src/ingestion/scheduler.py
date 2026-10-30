"""Scheduler de actualización: ingesta incremental justo después de cada cierre de vela."""

from __future__ import annotations

import asyncio
import contextlib
import signal
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.features.pipeline import run_cycle
from src.features.repository import FeatureRepository
from src.ingestion.binance import interval_to_timedelta
from src.ingestion.freshness import check_freshness, funding_is_stale
from src.ingestion.funding_repository import FundingRepository

if TYPE_CHECKING:
    from src.config import Settings
    from src.ingestion.repository import IngestionRepository

log = structlog.get_logger(__name__)

CLOSE_DELAY_MINUTES = 1  # margen para que Binance publique la vela recién cerrada


def build_trigger(interval: str) -> CronTrigger:
    """Cron UTC alineado al intervalo de las velas (4h -> 00:01, 04:01, 08:01 ...)."""
    seconds = int(interval_to_timedelta(interval).total_seconds())
    if seconds % 3600 == 0 and 86400 % seconds == 0:
        return CronTrigger(
            hour=f"*/{seconds // 3600}" if seconds < 86400 else 0,
            minute=CLOSE_DELAY_MINUTES,
            timezone="UTC",
        )
    if seconds % 60 == 0 and 3600 % seconds == 0:
        return CronTrigger(minute=f"*/{seconds // 60}", second=30, timezone="UTC")
    msg = f"Intervalo no soportado por el scheduler: {interval}"
    raise ValueError(msg)


async def _safe_run(repo: IngestionRepository, settings: Settings, lock: asyncio.Lock) -> None:
    if lock.locked():
        log.info("ingestion_already_running")
        return
    try:
        async with lock:
            await run_cycle(repo, settings)
    except Exception:
        # El siguiente disparo reintenta; la ingesta es idempotente y repara huecos.
        log.exception("scheduled_ingestion_failed")


async def _watchdog_tick(
    repo: IngestionRepository,
    feature_repo: FeatureRepository,
    settings: Settings,
    lock: asyncio.Lock,
    now: datetime,
) -> None:
    latest = await repo.last_open_time(settings.btc_symbol, settings.default_interval)
    freshness = check_freshness(latest, now, interval_to_timedelta(settings.default_interval))
    if freshness.premature_candle:
        log.error("premature_candle_detected", latest=str(latest))
        return
    latest_feature = await feature_repo.last_feature_open_time(
        settings.btc_symbol, settings.default_interval
    )
    features_stale = latest is not None and (latest_feature is None or latest_feature < latest)
    latest_funding = None
    funding_stale = False
    if settings.funding_enabled:
        latest_funding = await FundingRepository(repo.engine).last_funding_time(settings.btc_symbol)
        funding_stale = funding_is_stale(
            latest_funding, now, timedelta(hours=settings.funding_max_lag_hours)
        )
    if freshness.stale or features_stale or funding_stale:
        log.warning(
            "ingestion_recovery_needed",
            latest=str(latest),
            expected=str(freshness.expected_open_time),
            missing_recent=freshness.missing_recent,
            latest_feature=str(latest_feature),
            latest_funding=str(latest_funding),
            funding_stale=funding_stale,
        )
        await _safe_run(repo, settings, lock)


async def _watchdog(
    repo: IngestionRepository,
    settings: Settings,
    lock: asyncio.Lock,
    scheduler: AsyncIOScheduler,
    stop: asyncio.Event,
) -> None:
    feature_repo = FeatureRepository(repo.engine)
    while not stop.is_set():
        try:
            scheduler.wakeup()
            await _watchdog_tick(repo, feature_repo, settings, lock, datetime.now(UTC))
        except Exception:
            log.exception("ingestion_watchdog_failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.ingestion_watchdog_seconds)
        except TimeoutError:
            continue


async def serve(repo: IngestionRepository, settings: Settings) -> None:
    """Ingesta inicial (backfill o catch-up) y luego actualizaciones periódicas hasta SIGTERM."""
    lock = asyncio.Lock()
    await _safe_run(repo, settings, lock)

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        _safe_run,
        build_trigger(settings.default_interval),
        args=[repo, settings, lock],
        id="ingestion",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=600,
    )
    scheduler.start()
    log.info("scheduler_started", interval=settings.default_interval)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    watchdog = asyncio.create_task(_watchdog(repo, settings, lock, scheduler, stop))
    try:
        await stop.wait()
    finally:
        watchdog.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog
        scheduler.shutdown(wait=False)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
