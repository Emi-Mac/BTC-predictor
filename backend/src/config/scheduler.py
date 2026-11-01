"""Scheduler de predicciones integrado en el lifespan de FastAPI (task 4.10).

A diferencia del scheduler del `ml-worker` (que gestiona ingesta + features), este job
sólo orquesta la inferencia: toma la vela más reciente (ya con features calculadas por
el `ml-worker`), emite una predicción si falta y resuelve la predicción de la vela
anterior. También difunde la nueva predicción por WebSocket (`/ws/live`).
"""
from __future__ import annotations

from datetime import UTC, datetime

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.adapters.inbound.websocket_router import manager
from src.application.prediction_cycle import run_prediction_cycle
from src.config.dependencies import (
    get_candle_repository,
    get_feature_repository,
    get_model_inference_adapter,
    get_model_registry_repository,
    get_prediction_repository,
)
from src.config.settings import get_settings
from src.domain.interval import interval_to_timedelta

log = structlog.get_logger(__name__)

JOB_ID = "prediction_cycle"


async def run_prediction_tick() -> None:
    """Ejecuta un ciclo de predicción y difunde el resultado por WebSocket."""
    settings = get_settings()
    prediction_repo = get_prediction_repository()
    try:
        await run_prediction_cycle(
            candle_repo=get_candle_repository(),
            feature_repo=get_feature_repository(),
            model_registry=get_model_registry_repository(),
            inference=get_model_inference_adapter(),
            prediction_repo=prediction_repo,
            symbol=settings.btc_symbol,
            interval=settings.default_interval,
        )
        latest = await prediction_repo.get_latest()
        if latest is not None:
            await manager.broadcast(
                {
                    "type": "prediction",
                    "data": {
                        "direction": latest.direction,
                        "confidence": latest.confidence,
                        "candle_open_time": latest.candle_open_time.isoformat(),
                    },
                }
            )
    except Exception:
        # Un fallo aislado no debe tumbar el scheduler; el siguiente tick reintenta.
        log.exception("prediction_tick_failed")


def create_scheduler() -> AsyncIOScheduler:
    """Crea (sin iniciar) el scheduler con el job de predicción periódico."""
    scheduler = AsyncIOScheduler(timezone="UTC")
    settings = get_settings()
    period_seconds = int(interval_to_timedelta(settings.default_interval).total_seconds())
    poll_seconds = min(settings.prediction_poll_seconds, period_seconds)
    scheduler.add_job(
        run_prediction_tick,
        IntervalTrigger(seconds=poll_seconds),
        id=JOB_ID,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=600,
        next_run_time=datetime.now(UTC),
    )
    return scheduler
