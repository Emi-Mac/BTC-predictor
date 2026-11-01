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
    get_volatility_service,
)
from src.config.settings import get_settings
from src.domain.entities.volatility import Horizon
from src.domain.exceptions import NotFoundError
from src.domain.interval import interval_to_timedelta

log = structlog.get_logger(__name__)

JOB_ID = "prediction_cycle"
VOLATILITY_JOB_ID = "volatility_broadcast"
HORIZONS: tuple[Horizon, ...] = ("1d", "7d")
# Último pronóstico difundido por horizonte: evita repetir el mismo mensaje en cada sondeo.
_last_broadcast: dict[str, tuple[datetime, int]] = {}


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


async def run_volatility_tick() -> None:
    """Difunde por WebSocket cada pronóstico de volatilidad nuevo (uno por horizonte)."""
    try:
        service = get_volatility_service()
        for horizon in HORIZONS:
            try:
                forecast = await service.get_current(horizon)
            except NotFoundError:
                continue  # el ML worker aún no emite pronósticos de este horizonte
            marker = (forecast.candle_open_time, forecast.model_version)
            if _last_broadcast.get(horizon) == marker:
                continue
            await manager.broadcast(
                {
                    "type": "volatility",
                    "data": {
                        "horizon": horizon,
                        "vol_annual": forecast.vol_annual,
                        "regime": forecast.regime,
                        "bands": forecast.bands.model_dump() if forecast.bands else None,
                    },
                }
            )
            _last_broadcast[horizon] = marker
    except NotFoundError:
        return  # volatilidad deshabilitada
    except Exception:
        log.exception("volatility_tick_failed")


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
    if settings.volatility_enabled:
        scheduler.add_job(
            run_volatility_tick,
            IntervalTrigger(seconds=poll_seconds),
            id=VOLATILITY_JOB_ID,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=600,
            next_run_time=datetime.now(UTC),
        )
    return scheduler
