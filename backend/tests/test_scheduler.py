"""Tests de `config.scheduler`: construcción del job y orquestación del tick."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from apscheduler.triggers.interval import IntervalTrigger

from src.config import scheduler as scheduler_module
from src.domain.entities.prediction import Prediction


def test_create_scheduler_registers_prediction_job_with_interval_trigger():
    # `create_scheduler()` sólo construye y registra el job; no arranca el scheduler
    # (eso ocurre en el lifespan de FastAPI), por lo que no hace falta (ni es válido)
    # llamar a `shutdown()` aquí: APScheduler lanza `SchedulerNotRunningError` si se
    # apaga un scheduler que nunca se inició.
    scheduler = scheduler_module.create_scheduler()
    job = scheduler.get_job(scheduler_module.JOB_ID)
    assert job is not None
    assert isinstance(job.trigger, IntervalTrigger)
    # Sondeo corto (no una vez por vela) para no perder velas si las features llegan tarde.
    expected = min(scheduler_module.get_settings().prediction_poll_seconds, 4 * 3600)
    assert job.trigger.interval.total_seconds() == expected


def _patch_dependency_getters(monkeypatch, prediction_repo) -> None:
    """Sustituye los getters de `config.dependencies` usados por `run_prediction_tick`."""
    monkeypatch.setattr(scheduler_module, "get_prediction_repository", lambda: prediction_repo)
    monkeypatch.setattr(scheduler_module, "get_candle_repository", lambda: object())
    monkeypatch.setattr(scheduler_module, "get_feature_repository", lambda: object())
    monkeypatch.setattr(scheduler_module, "get_model_registry_repository", lambda: object())
    monkeypatch.setattr(scheduler_module, "get_model_inference_adapter", lambda: object())


@pytest.mark.asyncio
async def test_run_prediction_tick_broadcasts_the_latest_prediction(monkeypatch):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    prediction = Prediction(
        id=1,
        candle_id=1,
        candle_open_time=now,
        model_version=1,
        direction="UP",
        confidence=0.6,
        probability_up=0.6,
        predicted_at=now,
    )

    class FakePredictionRepo:
        async def get_latest(self):
            return prediction

    broadcasts: list[dict] = []

    async def fake_broadcast(message):
        broadcasts.append(message)

    monkeypatch.setattr(scheduler_module, "run_prediction_cycle", AsyncMock())
    _patch_dependency_getters(monkeypatch, FakePredictionRepo())
    monkeypatch.setattr(scheduler_module.manager, "broadcast", fake_broadcast)

    await scheduler_module.run_prediction_tick()

    assert len(broadcasts) == 1
    assert broadcasts[0]["type"] == "prediction"
    assert broadcasts[0]["data"]["direction"] == "UP"


@pytest.mark.asyncio
async def test_run_prediction_tick_does_not_broadcast_when_no_prediction_exists(monkeypatch):
    class FakePredictionRepo:
        async def get_latest(self):
            return None

    broadcasts: list[dict] = []

    async def fake_broadcast(message):
        broadcasts.append(message)

    monkeypatch.setattr(scheduler_module, "run_prediction_cycle", AsyncMock())
    _patch_dependency_getters(monkeypatch, FakePredictionRepo())
    monkeypatch.setattr(scheduler_module.manager, "broadcast", fake_broadcast)

    await scheduler_module.run_prediction_tick()

    assert broadcasts == []


@pytest.mark.asyncio
async def test_run_prediction_tick_swallows_exceptions_so_the_scheduler_keeps_running(monkeypatch):
    async def failing_cycle(**kwargs):
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(scheduler_module, "run_prediction_cycle", failing_cycle)
    _patch_dependency_getters(monkeypatch, object())

    # No debe propagar la excepción: el siguiente tick programado debe poder reintentar.
    await scheduler_module.run_prediction_tick()
