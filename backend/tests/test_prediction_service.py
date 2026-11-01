"""Tests de `PredictionService` usando un `PredictionRepositoryPort` en memoria."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.application.prediction_service import PredictionService
from src.domain.entities.metrics import MonthlyAccuracy
from src.domain.entities.prediction import Direction, Prediction
from src.domain.exceptions import NotFoundError
from src.domain.ports.prediction_repository_port import PredictionRepositoryPort


def _prediction(prediction_id: int, predicted_at: datetime) -> Prediction:
    return Prediction(
        id=prediction_id,
        candle_id=prediction_id,
        candle_open_time=predicted_at,
        model_version=1,
        direction="UP",
        confidence=0.6,
        probability_up=0.6,
        predicted_at=predicted_at,
    )


class FakePredictionRepository(PredictionRepositoryPort):
    def __init__(self, predictions: list[Prediction]) -> None:
        # Más reciente primero.
        self._predictions = predictions

    async def create(
        self,
        candle_id: int,
        model_version: int,
        direction: Direction,
        confidence: float,
        probability_up: float,
    ) -> Prediction:
        raise NotImplementedError

    async def exists_for_candle(self, candle_id: int, model_version: int) -> bool:
        raise NotImplementedError

    async def get_latest(self) -> Prediction | None:
        return self._predictions[0] if self._predictions else None

    async def list_history(self, limit: int, offset: int) -> tuple[list[Prediction], int]:
        page = self._predictions[offset : offset + limit]
        return page, len(self._predictions)

    async def resolve_for_candle(self, candle_id, actual_direction, resolved_at) -> int:
        raise NotImplementedError

    async def accuracy_since(self, since: datetime | None) -> tuple[int, int]:
        raise NotImplementedError

    async def accuracy_by_month(self) -> list[MonthlyAccuracy]:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_get_current_returns_latest_prediction():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    service = PredictionService(FakePredictionRepository([_prediction(1, now)]))

    result = await service.get_current()

    assert result.id == 1


@pytest.mark.asyncio
async def test_get_current_raises_not_found_when_empty():
    service = PredictionService(FakePredictionRepository([]))

    with pytest.raises(NotFoundError):
        await service.get_current()


@pytest.mark.asyncio
async def test_get_history_returns_paginated_page():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    predictions = [_prediction(i, now + timedelta(hours=i)) for i in range(3)]
    service = PredictionService(FakePredictionRepository(predictions))

    page = await service.get_history(limit=2, offset=0)

    assert len(page.data) == 2
    assert page.pagination.total == 3
    assert page.pagination.has_more is True
