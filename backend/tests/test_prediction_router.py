"""Tests de `prediction_router` con `PredictionPort` sobrescrito (sin DB real)."""
from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from httpx import ASGITransport

from src.config.dependencies import get_prediction_service
from src.domain.entities.pagination import Page, build_pagination
from src.domain.entities.prediction import Prediction
from src.domain.exceptions import NotFoundError
from src.domain.ports.prediction_port import PredictionPort
from src.main import app


def _prediction(prediction_id: int) -> Prediction:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Prediction(
        id=prediction_id,
        candle_id=prediction_id,
        candle_open_time=now,
        model_version=1,
        direction="UP",
        confidence=0.65,
        probability_up=0.65,
        predicted_at=now,
    )


class FakePredictionService(PredictionPort):
    def __init__(self, current=None, page=None, raise_not_found=False):
        self._current = current
        self._page = page
        self._raise_not_found = raise_not_found

    async def get_current(self):
        if self._raise_not_found:
            msg = "no predictions"
            raise NotFoundError(msg)
        return self._current

    async def get_history(self, limit, offset):
        return self._page


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_current_prediction_returns_payload():
    app.dependency_overrides[get_prediction_service] = lambda: FakePredictionService(
        current=_prediction(1)
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/predictions/current")

    assert response.status_code == 200
    assert response.json()["direction"] == "UP"


@pytest.mark.asyncio
async def test_get_current_prediction_returns_404_when_empty():
    app.dependency_overrides[get_prediction_service] = lambda: FakePredictionService(
        raise_not_found=True
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/predictions/current")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_prediction_history_returns_page():
    page = Page(data=[_prediction(1), _prediction(2)], pagination=build_pagination(50, 0, 2, 2))
    app.dependency_overrides[get_prediction_service] = lambda: FakePredictionService(page=page)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/predictions/history")

    assert response.status_code == 200
    assert len(response.json()["data"]) == 2
