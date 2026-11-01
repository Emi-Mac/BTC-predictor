"""Tests de `feature_router` con `FeaturePort` sobrescrito (sin DB real)."""
from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from httpx import ASGITransport

from src.config.dependencies import get_feature_service
from src.domain.entities.feature import FeatureSnapshot
from src.domain.exceptions import NotFoundError
from src.domain.ports.feature_port import FeaturePort
from src.main import app


class FakeFeatureService(FeaturePort):
    def __init__(self, snapshot=None, raise_not_found=False):
        self._snapshot = snapshot
        self._raise_not_found = raise_not_found

    async def get_latest_features(self, symbol, interval):
        if self._raise_not_found:
            msg = "no features"
            raise NotFoundError(msg)
        return self._snapshot


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_latest_features_returns_payload():
    snapshot = FeatureSnapshot(
        candle_id=1,
        symbol="BTCUSDT",
        interval="4h",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
        values={"rsi_14": 55.5},
    )
    app.dependency_overrides[get_feature_service] = lambda: FakeFeatureService(snapshot=snapshot)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/features/latest")

    assert response.status_code == 200
    assert response.json()["values"]["rsi_14"] == 55.5


@pytest.mark.asyncio
async def test_get_latest_features_returns_404_when_missing():
    app.dependency_overrides[get_feature_service] = lambda: FakeFeatureService(
        raise_not_found=True
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/features/latest")

    assert response.status_code == 404
