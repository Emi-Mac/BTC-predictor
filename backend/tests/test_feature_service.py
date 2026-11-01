"""Tests de `FeatureService` usando un `FeatureRepositoryPort` en memoria."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.application.feature_service import FeatureService
from src.domain.entities.feature import FeatureSnapshot
from src.domain.exceptions import NotFoundError
from src.domain.ports.feature_repository_port import FeatureRepositoryPort

SYMBOL = "BTCUSDT"
INTERVAL = "4h"


class FakeFeatureRepository(FeatureRepositoryPort):
    def __init__(self, snapshot: FeatureSnapshot | None) -> None:
        self._snapshot = snapshot

    async def get_latest(self, symbol: str, interval: str) -> FeatureSnapshot | None:
        return self._snapshot

    async def get_vector_for_candle(self, candle_id, feature_names):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_get_latest_features_returns_snapshot_when_present():
    snapshot = FeatureSnapshot(
        candle_id=1,
        symbol=SYMBOL,
        interval=INTERVAL,
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
        values={"rsi_14": 55.5},
    )
    service = FeatureService(FakeFeatureRepository(snapshot))

    result = await service.get_latest_features(SYMBOL, INTERVAL)

    assert result.values == {"rsi_14": 55.5}


@pytest.mark.asyncio
async def test_get_latest_features_raises_not_found_when_missing():
    service = FeatureService(FakeFeatureRepository(None))

    with pytest.raises(NotFoundError):
        await service.get_latest_features(SYMBOL, INTERVAL)
