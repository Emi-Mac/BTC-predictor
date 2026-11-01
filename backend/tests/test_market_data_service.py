"""Tests de `MarketDataService` usando un `CandleRepositoryPort` en memoria."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.application.market_data_service import MarketDataService
from src.domain.entities.candle import Candle
from src.domain.exceptions import NotFoundError
from src.domain.ports.candle_repository_port import CandleRepositoryPort

SYMBOL = "BTCUSDT"
INTERVAL = "4h"


def _candle(candle_id: int, open_time: datetime, close: float) -> Candle:
    return Candle(
        id=candle_id,
        symbol=SYMBOL,
        interval=INTERVAL,
        open_time=open_time,
        close_time=open_time + timedelta(hours=4),
        open=close - 1,
        high=close + 5,
        low=close - 5,
        close=close,
        volume=100.0,
        quote_volume=4_000_000.0,
        num_trades=1000,
    )


class FakeCandleRepository(CandleRepositoryPort):
    def __init__(self, candles: list[Candle]) -> None:
        # Más reciente primero, como lo devolvería el repositorio real.
        self._candles = candles

    async def list_candles(
        self, symbol: str, interval: str, limit: int, offset: int
    ) -> tuple[list[Candle], int]:
        page = self._candles[offset : offset + limit]
        return page, len(self._candles)

    async def get_latest(self, symbol: str, interval: str) -> Candle | None:
        return self._candles[0] if self._candles else None

    async def get_by_open_time(self, symbol, interval, open_time):
        for candle in self._candles:
            if candle.open_time == open_time:
                return candle
        return None

    async def get_by_id(self, candle_id: int) -> Candle | None:
        for candle in self._candles:
            if candle.id == candle_id:
                return candle
        return None


@pytest.mark.asyncio
async def test_get_candles_returns_page_with_pagination_metadata():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [_candle(i, base + timedelta(hours=4 * i), 100.0 + i) for i in range(5)]
    service = MarketDataService(FakeCandleRepository(list(reversed(candles))))

    page = await service.get_candles(SYMBOL, INTERVAL, limit=2, offset=0)

    assert len(page.data) == 2
    assert page.pagination.total == 5
    assert page.pagination.has_more is True


@pytest.mark.asyncio
async def test_get_latest_candle_computes_change_pct():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    previous = _candle(1, base, 100.0)
    latest = _candle(2, base + timedelta(hours=4), 110.0)
    service = MarketDataService(FakeCandleRepository([latest, previous]))

    result = await service.get_latest_candle(SYMBOL, INTERVAL)

    assert result.change_pct == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_get_latest_candle_change_pct_is_none_without_previous_candle():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    latest = _candle(1, base, 100.0)
    service = MarketDataService(FakeCandleRepository([latest]))

    result = await service.get_latest_candle(SYMBOL, INTERVAL)

    assert result.change_pct is None


@pytest.mark.asyncio
async def test_get_latest_candle_raises_not_found_when_empty():
    service = MarketDataService(FakeCandleRepository([]))

    with pytest.raises(NotFoundError):
        await service.get_latest_candle(SYMBOL, INTERVAL)
