"""Tests de `candle_router` con `MarketDataPort` sobrescrito (sin DB real)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from httpx import ASGITransport

from src.config.dependencies import get_market_data_service
from src.domain.entities.candle import Candle, CandleWithChange
from src.domain.entities.pagination import Page, build_pagination
from src.domain.exceptions import NotFoundError
from src.domain.ports.market_data_port import MarketDataPort
from src.main import app

SYMBOL = "BTCUSDT"
INTERVAL = "4h"


def _candle(candle_id: int) -> Candle:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Candle(
        id=candle_id,
        symbol=SYMBOL,
        interval=INTERVAL,
        open_time=now,
        close_time=now + timedelta(hours=4),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=10.0,
    )


class FakeMarketDataService(MarketDataPort):
    def __init__(
        self,
        page: Page[Candle] | None = None,
        latest: CandleWithChange | None = None,
        raise_not_found: bool = False,
    ) -> None:
        self._page = page
        self._latest = latest
        self._raise_not_found = raise_not_found

    async def get_candles(self, symbol, interval, limit, offset):
        return self._page

    async def get_latest_candle(self, symbol, interval):
        if self._raise_not_found:
            msg = "no data"
            raise NotFoundError(msg)
        return self._latest


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_candles_returns_page():
    page = Page(data=[_candle(1)], pagination=build_pagination(100, 0, 1, 1))
    app.dependency_overrides[get_market_data_service] = lambda: FakeMarketDataService(page=page)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/candles")

    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total"] == 1
    assert body["data"][0]["id"] == 1


@pytest.mark.asyncio
async def test_get_latest_candle_returns_payload():
    latest = CandleWithChange(**_candle(2).model_dump(), change_pct=5.0)
    app.dependency_overrides[get_market_data_service] = lambda: FakeMarketDataService(
        latest=latest
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/candles/latest")

    assert response.status_code == 200
    assert response.json()["change_pct"] == 5.0


@pytest.mark.asyncio
async def test_get_latest_candle_returns_404_when_not_found():
    app.dependency_overrides[get_market_data_service] = lambda: FakeMarketDataService(
        raise_not_found=True
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/candles/latest")

    assert response.status_code == 404
