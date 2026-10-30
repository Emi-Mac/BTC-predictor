"""Tests de los health checks de `main.py` (`/health`, `/readiness`)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import ASGITransport

from src import main


@pytest.mark.asyncio
async def test_health_endpoint_always_returns_ok():
    transport = ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "btc-predictor-api"}


@pytest.mark.asyncio
async def test_readiness_endpoint_returns_200_when_db_is_reachable(monkeypatch):
    monkeypatch.setattr(main, "check_db_connection", AsyncMock(return_value=True))
    transport = ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/readiness")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": True}


@pytest.mark.asyncio
async def test_readiness_endpoint_returns_503_when_db_is_unreachable(monkeypatch):
    monkeypatch.setattr(main, "check_db_connection", AsyncMock(return_value=False))
    transport = ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/readiness")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "database": False}
