"""Tests de los routers de volatilidad y riesgo con puertos sustituidos (sin DB)."""
from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from httpx import ASGITransport

from src.config.dependencies import get_risk_service, get_volatility_service
from src.domain.entities.pagination import Page, build_pagination
from src.domain.entities.volatility import (
    PositionSize,
    PriceBand,
    PriceBands,
    VolatilityForecast,
    VolatilityHistoryPoint,
    VolatilityMetrics,
    WindowCoverage,
)
from src.domain.exceptions import InvalidParameterError, NotFoundError
from src.domain.ports.risk_port import RiskPort
from src.domain.ports.volatility_port import VolatilityPort
from src.main import app

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def forecast(horizon="7d"):
    band = PriceBand(low=90_000.0, high=110_000.0)
    return VolatilityForecast(
        horizon=horizon,
        as_of=NOW,
        candle_open_time=NOW,
        reference_price=100_000.0,
        vol_annual=0.5,
        vol_horizon_pct=6.9,
        regime="normal",
        bands=PriceBands(p68=band, p90=band, p95=band),
        dvol=0.42,
        realized_vol_7d=0.4,
        realized_vol_30d=0.45,
        model_version=3,
        fallback=None,
        validation_status="in_validation",
    )


class FakeVolatility(VolatilityPort):
    def __init__(self, missing=False):
        self.missing = missing
        self.calls = []

    async def get_current(self, horizon):
        self.calls.append(("current", horizon))
        if self.missing:
            raise NotFoundError("sin pronósticos")
        return forecast(horizon)

    async def get_history(self, horizon, limit, offset):
        self.calls.append(("history", horizon, limit, offset))
        point = VolatilityHistoryPoint(
            candle_open_time=NOW,
            as_of=NOW,
            forecast_vol_annual=0.5,
            realized_vol_annual=None,
            regime="normal",
            model_version=3,
            fallback=None,
            resolved=False,
            inside_68=None,
            inside_90=None,
            inside_95=None,
        )
        return Page(data=[point], pagination=build_pagination(limit, offset, 1, 1))

    async def get_metrics(self, horizon):
        self.calls.append(("metrics", horizon))
        window = WindowCoverage(
            weeks=12, resolved=0, coverage_68=None, coverage_90=None, coverage_95=None
        )
        return VolatilityMetrics(
            horizon=horizon,
            resolved=10,
            coverage_68=0.7,
            coverage_90=0.9,
            coverage_95=0.95,
            qlike=0.2,
            qlike_naive=0.4,
            mz_slope=0.8,
            last_12w=window,
            validation_status="in_validation",
        )


class FakeRisk(RiskPort):
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def position_size(self, capital, risk_pct, stop_k, horizon, side):
        self.calls.append((capital, risk_pct, stop_k, horizon, side))
        if self.error:
            raise self.error
        return PositionSize(
            horizon=horizon,
            side=side,
            reference_price=100_000.0,
            vol_horizon_pct=2.0,
            stop_distance_pct=4.0,
            stop_price=96_000.0,
            risk_usd=100.0,
            position_usd=2_500.0,
            position_btc=0.025,
            exposure_pct=25.0,
            regime="normal",
            warnings=[],
        )


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


async def get(path, **params):
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.get(path, params=params)


@pytest.mark.asyncio
async def test_current_defaults_to_7d_and_returns_the_forecast_contract():
    fake = FakeVolatility()
    app.dependency_overrides[get_volatility_service] = lambda: fake
    response = await get("/api/v1/volatility/current")
    assert response.status_code == 200
    body = response.json()
    assert fake.calls == [("current", "7d")]
    assert set(body) >= {
        "horizon", "as_of", "reference_price", "vol_annual", "vol_horizon_pct", "regime",
        "bands", "dvol", "realized_vol_7d", "realized_vol_30d", "model_version", "fallback",
        "validation_status",
    }
    assert set(body["bands"]) == {"p68", "p90", "p95"}
    assert set(body["bands"]["p68"]) == {"low", "high"}
    one_day = await get("/api/v1/volatility/current", horizon="1d")
    assert one_day.json()["horizon"] == "1d"


@pytest.mark.asyncio
async def test_invalid_horizon_is_rejected_and_missing_forecast_is_404():
    app.dependency_overrides[get_volatility_service] = lambda: FakeVolatility()
    assert (await get("/api/v1/volatility/current", horizon="30d")).status_code == 422
    app.dependency_overrides[get_volatility_service] = lambda: FakeVolatility(missing=True)
    response = await get("/api/v1/volatility/current")
    assert response.status_code == 404
    assert response.json() == {"detail": "sin pronósticos"}


@pytest.mark.asyncio
async def test_history_is_paginated_and_validates_bounds():
    fake = FakeVolatility()
    app.dependency_overrides[get_volatility_service] = lambda: fake
    response = await get("/api/v1/volatility/history", horizon="1d", limit=10, offset=20)
    assert response.status_code == 200
    assert fake.calls == [("history", "1d", 10, 20)]
    assert response.json()["pagination"] == {
        "limit": 10, "offset": 20, "total": 1, "has_more": False,
    }
    for params in ({"limit": 0}, {"limit": 501}, {"offset": -1}):
        assert (await get("/api/v1/volatility/history", **params)).status_code == 422


@pytest.mark.asyncio
async def test_volatility_metrics_endpoint():
    fake = FakeVolatility()
    app.dependency_overrides[get_volatility_service] = lambda: fake
    response = await get("/api/v1/metrics/volatility", horizon="7d")
    assert response.status_code == 200
    body = response.json()
    assert body["coverage_95"] == 0.95 and body["last_12w"]["weeks"] == 12
    assert body["qlike"] < body["qlike_naive"]


@pytest.mark.asyncio
async def test_position_size_passes_parameters_and_applies_defaults():
    fake = FakeRisk()
    app.dependency_overrides[get_risk_service] = lambda: fake
    response = await get("/api/v1/risk/position-size", capital=10_000, risk_pct=1)
    assert response.status_code == 200
    assert fake.calls == [(10_000.0, 1.0, 2.0, "1d", "long")]
    body = response.json()
    assert body["position_usd"] == 2_500.0 and body["warnings"] == []
    await get(
        "/api/v1/risk/position-size",
        capital=500, risk_pct=0.5, stop_k=3, horizon="7d", side="short",
    )
    assert fake.calls[-1] == (500.0, 0.5, 3.0, "7d", "short")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"capital": 0, "risk_pct": 1},
        {"capital": -1, "risk_pct": 1},
        {"capital": 1000, "risk_pct": 0},
        {"capital": 1000, "risk_pct": 6},
        {"capital": 1000, "risk_pct": 1, "stop_k": 0},
        {"capital": 1000, "risk_pct": 1, "stop_k": 11},
        {"capital": 1000, "risk_pct": 1, "horizon": "3d"},
        {"capital": 1000, "risk_pct": 1, "side": "flat"},
        {"risk_pct": 1},
        {"capital": 1000},
    ],
)
async def test_position_size_rejects_invalid_query_parameters(params):
    fake = FakeRisk()
    app.dependency_overrides[get_risk_service] = lambda: fake
    assert (await get("/api/v1/risk/position-size", **params)).status_code == 422
    assert fake.calls == []


@pytest.mark.asyncio
async def test_domain_errors_from_the_risk_service_map_to_422_and_404():
    error = InvalidParameterError("stop nulo")
    app.dependency_overrides[get_risk_service] = lambda: FakeRisk(error)
    response = await get("/api/v1/risk/position-size", capital=1000, risk_pct=1)
    assert (response.status_code, response.json()) == (422, {"detail": "stop nulo"})
    app.dependency_overrides[get_risk_service] = lambda: FakeRisk(NotFoundError("sin pronóstico"))
    assert (await get("/api/v1/risk/position-size", capital=1000, risk_pct=1)).status_code == 404


@pytest.mark.asyncio
async def test_endpoints_are_hidden_while_volatility_is_disabled():
    # Sin overrides se usa la dependencia real: con VOLATILITY_ENABLED=false responde 404.
    for path, params in (
        ("/api/v1/volatility/current", {}),
        ("/api/v1/volatility/history", {}),
        ("/api/v1/metrics/volatility", {}),
        ("/api/v1/risk/position-size", {"capital": 1000, "risk_pct": 1}),
    ):
        response = await get(path, **params)
        assert response.status_code == 404
        assert "VOLATILITY_ENABLED" in response.json()["detail"]
