"""Tests de `VolatilityService` y `RiskService` con un repositorio en memoria."""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from src.application.risk_service import RiskService
from src.application.volatility_service import VolatilityService
from src.domain.entities.volatility import ResolvedForecast, StoredForecast
from src.domain.exceptions import InvalidParameterError, NotFoundError
from src.domain.ports.volatility_repository_port import VolatilityRepositoryPort

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
OPEN = datetime(2026, 11, 1, 8, 0, tzinfo=UTC)
STEP = timedelta(hours=4)


def stored(horizon=42, **changes):
    values = {
        "id": 1,
        "candle_id": 1,
        "candle_open_time": OPEN,
        "horizon_candles": horizon,
        "model_version": 3,
        "requested_model_version": 3,
        "fallback_reason": None,
        "forecast_var_annual": 0.25,
        "forecast_vol_annual": 0.5,
        "k68": 0.87,
        "k90": 1.87,
        "k95": 2.68,
        "regime": "normal",
        "reference_price": 100_000.0,
    }
    return StoredForecast(**{**values, **changes})


def resolved(offset_hours=0, **changes):
    values = {
        "candle_open_time": NOW - timedelta(hours=offset_hours),
        "forecast_var_annual": 0.25,
        "realized_var_annual": 0.2,
        "qlike": 0.05,
        "naive_var_annual": 0.1,
        "inside_68": True,
        "inside_90": True,
        "inside_95": True,
    }
    return ResolvedForecast(**{**values, **changes})


class FakeRepository(VolatilityRepositoryPort):
    def __init__(self, latest=None, history=(), rows=(), dvol=42.0, closes=None):
        self.latest, self.history, self.rows, self.dvol = latest, list(history), list(rows), dvol
        self.closes = closes if closes is not None else [100.0 + (i % 3) for i in range(181)]
        self.calls = []

    async def get_latest(self, horizon_candles):
        self.calls.append(("latest", horizon_candles))
        return self.latest

    async def list_history(self, horizon_candles, limit, offset):
        return self.history[offset : offset + limit], len(self.history)

    async def list_resolved(self, horizon_candles):
        return self.rows

    async def get_dvol(self, usable_until):
        self.calls.append(("dvol", usable_until))
        return self.dvol

    async def recent_closes(self, until_open_time, count):
        self.calls.append(("closes", count))
        return self.closes


def service(repo, **kwargs):
    return VolatilityService(repo, clock=lambda: NOW, **kwargs)


@pytest.mark.asyncio
async def test_current_forecast_includes_bands_context_and_validation_status():
    repo = FakeRepository(latest=stored())
    forecast = await service(repo).get_current("7d")
    sigma = 0.5 * math.sqrt(42 / 2190)
    assert forecast.horizon == "7d"
    assert forecast.as_of == OPEN + STEP
    assert forecast.vol_annual == 0.5
    assert forecast.vol_horizon_pct == pytest.approx(sigma * 100)
    assert forecast.bands.p95.high == pytest.approx(100_000 * math.exp(2.68 * sigma))
    assert forecast.bands.p68.low == pytest.approx(100_000 * math.exp(-0.87 * sigma))
    assert forecast.bands.p68.high < forecast.bands.p90.high < forecast.bands.p95.high
    assert forecast.dvol == pytest.approx(0.42)
    assert forecast.realized_vol_7d is not None and forecast.realized_vol_30d is not None
    assert forecast.model_version == 3 and forecast.fallback is None
    assert forecast.validation_status == "in_validation"
    assert ("latest", 42) in repo.calls and ("dvol", OPEN + STEP - timedelta(hours=1)) in repo.calls


@pytest.mark.asyncio
async def test_horizon_selects_candles_and_validated_status_is_configurable():
    repo = FakeRepository(latest=stored(6))
    forecast = await service(repo, validation_status="validated").get_current("1d")
    assert ("latest", 6) in repo.calls
    assert forecast.validation_status == "validated"


@pytest.mark.asyncio
async def test_current_without_calibration_or_context_degrades_gracefully():
    repo = FakeRepository(
        latest=stored(k68=None, k90=None, k95=None, fallback_reason="dvol_stale", model_version=4),
        dvol=None,
        closes=[100.0, 101.0],
    )
    forecast = await service(repo).get_current("7d")
    assert forecast.bands is None
    assert forecast.dvol is None
    assert forecast.realized_vol_7d is None and forecast.realized_vol_30d is None
    assert forecast.fallback == "dvol_stale"


@pytest.mark.asyncio
async def test_current_raises_not_found_without_forecasts():
    with pytest.raises(NotFoundError, match="1d"):
        await service(FakeRepository()).get_current("1d")


@pytest.mark.asyncio
async def test_history_maps_realized_volatility_and_paginates():
    rows = [
        stored(
            id=i,
            candle_open_time=OPEN - i * STEP,
            realized_var_annual=0.16 if i % 2 == 0 else None,
            resolved_at=NOW if i % 2 == 0 else None,
            inside_68=True if i % 2 == 0 else None,
        )
        for i in range(5)
    ]
    page = await service(FakeRepository(history=rows)).get_history("7d", limit=2, offset=0)
    assert [p.resolved for p in page.data] == [True, False]
    assert page.data[0].realized_vol_annual == pytest.approx(0.4)
    assert page.data[1].realized_vol_annual is None
    assert page.data[0].as_of == OPEN + STEP
    assert page.pagination.total == 5 and page.pagination.has_more
    last = await service(FakeRepository(history=rows)).get_history("7d", limit=2, offset=4)
    assert len(last.data) == 1 and not last.pagination.has_more


@pytest.mark.asyncio
async def test_metrics_report_coverage_qlike_against_naive_and_recent_window():
    rows = [resolved(offset_hours=h, inside_95=h % 2 == 0) for h in range(40)]
    rows.append(resolved(offset_hours=24 * 200, inside_68=False, naive_var_annual=None))
    metrics = await service(FakeRepository(rows=rows)).get_metrics("7d")
    assert metrics.resolved == 41
    assert metrics.coverage_68 == pytest.approx(40 / 41)
    assert metrics.coverage_95 == pytest.approx(21 / 41)
    assert metrics.qlike == pytest.approx(0.05)
    assert metrics.qlike_naive > metrics.qlike  # el ingenuo (0.1 vs 0.2 realizada) pierde más
    assert metrics.mz_slope is None or math.isfinite(metrics.mz_slope)
    assert metrics.last_12w.weeks == 12
    assert metrics.last_12w.resolved == 40  # excluye la de hace 200 días
    assert metrics.last_12w.coverage_68 == 1.0
    assert metrics.validation_status == "in_validation"


@pytest.mark.asyncio
async def test_metrics_without_resolved_forecasts_are_empty_not_zero():
    metrics = await service(FakeRepository()).get_metrics("1d")
    assert metrics.resolved == 0
    for value in (metrics.coverage_68, metrics.coverage_90, metrics.coverage_95, metrics.qlike):
        assert value is None
    assert metrics.qlike_naive is None and metrics.mz_slope is None
    assert metrics.last_12w.resolved == 0 and metrics.last_12w.coverage_95 is None


@pytest.mark.asyncio
async def test_metrics_mincer_zarnowitz_needs_enough_samples():
    few = [resolved(offset_hours=h, realized_var_annual=0.1 + h / 100) for h in range(10)]
    assert (await service(FakeRepository(rows=few)).get_metrics("7d")).mz_slope is None
    many = [
        resolved(
            offset_hours=h,
            forecast_var_annual=0.1 + h / 100,
            realized_var_annual=0.1 + h / 100,
        )
        for h in range(40)
    ]
    metrics = await service(FakeRepository(rows=many)).get_metrics("7d")
    assert metrics.mz_slope == pytest.approx(1.0)


# ── Calculadora de tamaño de posición ────────────────────────────────────────────────────


def risk(**kwargs):
    repo = FakeRepository(latest=stored(**kwargs))
    return RiskService(service(repo))


@pytest.mark.asyncio
async def test_position_size_risks_the_requested_fraction_of_capital():
    result = await risk().position_size(10_000, 1.0, 2.0, "7d", "long")
    sigma = 0.5 * math.sqrt(42 / 2190)
    distance = 1 - math.exp(-2 * sigma)
    assert result.stop_price == pytest.approx(100_000 * math.exp(-2 * sigma))
    assert result.stop_distance_pct == pytest.approx(distance * 100)
    assert result.risk_usd == pytest.approx(100.0)
    assert result.position_usd == pytest.approx(100.0 / distance)
    assert result.position_btc == pytest.approx(result.position_usd / 100_000)
    assert result.exposure_pct == pytest.approx(result.position_usd / 10_000 * 100)
    # Si el stop se activa se pierde exactamente el riesgo elegido.
    loss = result.position_usd * (1 - result.stop_price / result.reference_price)
    assert loss == pytest.approx(result.risk_usd)
    assert "El pronóstico de volatilidad sigue en validación." in result.warnings


@pytest.mark.asyncio
async def test_short_positions_place_the_stop_above_the_price():
    result = await risk().position_size(10_000, 1.0, 2.0, "7d", "short")
    assert result.stop_price > result.reference_price
    loss = result.position_usd * (result.stop_price / result.reference_price - 1)
    assert loss == pytest.approx(result.risk_usd)


@pytest.mark.asyncio
async def test_position_warnings_cover_leverage_aggression_noise_regime_and_fallback():
    result = await RiskService(
        service(
            FakeRepository(
                latest=stored(regime="extreme", fallback_reason="dvol_stale", model_version=4)
            ),
            validation_status="validated",
        )
    ).position_size(1_000, 5.0, 0.5, "1d", "long")
    text = " | ".join(result.warnings)
    assert result.exposure_pct > 100
    for fragment in ("apalancamiento", "agresivo", "menos de 1σ", "extreme", "respaldo"):
        assert fragment in text
    assert "validación" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capital", "risk_pct", "stop_k"),
    [
        (0, 1.0, 2.0),
        (-5, 1.0, 2.0),
        (math.nan, 1.0, 2.0),
        (math.inf, 1.0, 2.0),
        (1000, 0, 2.0),
        (1000, 5.01, 2.0),
        (1000, math.nan, 2.0),
        (1000, 1.0, 0),
        (1000, 1.0, 10.5),
        (1000, 1.0, math.nan),
    ],
)
async def test_position_size_rejects_out_of_range_inputs(capital, risk_pct, stop_k):
    with pytest.raises(InvalidParameterError):
        await risk().position_size(capital, risk_pct, stop_k, "7d", "long")


@pytest.mark.asyncio
async def test_position_size_rejects_a_null_stop_and_propagates_missing_forecasts():
    with pytest.raises(InvalidParameterError, match="nulo"):
        await risk(forecast_vol_annual=1e-12).position_size(1000, 1.0, 1e-9, "7d", "long")
    with pytest.raises(NotFoundError):
        await RiskService(service(FakeRepository())).position_size(1000, 1.0, 2.0, "7d", "long")
