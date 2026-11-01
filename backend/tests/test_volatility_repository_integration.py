"""Repositorio de volatilidad contra PostgreSQL real (requiere TEST_DATABASE_URL)."""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.schema import CreateSchema, DropSchema

from src.adapters.outbound.prediction_repository import PostgresPredictionRepository
from src.adapters.outbound.volatility_repository import PostgresVolatilityRepository
from src.application.risk_service import RiskService
from src.application.volatility_service import VolatilityService
from src.config.models import (
    Base,
    ImpliedVolatilityORM,
    ModelRegistryORM,
    OhlcvCandleORM,
    PredictionLogORM,
    VolatilityForecastORM,
    VolatilityModelORM,
)
from src.domain.volatility_math import realized_variance

SYMBOL, INTERVAL, STEP = "BTCUSDT", "4h", timedelta(hours=4)
START = datetime(2026, 9, 1, tzinfo=UTC)
CANDLES = 120


@pytest.fixture
async def schema_engine(test_database_url):
    url = test_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    schema = f"backend_vol_{uuid4().hex}"
    admin = create_async_engine(url)
    async with admin.begin() as conn:
        await conn.execute(CreateSchema(schema))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield engine
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(DropSchema(schema, cascade=True))
        await admin.dispose()


@pytest.fixture
def sessionmaker(schema_engine):
    return async_sessionmaker(bind=schema_engine, expire_on_commit=False)


@pytest.fixture
def repository(sessionmaker):
    return PostgresVolatilityRepository(sessionmaker, SYMBOL, INTERVAL, STEP)


def price(i: int) -> float:
    return 50_000 * math.exp(0.01 * math.sin(i / 3) + 0.0005 * i)


async def seed(sessionmaker):
    """Velas, dos modelos por horizonte y pronósticos (algunos resueltos)."""
    async with sessionmaker() as session:
        candles = [
            OhlcvCandleORM(
                symbol=SYMBOL, interval=INTERVAL, open_time=START + i * STEP,
                close_time=START + (i + 1) * STEP, open=price(i), high=price(i) + 1,
                low=price(i) - 1, close=price(i), volume=1.0,
            )
            for i in range(CANDLES)
        ]
        session.add_all(candles)
        models = {}
        for horizon in (6, 42):
            for role, kind in (("primary", "har_x_live"), ("fallback", "har")):
                model = VolatilityModelORM(
                    kind=kind, role=role, horizon_candles=horizon, artifact_path="/m.json",
                    artifact_sha256="a" * 64, feature_names=["x"], validation_metrics={},
                    is_active=True, trained_at=START,
                )
                session.add(model)
                models[(horizon, role)] = model
        await session.flush()
        for i in (0, 1, 2, 3):
            hour = START + timedelta(hours=i)
            session.add(ImpliedVolatilityORM(currency="BTC", open_time=hour, close=40 + i))
        await session.commit()
        return [c.id for c in candles], {k: m.version for k, m in models.items()}


def forecast_row(candle_id, versions, horizon=6, fallback=False, **changes):
    primary = versions[(horizon, "primary")]
    served = versions[(horizon, "fallback" if fallback else "primary")]
    values = {
        "candle_id": candle_id, "horizon_candles": horizon, "model_version": served,
        "requested_model_version": primary,
        "fallback_reason": "dvol_stale" if fallback else None, "features_as_of": {},
        "forecast_var_annual": 0.25, "forecast_vol_annual": 0.5, "k68": 0.87, "k90": 1.87,
        "k95": 2.68, "calibration_window_end": START, "regime": "normal",
        "reference_price": 50_000,
    }
    return VolatilityForecastORM(**{**values, **changes})


@pytest.mark.asyncio
async def test_latest_and_history_are_ordered_by_candle_per_horizon(sessionmaker, repository):
    ids, versions = await seed(sessionmaker)
    async with sessionmaker() as session:
        session.add_all(forecast_row(ids[i], versions) for i in (10, 12, 11))
        session.add(forecast_row(ids[50], versions, horizon=42))
        await session.commit()
    latest = await repository.get_latest(6)
    assert latest.candle_open_time == START + 12 * STEP
    assert latest.forecast_vol_annual == 0.5 and latest.k95 == pytest.approx(2.68)
    assert latest.fallback_reason is None and latest.resolved_at is None
    assert (await repository.get_latest(42)).candle_open_time == START + 50 * STEP
    page, total = await repository.list_history(6, limit=2, offset=0)
    assert total == 3
    assert [f.candle_open_time for f in page] == [START + 12 * STEP, START + 11 * STEP]
    assert [f.candle_open_time for f in (await repository.list_history(6, 2, 2))[0]] == [
        START + 10 * STEP
    ]
    only_7d, total_7d = await repository.list_history(42, 10, 0)
    assert total_7d == 1 and len(only_7d) == 1


@pytest.mark.asyncio
async def test_empty_repository_returns_nothing(repository):
    assert await repository.get_latest(6) is None
    assert await repository.list_history(6, 10, 0) == ([], 0)
    assert await repository.list_resolved(6) == []
    assert await repository.get_dvol(START) is None
    assert await repository.recent_closes(START, 5) == []


@pytest.mark.asyncio
async def test_resolved_forecasts_carry_the_naive_variance_of_the_previous_window(
    sessionmaker, repository
):
    ids, versions = await seed(sessionmaker)
    async with sessionmaker() as session:
        session.add(
            forecast_row(
                ids[20], versions, realized_var_annual=0.2, realized_log_return=0.01,
                inside_68=True, inside_90=False, inside_95=None, qlike=0.1, resolved_at=START,
            )
        )
        session.add(forecast_row(ids[3], versions, realized_var_annual=0.3, realized_log_return=0.0,
                                 qlike=0.2, resolved_at=START))  # sin 6 velas previas
        session.add(forecast_row(ids[21], versions))  # sin resolver
        await session.commit()
    rows = await repository.list_resolved(6)
    assert [r.candle_open_time for r in rows] == [START + 3 * STEP, START + 20 * STEP]
    first, second = rows
    assert first.naive_var_annual is None
    expected = realized_variance([price(i) for i in range(14, 21)])
    assert second.naive_var_annual == pytest.approx(expected)
    assert second.realized_var_annual == 0.2 and second.qlike == pytest.approx(0.1)
    assert (second.inside_68, second.inside_90, second.inside_95) == (True, False, None)


@pytest.mark.asyncio
async def test_dvol_is_the_latest_available_hour_and_closes_are_contiguous(
    sessionmaker, repository
):
    await seed(sessionmaker)
    assert await repository.get_dvol(START + timedelta(hours=2)) == 42.0
    assert await repository.get_dvol(START + timedelta(hours=10)) == 43.0
    assert await repository.get_dvol(START - timedelta(hours=1)) is None
    closes = await repository.recent_closes(START + 9 * STEP, 4)
    assert closes == pytest.approx([price(i) for i in range(6, 10)])
    assert len(await repository.recent_closes(START + 2 * STEP, 50)) == 3


@pytest.mark.asyncio
async def test_services_work_end_to_end_over_postgres(sessionmaker, repository):
    ids, versions = await seed(sessionmaker)
    async with sessionmaker() as session:
        session.add(forecast_row(ids[119], versions, horizon=42, k68=None, k90=None, k95=None,
                                 calibration_window_end=None))
        session.add(forecast_row(ids[110], versions, horizon=42, realized_var_annual=0.2,
                                 realized_log_return=0.01, inside_68=True, inside_90=True,
                                 inside_95=True, qlike=0.1, resolved_at=START))
        await session.commit()
    service = VolatilityService(repository, clock=lambda: START + 120 * STEP)
    current = await service.get_current("7d")
    assert current.bands is None  # sin calibración no hay bandas
    assert current.reference_price == 50_000 and current.dvol == pytest.approx(0.43)
    assert current.realized_vol_7d is not None
    metrics = await service.get_metrics("7d")
    assert metrics.resolved == 1 and metrics.coverage_95 == 1.0
    history = await service.get_history("7d", 10, 0)
    assert [p.resolved for p in history.data] == [False, True]
    position = await RiskService(service).position_size(10_000, 1.0, 2.0, "7d", "long")
    assert position.risk_usd == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_database_keeps_the_forecast_contract(sessionmaker):
    ids, versions = await seed(sessionmaker)

    async def insert(**changes):
        async with sessionmaker() as session:
            session.add(forecast_row(ids[5], versions, **changes))
            await session.commit()

    await insert()
    for changes in (
        {},  # duplicado (vela, horizonte)
        {
            "model_version": versions[(42, "primary")],
            "requested_model_version": versions[(42, "primary")],
        },
        {"fallback": True, "fallback_reason": None},
        {"regime": "wild"},
    ):
        with pytest.raises(IntegrityError):
            await insert(**changes)


@pytest.mark.asyncio
async def test_majority_direction_share_is_the_trivial_baseline(sessionmaker):
    repo = PostgresPredictionRepository(sessionmaker)
    assert await repo.majority_direction_share() is None
    ids, _ = await seed(sessionmaker)
    async with sessionmaker() as session:
        model = ModelRegistryORM(model_type="xgboost", hyperparameters={}, trained_at=START)
        session.add(model)
        await session.flush()
        for index, actual in enumerate(["UP", "UP", "UP", "DOWN", None]):
            session.add(
                PredictionLogORM(
                    candle_id=ids[index], model_version=model.version, predicted_direction="UP",
                    confidence_score=0.6, raw_probability_up=0.6, actual_direction=actual,
                    is_correct=None if actual is None else actual == "UP",
                    resolved_at=None if actual is None else START,
                )
            )
        await session.commit()
    assert await repo.majority_direction_share() == pytest.approx(0.75)
