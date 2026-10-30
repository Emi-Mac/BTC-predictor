"""Tests de integración de los repositorios outbound contra un PostgreSQL real.

Se saltan automáticamente si `TEST_DATABASE_URL` no está definido (ver `conftest.py`),
siguiendo la misma convención que los tests de integración de `ml/tests`.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.schema import CreateSchema, DropSchema

from src.adapters.outbound.candle_repository import PostgresCandleRepository
from src.adapters.outbound.feature_repository import PostgresFeatureRepository
from src.adapters.outbound.model_registry_repository import PostgresModelRegistryRepository
from src.adapters.outbound.prediction_repository import PostgresPredictionRepository
from src.config.models import Base, ComputedFeatureORM, ModelRegistryORM, OhlcvCandleORM

SYMBOL = "BTCUSDT"
INTERVAL = "4h"


@pytest.fixture
async def schema_engine(test_database_url):
    url = test_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    schema = f"backend_test_{uuid4().hex}"
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


async def _insert_candle(sessionmaker, open_time: datetime, close: float) -> int:
    async with sessionmaker() as session:
        candle = OhlcvCandleORM(
            symbol=SYMBOL,
            interval=INTERVAL,
            open_time=open_time,
            close_time=open_time + timedelta(hours=4),
            open=close - 1,
            high=close + 5,
            low=close - 5,
            close=close,
            volume=100.0,
        )
        session.add(candle)
        await session.commit()
        await session.refresh(candle)
        return candle.id


async def _insert_model(sessionmaker, **overrides) -> int:
    async with sessionmaker() as session:
        model = ModelRegistryORM(
            model_type=overrides.pop("model_type", "xgboost"),
            hyperparameters=overrides.pop("hyperparameters", {}),
            is_active=overrides.pop("is_active", True),
            **overrides,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)
        return model.version


async def test_candle_repository_lists_most_recent_first_and_finds_latest(sessionmaker):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    await _insert_candle(sessionmaker, base, 100.0)
    await _insert_candle(sessionmaker, base + timedelta(hours=4), 110.0)

    repo = PostgresCandleRepository(sessionmaker)
    candles, total = await repo.list_candles(SYMBOL, INTERVAL, limit=10, offset=0)

    assert total == 2
    assert [c.close for c in candles] == [110.0, 100.0]

    latest = await repo.get_latest(SYMBOL, INTERVAL)
    assert latest.close == 110.0


async def test_feature_repository_returns_vector_in_requested_order(sessionmaker):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    candle_id = await _insert_candle(sessionmaker, base, 100.0)
    async with sessionmaker() as session:
        session.add(ComputedFeatureORM(candle_id=candle_id, rsi_14=55.5, macd=0.2))
        await session.commit()

    repo = PostgresFeatureRepository(sessionmaker)
    vector = await repo.get_vector_for_candle(candle_id, ["macd", "rsi_14"])

    assert vector == [0.2, 55.5]


async def test_feature_repository_returns_none_when_a_requested_feature_is_null(sessionmaker):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    candle_id = await _insert_candle(sessionmaker, base, 100.0)
    async with sessionmaker() as session:
        session.add(ComputedFeatureORM(candle_id=candle_id, rsi_14=55.5))  # macd queda NULL
        await session.commit()

    repo = PostgresFeatureRepository(sessionmaker)
    vector = await repo.get_vector_for_candle(candle_id, ["macd", "rsi_14"])

    assert vector is None


async def test_feature_repository_get_latest_excludes_metadata_columns(sessionmaker):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    candle_id = await _insert_candle(sessionmaker, base, 100.0)
    async with sessionmaker() as session:
        session.add(ComputedFeatureORM(candle_id=candle_id, rsi_14=55.5, target_label=1))
        await session.commit()

    repo = PostgresFeatureRepository(sessionmaker)
    snapshot = await repo.get_latest(SYMBOL, INTERVAL)

    assert snapshot.values["rsi_14"] == 55.5
    assert "target_label" not in snapshot.values
    assert "candle_id" not in snapshot.values
    assert "created_at" not in snapshot.values


async def test_prediction_repository_create_exists_and_resolve(sessionmaker):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    candle_id = await _insert_candle(sessionmaker, base, 100.0)
    model_version = await _insert_model(sessionmaker)

    repo = PostgresPredictionRepository(sessionmaker)
    prediction = await repo.create(
        candle_id=candle_id,
        model_version=model_version,
        direction="UP",
        confidence=0.6,
        probability_up=0.6,
    )
    assert prediction.direction == "UP"
    assert prediction.candle_open_time == base

    assert await repo.exists_for_candle(candle_id, model_version) is True
    assert await repo.exists_for_candle(candle_id, model_version + 1) is False

    resolved = await repo.resolve_for_candle(candle_id, "UP", base + timedelta(hours=4))
    assert resolved == 1
    # Idempotente: una vez resuelta, una segunda llamada no debe volver a actualizarla.
    assert await repo.resolve_for_candle(candle_id, "DOWN", base + timedelta(hours=8)) == 0

    latest = await repo.get_latest()
    assert latest.is_correct is True
    assert latest.actual_direction == "UP"

    correct, total = await repo.accuracy_since(None)
    assert (correct, total) == (1, 1)


async def test_prediction_repository_accuracy_by_month_groups_by_predicted_at(sessionmaker):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    candle_id = await _insert_candle(sessionmaker, base, 100.0)
    model_version = await _insert_model(sessionmaker)

    repo = PostgresPredictionRepository(sessionmaker)
    await repo.create(
        candle_id=candle_id,
        model_version=model_version,
        direction="UP",
        confidence=0.6,
        probability_up=0.6,
    )
    await repo.resolve_for_candle(candle_id, "UP", base + timedelta(hours=4))

    by_month = await repo.accuracy_by_month()

    current_month = datetime.now(UTC).strftime("%Y-%m")
    assert len(by_month) == 1
    assert by_month[0].month == current_month
    assert by_month[0].accuracy == 1.0
    assert by_month[0].total_predictions == 1


async def test_model_registry_repository_returns_active_and_by_version(sessionmaker):
    await _insert_model(sessionmaker, model_type="xgboost", is_active=False, feature_names=["a"])
    active_version = await _insert_model(
        sessionmaker,
        model_type="lightgbm",
        is_active=True,
        feature_names=["b"],
        artifact_path="/artifacts/2/model.onnx",
    )

    repo = PostgresModelRegistryRepository(sessionmaker)
    active = await repo.get_active()
    assert active.version == active_version
    assert active.model_type == "lightgbm"

    by_version = await repo.get_by_version(active_version)
    assert by_version.artifact_path == "/artifacts/2/model.onnx"

    assert await repo.get_by_version(999_999) is None
