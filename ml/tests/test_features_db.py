"""Tests de persistencia en un esquema PostgreSQL aislado."""

from datetime import timedelta
from uuid import uuid4

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateSchema, DropSchema

from src.config import Settings
from src.features.compute import FEATURE_COLUMNS, STORED_COLUMNS, compute_features
from src.features.context_repository import load_contextual_features
from src.features.funding_context import FUNDING_FEATURE_COLUMNS
from src.features.pipeline import run_features
from src.features.price_context import PRICE_CONTEXT_COLUMNS
from src.features.repository import BOOL_COLUMNS, INT_COLUMNS, FeatureRepository
from src.features.taker_flow import TAKER_FEATURE_COLUMNS
from src.ingestion import run as ingestion_cli
from src.ingestion.funding_repository import FundingRepository
from src.ingestion.repository import IngestionRepository
from src.ingestion.tables import metadata, ohlcv_candles
from tests.test_features import STEP

pytest_plugins = ["tests.test_features"]


@pytest.fixture
async def feature_engine(test_database_url):
    url = test_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    schema = f"feature_test_{uuid4().hex}"
    admin = create_async_engine(url)
    async with admin.begin() as conn:
        await conn.execute(CreateSchema(schema))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    features = sa.Table(
        "computed_features",
        metadata,
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("candle_id", sa.BigInteger, sa.ForeignKey(ohlcv_candles.c.id), nullable=False),
        *[
            sa.Column(
                name,
                sa.Boolean
                if name in BOOL_COLUMNS
                else (sa.Integer if name in INT_COLUMNS else sa.Numeric(20, 8)),
            )
            for name in STORED_COLUMNS
        ],
        sa.UniqueConstraint("candle_id", name="uq_features_candle"),
        extend_existing=True,
    )
    try:
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        yield engine, features
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(DropSchema(schema, cascade=True))
        await admin.dispose()
        metadata.remove(features)


async def test_feature_pipeline_idempotence_and_atomic_rollback(
    feature_engine, market_data, test_database_url, monkeypatch, capsys
):
    engine, table = feature_engine
    candles, macro, fgi = market_data
    candles["close_time"] = candles["open_time"] + STEP - pd.Timedelta(milliseconds=1)
    candles["quote_volume"] = candles["volume"] * candles["close"]
    candles["num_trades"] = 100
    source = IngestionRepository(engine)
    await source.upsert_candles("BTCUSDT", "4h", candles)
    await source.upsert_macro(macro)
    await source.upsert_fear_greed(fgi)
    settings = Settings(database_url=test_database_url, _env_file=None)
    assert await run_features(engine, settings) == len(candles) - 199
    target = FeatureRepository(engine)
    data = await target.load_for_selection("BTCUSDT", "4h")
    assert len(data) == len(candles) - 199
    assert data[FEATURE_COLUMNS].notna().all().all()
    assert data["target_label"].isna().sum() == 1
    assert await target.last_feature_open_time("BTCUSDT", "4h") == candles["open_time"].iloc[-1]
    async with engine.connect() as conn:
        before = (
            await conn.execute(sa.select(table.c.id, table.c.macd).order_by(table.c.id))
        ).all()
    assert await run_features(engine, settings) == len(candles) - 199
    async with engine.connect() as conn:
        after = (await conn.execute(sa.select(table.c.id, table.c.macd).order_by(table.c.id))).all()
    assert before == after

    next_candle = candles.iloc[[-1]].copy()
    next_candle["open_time"] += STEP
    next_candle["close_time"] += STEP
    next_candle[["open", "high", "low", "close"]] += 10
    await source.upsert_candles("BTCUSDT", "4h", next_candle)
    assert await run_features(engine, settings) == len(candles) - 198
    data = await target.load_for_selection("BTCUSDT", "4h")
    assert data["target_label"].iloc[-2] == 1
    assert pd.isna(data["target_label"].iloc[-1])
    context = await load_contextual_features(engine, settings)
    assert pd.isna(context["target_label"].iloc[-1])
    assert context[PRICE_CONTEXT_COLUMNS].iloc[-1].notna().all()
    assert context["return_1"].iloc[-1] == pytest.approx(
        next_candle["close"].iloc[0] / candles["close"].iloc[-1] - 1
    )
    unknown_flow = await load_contextual_features(engine, settings, include_taker=True)
    assert unknown_flow[TAKER_FEATURE_COLUMNS].isna().all().all()
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "UPDATE ohlcv_candles SET taker_buy_base_volume=volume*0.4, "
                "taker_buy_quote_volume=quote_volume*0.4"
            )
        )
    known_flow = await load_contextual_features(engine, settings, include_taker=True)
    assert known_flow[TAKER_FEATURE_COLUMNS].iloc[-1].tolist() == pytest.approx([0.4] * 4)
    funding_times = pd.date_range("2023-12-25", periods=350, freq="8h", tz="UTC")
    await FundingRepository(engine).upsert(
        pd.DataFrame(
            {
                "symbol": "BTCUSDT",
                "funding_time": funding_times,
                "funding_rate": ["0.0001"] * len(funding_times),
                "mark_price": ["40000"] * len(funding_times),
            }
        )
    )
    with_funding = await load_contextual_features(
        engine, settings, include_taker=True, include_funding=True
    )
    assert with_funding[FUNDING_FEATURE_COLUMNS].iloc[-1].notna().all()
    assert with_funding["funding_available_at"].iloc[-1] <= (
        with_funding["open_time"].iloc[-1] + STEP
    )

    latest_day = next_candle["open_time"].iloc[0].date()
    correction = macro[
        (macro["symbol"] == "^GSPC") & (macro["trade_date"] == latest_day - timedelta(days=1))
    ].copy()
    correction["close"] += 50
    await source.upsert_macro(correction)
    await run_features(engine, settings)
    data = await target.load_for_selection("BTCUSDT", "4h")
    assert data["sp500_close"].iloc[-1] == pytest.approx(correction["close"].iloc[0])

    async with engine.connect() as conn:
        before = (
            await conn.execute(sa.select(table.c.id, table.c.macd).order_by(table.c.id))
        ).all()
    calculated = compute_features(
        await target.load_candles("BTCUSDT", "4h"),
        await source.load_macro(),
        await source.load_fear_greed(),
        STEP,
    ).dropna(subset=FEATURE_COLUMNS)
    calculated.iloc[0, calculated.columns.get_loc("macd")] = 99999
    calculated.iloc[-1, calculated.columns.get_loc("bb_width")] = 10**30
    with pytest.raises(sa.exc.DBAPIError):
        await target.save("BTCUSDT", "4h", calculated)
    async with engine.connect() as conn:
        rolled_back = (
            await conn.execute(sa.select(table.c.id, table.c.macd).order_by(table.c.id))
        ).all()
    assert rolled_back == before

    calculated = calculated.iloc[1:].copy()
    calculated["bb_width"] = 0.1
    await target.save("BTCUSDT", "4h", calculated)
    assert len(await target.load_for_selection("BTCUSDT", "4h")) == len(calculated)
    await run_features(engine, settings)
    assert len(await target.load_for_selection("BTCUSDT", "4h")) == len(calculated) + 1
    settings.funding_enabled = False
    monkeypatch.setattr(ingestion_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(ingestion_cli, "create_async_engine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(ingestion_cli, "MIN_HISTORY_DAYS", 0)
    assert await ingestion_cli._run("check") == 1
    assert "velas recientes faltantes" in capsys.readouterr().out
