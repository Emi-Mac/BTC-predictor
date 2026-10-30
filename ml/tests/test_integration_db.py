"""Tests de integración contra PostgreSQL real (requieren TEST_DATABASE_URL)."""

from datetime import UTC, date, datetime, timedelta

import httpx
import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import Settings
from src.ingestion import pipeline
from src.ingestion.backfill_taker import backfill_taker
from src.ingestion.binance import BinanceKlinesClient
from src.ingestion.funding_repository import FundingRepository
from src.ingestion.integrity import check_candle_integrity
from src.ingestion.repository import IngestionRepository
from src.ingestion.tables import metadata
from tests.conftest import STEP, make_candles

STEP_MS = int(STEP.total_seconds() * 1000)


@pytest.fixture
async def engine(test_database_url):
    url = test_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(metadata.drop_all)
        await conn.run_sync(metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(metadata.drop_all)
    await eng.dispose()


@pytest.fixture
def repo(engine):
    return IngestionRepository(engine)


async def test_candle_upsert_is_idempotent_and_roundtrips(repo):
    df = make_candles(50)
    assert await repo.upsert_candles("BTCUSDT", "4h", df) == 50
    assert await repo.upsert_candles("BTCUSDT", "4h", df) == 0  # ON CONFLICT DO NOTHING
    loaded = await repo.load_candles("BTCUSDT", "4h")
    assert len(loaded) == 50
    assert loaded["close"].iloc[0] == df["close"].iloc[0]
    assert loaded["open_time"].iloc[-1] == df["open_time"].iloc[-1]
    assert await repo.last_open_time("BTCUSDT", "4h") == df["open_time"].iloc[-1].to_pydatetime()
    assert check_candle_integrity(loaded, STEP).ok


async def test_backfill_only_adds_taker_volumes_and_is_idempotent(repo, engine):
    candles = make_candles(30)
    assert await repo.upsert_candles("BTCUSDT", "4h", candles) == 30
    async with engine.connect() as conn:
        before = (
            await conn.execute(
                sa.text(
                    "SELECT id, open_time, open, high, low, close, volume, quote_volume, "
                    "num_trades, created_at FROM ohlcv_candles ORDER BY open_time"
                )
            )
        ).all()
    incoming = candles.copy()
    incoming["taker_buy_base_volume"] = incoming["volume"] * 0.4
    incoming["taker_buy_quote_volume"] = incoming["quote_volume"] * 0.4
    assert await repo.backfill_taker_volumes("BTCUSDT", "4h", incoming) == 30
    assert await repo.backfill_taker_volumes("BTCUSDT", "4h", incoming) == 0
    async with engine.connect() as conn:
        after = (
            await conn.execute(
                sa.text(
                    "SELECT id, open_time, open, high, low, close, volume, quote_volume, "
                    "num_trades, created_at FROM ohlcv_candles ORDER BY open_time"
                )
            )
        ).all()
    assert before == after
    loaded = await repo.load_candles("BTCUSDT", "4h")
    assert loaded["taker_buy_base_volume"].tolist() == pytest.approx(
        incoming["taker_buy_base_volume"].tolist()
    )
    assert await repo.missing_taker_bounds("BTCUSDT", "4h") == (None, None, 0)


async def test_backfill_rejects_mismatches_and_missing_candles_atomically(repo):
    candles = make_candles(30)
    await repo.upsert_candles("BTCUSDT", "4h", candles)
    incoming = candles.copy()
    incoming["taker_buy_base_volume"] = incoming["volume"] * 0.4
    incoming["taker_buy_quote_volume"] = incoming["quote_volume"] * 0.4
    mismatch = incoming.copy()
    mismatch.loc[20, "volume"] += 1
    with pytest.raises(ValueError, match="no coinciden"):
        await repo.backfill_taker_volumes("BTCUSDT", "4h", mismatch)
    assert (await repo.load_candles("BTCUSDT", "4h"))["taker_buy_base_volume"].isna().all()
    with pytest.raises(ValueError, match="no devolvió"):
        await repo.backfill_taker_volumes("BTCUSDT", "4h", incoming.drop(index=20))
    assert (await repo.load_candles("BTCUSDT", "4h"))["taker_buy_base_volume"].isna().all()


async def test_incremental_ingestion_fills_existing_missing_taker_volumes(repo):
    candles = make_candles(5)
    await repo.upsert_candles("BTCUSDT", "4h", candles)
    incoming = candles.copy()
    incoming["taker_buy_base_volume"] = incoming["volume"] * 0.3
    incoming["taker_buy_quote_volume"] = incoming["quote_volume"] * 0.3
    assert await repo.upsert_candles("BTCUSDT", "4h", incoming) == 0
    assert (await repo.load_candles("BTCUSDT", "4h"))["taker_buy_base_volume"].notna().all()


async def test_database_rejects_invalid_taker_volume_pairs(repo, engine):
    await repo.upsert_candles("BTCUSDT", "4h", make_candles(1))
    with pytest.raises(sa.exc.IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "UPDATE ohlcv_candles SET taker_buy_base_volume = volume + 1, "
                    "taker_buy_quote_volume = 0"
                )
            )
    with pytest.raises(sa.exc.IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(sa.text("UPDATE ohlcv_candles SET taker_buy_base_volume = 1"))


async def test_funding_history_is_idempotent_and_does_not_replace_past_values(engine):
    funding = FundingRepository(engine)
    times = pd.date_range("2024-01-01", periods=2500, freq="8h", tz="UTC")
    data = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "funding_time": times,
            "funding_rate": ["0.00012345"] * len(times),
            "mark_price": ["50000.12345678"] * len(times),
        }
    )
    assert await funding.upsert(data) == 2500
    assert await funding.upsert(data) == 0
    corrected = data.iloc[[0]].copy()
    corrected["funding_rate"] = "0.00099999"
    assert await funding.upsert(corrected) == 0
    loaded = await funding.load("BTCUSDT")
    assert len(loaded) == 2500
    assert loaded["funding_rate"].iloc[0] == pytest.approx(0.00012345)
    assert await funding.last_funding_time("BTCUSDT") == times[-1].to_pydatetime()


async def test_ingest_funding_backfills_then_reuses_overlap(repo, settings):
    candles = make_candles(10)
    await repo.upsert_candles("BTCUSDT", "4h", candles)
    settings.funding_enabled = True
    event_time = candles["open_time"].iloc[0].to_pydatetime()
    starts = []

    class Client:
        async def fetch(self, symbol, start):
            starts.append(start)
            return pd.DataFrame(
                {
                    "symbol": [symbol],
                    "funding_time": [event_time],
                    "funding_rate": ["0.0001"],
                    "mark_price": ["40000.0"],
                }
            )

    summary = pipeline.IngestionSummary()
    await pipeline.ingest_funding(repo, settings, summary, Client())
    await pipeline.ingest_funding(repo, settings, summary, Client())
    assert summary.funding_rows == 1
    assert starts == [event_time - timedelta(days=7), event_time - timedelta(days=1)]


async def test_upsert_chunks_large_batches(repo):
    df = make_candles(2500)
    assert await repo.upsert_candles("BTCUSDT", "4h", df) == 2500


async def test_macro_and_fear_greed_upsert_updates_existing_rows(repo, engine):
    macro = pd.DataFrame(
        {
            "symbol": ["^VIX", "^VIX"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "close": [10.0, 11.0],
        }
    )
    await repo.upsert_macro(macro)
    macro.loc[1, "close"] = 12.5
    await repo.upsert_macro(macro)
    loaded = await repo.load_macro()
    assert loaded["close"].tolist() == [10.0, 12.5]
    assert await repo.last_macro_date() == date(2024, 1, 3)

    fng = pd.DataFrame({"obs_date": [date(2024, 1, 3)], "value": [40], "classification": ["Fear"]})
    await repo.upsert_fear_greed(fng)
    fng["value"] = 41
    await repo.upsert_fear_greed(fng)
    assert (await repo.load_fear_greed())["value"].tolist() == [41]
    assert await repo.last_fear_greed_date() == date(2024, 1, 3)
    assert await repo.last_macro_date() is not None
    async with engine.connect() as conn:
        assert (await conn.execute(sa.text("select count(*) from macro_daily"))).scalar() == 2


def fake_binance(available_from: datetime, missing: set[int] | None = None) -> BinanceKlinesClient:
    missing = missing or set()
    base_ms = int(available_from.timestamp() * 1000)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    def handler(request: httpx.Request) -> httpx.Response:
        start = max(int(request.url.params["startTime"]), base_ms)
        end = int(request.url.params["endTime"])
        idx = -(-(start - base_ms) // STEP_MS)
        rows = []
        while len(rows) < 1000:
            open_ms = base_ms + idx * STEP_MS
            if open_ms > end or open_ms > now_ms:
                break
            if idx not in missing:
                p = 40_000 + idx
                close_ms = open_ms + STEP_MS - 1
                rows.append(
                    [
                        open_ms,
                        str(p),
                        str(p + 5),
                        str(p - 5),
                        str(p + 1),
                        "10",
                        close_ms,
                        "4e5",
                        5,
                        "4",
                        "160000",
                        "0",
                    ]
                )
            idx += 1
        return httpx.Response(200, json=rows)

    http = httpx.AsyncClient(
        base_url="https://api.binance.com", transport=httpx.MockTransport(handler)
    )
    return BinanceKlinesClient(client=http, rate_limit_per_min=1_000_000)


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")

    async def no_macro(*args, **kwargs):
        return pd.DataFrame(columns=["symbol", "trade_date", "close"])

    async def no_fng(*args, **kwargs):
        return pd.DataFrame(columns=["obs_date", "value", "classification"])

    monkeypatch.setattr(pipeline, "fetch_macro_series", no_macro)
    monkeypatch.setattr(pipeline, "fetch_fear_greed", no_fng)
    return Settings(_env_file=None, history_days=60, funding_enabled=False)


async def test_backfill_then_update_is_idempotent_and_complete(repo, settings):
    start = datetime.now(UTC) - timedelta(days=settings.history_days + 1)
    client = fake_binance(pipeline.floor_to_step(start, STEP))
    first = await pipeline.run_ingestion(repo, settings, client)
    assert first.candles_inserted >= settings.history_days * 6 - 6

    second = await pipeline.run_ingestion(repo, settings, client)
    assert second.candles_inserted == 0
    await client.aclose()

    stored = await repo.load_candles(settings.btc_symbol, settings.default_interval)
    report = check_candle_integrity(stored, STEP, min_span_days=settings.history_days - 1)
    assert report.ok, report.problems


async def test_update_repairs_historical_gaps(repo, settings):
    start = pipeline.floor_to_step(datetime.now(UTC) - timedelta(days=30), STEP)
    full = fake_binance(start)
    await pipeline.run_ingestion(repo, settings, full)

    stored = await repo.load_candles(settings.btc_symbol, settings.default_interval)
    hole = stored["open_time"].iloc[40:43]  # hueco de 3 velas en medio del histórico
    async with repo._engine.begin() as conn:
        await conn.execute(
            sa.text("delete from ohlcv_candles where open_time = any(:t)"),
            {"t": [t.to_pydatetime() for t in hole]},
        )
    broken = await repo.load_candles(settings.btc_symbol, settings.default_interval)
    assert not check_candle_integrity(broken, STEP).ok

    summary = await pipeline.run_ingestion(repo, settings, full)
    await full.aclose()
    assert summary.candles_inserted == 3
    repaired = await repo.load_candles(settings.btc_symbol, settings.default_interval)
    assert check_candle_integrity(repaired, STEP).ok


async def test_backfill_taker_fetches_historical_missing_volumes(repo, settings):
    start = pipeline.floor_to_step(datetime.now(UTC) - timedelta(days=30), STEP)
    client = fake_binance(start)
    await pipeline.run_ingestion(repo, settings, client)
    async with repo.engine.begin() as conn:
        await conn.execute(
            sa.text(
                "UPDATE ohlcv_candles SET taker_buy_base_volume=NULL, "
                "taker_buy_quote_volume=NULL WHERE id IN "
                "(SELECT id FROM ohlcv_candles ORDER BY open_time LIMIT 10)"
            )
        )
    assert await backfill_taker(repo, settings, client) == 10
    assert await backfill_taker(repo, settings, client) == 0
    await client.aclose()
    assert (await repo.load_candles("BTCUSDT", "4h"))["taker_buy_base_volume"].notna().all()
