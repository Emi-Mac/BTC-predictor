"""Inferencia y resolución de volatilidad contra PostgreSQL real (requiere TEST_DATABASE_URL)."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import Settings
from src.features import pipeline as features_pipeline
from src.ingestion.live_sources import LiveSourcesRepository
from src.ingestion.repository import IngestionRepository
from src.ingestion.tables import metadata, volatility_forecasts
from src.research import volatility_study as e12
from src.volatility import activation, inference, registry, resolution
from src.volatility.features import STEP
from src.volatility.model import fit_artifact
from src.volatility_config import VolatilitySettings
from tests.conftest import make_candles
from tests.test_live_sources import dvol_frame, events_frame
from tests.test_volatility_live_model import synthetic_bundle

NOW = datetime(2026, 11, 1, 8, 30, tzinfo=UTC)
LAST_OPEN = pd.Timestamp("2026-11-01 04:00", tz="UTC")
DECISION = LAST_OPEN + STEP
CANDLES = 720
COMPUTED_FEATURES = """
    create table computed_features (
        candle_id bigint primary key references ohlcv_candles(id) on delete cascade,
        vix_close numeric, volume_sma_ratio numeric)
"""


@pytest.fixture
async def engine(test_database_url):
    url = test_database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.execute(sa.text("drop table if exists computed_features"))
        await conn.run_sync(metadata.drop_all)
        await conn.run_sync(metadata.create_all)
        await conn.execute(sa.text(COMPUTED_FEATURES))
    yield eng
    async with eng.begin() as conn:
        await conn.execute(sa.text("drop table if exists computed_features"))
        await conn.run_sync(metadata.drop_all)
    await eng.dispose()


@pytest.fixture
def settings():
    return Settings(_env_file=None, database_url="postgresql://x/x", funding_enabled=False)


@pytest.fixture
def live():
    return VolatilitySettings(_env_file=None, volatility_sources_enabled=True)


def random_walk(start, n, seed=0):
    rng = np.random.default_rng(seed)
    frame = make_candles(n, start=str(start.date()))
    frame["open_time"] = [start + i * STEP for i in range(n)]
    frame["close_time"] = frame["open_time"] + STEP - pd.Timedelta(milliseconds=1)
    price = 40_000 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    for column in ("open", "high", "low", "close"):
        frame[column] = price
    return frame


async def seed_market(engine, settings, last_open=LAST_OPEN, dvol_hours=24 * 40, fomc=True):
    repo = IngestionRepository(engine)
    start = last_open - (CANDLES - 1) * STEP
    await repo.upsert_candles(
        settings.btc_symbol, settings.default_interval, random_walk(start, CANDLES)
    )
    async with engine.begin() as conn:
        await conn.execute(
            sa.text("insert into computed_features select id, 20, 1 from ohlcv_candles")
        )
        funding_times = pd.date_range(last_open - pd.Timedelta(days=60), last_open, freq="8h")
        for stamp in funding_times:
            await conn.execute(
                sa.text(
                    "insert into funding_rates (symbol, funding_time, funding_rate) "
                    "values ('BTCUSDT', :t, 0.0001)"
                ),
                {"t": stamp.to_pydatetime()},
            )
        await conn.execute(
            sa.text("insert into macro_daily (symbol, trade_date, close) values ('^VIX', :d, 20)"),
            {"d": (last_open - pd.Timedelta(days=1)).date()},
        )
    store = LiveSourcesRepository(engine)
    closed_hours = DECISION - pd.Timedelta(hours=1)
    await store.upsert_dvol(
        dvol_frame(closed_hours - pd.Timedelta(hours=dvol_hours - 1), dvol_hours, close=50.0)
    )
    if fomc:
        await store.upsert_events(
            events_frame(
                datetime(2026, 1, 28, 19, tzinfo=UTC), datetime(2026, 11, 4, 19, tzinfo=UTC)
            )
        )


@pytest.fixture(scope="module")
def development():
    data = synthetic_bundle()[4]
    return e12.development_rows(data, data["open_time"].iloc[-1])


def with_seed(artifact, candles, count=1500):
    issued = [DECISION - (candles + 1 + i) * STEP for i in range(count)][::-1]
    rng = np.random.default_rng(candles)
    seed = {
        "issued_at": [t.isoformat() for t in issued],
        "resolved_at": [(t + candles * STEP).isoformat() for t in issued],
        "z": [float(v) for v in np.abs(rng.normal(size=count))],
    }
    return replace(artifact, calibration_seed=seed, validation={"qlike": 0.2})


async def activate_models(engine, development, tmp_path, with_calibration=True):
    versions = {}
    for horizon, candles in (("1d", 6), ("7d", 42)):
        for role, kind in (("primary", "har_x_live"), ("fallback", "har")):
            artifact = fit_artifact(kind, development, horizon)
            if with_calibration:
                artifact = with_seed(artifact, candles)
            path = tmp_path / f"h{candles}-{role}.json"
            artifact.save(path)
            version = await registry.register(engine, path, role)
            await activation.activate(engine, version, "prueba")
            versions[(candles, role)] = version
    return versions


async def forecasts(engine):
    async with engine.connect() as conn:
        found = await conn.execute(
            sa.select(volatility_forecasts).order_by("horizon_candles", "candle_id")
        )
        return [dict(row) for row in found.mappings()]


async def test_inference_stores_one_primary_forecast_per_horizon_and_is_idempotent(
    engine, settings, live, development, tmp_path
):
    versions = await activate_models(engine, development, tmp_path)
    await seed_market(engine, settings)
    assert await inference.run_inference(engine, settings, live, NOW) == 2
    rows = await forecasts(engine)
    assert [r["horizon_candles"] for r in rows] == [6, 42]
    for row in rows:
        horizon = row["horizon_candles"]
        assert (
            row["model_version"] == row["requested_model_version"] == versions[(horizon, "primary")]
        )
        assert row["fallback_reason"] is None
        assert row["forecast_var_annual"] > 0
        assert row["forecast_vol_annual"] == pytest.approx(
            Decimal(str(float(row["forecast_var_annual"]) ** 0.5)), rel=Decimal("1e-8")
        )
        assert row["regime"] in {"low", "normal", "high", "extreme"}
        assert row["k68"] < row["k90"] < row["k95"]
        assert row["calibration_window_end"] <= DECISION
        assert row["resolved_at"] is None
        assert float(row["reference_price"]) > 0
        as_of = row["features_as_of"]
        assert as_of["decision_time"] == DECISION.isoformat()
        assert as_of["fomc_next"].startswith("2026-11-04")
        assert as_of["dvol_open_time"].startswith("2026-11-01T07:00")
        assert as_of["funding_time"].startswith("2026-11-01T04:00")
        assert as_of["vix_trade_date"] == "2026-10-31"
    assert await inference.run_inference(engine, settings, live, NOW) == 0
    assert len(await forecasts(engine)) == 2


async def test_stale_or_missing_sources_serve_the_fallback_with_reasons(
    engine, settings, live, development, tmp_path
):
    versions = await activate_models(engine, development, tmp_path)
    await seed_market(engine, settings, dvol_hours=24 * 40, fomc=False)
    async with engine.begin() as conn:  # DVOL con más de 6 h de antigüedad
        await conn.execute(
            sa.text("delete from implied_volatility where open_time > :t"),
            {"t": (DECISION - pd.Timedelta(hours=10)).to_pydatetime()},
        )
    assert await inference.run_inference(engine, settings, live, NOW) == 2
    for row in await forecasts(engine):
        horizon = row["horizon_candles"]
        assert row["model_version"] == versions[(horizon, "fallback")]
        assert row["requested_model_version"] == versions[(horizon, "primary")]
        assert row["fallback_reason"] == "dvol_stale,fomc_calendar_missing"
        assert row["features_as_of"]["fomc_next"] is None


async def test_missing_features_row_and_funding_are_reported(
    engine, settings, live, development, tmp_path
):
    await activate_models(engine, development, tmp_path)
    await seed_market(engine, settings)
    async with engine.begin() as conn:
        await conn.execute(sa.text("delete from computed_features"))
        await conn.execute(sa.text("delete from funding_rates"))
    await inference.run_inference(engine, settings, live, NOW)
    reasons = {r["fallback_reason"] for r in await forecasts(engine)}
    assert reasons == {"features_missing,funding_stale"}


async def test_inference_skips_without_current_candle_models_or_history(
    engine, settings, live, development, tmp_path
):
    await seed_market(engine, settings)
    assert await inference.run_inference(engine, settings, live, NOW) == 0  # sin modelos activos
    await activate_models(engine, development, tmp_path)
    stale_now = NOW + timedelta(hours=4)  # falta la vela más reciente
    assert await inference.run_inference(engine, settings, live, stale_now) == 0
    async with engine.begin() as conn:
        await conn.execute(
            sa.text("delete from ohlcv_candles where open_time < :t"),
            {"t": (LAST_OPEN - 100 * STEP).to_pydatetime()},
        )
    assert await inference.run_inference(engine, settings, live, NOW) == 0  # < 181 velas
    assert await forecasts(engine) == []


async def test_candle_gaps_keep_only_the_contiguous_tail(
    engine, settings, live, development, tmp_path
):
    await activate_models(engine, development, tmp_path)
    await seed_market(engine, settings)
    async with engine.begin() as conn:  # hueco a 150 velas del final
        await conn.execute(
            sa.text("delete from ohlcv_candles where open_time = :t"),
            {"t": (LAST_OPEN - 150 * STEP).to_pydatetime()},
        )
    assert await inference.run_inference(engine, settings, live, NOW) == 0  # tramo < 181
    candles = await inference._recent_candles(engine, settings, NOW)
    assert len(inference.contiguous_tail(candles)) == 150


async def test_missing_calibration_leaves_bands_empty_but_still_forecasts(
    engine, settings, live, development, tmp_path
):
    await activate_models(engine, development, tmp_path, with_calibration=False)
    await seed_market(engine, settings)
    assert await inference.run_inference(engine, settings, live, NOW) == 2
    for row in await forecasts(engine):
        assert row["k68"] is row["k90"] is row["k95"] is row["calibration_window_end"] is None


async def test_forecasts_resolve_with_the_training_definition_and_only_when_complete(
    engine, settings, live, development, tmp_path
):
    await activate_models(engine, development, tmp_path)
    await seed_market(engine, settings)
    await inference.run_inference(engine, settings, live, NOW)
    assert await resolution.resolve_forecasts(engine, settings, NOW) == 0  # aún sin velas futuras

    repo = IngestionRepository(engine)
    rng = np.random.default_rng(9)
    following = random_walk(LAST_OPEN + STEP, 42, seed=1)
    following["close"] = float(await _last_close(engine)) * np.exp(
        np.cumsum(rng.normal(0, 0.01, 42))
    )
    await repo.upsert_candles(settings.btc_symbol, settings.default_interval, following)
    one_day = LAST_OPEN + 7 * STEP  # cierre de la vela t+6
    before = one_day - timedelta(minutes=1)
    assert await resolution.resolve_forecasts(engine, settings, before) == 0  # t+6 no cerró
    after = one_day
    assert await resolution.resolve_forecasts(engine, settings, after) == 1
    rows = {r["horizon_candles"]: r for r in await forecasts(engine)}
    assert rows[6]["resolved_at"] == after and rows[42]["resolved_at"] is None
    closes = await _closes(engine)
    expected_var, expected_ret = resolution.realized_statistics(
        np.array([closes[LAST_OPEN + i * STEP] for i in range(7)])
    )
    assert float(rows[6]["realized_var_annual"]) == pytest.approx(expected_var, rel=1e-8)
    assert float(rows[6]["realized_log_return"]) == pytest.approx(expected_ret, rel=1e-8)
    assert rows[6]["qlike"] is not None
    assert {rows[6]["inside_68"], rows[6]["inside_90"], rows[6]["inside_95"]} <= {True, False}
    done = LAST_OPEN + 43 * STEP
    assert await resolution.resolve_forecasts(engine, settings, done) == 1
    assert await resolution.resolve_forecasts(engine, settings, done) == 0  # idempotente
    assert all(r["resolved_at"] is not None for r in await forecasts(engine))


async def test_resolution_waits_when_a_future_candle_is_missing(
    engine, settings, live, development, tmp_path
):
    await activate_models(engine, development, tmp_path)
    await seed_market(engine, settings)
    await inference.run_inference(engine, settings, live, NOW)
    following = random_walk(LAST_OPEN + STEP, 42, seed=2).drop(index=3)
    await IngestionRepository(engine).upsert_candles(
        settings.btc_symbol, settings.default_interval, following
    )
    assert await resolution.resolve_forecasts(engine, settings, NOW + timedelta(days=30)) == 0
    assert all(r["resolved_at"] is None for r in await forecasts(engine))


async def test_live_resolved_forecasts_feed_the_next_calibration(
    engine, settings, live, development, tmp_path
):
    await activate_models(engine, development, tmp_path, with_calibration=False)
    await seed_market(engine, settings)
    models = await inference.load_active_models(engine)
    model = models[(6, "primary")]
    k, end = await inference.band_multipliers(engine, model, DECISION)
    assert end is None and all(np.isnan(v) for v in k.values())
    issued, resolved, z = await inference._live_calibration(engine, model.version, 6)
    assert len(issued) == len(resolved) == len(z) == 0


async def test_live_resolved_forecasts_extend_the_seed_without_overlap(
    engine, settings, live, development, tmp_path, monkeypatch
):
    await activate_models(engine, development, tmp_path)
    await seed_market(engine, settings)
    monkeypatch.setattr(inference.calibration, "MIN_RESOLVED", 60)
    model = (await inference.load_active_models(engine))[(6, "primary")]
    short = replace(
        model.artifact,
        calibration_seed={k: v[:50] for k, v in model.artifact.calibration_seed.items()},
    )
    model = replace(model, artifact=short)
    k, end = await inference.band_multipliers(engine, model, DECISION)
    assert end is None and all(np.isnan(v) for v in k.values())  # 50 < 60

    async with engine.connect() as conn:
        ids = (
            (
                await conn.execute(
                    sa.text("select id from ohlcv_candles order by open_time limit 20")
                )
            )
            .scalars()
            .all()
        )
    async with engine.begin() as conn:
        for candle_id in ids:
            await conn.execute(
                volatility_forecasts.insert().values(
                    candle_id=candle_id,
                    horizon_candles=6,
                    model_version=model.version,
                    requested_model_version=model.version,
                    features_as_of={},
                    forecast_var_annual=Decimal("0.25"),
                    forecast_vol_annual=Decimal("0.5"),
                    regime="normal",
                    reference_price=Decimal("40000"),
                    realized_var_annual=Decimal("0.2"),
                    realized_log_return=Decimal("0.01"),
                    qlike=Decimal("0.1"),
                    resolved_at=NOW,
                )
            )
    issued, resolved, z = await inference._live_calibration(engine, model.version, 6)
    assert len(z) == 20 and (resolved - issued == 6 * STEP).all()
    k, end = await inference.band_multipliers(engine, model, DECISION)
    assert end is not None and end <= DECISION
    assert k[0.68] <= k[0.90] <= k[0.95]


async def test_database_protects_forecast_history_and_invariants(
    engine, settings, live, development, tmp_path
):
    versions = await activate_models(engine, development, tmp_path)
    await seed_market(engine, settings)
    await inference.run_inference(engine, settings, live, NOW)
    base = (await forecasts(engine))[0]
    clone = {k: v for k, v in base.items() if k not in ("id", "created_at")}

    async def insert(**changes):
        async with engine.begin() as conn:
            await conn.execute(volatility_forecasts.insert().values({**clone, **changes}))

    with pytest.raises(IntegrityError):  # único por (vela, horizonte)
        await insert()
    other = {"candle_id": base["candle_id"] - 1}
    with pytest.raises(IntegrityError):  # modelo de otro horizonte
        await insert(
            **other,
            model_version=versions[(42, "primary")],
            requested_model_version=versions[(42, "primary")],
        )
    with pytest.raises(IntegrityError):  # respaldo sin motivo
        await insert(**other, model_version=versions[(6, "fallback")])
    with pytest.raises(IntegrityError):  # resolución a medias
        await insert(**other, realized_var_annual=Decimal("0.1"))
    with pytest.raises(IntegrityError):  # bandas a medias
        await insert(**other, k68=None)
    with pytest.raises(IntegrityError):  # régimen inválido
        await insert(**other, regime="wild")
    with pytest.raises(IntegrityError):  # el historial de evaluación no se borra
        async with engine.begin() as conn:
            await conn.execute(sa.text("delete from volatility_models"))


async def test_cycle_isolates_failures_and_reports_a_summary(
    engine, settings, live, development, tmp_path, monkeypatch
):
    await activate_models(engine, development, tmp_path)
    await seed_market(engine, settings)

    async def boom(*_args, **_kwargs):
        raise RuntimeError("resolución caída")

    monkeypatch.setattr(inference, "resolve_forecasts", boom)
    assert await inference.run_volatility_cycle(engine, settings, live, NOW) == {
        "resolved": 0,
        "created": 2,
    }
    monkeypatch.setattr(inference, "run_inference", boom)
    assert await inference.run_volatility_cycle(engine, settings, live, NOW) == {
        "resolved": 0,
        "created": 0,
    }


async def test_unavailable_artifacts_are_ignored(engine, development, tmp_path):
    versions = await activate_models(engine, development, tmp_path)
    (tmp_path / "h6-primary.json").unlink()
    (tmp_path / "h42-primary.json").write_text("{}")
    models = await inference.load_active_models(engine)
    assert set(models) == {(6, "fallback"), (42, "fallback")}
    assert models[(6, "fallback")].version == versions[(6, "fallback")]


async def test_features_pipeline_runs_the_cycle_only_when_enabled(monkeypatch):
    calls = []

    class Frame:
        empty = False

        def dropna(self, subset):
            return self

        def __len__(self):
            return 1

    class Source:
        def __init__(self, engine):
            pass

        async def load_macro(self):
            return None

        async def load_fear_greed(self):
            return None

    class Target:
        def __init__(self, engine):
            pass

        async def load_candles(self, symbol, interval):
            return None

        async def save(self, symbol, interval, frame):
            return 1

    async def context(engine, settings):
        return 0

    async def cycle(engine, settings, live):
        calls.append(live.volatility_sources_enabled)

    monkeypatch.setattr(features_pipeline, "IngestionRepository", Source)
    monkeypatch.setattr(features_pipeline, "FeatureRepository", Target)
    monkeypatch.setattr(features_pipeline, "compute_features", lambda *a: Frame())
    monkeypatch.setattr(features_pipeline, "save_contextual_features", context)
    monkeypatch.setattr(features_pipeline, "run_volatility_cycle", cycle)
    settings = Settings(_env_file=None, database_url="postgresql://x/x")
    for enabled in (False, True):
        monkeypatch.setattr(
            features_pipeline,
            "get_volatility_settings",
            lambda enabled=enabled: VolatilitySettings(
                _env_file=None, volatility_sources_enabled=enabled
            ),
        )
        assert await features_pipeline.run_features(object(), settings) == 1
    assert calls == [True]


async def _last_close(engine):
    async with engine.connect() as conn:
        return (
            await conn.execute(
                sa.text("select close from ohlcv_candles order by open_time desc limit 1")
            )
        ).scalar()


async def _closes(engine):
    async with engine.connect() as conn:
        rows = (await conn.execute(sa.text("select open_time, close from ohlcv_candles"))).all()
    return {pd.Timestamp(t): float(c) for t, c in rows}


async def test_replay_rebuilds_history_only_on_development_databases(
    engine, settings, live, development, tmp_path
):
    from src.volatility import replay

    for name in ("btcshadow", "btcpredictor", "btcpredictor_test", "", None):
        with pytest.raises(ValueError, match="desarrollo"):
            replay.assert_development_database(name)
    replay.assert_development_database("btc_vol_dev")

    await activate_models(engine, development, tmp_path)
    await seed_market(engine, settings)
    with pytest.raises(ValueError, match="desarrollo"):  # la base de pruebas no se llama «dev»
        await replay.replay(engine, settings, live, candles=3)
    with pytest.raises(ValueError, match="positivo"):
        await replay.replay(engine, settings, live, candles=0, enforce_dev_database=False)

    result = await replay.replay(
        engine, settings, live, candles=3, resolve_at=NOW, enforce_dev_database=False
    )
    assert result["candles"] == 3 and result["created"] == 6
    rows = await forecasts(engine)
    opened = sorted({LAST_OPEN - i * STEP for i in (1, 2, 3)})
    async with engine.connect() as conn:
        times = (
            await conn.execute(
                sa.text(
                    "select distinct c.open_time from volatility_forecasts f "
                    "join ohlcv_candles c on c.id = f.candle_id order by 1"
                )
            )
        ).scalars().all()
    assert [pd.Timestamp(t) for t in times] == opened  # la última vela queda para el worker
    assert len(rows) == 6 and all(r["resolved_at"] is None for r in rows)
