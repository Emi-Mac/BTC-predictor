"""Sombra prospectiva: estadística pura y evaluación contra PostgreSQL (TEST_DATABASE_URL)."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from src.ingestion.repository import IngestionRepository
from src.ingestion.tables import (
    implied_volatility,
    metadata,
    open_interest,
    volatility_forecasts,
    volatility_models,
)
from src.volatility import shadow
from src.volatility.features import STEP
from tests.conftest import make_candles

START = datetime(2026, 11, 2, tzinfo=UTC)  # lunes 00:00 UTC
END = START + timedelta(weeks=12)
NOW = END + timedelta(days=3)


def test_exact_interval_matches_known_values_and_validates_inputs():
    from scipy.stats import beta

    low, high = shadow.exact_interval(57, 84)
    # Clopper-Pearson por la relación con la distribución beta
    assert low == pytest.approx(beta.ppf(0.025, 57, 28), abs=1e-9)
    assert high == pytest.approx(beta.ppf(0.975, 58, 27), abs=1e-9)
    assert high - low == pytest.approx(0.21, abs=0.02)  # ≈ ±10 pp con 84 observaciones
    assert shadow.exact_interval(0, 10)[0] == 0.0 and shadow.exact_interval(10, 10)[1] == 1.0
    for hits, n in ((1, 0), (-1, 5), (6, 5)):
        with pytest.raises(ValueError, match="n > 0"):
            shadow.exact_interval(hits, n)


def test_coverage_check_flags_values_outside_the_interval():
    ok = shadow.coverage_check(pd.Series([True] * 57 + [False] * 27), 0.68, 0.95)
    assert ok["n"] == 84 and ok["hits"] == 57 and ok["ok"] is True
    bad = shadow.coverage_check(pd.Series([True] * 84), 0.68, 0.95)
    assert bad["coverage"] == 1.0 and bad["ok"] is False
    empty = shadow.coverage_check(pd.Series([None, None], dtype=object), 0.9, 0.95)
    assert empty["n"] == 0 and empty["ok"] is None
    assert shadow.coverage_check(pd.Series([True, None, False]), 0.5, 0.95)["n"] == 2


def test_non_overlapping_samples_pick_daily_midnight_and_monday_candles():
    times = pd.date_range(START, periods=6 * 21, freq="4h", tz="UTC")
    frame = pd.DataFrame([{"open_time": t, "horizon_candles": h} for t in times for h in (6, 42)])
    daily = shadow.non_overlapping(frame, 6)
    weekly = shadow.non_overlapping(frame, 42)
    assert len(daily) == 21 and (daily["open_time"].dt.hour == 0).all()
    assert (daily["open_time"].diff().dropna() == timedelta(days=1)).all()
    assert len(weekly) == 3 and (weekly["open_time"].dt.day_of_week == 0).all()


def test_max_gap_and_expected_observations():
    times = pd.Series(
        pd.to_datetime(["2026-11-02 00:00", "2026-11-02 04:00", "2026-11-02 12:00"], utc=True)
    )
    assert shadow.max_gap(times) == 8.0
    assert shadow.max_gap(times.iloc[:1]) == 0.0
    assert shadow.expected_observations(START, END, 6) == 83  # la última resuelve tras 28 h
    assert shadow.expected_observations(START, END, 42) == 11
    assert shadow.expected_observations(START, START, 6) == 0


def test_manifest_requires_midnight_utc_and_both_primaries():
    models = [
        {
            "version": 1,
            "kind": "har_x_live",
            "role": "primary",
            "horizon_candles": 6,
            "artifact_sha256": "a",
        },
        {
            "version": 3,
            "kind": "har_x_live",
            "role": "primary",
            "horizon_candles": 42,
            "artifact_sha256": "b",
        },
    ]
    manifest = shadow.build_manifest(START, models, NOW, "abc123")
    assert manifest["protocol"] == "volatility_shadow_v1" and manifest["git_commit"] == "abc123"
    assert manifest["criteria"]["weeks"] == 12
    json.dumps(manifest)
    with pytest.raises(ValueError, match="medianoche"):
        shadow.build_manifest(START + timedelta(hours=1), models, NOW, None)
    with pytest.raises(ValueError, match="medianoche"):
        shadow.build_manifest(datetime(2026, 11, 2), models, NOW, None)
    with pytest.raises(ValueError, match="primario"):
        shadow.build_manifest(START, models[:1], NOW, None)


# ── Integración ──────────────────────────────────────────────────────────────────────────


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


async def seed(engine, fallback_share=0.0, lag_hours=0.5, miss_forecasts=()):
    """12 semanas de velas, modelos activos, pronósticos resueltos y recepción de fuentes."""
    rng = np.random.default_rng(3)
    count = 6 * 7 * 12 + 12
    candles = make_candles(count, start="2026-11-02")
    await IngestionRepository(engine).upsert_candles("BTCUSDT", "4h", candles)
    async with engine.begin() as conn:
        for horizon in (6, 42):
            for role, kind in (("primary", "har_x_live"), ("fallback", "har")):
                await conn.execute(
                    volatility_models.insert().values(
                        kind=kind,
                        role=role,
                        horizon_candles=horizon,
                        artifact_path="/m.json",
                        artifact_sha256="a" * 64,
                        feature_names=["x"],
                        validation_metrics={},
                        is_active=True,
                        trained_at=START,
                    )
                )
        versions = {
            (r.horizon_candles, r.role): r.version
            for r in (await conn.execute(sa.select(volatility_models))).all()
        }
        ids = (
            await conn.execute(sa.text("select id, open_time from ohlcv_candles order by 2"))
        ).all()
        rows = []
        for index, (candle_id, opened) in enumerate(ids):
            if index in miss_forecasts:
                continue
            resolved = opened + timedelta(hours=4 * 45) <= NOW
            for horizon in (6, 42):
                use_fallback = rng.random() < fallback_share
                served = versions[(horizon, "fallback" if use_fallback else "primary")]
                rows.append(
                    {
                        "candle_id": candle_id,
                        "horizon_candles": horizon,
                        "model_version": served,
                        "requested_model_version": versions[(horizon, "primary")],
                        "fallback_reason": "dvol_stale" if use_fallback else None,
                        "features_as_of": {},
                        "forecast_var_annual": Decimal("0.25"),
                        "forecast_vol_annual": Decimal("0.5"),
                        "regime": "normal",
                        "reference_price": Decimal("50000"),
                        "realized_var_annual": Decimal("0.2") if resolved else None,
                        "realized_log_return": Decimal("0.01") if resolved else None,
                        "qlike": Decimal("0.1") if resolved else None,
                        "resolved_at": NOW if resolved else None,
                        # Patrón determinista por día con cobertura exactamente nominal
                        "inside_68": (index // 6) % 25 < 17 if resolved else None,
                        "inside_90": (index // 6) % 10 != 0 if resolved else None,
                        "inside_95": (index // 6) % 20 != 0 if resolved else None,
                    }
                )
        await conn.execute(volatility_forecasts.insert(), rows)
        hours = pd.date_range(START, END, freq="1h", tz="UTC")
        await conn.execute(
            implied_volatility.insert(),
            [
                {
                    "currency": "BTC",
                    "open_time": t.to_pydatetime(),
                    "close": Decimal("45"),
                    "recorded_at": (t + HOUR + timedelta(hours=lag_hours)).to_pydatetime(),
                }
                for t in hours
            ],
        )
        steps = pd.date_range(START, END, freq="30min", tz="UTC")
        await conn.execute(
            open_interest.insert(),
            [
                {
                    "symbol": "BTCUSDT",
                    "create_time": t.to_pydatetime(),
                    "sum_open_interest": Decimal("1"),
                    "sum_open_interest_value": Decimal("1"),
                    "recorded_at": (t + timedelta(hours=lag_hours)).to_pydatetime(),
                }
                for t in steps
            ],
        )


HOUR = timedelta(hours=1)


async def run(engine, tmp_path, now=NOW, **kwargs):
    await seed(engine, **kwargs)
    await shadow.init_shadow(engine, tmp_path, START, now=START - timedelta(days=1))
    return await shadow.make_report(engine, tmp_path, now=now)


async def test_a_clean_twelve_weeks_pass_every_criterion(engine, tmp_path):
    report = await run(engine, tmp_path)
    assert report["status"] == "passed", report["checks"]
    assert all(v is True for v in report["checks"].values())
    assert report["one_day"]["observations"] >= 75
    assert report["seven_day"]["observations"] >= 9
    assert report["seven_day"]["validated"] is False  # faltan 26 semanas
    assert report["recommended_validation_status"] == "in_validation"
    assert report["fallback"]["share"] == 0 and report["ingestion"]["max_forecast_gap_hours"] == 4.0
    assert (tmp_path / "shadow" / "manifest.json").exists()
    assert list((tmp_path / "shadow").glob("report-*.json"))
    json.dumps(report)


async def test_partial_window_is_in_progress_and_flags_failures_early(engine, tmp_path):
    report = await run(engine, tmp_path, now=START + timedelta(weeks=4), fallback_share=0.3)
    assert report["status"] == "in_progress"
    assert report["interim"] == "failing"
    assert report["checks"]["fallback_share_ok"] is False
    assert report["elapsed_weeks"] == 4.0


@pytest.mark.parametrize(
    ("kwargs", "check"),
    [
        ({"fallback_share": 0.2}, "fallback_share_ok"),
        ({"lag_hours": 9}, "ingestion_lag_ok"),
        ({"miss_forecasts": {100, 101, 300}}, "forecast_gaps_ok"),
    ],
)
async def test_each_failure_mode_fails_the_shadow(engine, tmp_path, kwargs, check):
    report = await run(engine, tmp_path, **kwargs)
    assert report["status"] == "failed"
    assert report["checks"][check] is False


async def test_models_must_stay_frozen_during_the_shadow(engine, tmp_path):
    await seed(engine)
    await shadow.init_shadow(engine, tmp_path, START, now=START - timedelta(days=1))
    async with engine.begin() as conn:
        await conn.execute(
            volatility_models.insert().values(
                kind="har",
                role="fallback",
                horizon_candles=6,
                artifact_path="/n.json",
                artifact_sha256="b" * 64,
                feature_names=["x"],
                validation_metrics={},
                is_active=False,
                trained_at=START,
            )
        )
        newest = (
            await conn.execute(sa.text("select max(version) from volatility_models"))
        ).scalar()
        await conn.execute(
            sa.text(
                "update volatility_forecasts set model_version=:v, requested_model_version=:v "
                "where id=(select min(id) from volatility_forecasts where horizon_candles=6)"
            ),
            {"v": newest},
        )
    report = await shadow.make_report(engine, tmp_path, now=NOW, save=False)
    assert report["checks"]["models_frozen"] is False
    assert report["status"] == "failed"


async def test_empty_database_reports_missing_evidence_not_success(engine, tmp_path):
    async with engine.begin() as conn:
        for horizon in (6, 42):
            await conn.execute(
                volatility_models.insert().values(
                    kind="har_x_live",
                    role="primary",
                    horizon_candles=horizon,
                    artifact_path="/m",
                    artifact_sha256="a" * 64,
                    feature_names=["x"],
                    validation_metrics={},
                    is_active=True,
                    trained_at=START,
                )
            )
    await shadow.init_shadow(engine, tmp_path, START, now=START)
    done = await shadow.make_report(engine, tmp_path, now=NOW, save=False)
    assert done["status"] == "failed" and done["one_day"]["observations"] == 0
    early = await shadow.make_report(engine, tmp_path, now=START + timedelta(days=2), save=False)
    assert early["status"] == "in_progress" and early["interim"] == "on_track"


async def test_protocol_cannot_be_rewritten_once_started(engine, tmp_path):
    await seed(engine)
    first = await shadow.init_shadow(
        engine, tmp_path, None, now=datetime(2026, 11, 1, 18, tzinfo=UTC)
    )
    assert first["start"] == "2026-11-02T00:00:00+00:00"  # próxima medianoche
    with pytest.raises(FileExistsError, match="ya está iniciada"):
        await shadow.init_shadow(engine, tmp_path, START)
    assert STEP == timedelta(hours=4)


def test_git_commit_prefers_the_declared_environment_value(monkeypatch):
    monkeypatch.setenv("SHADOW_GIT_COMMIT", "abc123-dirty")
    assert shadow.git_commit() == "abc123-dirty"
    monkeypatch.setenv("SHADOW_GIT_COMMIT", "unknown")
    assert shadow.git_commit() != "unknown"  # cae a git o a None
