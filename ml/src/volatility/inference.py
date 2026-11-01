"""Inferencia de volatilidad en el ML worker: un pronóstico por horizonte para la última vela.

La inferencia vive en el worker (no en el backend) porque sus entradas requieren pandas y el
pipeline de features; así no hay código duplicado ni diferencias entre entrenamiento y
producción. Usa `live_inputs` (el mismo código de la investigación). Si alguna fuente exógena
incumple su regla de frescura se sirve el respaldo (HAR) y se guardan `fallback_reason` y
`features_as_of`. Es idempotente por (vela, horizonte).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import sqlalchemy as sa
import structlog
from sqlalchemy.dialects.postgresql import insert

from src.ingestion.live_sources import LiveSourcesRepository
from src.ingestion.tables import (
    funding_rates,
    macro_daily,
    ohlcv_candles,
    volatility_forecasts,
    volatility_models,
)
from src.ingestion.time_utils import floor_to_step
from src.volatility import calibration
from src.volatility.features import (
    HAR_COLUMNS,
    HAR_X_LIVE_EXTRA,
    HORIZONS,
    LIVE_WINDOW_CANDLES,
    STEP,
    live_inputs,
)
from src.volatility.model import VolatilityArtifact, horizon_sigma, predict_variance, regime
from src.volatility.registry import file_sha256
from src.volatility.resolution import resolve_forecasts

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.config import Settings
    from src.training.types import FloatArray
    from src.volatility_config import VolatilitySettings

log = structlog.get_logger(__name__)
EVENT_TYPE = "fomc"
VIX_SYMBOL = "^VIX"
FUNDING_EVENTS = 120
DVOL_LOOKBACK = pd.Timedelta(days=30)
REASON_BY_COLUMN = {
    "log_dvol_var": "dvol_stale",
    "log_vix": "features_missing",
    "volume_sma_ratio": "features_missing",
    "funding_sum_7d": "funding_stale",
    "drawdown_90d": "insufficient_history",
    "abs_return_1d": "insufficient_history",
    "fomc_next_72h": "fomc_calendar_missing",
}
REASON_LIMIT = 100
MIN_CLOSES = 181  # 180 retornos para la varianza de 30 días


@dataclass(frozen=True)
class ActiveModel:
    version: int
    kind: str
    role: str
    horizon_candles: int
    artifact: VolatilityArtifact


@dataclass(frozen=True)
class Inputs:
    candle_id: int
    open_time: pd.Timestamp
    reference_price: float
    row: pd.DataFrame
    fallback_reasons: list[str]
    features_as_of: dict[str, Any]


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def _iso(value: pd.Timestamp | datetime | None) -> str | None:
    if value is None or value is pd.NaT:
        return None
    text: str = pd.Timestamp(value).isoformat()
    return text


async def load_active_models(engine: AsyncEngine) -> dict[tuple[int, str], ActiveModel]:
    """Modelos activos por (horizonte, rol); descarta artefactos ausentes o alterados."""
    async with engine.connect() as conn:
        rows = (
            (await conn.execute(sa.select(volatility_models).where(volatility_models.c.is_active)))
            .mappings()
            .all()
        )
    models: dict[tuple[int, str], ActiveModel] = {}
    for row in rows:
        path = Path(row["artifact_path"])
        if not path.is_file() or file_sha256(path) != row["artifact_sha256"]:
            log.error("volatility_artifact_unavailable", version=row["version"], path=str(path))
            continue
        artifact = VolatilityArtifact.load(path)
        problems = artifact.problems()
        if problems:
            log.error("volatility_artifact_invalid", version=row["version"], problems=problems)
            continue
        models[(row["horizon_candles"], row["role"])] = ActiveModel(
            row["version"], row["kind"], row["role"], row["horizon_candles"], artifact
        )
    return models


async def _recent_candles(engine: AsyncEngine, settings: Settings, now: datetime) -> pd.DataFrame:
    query = (
        sa.select(ohlcv_candles.c.id, ohlcv_candles.c.open_time, ohlcv_candles.c.close)
        .where(
            ohlcv_candles.c.symbol == settings.btc_symbol,
            ohlcv_candles.c.interval == settings.default_interval,
            ohlcv_candles.c.open_time + STEP.to_pytimedelta() <= now,
        )
        .order_by(ohlcv_candles.c.open_time.desc())
        .limit(LIVE_WINDOW_CANDLES + 50)
    )
    async with engine.connect() as conn:
        rows = (await conn.execute(query)).all()
    frame = pd.DataFrame(rows, columns=["id", "open_time", "close"]).iloc[::-1]
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame["close"] = frame["close"].astype(float)
    return frame.reset_index(drop=True)


def contiguous_tail(candles: pd.DataFrame) -> pd.DataFrame:
    """Último tramo sin huecos de velas de 4h (los huecos anteriores no sirven a la rejilla)."""
    gaps = np.flatnonzero(candles["open_time"].diff().to_numpy() != STEP.to_timedelta64())
    gaps = gaps[gaps > 0]
    start = int(gaps[-1]) if len(gaps) else 0
    return candles.iloc[start:].reset_index(drop=True)


async def _base_features(engine: AsyncEngine, candle_ids: list[int]) -> pd.DataFrame:
    table = sa.table(
        "computed_features",
        sa.column("candle_id"),
        sa.column("vix_close"),
        sa.column("volume_sma_ratio"),
    )
    query = sa.select(table.c.candle_id, table.c.vix_close, table.c.volume_sma_ratio).where(
        table.c.candle_id.in_(candle_ids)
    )
    async with engine.connect() as conn:
        rows = (await conn.execute(query)).all()
    frame = pd.DataFrame(rows, columns=["id", "vix_close", "volume_sma_ratio"])
    for name in ("vix_close", "volume_sma_ratio"):
        frame[name] = frame[name].astype(float)
    return frame


async def _funding(engine: AsyncEngine, symbol: str, until: datetime) -> pd.DataFrame:
    query = (
        sa.select(funding_rates.c.funding_time, funding_rates.c.funding_rate)
        .where(funding_rates.c.symbol == symbol, funding_rates.c.funding_time <= until)
        .order_by(funding_rates.c.funding_time.desc())
        .limit(FUNDING_EVENTS)
    )
    async with engine.connect() as conn:
        rows = (await conn.execute(query)).all()
    frame = pd.DataFrame(rows, columns=["funding_time", "funding_rate"]).iloc[::-1]
    frame["funding_time"] = pd.to_datetime(frame["funding_time"], utc=True)
    frame["funding_rate"] = frame["funding_rate"].astype(float)
    return frame.reset_index(drop=True)


async def _latest_vix_date(engine: AsyncEngine) -> date | None:
    query = sa.select(sa.func.max(macro_daily.c.trade_date)).where(
        macro_daily.c.symbol == VIX_SYMBOL
    )
    async with engine.connect() as conn:
        latest: date | None = (await conn.execute(query)).scalar()
    return latest


async def build_inputs(
    engine: AsyncEngine, settings: Settings, live: VolatilitySettings, now: datetime
) -> Inputs | None:
    """Entradas para la última vela cerrada, o `None` si no hay datos para pronosticar."""
    candles = contiguous_tail(await _recent_candles(engine, settings, now))
    expected = floor_to_step(now, STEP) - STEP
    if candles.empty or candles["open_time"].iloc[-1] != expected:
        log.warning("volatility_inference_skipped", reason="last_candle_not_current")
        return None
    if len(candles) < MIN_CLOSES:
        log.warning(
            "volatility_inference_skipped", reason="insufficient_candles", rows=len(candles)
        )
        return None
    decision = candles["open_time"].iloc[-1] + STEP
    features = await _base_features(engine, [int(i) for i in candles["id"]])
    base = candles[["id", "open_time"]].merge(features, on="id", how="left").drop(columns="id")
    store = LiveSourcesRepository(engine)
    funding = await _funding(engine, settings.btc_symbol, decision.to_pydatetime())
    # Una vela horaria solo está disponible al cerrar: `open_time + 1h ≤ decision`.
    usable_until = decision - pd.Timedelta(hours=1)
    dvol = await store.load_dvol(
        live.dvol_currency,
        (decision - DVOL_LOOKBACK).to_pydatetime(),
        usable_until.to_pydatetime(),
    )
    events = await store.load_event_times(EVENT_TYPE)
    upcoming = events.loc[events["decision_time"] > decision, "decision_time"]
    fomc = events if not upcoming.empty else None  # sin reunión futura el calendario no cubre
    close = pd.Series(candles["close"].to_numpy(), index=pd.DatetimeIndex(candles["open_time"]))
    row = live_inputs(base, close, funding, dvol, fomc).iloc[[-1]].reset_index(drop=True)
    columns = [*HAR_COLUMNS, *HAR_X_LIVE_EXTRA]
    if not np.isfinite(row[HAR_COLUMNS].to_numpy(dtype=float)).all():
        log.warning("volatility_inference_skipped", reason="har_inputs_unavailable")
        return None
    missing = [c for c in columns if not np.isfinite(row.loc[0, c])]
    reasons = sorted({REASON_BY_COLUMN[c] for c in missing})
    vix_date = await _latest_vix_date(engine)
    funding_available = funding["funding_time"] + STEP
    as_of = {
        "decision_time": _iso(decision),
        "candle_open_time": _iso(candles["open_time"].iloc[-1]),
        "dvol_open_time": _iso(dvol["open_time"].max()) if not dvol.empty else None,
        "funding_time": _iso(funding.loc[funding_available <= decision, "funding_time"].max())
        if not funding.empty
        else None,
        "vix_trade_date": None if vix_date is None else vix_date.isoformat(),
        "fomc_next": _iso(upcoming.min()) if not upcoming.empty else None,
    }
    return Inputs(
        candle_id=int(candles["id"].iloc[-1]),
        open_time=candles["open_time"].iloc[-1],
        reference_price=float(candles["close"].iloc[-1]),
        row=row,
        fallback_reasons=reasons,
        features_as_of=as_of,
    )


async def _live_calibration(
    engine: AsyncEngine, version: int, horizon: int
) -> tuple[pd.Series, pd.Series, FloatArray]:
    """Pronósticos ya resueltos del modelo: (emitido, resuelto, z) en tiempos nominales."""
    table = volatility_forecasts
    query = (
        sa.select(
            ohlcv_candles.c.open_time, table.c.forecast_var_annual, table.c.realized_log_return
        )
        .join(ohlcv_candles, table.c.candle_id == ohlcv_candles.c.id)
        .where(
            table.c.model_version == version,
            table.c.horizon_candles == horizon,
            table.c.resolved_at.is_not(None),
        )
        .order_by(ohlcv_candles.c.open_time)
    )
    async with engine.connect() as conn:
        rows = (await conn.execute(query)).all()
    issued = pd.Series(pd.to_datetime([r[0] for r in rows], utc=True)) + STEP
    var = np.array([float(r[1]) for r in rows])
    returns = np.array([float(r[2]) for r in rows])
    z = calibration.standardized_residual(returns, horizon_sigma(var, horizon))
    return issued.reset_index(drop=True), (issued + horizon * STEP).reset_index(drop=True), z


async def band_multipliers(
    engine: AsyncEngine, model: ActiveModel, decision: pd.Timestamp
) -> tuple[dict[float, float], pd.Timestamp | None]:
    """Multiplicadores y fin de ventana: semilla del modelo más sus pronósticos en vivo."""
    seed = model.artifact.calibration_seed
    issued = pd.Series(pd.to_datetime(seed.get("issued_at", []), utc=True))
    resolved = pd.Series(pd.to_datetime(seed.get("resolved_at", []), utc=True))
    z = np.asarray(seed.get("z", []), dtype=float)
    live_issued, live_resolved, live_z = await _live_calibration(
        engine, model.version, model.horizon_candles
    )
    if len(live_issued):
        newer = (
            (live_issued > issued.max()).to_numpy() if len(issued) else np.ones(len(live_z), bool)
        )
        issued = pd.concat([issued, live_issued[newer]], ignore_index=True)
        resolved = pd.concat([resolved, live_resolved[newer]], ignore_index=True)
        z = np.concatenate([z, live_z[newer]])
    k = calibration.multipliers_at(
        decision, issued, resolved, z, min_count=calibration.MIN_RESOLVED
    )
    if any(np.isnan(value) for value in k.values()):
        return k, None
    usable = calibration.eligible(decision, issued, resolved, z)
    return k, pd.Timestamp(resolved[usable].max())


def _forecast_row(
    inputs: Inputs,
    horizon: int,
    primary: ActiveModel,
    served: ActiveModel,
    var: float,
    k: dict[float, float],
    window_end: pd.Timestamp | None,
) -> dict[str, Any]:
    vol = float(np.sqrt(var))
    reasons = ",".join(inputs.fallback_reasons)[:REASON_LIMIT] if served is not primary else None
    known = window_end is not None
    return {
        "candle_id": inputs.candle_id,
        "horizon_candles": horizon,
        "model_version": served.version,
        "requested_model_version": primary.version,
        "fallback_reason": reasons,
        "features_as_of": json.loads(json.dumps(inputs.features_as_of)),
        "forecast_var_annual": _decimal(var),
        "forecast_vol_annual": _decimal(vol),
        "k68": _decimal(k[0.68]) if known else None,
        "k90": _decimal(k[0.90]) if known else None,
        "k95": _decimal(k[0.95]) if known else None,
        "calibration_window_end": window_end.to_pydatetime() if known and window_end else None,
        "regime": regime(vol, served.artifact.regime_thresholds),
        "reference_price": _decimal(inputs.reference_price),
    }


async def run_inference(
    engine: AsyncEngine,
    settings: Settings,
    live: VolatilitySettings,
    now: datetime | None = None,
) -> int:
    """Guarda el pronóstico de la última vela cerrada para cada horizonte; devuelve los nuevos."""
    now = now or datetime.now(UTC)
    models = await load_active_models(engine)
    if not models:
        log.info("volatility_inference_skipped", reason="no_active_models")
        return 0
    inputs = await build_inputs(engine, settings, live, now)
    if inputs is None:
        return 0
    decision = inputs.open_time + STEP
    created = 0
    for horizon in sorted(HORIZONS.values()):
        primary = models.get((horizon, "primary"))
        fallback = models.get((horizon, "fallback"))
        served = primary if not inputs.fallback_reasons else fallback
        if primary is None or served is None:
            log.warning("volatility_inference_skipped", horizon=horizon, reason="model_unavailable")
            continue
        var = float(predict_variance(served.artifact, inputs.row)[0])
        if not np.isfinite(var) or var <= 0:
            log.error("volatility_forecast_invalid", horizon=horizon, version=served.version)
            continue
        k, window_end = await band_multipliers(engine, served, decision)
        row = _forecast_row(inputs, horizon, primary, served, var, k, window_end)
        statement = insert(volatility_forecasts).values(row)
        statement = statement.on_conflict_do_nothing(
            index_elements=["candle_id", "horizon_candles"]
        )
        async with engine.begin() as conn:
            inserted = (await conn.execute(statement)).rowcount or 0
        created += inserted
        log.info(
            "volatility_forecast_saved" if inserted else "volatility_forecast_exists",
            horizon=horizon,
            version=served.version,
            fallback=row["fallback_reason"],
            vol_annual=round(float(row["forecast_vol_annual"]), 4),
        )
    return created


async def run_volatility_cycle(
    engine: AsyncEngine,
    settings: Settings,
    live: VolatilitySettings,
    now: datetime | None = None,
) -> dict[str, int]:
    """Resuelve los pronósticos vencidos y emite los nuevos (cada fase es independiente)."""
    summary = {"resolved": 0, "created": 0}
    try:
        summary["resolved"] = await resolve_forecasts(engine, settings, now)
    except Exception:
        log.exception("volatility_resolution_failed")
    try:
        summary["created"] = await run_inference(engine, settings, live, now)
    except Exception:
        log.exception("volatility_inference_failed")
    return summary
