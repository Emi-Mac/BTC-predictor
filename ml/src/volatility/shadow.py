"""Sombra prospectiva de 12 semanas (criterio 4 del plan de integración de volatilidad).

Un protocolo congelado en un manifiesto (`init`) y un evaluador (`report`) que lee solo lo que
el ML worker guardó en `volatility_forecasts` y en las tablas de ingesta, sin recalcular nada:

- **1 día:** una observación sin solapamiento por día (el pronóstico emitido en la vela de
  00:00 UTC; 84 en 12 semanas). Pasa si el IC binomial exacto del 95% de la cobertura de las
  bandas de 68% y 90% contiene el nivel nominal. Con 84 datos el IC es de ±10 pp: es una
  prueba de cordura, no de precisión.
- **7 días:** una observación semanal (vela de 00:00 UTC del lunes; 12 en 12 semanas). Solo se
  monitorea; el horizonte de 7 días sigue «en validación» hasta acumular 26 semanas.
- **Ingesta:** ninguna fuente (DVOL, OI) registrada con más de 6 h de rezago respecto del
  dato, ni huecos de más de 6 h en los pronósticos.
- **Respaldo:** menos de 5% de los pronósticos servidos con el modelo de respaldo.

Mientras no se cumplan las 12 semanas el informe es provisional (`in_progress`).

    python -m src.volatility.shadow init --start 2026-11-03
    python -m src.volatility.shadow report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa
import structlog
from scipy.stats import binomtest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.config import get_settings
from src.ingestion.run import configure_logging
from src.ingestion.tables import (
    implied_volatility,
    ohlcv_candles,
    open_interest,
    volatility_forecasts,
    volatility_models,
)

log = structlog.get_logger(__name__)
SHADOW_DIR = "shadow"
MANIFEST_NAME = "manifest.json"
ONE_DAY, SEVEN_DAY = 6, 42
HOUR = timedelta(hours=1)


@dataclass(frozen=True)
class Criteria:
    """Criterios preespecificados; se guardan en el manifiesto y no cambian durante la sombra."""

    weeks: int = 12
    weeks_for_seven_day_validation: int = 26
    confidence: float = 0.95
    one_day_levels: tuple[float, ...] = (0.68, 0.90)
    seven_day_levels: tuple[float, ...] = (0.68, 0.90)
    min_sample_share: float = 0.9
    max_ingestion_lag_hours: float = 6.0
    max_forecast_gap_hours: float = 6.0
    max_fallback_share: float = 0.05
    sample_hour_utc: int = 0
    seven_day_weekday: int = 0  # lunes


CRITERIA = Criteria()


# ─── Estadística pura ────────────────────────────────────────────────────────


def exact_interval(hits: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """IC binomial exacto (Clopper-Pearson) de la proporción."""
    if n <= 0 or not 0 <= hits <= n:
        raise ValueError("Se requieren n > 0 y 0 <= aciertos <= n")
    interval = binomtest(hits, n).proportion_ci(confidence_level=confidence, method="exact")
    return float(interval.low), float(interval.high)


def coverage_check(flags: pd.Series, nominal: float, confidence: float) -> dict[str, Any]:
    """Cobertura observada, su IC exacto y si contiene el nivel nominal."""
    known = flags.dropna().astype(bool)
    n = len(known)
    if n == 0:
        return {"nominal": nominal, "n": 0, "hits": 0, "coverage": None, "ci": None, "ok": None}
    hits = int(known.sum())
    low, high = exact_interval(hits, n, confidence)
    return {
        "nominal": nominal,
        "n": n,
        "hits": hits,
        "coverage": hits / n,
        "ci": [low, high],
        "ok": bool(low <= nominal <= high),
    }


def non_overlapping(
    frame: pd.DataFrame, horizon: int, criteria: Criteria = CRITERIA
) -> pd.DataFrame:
    """Observaciones sin solapamiento: 00:00 UTC diario (1d) o lunes 00:00 UTC (7d)."""
    rows = frame[frame["horizon_candles"] == horizon]
    opened = rows["open_time"]
    mask = opened.dt.hour == criteria.sample_hour_utc
    if horizon == SEVEN_DAY:
        mask &= opened.dt.day_of_week == criteria.seven_day_weekday
    return rows[mask & (opened.dt.minute == 0)].sort_values("open_time")


def max_gap(times: pd.Series) -> float:
    """Mayor hueco, en horas, entre instantes consecutivos (0 con menos de dos)."""
    ordered = pd.to_datetime(times, utc=True).sort_values()
    if len(ordered) < 2:
        return 0.0
    return float(ordered.diff().dropna().max() / HOUR)


def expected_observations(start: datetime, end: datetime, horizon: int) -> int:
    """Observaciones sin solapamiento (una por día o semana desde `start`) ya resueltas."""
    resolution = timedelta(hours=4 * (horizon + 1))
    span = end - resolution - start
    if span < timedelta(0):
        return 0
    period = timedelta(days=1) if horizon == ONE_DAY else timedelta(weeks=1)
    return int(span / period) + 1


# ─── Manifiesto ──────────────────────────────────────────────────────────────


def manifest_path(artifact_root: Path) -> Path:
    return artifact_root / SHADOW_DIR / MANIFEST_NAME


def git_commit() -> str | None:
    """Commit evaluado: `SHADOW_GIT_COMMIT` (lo fija shadow.sh; el contenedor no tiene git)."""
    declared = os.environ.get("SHADOW_GIT_COMMIT", "").strip()
    if declared and declared != "unknown":
        return declared
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


async def active_models(engine: AsyncEngine) -> list[dict[str, Any]]:
    query = (
        sa.select(
            volatility_models.c.version,
            volatility_models.c.kind,
            volatility_models.c.role,
            volatility_models.c.horizon_candles,
            volatility_models.c.artifact_sha256,
        )
        .where(volatility_models.c.is_active)
        .order_by(volatility_models.c.horizon_candles, volatility_models.c.role)
    )
    async with engine.connect() as conn:
        return [dict(row) for row in (await conn.execute(query)).mappings()]


def build_manifest(
    start: datetime, models: list[dict[str, Any]], created_at: datetime, commit: str | None
) -> dict[str, Any]:
    if start.tzinfo is None or start != start.replace(hour=0, minute=0, second=0, microsecond=0):
        raise ValueError("El inicio debe ser medianoche UTC con zona horaria")
    primaries = {m["horizon_candles"] for m in models if m["role"] == "primary"}
    if primaries != {ONE_DAY, SEVEN_DAY}:
        raise ValueError("Se requiere un primario activo para 1d y para 7d")
    return {
        "protocol": "volatility_shadow_v1",
        "start": start.isoformat(),
        "created_at": created_at.isoformat(),
        "criteria": asdict(CRITERIA),
        "models": models,
        "git_commit": commit,
        "note": "Modelos congelados durante la sombra; un cambio de versión invalida la prueba.",
    }


# ─── Lectura de datos ────────────────────────────────────────────────────────


async def load_frames(engine: AsyncEngine, start: datetime) -> dict[str, pd.DataFrame]:
    """Pronósticos emitidos desde `start` y marcas de recepción de las fuentes."""
    forecasts = (
        sa.select(
            ohlcv_candles.c.open_time,
            volatility_forecasts.c.horizon_candles,
            volatility_forecasts.c.model_version,
            volatility_forecasts.c.requested_model_version,
            volatility_forecasts.c.fallback_reason,
            volatility_forecasts.c.inside_68,
            volatility_forecasts.c.inside_90,
            volatility_forecasts.c.inside_95,
            volatility_forecasts.c.resolved_at,
        )
        .join(ohlcv_candles, volatility_forecasts.c.candle_id == ohlcv_candles.c.id)
        .where(ohlcv_candles.c.open_time >= start)
        .order_by(ohlcv_candles.c.open_time)
    )
    dvol = sa.select(implied_volatility.c.open_time, implied_volatility.c.recorded_at).where(
        implied_volatility.c.open_time >= start
    )
    oi = sa.select(open_interest.c.create_time, open_interest.c.recorded_at).where(
        open_interest.c.create_time >= start
    )
    async with engine.connect() as conn:
        frames = {
            "forecasts": pd.DataFrame((await conn.execute(forecasts)).all()),
            "dvol": pd.DataFrame((await conn.execute(dvol)).all()),
            "oi": pd.DataFrame((await conn.execute(oi)).all()),
        }
    for name, columns in (
        (
            "forecasts",
            [
                "open_time",
                "horizon_candles",
                "model_version",
                "requested_model_version",
                "fallback_reason",
                "inside_68",
                "inside_90",
                "inside_95",
                "resolved_at",
            ],
        ),
        ("dvol", ["open_time", "recorded_at"]),
        ("oi", ["create_time", "recorded_at"]),
    ):
        if frames[name].empty:
            frames[name] = pd.DataFrame(columns=columns)
    for name, columns in (
        ("forecasts", ["open_time", "resolved_at"]),
        ("dvol", ["open_time", "recorded_at"]),
        ("oi", ["create_time", "recorded_at"]),
    ):
        for column in columns:
            frames[name][column] = pd.to_datetime(frames[name][column], utc=True)
    return frames


# ─── Informe ─────────────────────────────────────────────────────────────────


def ingestion_lag(
    frame: pd.DataFrame, time_column: str, available_after: timedelta
) -> dict[str, Any]:
    """Rezago entre que el dato estuvo disponible y que el worker lo registró."""
    if frame.empty:
        return {"rows": 0, "max_lag_hours": None}
    lag = frame["recorded_at"] - (frame[time_column] + available_after)
    return {"rows": len(frame), "max_lag_hours": float(lag.max() / HOUR)}


def horizon_section(
    frame: pd.DataFrame,
    horizon: int,
    levels: tuple[float, ...],
    start: datetime,
    now: datetime,
    criteria: Criteria,
) -> dict[str, Any]:
    sample = non_overlapping(frame, horizon, criteria)
    resolved = sample[sample["resolved_at"].notna()]
    expected = expected_observations(start, now, horizon)
    checks = {
        f"coverage_{round(level * 100)}": coverage_check(
            resolved[f"inside_{round(level * 100)}"], level, criteria.confidence
        )
        for level in levels
    }
    informative = coverage_check(resolved["inside_95"], 0.95, criteria.confidence)
    return {
        "horizon_candles": horizon,
        "observations": len(resolved),
        "expected_observations": expected,
        "sample_share": (len(resolved) / expected) if expected else None,
        "checks": checks,
        "informative_coverage_95": informative,
    }


def build_report(
    manifest: dict[str, Any], frames: dict[str, pd.DataFrame], now: datetime
) -> dict[str, Any]:
    criteria = Criteria(
        **{
            **manifest["criteria"],
            "one_day_levels": tuple(manifest["criteria"]["one_day_levels"]),
            "seven_day_levels": tuple(manifest["criteria"]["seven_day_levels"]),
        }
    )
    start = datetime.fromisoformat(manifest["start"])
    end = start + timedelta(weeks=criteria.weeks)
    elapsed_weeks = max(0.0, (now - start) / timedelta(weeks=1))
    complete = now >= end
    forecasts = frames["forecasts"]
    window_end = min(now, end)

    one_day = horizon_section(
        forecasts, ONE_DAY, criteria.one_day_levels, start, window_end, criteria
    )
    seven_day = horizon_section(
        forecasts, SEVEN_DAY, criteria.seven_day_levels, start, window_end, criteria
    )

    lag_dvol = ingestion_lag(frames["dvol"], "open_time", HOUR)
    lag_oi = ingestion_lag(frames["oi"], "create_time", timedelta(0))
    lags = [x["max_lag_hours"] for x in (lag_dvol, lag_oi) if x["max_lag_hours"] is not None]
    emitted = forecasts[forecasts["horizon_candles"] == ONE_DAY]
    forecast_gap = max_gap(emitted["open_time"]) if len(emitted) else None
    served = len(forecasts)
    fallback = int(forecasts["fallback_reason"].notna().sum()) if served else 0
    fallback_share = fallback / served if served else None
    versions = sorted(set(forecasts["model_version"])) if served else []
    frozen = {m["version"] for m in manifest["models"]}

    checks = {
        # Solo es exigible al cerrar la ventana: antes, la muestra aún se está acumulando.
        "one_day_sample_sufficient": (
            None
            if not complete or one_day["sample_share"] is None
            else one_day["sample_share"] >= criteria.min_sample_share
        ),
        **{f"one_day_{k}": v["ok"] for k, v in one_day["checks"].items()},
        "ingestion_lag_ok": (None if not lags else max(lags) <= criteria.max_ingestion_lag_hours),
        "forecast_gaps_ok": (
            None if forecast_gap is None else forecast_gap <= criteria.max_forecast_gap_hours
        ),
        "fallback_share_ok": (
            None if fallback_share is None else fallback_share < criteria.max_fallback_share
        ),
        "models_frozen": set(versions) <= frozen if versions else None,
    }
    known = [v for v in checks.values() if v is not None]
    if not complete:
        status = "in_progress"
        interim = "failing" if any(v is False for v in known) else "on_track"
    else:
        status = "passed" if known and all(known) and len(known) == len(checks) else "failed"
        interim = status
    seven_validated = elapsed_weeks >= criteria.weeks_for_seven_day_validation and all(
        v["ok"] for v in seven_day["checks"].values()
    )
    return {
        "protocol": manifest["protocol"],
        "generated_at": now.isoformat(),
        "start": manifest["start"],
        "end": end.isoformat(),
        "elapsed_weeks": round(elapsed_weeks, 2),
        "status": status,
        "interim": interim,
        "checks": checks,
        "one_day": one_day,
        "seven_day": {
            **seven_day,
            "monitor_only": True,
            "validated": bool(seven_validated),
            "weeks_until_validation": max(
                0.0, round(criteria.weeks_for_seven_day_validation - elapsed_weeks, 2)
            ),
        },
        "ingestion": {
            "dvol": lag_dvol,
            "open_interest": lag_oi,
            "max_forecast_gap_hours": forecast_gap,
        },
        "fallback": {"served": served, "fallback": fallback, "share": fallback_share},
        "model_versions_seen": versions,
        "recommended_validation_status": (
            "validated" if status == "passed" and seven_validated else "in_validation"
        ),
        "caveat": (
            "Con 84 observaciones el IC es de ±10 pp: es una prueba de cordura, no de precisión. "
            "El horizonte de 7 días sigue en validación hasta acumular 26 semanas."
        ),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _next_midnight(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


async def init_shadow(
    engine: AsyncEngine, root: Path, start: datetime | None, now: datetime | None = None
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    path = manifest_path(root)
    if path.exists():
        raise FileExistsError(f"La sombra ya está iniciada ({path}); el protocolo no se reescribe")
    manifest = build_manifest(
        start or _next_midnight(now), await active_models(engine), now, git_commit()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("shadow_initialized", path=str(path), start=manifest["start"])
    return manifest


async def make_report(
    engine: AsyncEngine, root: Path, now: datetime | None = None, save: bool = True
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    manifest = json.loads(manifest_path(root).read_text(encoding="utf-8"))
    frames = await load_frames(engine, datetime.fromisoformat(manifest["start"]))
    report = build_report(manifest, frames, now)
    if save:
        target = root / SHADOW_DIR / f"report-{now:%Y%m%dT%H%M%SZ}.json"
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


async def _main(args: argparse.Namespace) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    root = Path(settings.model_artifact_dir)
    try:
        if args.command == "init":
            start = datetime.fromisoformat(args.start).replace(tzinfo=UTC) if args.start else None
            result = await init_shadow(engine, root, start)
        else:
            result = await make_report(engine, root)
        print(json.dumps(result, indent=2))  # noqa: T201
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sombra prospectiva de volatilidad (12 semanas)")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="congela el protocolo y los modelos activos")
    init.add_argument("--start", help="YYYY-MM-DD (UTC); por defecto, la próxima medianoche")
    sub.add_parser("report", help="evalúa los criterios con los datos acumulados")
    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
