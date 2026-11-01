"""Resolución de pronósticos de volatilidad cuando se cumple su horizonte.

Invariantes (probados contra `volatility_frame`, la definición del objetivo de entrenamiento):

- Varianza realizada: media de los cuadrados de los retornos log de las velas `t+1 … t+h`
  (`log c[t+i] − log c[t+i−1]`, i = 1…h), anualizada con 6·365 periodos.
- Cobertura de las bandas: `log(c[t+h] / c[t])` frente a `k · σ̂_h`.
- Solo se resuelve cuando existen las `h` velas siguientes, sin huecos y ya cerradas.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import sqlalchemy as sa
import structlog

from src.evaluation.economic import PERIODS_PER_YEAR_4H
from src.ingestion.tables import ohlcv_candles, volatility_forecasts
from src.volatility.features import STEP
from src.volatility.model import horizon_sigma

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.config import Settings
    from src.training.types import FloatArray

log = structlog.get_logger(__name__)
EPSILON = 1e-12
LEVELS = ("68", "90", "95")


def realized_statistics(closes: FloatArray) -> tuple[float, float]:
    """(varianza realizada anualizada, retorno log total) de `h+1` cierres consecutivos."""
    values = np.asarray(closes, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Se requieren al menos dos cierres positivos y finitos")
    returns = np.diff(np.log(values))
    variance = float(np.mean(returns**2) * PERIODS_PER_YEAR_4H)
    return variance, float(np.log(values[-1] / values[0]))


def qlike_loss(realized_var: float, forecast_var: float) -> float:
    ratio = max(realized_var / max(forecast_var, EPSILON), EPSILON)
    return float(ratio - np.log(ratio) - 1)


def resolution_values(
    closes: FloatArray, horizon: int, forecast_var: float, k: dict[str, float | None]
) -> dict[str, Any]:
    """Campos de resolución de un pronóstico, dados los cierres de `t … t+h`."""
    if len(closes) != horizon + 1:
        raise ValueError("Se requieren exactamente h+1 cierres")
    realized_var, log_return = realized_statistics(closes)
    sigma = float(horizon_sigma(np.array([forecast_var]), horizon)[0])
    z = abs(log_return) / max(sigma, EPSILON)
    inside = {
        f"inside_{level}": None if k[level] is None else bool(z <= float(k[level] or 0))
        for level in LEVELS
    }
    return {
        "realized_var_annual": realized_var,
        "realized_log_return": log_return,
        "qlike": qlike_loss(realized_var, forecast_var),
        **inside,
    }


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


async def resolve_forecasts(
    engine: AsyncEngine, settings: Settings, now: datetime | None = None
) -> int:
    """Resuelve los pronósticos cuyo horizonte ya se cumplió; devuelve cuántos resolvió."""
    now = now or datetime.now(UTC)
    table = volatility_forecasts
    pending_query = (
        sa.select(
            table.c.id,
            table.c.horizon_candles,
            table.c.forecast_var_annual,
            table.c.k68,
            table.c.k90,
            table.c.k95,
            ohlcv_candles.c.open_time,
        )
        .join(ohlcv_candles, table.c.candle_id == ohlcv_candles.c.id)
        .where(table.c.resolved_at.is_(None))
        .order_by(ohlcv_candles.c.open_time)
    )
    async with engine.connect() as conn:
        pending = (await conn.execute(pending_query)).mappings().all()
        if not pending:
            return 0
        first = min(row["open_time"] for row in pending)
        last = max(row["open_time"] + row["horizon_candles"] * STEP for row in pending)
        candles = (
            await conn.execute(
                sa.select(ohlcv_candles.c.open_time, ohlcv_candles.c.close).where(
                    ohlcv_candles.c.symbol == settings.btc_symbol,
                    ohlcv_candles.c.interval == settings.default_interval,
                    ohlcv_candles.c.open_time >= first,
                    ohlcv_candles.c.open_time <= last,
                )
            )
        ).all()
    close = {pd.Timestamp(time): float(price) for time, price in candles}
    updates: list[tuple[int, dict[str, Any]]] = []
    for row in pending:
        start = pd.Timestamp(row["open_time"])
        horizon = row["horizon_candles"]
        times = [start + i * STEP for i in range(horizon + 1)]
        if times[-1] + STEP > pd.Timestamp(now):  # la última vela aún no cerró
            continue
        if any(time not in close for time in times):
            log.warning("volatility_forecast_unresolvable_gap", forecast_id=row["id"])
            continue
        k = {
            level: None if row[f"k{level}"] is None else float(row[f"k{level}"]) for level in LEVELS
        }
        values = resolution_values(
            np.array([close[time] for time in times]), horizon, float(row["forecast_var_annual"]), k
        )
        updates.append((row["id"], values))
    async with engine.begin() as conn:
        for forecast_id, values in updates:
            payload = {
                **values,
                "realized_var_annual": _decimal(values["realized_var_annual"]),
                "realized_log_return": _decimal(values["realized_log_return"]),
                "qlike": _decimal(values["qlike"]),
                "resolved_at": now,
            }
            await conn.execute(
                table.update()
                .where(table.c.id == forecast_id, table.c.resolved_at.is_(None))
                .values(**payload)
            )
    log.info("volatility_forecasts_resolved", resolved=len(updates), pending=len(pending))
    return len(updates)
