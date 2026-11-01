"""Adaptador outbound: `VolatilityRepositoryPort` sobre PostgreSQL (SQLAlchemy 2.0 async)."""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config.models import (
    ImpliedVolatilityORM,
    OhlcvCandleORM,
    VolatilityForecastORM,
)
from src.domain.entities.volatility import ResolvedForecast, StoredForecast
from src.domain.ports.volatility_repository_port import VolatilityRepositoryPort
from src.domain.volatility_math import realized_variance

DVOL_CURRENCY = "BTC"


def _opt(value: float | Decimal | None) -> float | None:
    return None if value is None else float(value)


def _to_entity(row: VolatilityForecastORM, open_time: datetime) -> StoredForecast:
    return StoredForecast(
        id=row.id,
        candle_id=row.candle_id,
        candle_open_time=open_time,
        horizon_candles=row.horizon_candles,
        model_version=row.model_version,
        requested_model_version=row.requested_model_version,
        fallback_reason=row.fallback_reason,
        forecast_var_annual=float(row.forecast_var_annual),
        forecast_vol_annual=float(row.forecast_vol_annual),
        k68=_opt(row.k68),
        k90=_opt(row.k90),
        k95=_opt(row.k95),
        regime=row.regime,
        reference_price=float(row.reference_price),
        realized_var_annual=_opt(row.realized_var_annual),
        realized_log_return=_opt(row.realized_log_return),
        inside_68=row.inside_68,
        inside_90=row.inside_90,
        inside_95=row.inside_95,
        resolved_at=row.resolved_at,
    )


class PostgresVolatilityRepository(VolatilityRepositoryPort):
    """Lectura de pronósticos de volatilidad (el ML worker es el único que escribe)."""

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        symbol: str,
        interval: str,
        step: timedelta = timedelta(hours=4),
    ) -> None:
        self._sessionmaker = sessionmaker
        self._symbol = symbol
        self._interval = interval
        self._step = step

    async def get_latest(self, horizon_candles: int) -> StoredForecast | None:
        async with self._sessionmaker() as session:
            result = await session.execute(
                sa.select(VolatilityForecastORM, OhlcvCandleORM.open_time)
                .join(OhlcvCandleORM, VolatilityForecastORM.candle_id == OhlcvCandleORM.id)
                .where(VolatilityForecastORM.horizon_candles == horizon_candles)
                .order_by(OhlcvCandleORM.open_time.desc())
                .limit(1)
            )
            record = result.first()
            return None if record is None else _to_entity(*record)

    async def list_history(
        self, horizon_candles: int, limit: int, offset: int
    ) -> tuple[list[StoredForecast], int]:
        async with self._sessionmaker() as session:
            total = await session.scalar(
                sa.select(sa.func.count())
                .select_from(VolatilityForecastORM)
                .where(VolatilityForecastORM.horizon_candles == horizon_candles)
            )
            result = await session.execute(
                sa.select(VolatilityForecastORM, OhlcvCandleORM.open_time)
                .join(OhlcvCandleORM, VolatilityForecastORM.candle_id == OhlcvCandleORM.id)
                .where(VolatilityForecastORM.horizon_candles == horizon_candles)
                .order_by(OhlcvCandleORM.open_time.desc())
                .limit(limit)
                .offset(offset)
            )
            return [_to_entity(row, time) for row, time in result], int(total or 0)

    async def list_resolved(self, horizon_candles: int) -> list[ResolvedForecast]:
        async with self._sessionmaker() as session:
            result = await session.execute(
                sa.select(VolatilityForecastORM, OhlcvCandleORM.open_time)
                .join(OhlcvCandleORM, VolatilityForecastORM.candle_id == OhlcvCandleORM.id)
                .where(
                    VolatilityForecastORM.horizon_candles == horizon_candles,
                    VolatilityForecastORM.resolved_at.is_not(None),
                )
                .order_by(OhlcvCandleORM.open_time)
            )
            rows = [(row, time) for row, time in result]
            if not rows:
                return []
            first = rows[0][1] - horizon_candles * self._step
            last = rows[-1][1]
            candles = await session.execute(
                sa.select(OhlcvCandleORM.open_time, OhlcvCandleORM.close).where(
                    OhlcvCandleORM.symbol == self._symbol,
                    OhlcvCandleORM.interval == self._interval,
                    OhlcvCandleORM.open_time >= first,
                    OhlcvCandleORM.open_time <= last,
                )
            )
            close = {time: float(price) for time, price in candles}
        return [self._resolved(row, time, horizon_candles, close) for row, time in rows]

    def _resolved(
        self,
        row: VolatilityForecastORM,
        open_time: datetime,
        horizon_candles: int,
        close: dict[datetime, float],
    ) -> ResolvedForecast:
        # Pronóstico ingenuo: varianza realizada de las `h` velas previas a la emisión.
        times = [open_time - (horizon_candles - i) * self._step for i in range(horizon_candles + 1)]
        window = [close[t] for t in times if t in close]
        naive = realized_variance(window) if len(window) == len(times) else None
        return ResolvedForecast(
            candle_open_time=open_time,
            forecast_var_annual=float(row.forecast_var_annual),
            realized_var_annual=float(row.realized_var_annual or 0.0),
            qlike=float(row.qlike or 0.0),
            naive_var_annual=naive,
            inside_68=row.inside_68,
            inside_90=row.inside_90,
            inside_95=row.inside_95,
        )

    async def get_dvol(self, usable_until: datetime) -> float | None:
        async with self._sessionmaker() as session:
            value = await session.scalar(
                sa.select(ImpliedVolatilityORM.close)
                .where(
                    ImpliedVolatilityORM.currency == DVOL_CURRENCY,
                    ImpliedVolatilityORM.open_time <= usable_until,
                )
                .order_by(ImpliedVolatilityORM.open_time.desc())
                .limit(1)
            )
            return _opt(value)

    async def recent_closes(self, until_open_time: datetime, count: int) -> list[float]:
        async with self._sessionmaker() as session:
            result = await session.execute(
                sa.select(OhlcvCandleORM.close)
                .where(
                    OhlcvCandleORM.symbol == self._symbol,
                    OhlcvCandleORM.interval == self._interval,
                    OhlcvCandleORM.open_time <= until_open_time,
                )
                .order_by(OhlcvCandleORM.open_time.desc())
                .limit(count)
            )
            return [float(value) for value in reversed(result.scalars().all())]
