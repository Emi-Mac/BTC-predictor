"""Adaptador outbound: `CandleRepositoryPort` sobre PostgreSQL (SQLAlchemy 2.0 async)."""
from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config.models import OhlcvCandleORM
from src.domain.entities.candle import Candle
from src.domain.ports.candle_repository_port import CandleRepositoryPort


def _to_entity(row: OhlcvCandleORM) -> Candle:
    return Candle(
        id=row.id,
        symbol=row.symbol,
        interval=row.interval,
        open_time=row.open_time,
        close_time=row.close_time,
        open=float(row.open),
        high=float(row.high),
        low=float(row.low),
        close=float(row.close),
        volume=float(row.volume),
        quote_volume=float(row.quote_volume) if row.quote_volume is not None else None,
        num_trades=row.num_trades,
    )


class PostgresCandleRepository(CandleRepositoryPort):
    """Repositorio de lectura de velas OHLCV."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def list_candles(
        self, symbol: str, interval: str, limit: int, offset: int
    ) -> tuple[list[Candle], int]:
        async with self._sessionmaker() as session:
            filters = (OhlcvCandleORM.symbol == symbol, OhlcvCandleORM.interval == interval)
            total = await session.scalar(
                sa.select(sa.func.count()).select_from(OhlcvCandleORM).where(*filters)
            )
            result = await session.scalars(
                sa.select(OhlcvCandleORM)
                .where(*filters)
                .order_by(OhlcvCandleORM.open_time.desc())
                .limit(limit)
                .offset(offset)
            )
            candles = [_to_entity(row) for row in result]
            return candles, int(total or 0)

    async def get_latest(self, symbol: str, interval: str) -> Candle | None:
        async with self._sessionmaker() as session:
            row = await session.scalar(
                sa.select(OhlcvCandleORM)
                .where(OhlcvCandleORM.symbol == symbol, OhlcvCandleORM.interval == interval)
                .order_by(OhlcvCandleORM.open_time.desc())
                .limit(1)
            )
            return _to_entity(row) if row is not None else None

    async def get_by_open_time(
        self, symbol: str, interval: str, open_time: datetime
    ) -> Candle | None:
        async with self._sessionmaker() as session:
            row = await session.scalar(
                sa.select(OhlcvCandleORM).where(
                    OhlcvCandleORM.symbol == symbol,
                    OhlcvCandleORM.interval == interval,
                    OhlcvCandleORM.open_time == open_time,
                )
            )
            return _to_entity(row) if row is not None else None

    async def get_by_id(self, candle_id: int) -> Candle | None:
        async with self._sessionmaker() as session:
            row = await session.get(OhlcvCandleORM, candle_id)
            return _to_entity(row) if row is not None else None
