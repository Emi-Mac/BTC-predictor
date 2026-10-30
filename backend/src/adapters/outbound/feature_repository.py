"""Adaptador outbound: `FeatureRepositoryPort` sobre PostgreSQL (SQLAlchemy 2.0 async)."""
from __future__ import annotations

from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config.models import ComputedFeatureORM, OhlcvCandleORM
from src.domain.entities.feature import FeatureSnapshot
from src.domain.ports.feature_repository_port import FeatureRepositoryPort

# Columnas de `computed_features` que no son features de entrada del modelo.
_NON_FEATURE_COLUMNS = frozenset({"id", "candle_id", "created_at", "target_label"})


def _feature_column_names() -> list[str]:
    mapper = sa.inspect(ComputedFeatureORM)
    return [attr.key for attr in mapper.column_attrs if attr.key not in _NON_FEATURE_COLUMNS]


def _extract_values(row: ComputedFeatureORM) -> dict[str, float | int | bool | None]:
    values: dict[str, float | int | bool | None] = {}
    for name in _feature_column_names():
        value = getattr(row, name)
        values[name] = float(value) if isinstance(value, Decimal) else value
    return values


class PostgresFeatureRepository(FeatureRepositoryPort):
    """Repositorio de lectura de features calculadas (`computed_features`)."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def get_latest(self, symbol: str, interval: str) -> FeatureSnapshot | None:
        async with self._sessionmaker() as session:
            result = await session.execute(
                sa.select(ComputedFeatureORM, OhlcvCandleORM.open_time)
                .join(OhlcvCandleORM, ComputedFeatureORM.candle_id == OhlcvCandleORM.id)
                .where(OhlcvCandleORM.symbol == symbol, OhlcvCandleORM.interval == interval)
                .order_by(OhlcvCandleORM.open_time.desc())
                .limit(1)
            )
            record = result.first()
            if record is None:
                return None
            feature_row, open_time = record
            return FeatureSnapshot(
                candle_id=feature_row.candle_id,
                symbol=symbol,
                interval=interval,
                open_time=open_time,
                values=_extract_values(feature_row),
            )

    async def get_vector_for_candle(
        self, candle_id: int, feature_names: list[str]
    ) -> list[float] | None:
        async with self._sessionmaker() as session:
            columns = [getattr(ComputedFeatureORM, name) for name in feature_names]
            result = await session.execute(
                sa.select(*columns).where(ComputedFeatureORM.candle_id == candle_id)
            )
            record = result.first()
            if record is None or any(value is None for value in record):
                return None
            return [float(value) for value in record]
