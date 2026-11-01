"""Adaptador outbound: `FeatureRepositoryPort` sobre PostgreSQL (SQLAlchemy 2.0 async)."""
from __future__ import annotations

from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config.models import ComputedFeatureORM, ContextualFeatureORM, OhlcvCandleORM
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
        """Vector en el orden del modelo; combina `computed_features` y `contextual_features`.

        Devuelve `None` si falta la fila o cualquier valor (la vela aún no está lista).
        """
        stored_names = set(_feature_column_names())
        stored = [name for name in feature_names if name in stored_names]
        contextual = [name for name in feature_names if name not in stored_names]
        values: dict[str, object] = {}
        async with self._sessionmaker() as session:
            if stored:
                result = await session.execute(
                    sa.select(*(getattr(ComputedFeatureORM, name) for name in stored)).where(
                        ComputedFeatureORM.candle_id == candle_id
                    )
                )
                record = result.first()
                if record is None:
                    return None
                values.update(zip(stored, record, strict=True))
            if contextual:
                payload = await session.scalar(
                    sa.select(ContextualFeatureORM.feature_values).where(
                        ContextualFeatureORM.candle_id == candle_id
                    )
                )
                if payload is None:
                    return None
                values.update({name: payload.get(name) for name in contextual})
        if any(values[name] is None for name in feature_names):
            return None
        return [float(values[name]) for name in feature_names]  # type: ignore[arg-type]
