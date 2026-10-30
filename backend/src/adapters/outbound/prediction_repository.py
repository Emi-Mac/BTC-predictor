"""Adaptador outbound: `PredictionRepositoryPort` sobre PostgreSQL (SQLAlchemy 2.0 async)."""
from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config.models import OhlcvCandleORM, PredictionLogORM
from src.domain.entities.metrics import MonthlyAccuracy
from src.domain.entities.prediction import Direction, Prediction
from src.domain.ports.prediction_repository_port import PredictionRepositoryPort


def _to_entity(row: PredictionLogORM, candle_open_time: datetime) -> Prediction:
    return Prediction(
        id=row.id,
        candle_id=row.candle_id,
        candle_open_time=candle_open_time,
        model_version=row.model_version,
        direction=row.predicted_direction,
        confidence=float(row.confidence_score),
        probability_up=float(row.raw_probability_up)
        if row.raw_probability_up is not None
        else None,
        predicted_at=row.predicted_at,
        actual_direction=row.actual_direction,
        is_correct=row.is_correct,
        resolved_at=row.resolved_at,
    )


class PostgresPredictionRepository(PredictionRepositoryPort):
    """Repositorio de lectura/escritura del historial de predicciones."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def create(
        self,
        candle_id: int,
        model_version: int,
        direction: Direction,
        confidence: float,
        probability_up: float,
    ) -> Prediction:
        async with self._sessionmaker() as session:
            entity = PredictionLogORM(
                candle_id=candle_id,
                model_version=model_version,
                predicted_direction=direction,
                confidence_score=confidence,
                raw_probability_up=probability_up,
            )
            session.add(entity)
            await session.commit()
            await session.refresh(entity)
            candle_open_time = await session.scalar(
                sa.select(OhlcvCandleORM.open_time).where(OhlcvCandleORM.id == candle_id)
            )
            return _to_entity(entity, candle_open_time)

    async def exists_for_candle(self, candle_id: int, model_version: int) -> bool:
        async with self._sessionmaker() as session:
            row_id = await session.scalar(
                sa.select(PredictionLogORM.id)
                .where(
                    PredictionLogORM.candle_id == candle_id,
                    PredictionLogORM.model_version == model_version,
                )
                .limit(1)
            )
            return row_id is not None

    async def get_latest(self) -> Prediction | None:
        async with self._sessionmaker() as session:
            result = await session.execute(
                sa.select(PredictionLogORM, OhlcvCandleORM.open_time)
                .join(OhlcvCandleORM, PredictionLogORM.candle_id == OhlcvCandleORM.id)
                .order_by(PredictionLogORM.predicted_at.desc())
                .limit(1)
            )
            record = result.first()
            if record is None:
                return None
            row, open_time = record
            return _to_entity(row, open_time)

    async def list_history(self, limit: int, offset: int) -> tuple[list[Prediction], int]:
        async with self._sessionmaker() as session:
            total = await session.scalar(sa.select(sa.func.count()).select_from(PredictionLogORM))
            result = await session.execute(
                sa.select(PredictionLogORM, OhlcvCandleORM.open_time)
                .join(OhlcvCandleORM, PredictionLogORM.candle_id == OhlcvCandleORM.id)
                .order_by(PredictionLogORM.predicted_at.desc())
                .limit(limit)
                .offset(offset)
            )
            predictions = [_to_entity(row, open_time) for row, open_time in result]
            return predictions, int(total or 0)

    async def resolve_for_candle(
        self, candle_id: int, actual_direction: Direction, resolved_at: datetime
    ) -> int:
        async with self._sessionmaker() as session:
            result = await session.execute(
                sa.update(PredictionLogORM)
                .where(
                    PredictionLogORM.candle_id == candle_id,
                    PredictionLogORM.actual_direction.is_(None),
                )
                .values(
                    actual_direction=actual_direction,
                    is_correct=(PredictionLogORM.predicted_direction == actual_direction),
                    resolved_at=resolved_at,
                )
            )
            await session.commit()
            return int(result.rowcount or 0)

    async def accuracy_since(self, since: datetime | None) -> tuple[int, int]:
        async with self._sessionmaker() as session:
            filters = [PredictionLogORM.resolved_at.is_not(None)]
            if since is not None:
                filters.append(PredictionLogORM.predicted_at >= since)
            result = await session.execute(
                sa.select(
                    sa.func.count(),
                    sa.func.sum(sa.cast(PredictionLogORM.is_correct, sa.Integer)),
                ).where(*filters)
            )
            total, correct = result.one()
            return int(correct or 0), int(total or 0)

    async def accuracy_by_month(self) -> list[MonthlyAccuracy]:
        async with self._sessionmaker() as session:
            month = sa.func.to_char(PredictionLogORM.predicted_at, "YYYY-MM").label("month")
            result = await session.execute(
                sa.select(
                    month,
                    sa.func.count(),
                    sa.func.sum(sa.cast(PredictionLogORM.is_correct, sa.Integer)),
                )
                .where(PredictionLogORM.resolved_at.is_not(None))
                .group_by(month)
                .order_by(month)
            )
            return [
                MonthlyAccuracy(
                    month=row_month,
                    accuracy=(correct / total) if total else None,
                    total_predictions=int(total or 0),
                )
                for row_month, total, correct in result
            ]
