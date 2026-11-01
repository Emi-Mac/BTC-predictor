"""Adaptador outbound: `ModelRegistryPort` sobre PostgreSQL (SQLAlchemy 2.0 async)."""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config.models import ModelRegistryORM
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.ports.model_registry_port import ModelRegistryPort


def _to_entity(row: ModelRegistryORM) -> ModelMetadata:
    return ModelMetadata(
        version=row.version,
        model_type=row.model_type,
        val_accuracy=float(row.val_accuracy) if row.val_accuracy is not None else None,
        val_f1=float(row.val_f1) if row.val_f1 is not None else None,
        val_precision=float(row.val_precision) if row.val_precision is not None else None,
        val_recall=float(row.val_recall) if row.val_recall is not None else None,
        baseline_accuracy=float(row.baseline_accuracy)
        if row.baseline_accuracy is not None
        else None,
        trained_at=row.trained_at,
        is_active=row.is_active,
        feature_names=list(row.feature_names or []),
        artifact_path=row.artifact_path,
        scaler_path=row.scaler_path,
    )


class PostgresModelRegistryRepository(ModelRegistryPort):
    """Repositorio de lectura del registro de modelos entrenados."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def get_active(self) -> ModelMetadata | None:
        async with self._sessionmaker() as session:
            row = await session.scalar(
                sa.select(ModelRegistryORM)
                .where(ModelRegistryORM.is_active.is_(True))
                .order_by(ModelRegistryORM.version.desc())
                .limit(1)
            )
            return _to_entity(row) if row is not None else None

    async def get_by_version(self, version: int) -> ModelMetadata | None:
        async with self._sessionmaker() as session:
            row = await session.get(ModelRegistryORM, version)
            return _to_entity(row) if row is not None else None
