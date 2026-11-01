"""Registro de modelos de volatilidad (inactivos al registrar; la activación es explícita)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import sqlalchemy as sa
import structlog

from src.ingestion.tables import volatility_models
from src.volatility.model import VolatilityArtifact

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

log = structlog.get_logger(__name__)
ROLES = ("primary", "fallback")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def register(
    engine: AsyncEngine, artifact_file: Path, role: str, notes: str | None = None
) -> int:
    """Inserta el artefacto como inactivo y devuelve su versión."""
    if role not in ROLES:
        raise ValueError(f"Rol desconocido: {role}")
    artifact = VolatilityArtifact.load(artifact_file)
    problems = artifact.problems()
    if problems:
        raise ValueError("Artefacto inválido: " + "; ".join(problems))
    statement = (
        sa.insert(volatility_models)
        .values(
            kind=artifact.kind,
            role=role,
            horizon_candles=artifact.horizon_candles,
            artifact_path=str(artifact_file),
            artifact_sha256=file_sha256(artifact_file),
            feature_names=artifact.feature_names,
            validation_metrics=artifact.validation,
            is_active=False,
            notes=notes,
            trained_at=datetime.now(UTC),
        )
        .returning(volatility_models.c.version)
    )
    async with engine.begin() as conn:
        version: int = (await conn.execute(statement)).scalar_one()
    log.info(
        "volatility_model_registered",
        version=version,
        kind=artifact.kind,
        role=role,
        horizon=artifact.horizon_candles,
    )
    return version
