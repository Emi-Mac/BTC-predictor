"""Activación explícita de un modelo de volatilidad (decisión de producto).

Valida el artefacto de volatilidad (no los ONNX direccionales; `src.training.activation` no
cambia) y desactiva solo la versión activa del mismo horizonte y rol.

Uso (dentro del contenedor del ML worker, donde `artifact_path` es válido):

    python -m src.volatility.activation --version 3 --reason "E-12b aceptado"
    python -m src.volatility.activation --version 3 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
import structlog
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import get_settings
from src.ingestion.run import configure_logging
from src.ingestion.tables import volatility_models
from src.volatility.model import VolatilityArtifact
from src.volatility.registry import file_sha256

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncEngine

log = structlog.get_logger(__name__)
NOTES_LIMIT = 1000


def activation_problems(row: Mapping[str, Any]) -> list[str]:
    """Motivos por los que el modelo registrado no puede servirse (vacío si puede)."""
    path = Path(row["artifact_path"] or "")
    if not path.is_file():
        return [f"No existe el artefacto {path}"]
    if file_sha256(path) != row["artifact_sha256"]:
        return ["La huella del artefacto no coincide con la registrada"]
    artifact = VolatilityArtifact.load(path)
    problems = artifact.problems()
    if artifact.kind != row["kind"]:
        problems.append("kind del registro y del artefacto no coinciden")
    if artifact.horizon_candles != row["horizon_candles"]:
        problems.append("horizonte del registro y del artefacto no coinciden")
    if list(row["feature_names"]) != artifact.feature_names:
        problems.append("feature_names del registro y del artefacto no coinciden")
    return problems


async def activate(
    engine: AsyncEngine, version: int, reason: str, dry_run: bool = False
) -> dict[str, Any]:
    if not reason.strip():
        raise ValueError("La activación requiere un motivo")
    table = volatility_models
    async with engine.begin() as conn:
        row = (
            (await conn.execute(sa.select(table).where(table.c.version == version)))
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise ValueError(f"No existe la versión {version} en volatility_models")
        problems = activation_problems(dict(row))
        if problems:
            raise ValueError("No se puede activar: " + "; ".join(problems))
        same_slot = (table.c.horizon_candles == row["horizon_candles"]) & (
            table.c.role == row["role"]
        )
        previous = list(
            (
                await conn.execute(sa.select(table.c.version).where(same_slot, table.c.is_active))
            ).scalars()
        )
        stamp = datetime.now(UTC).strftime("%Y-%m-%d")
        note = f"{row['notes'] or ''} | Activado manualmente {stamp}: {reason.strip()}"
        summary = {
            "version": version,
            "kind": row["kind"],
            "role": row["role"],
            "horizon_candles": row["horizon_candles"],
            "previously_active": previous,
            "dry_run": dry_run,
        }
        if dry_run:
            return summary
        await conn.execute(
            table.update().where(same_slot, table.c.is_active).values(is_active=False)
        )
        await conn.execute(
            table.update()
            .where(table.c.version == version)
            .values(is_active=True, notes=note.strip(" |")[-NOTES_LIMIT:])
        )
    log.info("volatility_model_activated", **summary)
    return summary


async def _main(version: int, reason: str, dry_run: bool) -> None:
    engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
    try:
        print(json.dumps(await activate(engine, version, reason, dry_run), indent=2))  # noqa: T201
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Activa una versión de volatility_models")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--reason", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.reason.strip():
        parser.error("--reason es obligatorio salvo en --dry-run")
    configure_logging(get_settings().log_level)
    asyncio.run(_main(args.version, args.reason or "dry-run", args.dry_run))


if __name__ == "__main__":
    main()
