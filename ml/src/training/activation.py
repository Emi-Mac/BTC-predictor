"""Activación explícita de una versión del registro de modelos (decisión de producto).

El entrenamiento nunca activa modelos (ver `registry.py`). Este comando lo hace de forma
deliberada y auditable: valida que el backend pueda servir el modelo, deja una única versión
activa y anota el motivo en `notes`. Activar no cambia la evidencia de validación registrada.

Uso (dentro del contenedor del ML worker, donde `artifact_path` es válido):

    python -m src.training.activation --version 2 --reason "Plan original: mejor holdout"
    python -m src.training.activation --version 2 --dry-run
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
from src.features.compute import FEATURE_COLUMNS
from src.features.price_context import PRICE_CONTEXT_COLUMNS
from src.ingestion.run import configure_logging

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

log = structlog.get_logger(__name__)
SERVABLE_FEATURES = frozenset([*FEATURE_COLUMNS, *PRICE_CONTEXT_COLUMNS])
REQUIRED_FILES = ("model.onnx", "preprocessor.json", "calibration.json", "inference.json")
UNSUPPORTED_FAMILIES = frozenset({"lstm"})
NOTES_LIMIT = 1000


def activation_problems(feature_names: list[str], artifact_path: str | None) -> list[str]:
    """Motivos por los que el backend no podría servir el modelo (lista vacía si puede)."""
    problems: list[str] = []
    if not feature_names:
        problems.append("El modelo no registra feature_names")
    missing = sorted(set(feature_names) - SERVABLE_FEATURES)
    if missing:
        problems.append(f"Features no persistidas para inferencia: {missing}")
    if not artifact_path:
        problems.append("El modelo no tiene artifact_path")
        return problems
    directory = Path(artifact_path).parent
    absent = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if absent:
        problems.append(f"Faltan archivos del artefacto en {directory}: {absent}")
        return problems
    family = json.loads((directory / "inference.json").read_text(encoding="utf-8"))["family"]
    if family in UNSUPPORTED_FAMILIES:
        problems.append(f"El backend no sirve la familia {family}")
    preprocessor = json.loads((directory / "preprocessor.json").read_text(encoding="utf-8"))
    if list(preprocessor["feature_names"]) != list(feature_names):
        problems.append("feature_names del registro y del preprocesador no coinciden")
    return problems


def _registry(sync: sa.Connection) -> sa.Table:
    return sa.Table("model_registry", sa.MetaData(), autoload_with=sync)


async def activate(
    engine: AsyncEngine, version: int, reason: str, dry_run: bool = False
) -> dict[str, Any]:
    if not reason.strip():
        raise ValueError("La activación requiere un motivo")
    async with engine.begin() as conn:
        table = await conn.run_sync(_registry)
        row = (
            (await conn.execute(sa.select(table).where(table.c.version == version)))
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise ValueError(f"No existe la versión {version} en model_registry")
        problems = activation_problems(list(row["feature_names"] or []), row["artifact_path"])
        if problems:
            raise ValueError("No se puede activar: " + "; ".join(problems))
        previous = list(
            (await conn.execute(sa.select(table.c.version).where(table.c.is_active))).scalars()
        )
        stamp = datetime.now(UTC).strftime("%Y-%m-%d")
        note = f"{row['notes'] or ''} | Activado manualmente {stamp}: {reason.strip()}"
        summary = {
            "version": version,
            "model_type": row["model_type"],
            "val_accuracy": float(row["val_accuracy"]) if row["val_accuracy"] is not None else None,
            "previously_active": previous,
            "dry_run": dry_run,
        }
        if dry_run:
            return summary
        await conn.execute(table.update().where(table.c.is_active).values(is_active=False))
        await conn.execute(
            table.update()
            .where(table.c.version == version)
            .values(is_active=True, notes=note.strip(" |")[-NOTES_LIMIT:])
        )
    log.info("model_activated", **summary)
    return summary


async def _main(version: int, reason: str, dry_run: bool) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        print(json.dumps(await activate(engine, version, reason, dry_run), indent=2))
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Activa una versión del registro de modelos")
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
