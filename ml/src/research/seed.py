"""Reutiliza fuentes históricas verificadas al extender un dataset aislado."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.dataset import verify_dataset
from src.research.source_snapshot import SOURCE_FILES, file_sha256


def verify_seed(directory: Path, spec: dict[str, Any]) -> dict[str, Any]:
    manifest = verify_dataset(directory)
    if manifest["symbol"] != spec["symbol"] or manifest["start"] != spec["start"]:
        raise ValueError("El dataset base debe tener el mismo símbolo y fecha inicial")
    if date.fromisoformat(manifest["end_exclusive"]) >= date.fromisoformat(spec["end_exclusive"]):
        raise ValueError("El dataset extendido debe terminar después del dataset base")
    if (
        manifest["availability_assumption"]["metrics_interval_end_minutes"]
        != spec["assumed_interval_end_minutes"]
        or manifest["availability_assumption"]["metrics_delay_hours"]
        != spec["assumed_availability_delay_hours"]
    ):
        raise ValueError("Los supuestos temporales no coinciden con el dataset base")
    expected = spec.get("seed_manifest_sha256")
    if expected is not None and expected != manifest["manifest_sha256"]:
        raise ValueError("El manifiesto del dataset base cambió")
    return manifest


def copy_seed_archives(seed: Path, target: Path) -> None:
    archive_manifest = json.loads((seed / "archive-manifest.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    for item in archive_manifest["files"]:
        filename = item["filename"]
        if Path(filename).name != filename:
            raise ValueError("El manifiesto base contiene un nombre de archivo inválido")
        destination = target / filename
        if destination.exists():
            if file_sha256(destination) != item["sha256"]:
                raise ValueError(f"El archivo reutilizado no coincide con su huella: {filename}")
        else:
            temporary = target / f".{filename}.part"
            shutil.copyfile(seed / "archives" / filename, temporary)
            if file_sha256(temporary) != item["sha256"]:
                raise ValueError(f"El archivo copiado no coincide con su huella: {filename}")
            temporary.replace(destination)
        metadata_path = target / f"{filename}.json"
        if metadata_path.exists():
            if json.loads(metadata_path.read_text(encoding="utf-8")) != item:
                raise ValueError(f"Los metadatos reutilizados cambiaron: {filename}")
        else:
            temporary_metadata = target / f".{filename}.json.part"
            temporary_metadata.write_text(
                json.dumps(item, indent=2, allow_nan=False), encoding="utf-8"
            )
            temporary_metadata.replace(metadata_path)


def load_seed_sources(directory: Path) -> dict[str, pd.DataFrame]:
    metadata = json.loads((directory / "source-manifest.json").read_text(encoding="utf-8"))
    return {
        name: pd.read_parquet(directory / metadata["files"][name]["path"]) for name in SOURCE_FILES
    }


def merge_seed_sources(
    historical: dict[str, pd.DataFrame],
    current: dict[str, pd.DataFrame],
    boundary: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    result = {}
    for name in SOURCE_FILES:
        column = {
            "candles": "open_time",
            "macro": "trade_date",
            "sentiment": "obs_date",
            "funding": "funding_time",
        }[name]
        cutoff: date | pd.Timestamp = (
            boundary.date() if name in ("macro", "sentiment") else boundary
        )
        old = historical[name].loc[historical[name][column] < cutoff]
        new = current[name].loc[current[name][column] >= cutoff]
        result[name] = pd.concat([old, new], ignore_index=True)
    return result
