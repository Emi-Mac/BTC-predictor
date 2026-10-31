"""Coordinación de períodos reservados entre los flujos de investigación."""

from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


def frozen_digest(metadata: dict[str, Any]) -> str:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_frozen(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Falta la reserva: {path}")
    metadata: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    expected = metadata.pop("frozen_sha256", None)
    if expected != frozen_digest(metadata):
        raise ValueError(f"La reserva fue modificada: {path}")
    if metadata["status"] != "frozen":
        raise ValueError(f"La reserva tiene un estado inválido: {path}")
    start = pd.Timestamp(metadata["holdout_start"])
    end = pd.Timestamp(metadata["holdout_end_exclusive"])
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError(f"La reserva tiene fechas inválidas: {path}")
    metadata["frozen_sha256"] = expected
    return metadata


@contextmanager
def reservation_lock(root: Path) -> Generator[None, None, None]:
    with (root / ".holdout-reservations.lock").open("a", encoding="utf-8") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def assert_unused_holdout(
    artifact_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    exclude: Path | None = None,
) -> None:
    """Impide volver a reservar o puntuar un período observado o reservado."""
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("El holdout requiere fechas válidas con zona horaria")
    for metadata_path in artifact_root.glob("research-*/development.json"):
        directory = metadata_path.parent
        if (exclude and directory.resolve() == exclude.resolve()) or not (
            directory / "holdout.lock"
        ).exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prior_start = pd.Timestamp(metadata["holdout_start"])
        prior_end = pd.Timestamp(metadata["holdout_end_exclusive"])
        if start < prior_end and prior_start < end:
            raise ValueError(f"El holdout se solapa con una evaluación previa: {directory}")
    for path in artifact_root.glob("oi-ablation-*/frozen.json"):
        if exclude and path.parent.resolve() == exclude.resolve():
            continue
        metadata = read_frozen(path)
        prior_start = pd.Timestamp(metadata["holdout_start"])
        prior_end = pd.Timestamp(metadata["holdout_end_exclusive"])
        if start < prior_end and prior_start < end:
            raise ValueError(f"El holdout se solapa con una reserva existente: {path.parent}")


def assert_no_reserved_training_data(
    artifact_root: Path, open_times: pd.Series, step: timedelta
) -> None:
    """Protege reservas no evaluadas de su uso accidental en otros entrenamientos."""
    if step <= timedelta(0):
        raise ValueError("El intervalo de entrenamiento debe ser positivo")
    times = pd.to_datetime(open_times, utc=True)
    if times.isna().any():
        raise ValueError("Hay timestamps inválidos en el entrenamiento")
    for path in artifact_root.glob("oi-ablation-*/frozen.json"):
        metadata = read_frozen(path)
        result_path = path.parent / "holdout.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                not (path.parent / "holdout.lock").exists()
                or result.get("selected_variant") != metadata["selected_variant"]
            ):
                raise ValueError(f"Resultado de holdout incompleto o incompatible: {result_path}")
            continue
        start = pd.Timestamp(metadata["holdout_start"])
        end = pd.Timestamp(metadata["holdout_end_exclusive"])
        if ((times + 2 * pd.Timedelta(step) > start) & (times < end)).any():
            raise ValueError(
                f"El entrenamiento contiene datos de un holdout aún reservado: {path.parent}"
            )
