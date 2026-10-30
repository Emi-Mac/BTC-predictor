"""Entidad de dominio para la metadata de un modelo entrenado registrado."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ModelMetadata(BaseModel):
    """Fila de `model_registry` expuesta al dominio."""

    model_config = ConfigDict(frozen=True)

    version: int
    model_type: str
    val_accuracy: float | None
    val_f1: float | None
    val_precision: float | None
    val_recall: float | None
    baseline_accuracy: float | None
    trained_at: datetime
    is_active: bool
    feature_names: list[str]
    artifact_path: str | None
    scaler_path: str | None
