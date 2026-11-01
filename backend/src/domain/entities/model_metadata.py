"""Entidad de dominio para la metadata de un modelo entrenado registrado."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from src.domain.entities.prediction import DirectionStatus


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
    # Estado honesto: derivado de la evidencia, no del accuracy de validación por sí solo.
    best_baseline_accuracy: float | None = None
    beats_baseline: bool | None = None
    status: DirectionStatus = "experimental"
    evidence_url: str | None = None
