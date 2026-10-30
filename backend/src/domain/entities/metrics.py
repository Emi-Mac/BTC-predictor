"""Entidades de dominio para métricas de accuracy histórico."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MonthlyAccuracy(BaseModel):
    """Accuracy agregado de un mes calendario."""

    model_config = ConfigDict(frozen=True)

    month: str  # Formato "YYYY-MM"
    accuracy: float | None
    total_predictions: int


class AccuracyMetrics(BaseModel):
    """Accuracy histórico agregado por ventana temporal."""

    model_config = ConfigDict(frozen=True)

    overall: float | None
    last_30d: float | None
    last_7d: float | None
    total_resolved: int
    by_month: list[MonthlyAccuracy]
