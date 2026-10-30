"""Entidades de dominio relacionadas con predicciones."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

Direction = Literal["UP", "DOWN"]


class Prediction(BaseModel):
    """Una predicción emitida y persistida en `predictions_log`."""

    model_config = ConfigDict(frozen=True)

    id: int
    candle_id: int
    candle_open_time: datetime
    model_version: int
    direction: Direction
    confidence: float
    probability_up: float | None
    predicted_at: datetime
    actual_direction: Direction | None = None
    is_correct: bool | None = None
    resolved_at: datetime | None = None
