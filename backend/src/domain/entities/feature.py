"""Entidad de dominio para el snapshot de features calculadas de una vela."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FeatureSnapshot(BaseModel):
    """Features técnicas/macro/sentimiento calculadas para una vela concreta."""

    model_config = ConfigDict(frozen=True)

    candle_id: int
    symbol: str
    interval: str
    open_time: datetime
    values: dict[str, float | int | bool | None]
