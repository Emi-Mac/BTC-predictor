"""Entidad de dominio para una vela OHLCV. Inmutable (Pydantic frozen)."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Candle(BaseModel):
    """Vela OHLCV tal como se sirve a los consumidores de la API."""

    model_config = ConfigDict(frozen=True)

    id: int
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float | None = None
    num_trades: int | None = None


class CandleWithChange(Candle):
    """Última vela junto con la variación porcentual respecto a la vela anterior."""

    change_pct: float | None = None
