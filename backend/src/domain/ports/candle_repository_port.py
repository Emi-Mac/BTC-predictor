"""Puerto outbound: acceso de solo lectura a velas OHLCV persistidas."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from src.domain.entities.candle import Candle


class CandleRepositoryPort(ABC):
    """Operaciones de lectura sobre `ohlcv_candles`."""

    @abstractmethod
    async def list_candles(
        self, symbol: str, interval: str, limit: int, offset: int
    ) -> tuple[list[Candle], int]:
        """Devuelve una página de velas (más recientes primero) y el total disponible."""

    @abstractmethod
    async def get_latest(self, symbol: str, interval: str) -> Candle | None:
        """Devuelve la vela más reciente para el símbolo/intervalo indicado."""

    @abstractmethod
    async def get_by_open_time(
        self, symbol: str, interval: str, open_time: datetime
    ) -> Candle | None:
        """Devuelve la vela cuyo `open_time` coincide exactamente, si existe."""

    @abstractmethod
    async def get_by_id(self, candle_id: int) -> Candle | None:
        """Devuelve una vela por su id primario."""
