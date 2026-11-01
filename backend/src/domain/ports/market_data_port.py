"""Puerto inbound: datos de mercado (velas OHLCV) para los routers."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.candle import Candle, CandleWithChange
from src.domain.entities.pagination import Page


class MarketDataPort(ABC):
    """Caso de uso de consulta de velas, consumido por `candle_router`."""

    @abstractmethod
    async def get_candles(
        self, symbol: str, interval: str, limit: int, offset: int
    ) -> Page[Candle]:
        """Página de velas OHLCV, más recientes primero."""

    @abstractmethod
    async def get_latest_candle(self, symbol: str, interval: str) -> CandleWithChange:
        """Última vela disponible junto con su variación porcentual.

        Lanza `NotFoundError` si no hay ninguna vela todavía.
        """
