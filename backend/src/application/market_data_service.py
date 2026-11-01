"""Servicio de datos de mercado: implementa `MarketDataPort` sobre `CandleRepositoryPort`."""
from __future__ import annotations

from src.domain.entities.candle import Candle, CandleWithChange
from src.domain.entities.pagination import Page, build_pagination
from src.domain.exceptions import NotFoundError
from src.domain.ports.candle_repository_port import CandleRepositoryPort
from src.domain.ports.market_data_port import MarketDataPort


class MarketDataService(MarketDataPort):
    """Caso de uso de consulta de velas OHLCV."""

    def __init__(self, candle_repo: CandleRepositoryPort) -> None:
        self._candle_repo = candle_repo

    async def get_candles(
        self, symbol: str, interval: str, limit: int, offset: int
    ) -> Page[Candle]:
        candles, total = await self._candle_repo.list_candles(symbol, interval, limit, offset)
        return Page(
            data=candles,
            pagination=build_pagination(limit, offset, total, len(candles)),
        )

    async def get_latest_candle(self, symbol: str, interval: str) -> CandleWithChange:
        # Se piden 2 velas para poder calcular la variación porcentual respecto a la anterior.
        candles, _ = await self._candle_repo.list_candles(symbol, interval, limit=2, offset=0)
        if not candles:
            msg = f"No hay velas disponibles para {symbol}/{interval}"
            raise NotFoundError(msg)
        latest, *rest = candles
        change_pct: float | None = None
        if rest:
            previous = rest[0]
            if previous.close:
                change_pct = (latest.close - previous.close) / previous.close * 100
        return CandleWithChange(**latest.model_dump(), change_pct=change_pct)
