"""Router REST para datos de mercado: velas OHLCV."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.config.dependencies import get_market_data_service
from src.config.settings import get_settings
from src.domain.entities.candle import Candle, CandleWithChange
from src.domain.entities.pagination import Page
from src.domain.ports.market_data_port import MarketDataPort

router = APIRouter(prefix="/api/v1/candles", tags=["Candles"])


@router.get("", summary="OHLCV paginado")
async def list_candles(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    service: MarketDataPort = Depends(get_market_data_service),
) -> Page[Candle]:
    settings = get_settings()
    return await service.get_candles(
        settings.btc_symbol, settings.default_interval, limit, offset
    )


@router.get("/latest", summary="Última vela + variación")
async def get_latest_candle(
    service: MarketDataPort = Depends(get_market_data_service),
) -> CandleWithChange:
    settings = get_settings()
    return await service.get_latest_candle(settings.btc_symbol, settings.default_interval)
