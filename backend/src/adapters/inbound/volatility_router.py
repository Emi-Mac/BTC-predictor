"""Router REST de volatilidad: pronóstico vigente, historial y tamaño de posición.

La volatilidad no es una señal direccional: ninguna respuesta indica comprar o vender.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.config.dependencies import get_risk_service, get_volatility_service
from src.domain.entities.pagination import Page
from src.domain.entities.volatility import (
    Horizon,
    PositionSize,
    Side,
    VolatilityForecast,
    VolatilityHistoryPoint,
)
from src.domain.ports.risk_port import RiskPort
from src.domain.ports.volatility_port import VolatilityPort

router = APIRouter(prefix="/api/v1/volatility", tags=["Volatility"])
risk_router = APIRouter(prefix="/api/v1/risk", tags=["Risk"])


@router.get("/current", summary="Pronóstico de volatilidad vigente y rango de precio probable")
async def get_current_volatility(
    horizon: Horizon = Query(default="7d", description="Horizonte del pronóstico"),
    service: VolatilityPort = Depends(get_volatility_service),
) -> VolatilityForecast:
    return await service.get_current(horizon)


@router.get("/history", summary="Volatilidad pronosticada frente a realizada")
async def get_volatility_history(
    horizon: Horizon = Query(default="7d"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: VolatilityPort = Depends(get_volatility_service),
) -> Page[VolatilityHistoryPoint]:
    return await service.get_history(horizon, limit, offset)


@risk_router.get(
    "/position-size",
    summary="Tamaño de posición con stop basado en volatilidad (herramienta educativa)",
)
async def get_position_size(
    capital: float = Query(gt=0, le=1e12, description="Capital en USD"),
    risk_pct: float = Query(gt=0, le=5, description="% del capital a arriesgar (máx. 5)"),
    stop_k: float = Query(default=2.0, gt=0, le=10, description="Stop a k·σ del horizonte"),
    horizon: Horizon = Query(default="1d"),
    side: Side = Query(default="long"),
    service: RiskPort = Depends(get_risk_service),
) -> PositionSize:
    return await service.position_size(capital, risk_pct, stop_k, horizon, side)
