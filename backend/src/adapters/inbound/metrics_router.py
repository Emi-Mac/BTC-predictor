"""Router REST para métricas: accuracy histórico y metadata del modelo activo."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.config.dependencies import get_metrics_service, get_volatility_service
from src.domain.entities.metrics import AccuracyMetrics
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.entities.volatility import Horizon, VolatilityMetrics
from src.domain.ports.metrics_port import MetricsPort
from src.domain.ports.volatility_port import VolatilityPort

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics"])


@router.get("/accuracy", summary="Accuracy histórico y por período")
async def get_accuracy(
    service: MetricsPort = Depends(get_metrics_service),
) -> AccuracyMetrics:
    return await service.get_accuracy()


@router.get("/model", summary="Metadata del modelo activo")
async def get_model_metadata(
    service: MetricsPort = Depends(get_metrics_service),
) -> ModelMetadata:
    return await service.get_active_model()


@router.get("/volatility", summary="Calibración del pronóstico de volatilidad y de sus bandas")
async def get_volatility_metrics(
    horizon: Horizon = Query(default="7d"),
    service: VolatilityPort = Depends(get_volatility_service),
) -> VolatilityMetrics:
    return await service.get_metrics(horizon)
