"""Router REST para métricas: accuracy histórico y metadata del modelo activo."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.config.dependencies import get_metrics_service
from src.domain.entities.metrics import AccuracyMetrics
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.ports.metrics_port import MetricsPort

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
