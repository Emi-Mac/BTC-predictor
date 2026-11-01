"""Router REST para predicciones: activa e histórico."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.config.dependencies import get_prediction_service
from src.domain.entities.pagination import Page
from src.domain.entities.prediction import Prediction
from src.domain.ports.prediction_port import PredictionPort

router = APIRouter(prefix="/api/v1/predictions", tags=["Predictions"])


@router.get("/current", summary="Predicción activa")
async def get_current_prediction(
    service: PredictionPort = Depends(get_prediction_service),
) -> Prediction:
    return await service.get_current()


@router.get("/history", summary="Historial de predicciones")
async def get_prediction_history(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: PredictionPort = Depends(get_prediction_service),
) -> Page[Prediction]:
    return await service.get_history(limit, offset)
