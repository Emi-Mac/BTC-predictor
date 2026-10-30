"""Servicio de predicciones: implementa `PredictionPort` sobre `PredictionRepositoryPort`."""
from __future__ import annotations

from src.domain.entities.pagination import Page, build_pagination
from src.domain.entities.prediction import Prediction
from src.domain.exceptions import NotFoundError
from src.domain.ports.prediction_port import PredictionPort
from src.domain.ports.prediction_repository_port import PredictionRepositoryPort


class PredictionService(PredictionPort):
    """Caso de uso de consulta de predicciones."""

    def __init__(self, prediction_repo: PredictionRepositoryPort) -> None:
        self._prediction_repo = prediction_repo

    async def get_current(self) -> Prediction:
        prediction = await self._prediction_repo.get_latest()
        if prediction is None:
            msg = "No hay predicciones emitidas todavía"
            raise NotFoundError(msg)
        return prediction

    async def get_history(self, limit: int, offset: int) -> Page[Prediction]:
        predictions, total = await self._prediction_repo.list_history(limit, offset)
        return Page(
            data=predictions,
            pagination=build_pagination(limit, offset, total, len(predictions)),
        )
