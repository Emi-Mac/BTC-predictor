"""Servicio de predicciones: implementa `PredictionPort` sobre `PredictionRepositoryPort`."""
from __future__ import annotations

from src.domain.entities.pagination import Page, build_pagination
from src.domain.entities.prediction import DirectionStatus, Prediction
from src.domain.exceptions import NotFoundError
from src.domain.ports.prediction_port import PredictionPort
from src.domain.ports.prediction_repository_port import PredictionRepositoryPort


class PredictionService(PredictionPort):
    """Caso de uso de consulta de predicciones."""

    def __init__(
        self,
        prediction_repo: PredictionRepositoryPort,
        model_status: DirectionStatus = "experimental",
    ) -> None:
        self._prediction_repo = prediction_repo
        self._model_status = model_status

    async def get_current(self) -> Prediction:
        prediction = await self._prediction_repo.get_latest()
        if prediction is None:
            msg = "No hay predicciones emitidas todavía"
            raise NotFoundError(msg)
        return prediction.model_copy(update={"status": self._model_status})

    async def get_history(self, limit: int, offset: int) -> Page[Prediction]:
        predictions, total = await self._prediction_repo.list_history(limit, offset)
        return Page(
            data=[p.model_copy(update={"status": self._model_status}) for p in predictions],
            pagination=build_pagination(limit, offset, total, len(predictions)),
        )
