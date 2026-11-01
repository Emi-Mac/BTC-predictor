"""Puerto inbound: predicciones, consumido por `prediction_router`."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.pagination import Page
from src.domain.entities.prediction import Prediction


class PredictionPort(ABC):
    """Caso de uso de consulta de predicciones."""

    @abstractmethod
    async def get_current(self) -> Prediction:
        """Última predicción emitida. Lanza `NotFoundError` si no hay ninguna aún."""

    @abstractmethod
    async def get_history(self, limit: int, offset: int) -> Page[Prediction]:
        """Historial paginado de predicciones, más recientes primero."""
