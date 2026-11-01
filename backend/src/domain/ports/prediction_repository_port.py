"""Puerto outbound: lectura y escritura de `predictions_log`."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from src.domain.entities.metrics import MonthlyAccuracy
from src.domain.entities.prediction import Direction, Prediction


class PredictionRepositoryPort(ABC):
    """Operaciones sobre el historial de predicciones emitidas."""

    @abstractmethod
    async def create(
        self,
        candle_id: int,
        model_version: int,
        direction: Direction,
        confidence: float,
        probability_up: float,
    ) -> Prediction:
        """Persiste una nueva predicción para una vela y versión de modelo dadas."""

    @abstractmethod
    async def exists_for_candle(self, candle_id: int, model_version: int) -> bool:
        """True si ya existe una predicción para esa vela con esa versión de modelo."""

    @abstractmethod
    async def get_latest(self) -> Prediction | None:
        """Devuelve la predicción más reciente emitida (cualquier versión de modelo)."""

    @abstractmethod
    async def list_history(self, limit: int, offset: int) -> tuple[list[Prediction], int]:
        """Devuelve una página del historial (más recientes primero) y el total."""

    @abstractmethod
    async def resolve_for_candle(
        self, candle_id: int, actual_direction: Direction, resolved_at: datetime
    ) -> int:
        """Resuelve las predicciones pendientes hechas sobre `candle_id`.

        Devuelve el número de filas resueltas (normalmente 0 o 1).
        """

    @abstractmethod
    async def accuracy_since(self, since: datetime | None) -> tuple[int, int]:
        """Devuelve (aciertos, total_resueltas) desde una fecha (o todo el histórico)."""

    @abstractmethod
    async def accuracy_by_month(self) -> list[MonthlyAccuracy]:
        """Devuelve el accuracy agregado por mes calendario, de más antiguo a más reciente."""
