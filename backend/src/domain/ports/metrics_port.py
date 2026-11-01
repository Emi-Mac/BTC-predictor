"""Puerto inbound: métricas históricas y metadata del modelo activo."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.metrics import AccuracyMetrics
from src.domain.entities.model_metadata import ModelMetadata


class MetricsPort(ABC):
    """Caso de uso consumido por `metrics_router`."""

    @abstractmethod
    async def get_accuracy(self) -> AccuracyMetrics:
        """Accuracy histórico agregado (overall, 30d, 7d, por mes)."""

    @abstractmethod
    async def get_active_model(self) -> ModelMetadata:
        """Metadata del modelo activo. Lanza `NotFoundError` si no hay ninguno activo."""
