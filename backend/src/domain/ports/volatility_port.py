"""Puerto inbound: pronóstico de volatilidad, historial y calibración."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.pagination import Page
from src.domain.entities.volatility import (
    Horizon,
    VolatilityForecast,
    VolatilityHistoryPoint,
    VolatilityMetrics,
)


class VolatilityPort(ABC):
    """Casos de uso consumidos por los routers de volatilidad."""

    @abstractmethod
    async def get_current(self, horizon: Horizon) -> VolatilityForecast:
        """Pronóstico vigente. Lanza `NotFoundError` si aún no hay ninguno."""

    @abstractmethod
    async def get_history(
        self, horizon: Horizon, limit: int, offset: int
    ) -> Page[VolatilityHistoryPoint]:
        """Pronóstico frente a realizado, paginado del más reciente al más antiguo."""

    @abstractmethod
    async def get_metrics(self, horizon: Horizon) -> VolatilityMetrics:
        """Cobertura real de las bandas, QLIKE y pendiente de Mincer-Zarnowitz."""
