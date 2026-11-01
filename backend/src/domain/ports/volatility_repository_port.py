"""Puerto outbound: lectura de pronósticos de volatilidad y datos de contexto."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from src.domain.entities.volatility import ResolvedForecast, StoredForecast


class VolatilityRepositoryPort(ABC):
    """Lee `volatility_forecasts` (escritos por el ML worker) y el contexto para exponerlos."""

    @abstractmethod
    async def get_latest(self, horizon_candles: int) -> StoredForecast | None:
        """Pronóstico más reciente del horizonte (por vela)."""

    @abstractmethod
    async def list_history(
        self, horizon_candles: int, limit: int, offset: int
    ) -> tuple[list[StoredForecast], int]:
        """Pronósticos del horizonte, del más reciente al más antiguo, y el total."""

    @abstractmethod
    async def list_resolved(self, horizon_candles: int) -> list[ResolvedForecast]:
        """Pronósticos resueltos con su pronóstico ingenuo (varianza de las `h` velas previas)."""

    @abstractmethod
    async def get_dvol(self, usable_until: datetime) -> float | None:
        """Último DVOL cuya vela horaria abrió a más tardar en `usable_until` (puntos de vol)."""

    @abstractmethod
    async def recent_closes(self, until_open_time: datetime, count: int) -> list[float]:
        """Hasta `count` cierres consecutivos que terminan en la vela `until_open_time`."""
