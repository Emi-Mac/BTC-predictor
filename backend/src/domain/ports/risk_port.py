"""Puerto inbound: calculadora de tamaño de posición basada en volatilidad."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.volatility import Horizon, PositionSize, Side


class RiskPort(ABC):
    @abstractmethod
    async def position_size(
        self, capital: float, risk_pct: float, stop_k: float, horizon: Horizon, side: Side
    ) -> PositionSize:
        """Tamaño de posición para arriesgar `risk_pct`% del capital con un stop de `stop_k`·σ."""
