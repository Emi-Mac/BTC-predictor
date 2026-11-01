"""Puerto inbound: features actuales, para transparencia del modelo."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.feature import FeatureSnapshot


class FeaturePort(ABC):
    """Caso de uso consumido por `feature_router`."""

    @abstractmethod
    async def get_latest_features(self, symbol: str, interval: str) -> FeatureSnapshot:
        """Features de la vela más reciente. Lanza `NotFoundError` si no hay ninguna."""
