"""Puerto outbound: acceso de solo lectura a features calculadas (`computed_features`)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.feature import FeatureSnapshot


class FeatureRepositoryPort(ABC):
    """Operaciones de lectura sobre `computed_features`."""

    @abstractmethod
    async def get_latest(self, symbol: str, interval: str) -> FeatureSnapshot | None:
        """Devuelve el snapshot de features de la vela más reciente."""

    @abstractmethod
    async def get_vector_for_candle(
        self, candle_id: int, feature_names: list[str]
    ) -> list[float] | None:
        """Devuelve los valores de `feature_names`, en ese orden, para una vela.

        Devuelve `None` si no existe fila de features para la vela, o si alguna de las
        features solicitadas está ausente (NULL) en esa fila.
        """
