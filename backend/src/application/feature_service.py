"""Servicio de features: implementa `FeaturePort` sobre `FeatureRepositoryPort`."""
from __future__ import annotations

from src.domain.entities.feature import FeatureSnapshot
from src.domain.exceptions import NotFoundError
from src.domain.ports.feature_port import FeaturePort
from src.domain.ports.feature_repository_port import FeatureRepositoryPort


class FeatureService(FeaturePort):
    """Caso de uso de transparencia de features (feature actual del modelo)."""

    def __init__(self, feature_repo: FeatureRepositoryPort) -> None:
        self._feature_repo = feature_repo

    async def get_latest_features(self, symbol: str, interval: str) -> FeatureSnapshot:
        snapshot = await self._feature_repo.get_latest(symbol, interval)
        if snapshot is None:
            msg = f"No hay features calculadas para {symbol}/{interval}"
            raise NotFoundError(msg)
        return snapshot
