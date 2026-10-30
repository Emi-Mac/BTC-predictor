"""Puerto outbound: acceso de solo lectura a `model_registry`."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.model_metadata import ModelMetadata


class ModelRegistryPort(ABC):
    """Operaciones de lectura sobre el registro de modelos entrenados."""

    @abstractmethod
    async def get_active(self) -> ModelMetadata | None:
        """Devuelve el modelo marcado como activo (`is_active=True`), si existe alguno."""

    @abstractmethod
    async def get_by_version(self, version: int) -> ModelMetadata | None:
        """Devuelve la metadata de una versión concreta de modelo."""
