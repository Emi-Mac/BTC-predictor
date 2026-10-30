"""Puerto outbound: inferencia de modelos ML servidos como artefactos ONNX."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple

from src.domain.entities.prediction import Direction


class InferenceOutcome(NamedTuple):
    """Resultado crudo de una inferencia, antes de persistirse como `Prediction`."""

    direction: Direction
    confidence: float
    probability_up: float


class ModelInferencePort(ABC):
    """Puerto para cargar y ejecutar un modelo de clasificación binaria UP/DOWN."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True si hay un modelo cargado y listo para `predict`."""

    @property
    @abstractmethod
    def loaded_version(self) -> int | None:
        """Versión de `model_registry` actualmente cargada en memoria, si hay alguna."""

    @abstractmethod
    def load(self, directory: Path, model_version: int) -> None:
        """Carga (o recarga) el modelo desde el directorio de artefactos indicado."""

    @abstractmethod
    def predict(self, feature_values: list[float]) -> InferenceOutcome:
        """Ejecuta inferencia sobre un vector de features ya ordenado."""
