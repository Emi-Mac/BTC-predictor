"""Puertos (interfaces ABC) de la arquitectura hexagonal.

- inbound (casos de uso, implementados en `src.application`): `MarketDataPort`,
  `PredictionPort`, `MetricsPort`, `FeaturePort`.
- outbound (dependencias externas, implementadas por `adapters.outbound`):
  `ModelInferencePort`, `CandleRepositoryPort`, `FeatureRepositoryPort`,
  `PredictionRepositoryPort`, `ModelRegistryPort`.
"""
from src.domain.ports.candle_repository_port import CandleRepositoryPort
from src.domain.ports.feature_port import FeaturePort
from src.domain.ports.feature_repository_port import FeatureRepositoryPort
from src.domain.ports.market_data_port import MarketDataPort
from src.domain.ports.metrics_port import MetricsPort
from src.domain.ports.model_inference_port import InferenceOutcome, ModelInferencePort
from src.domain.ports.model_registry_port import ModelRegistryPort
from src.domain.ports.prediction_port import PredictionPort
from src.domain.ports.prediction_repository_port import PredictionRepositoryPort

__all__ = [
    "CandleRepositoryPort",
    "FeaturePort",
    "FeatureRepositoryPort",
    "InferenceOutcome",
    "MarketDataPort",
    "MetricsPort",
    "ModelInferencePort",
    "ModelRegistryPort",
    "PredictionPort",
    "PredictionRepositoryPort",
]
