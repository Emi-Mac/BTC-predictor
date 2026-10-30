"""Adaptadores outbound: implementaciones concretas de los puertos de dominio."""
from src.adapters.outbound.candle_repository import PostgresCandleRepository
from src.adapters.outbound.feature_repository import PostgresFeatureRepository
from src.adapters.outbound.model_registry_repository import PostgresModelRegistryRepository
from src.adapters.outbound.onnx_model_adapter import OnnxModelAdapter
from src.adapters.outbound.prediction_repository import PostgresPredictionRepository

__all__ = [
    "OnnxModelAdapter",
    "PostgresCandleRepository",
    "PostgresFeatureRepository",
    "PostgresModelRegistryRepository",
    "PostgresPredictionRepository",
]
