"""Inyección de dependencias (hexagonal): conecta los puertos con sus implementaciones.

Los repositorios y el adaptador de inferencia son singletons cacheados con `lru_cache`:
abren una sesión de DB nueva por operación (ver `async_sessionmaker`) pero el propio
objeto repositorio no mantiene estado, por lo que es seguro compartirlo entre requests.
El adaptador ONNX sí mantiene estado (el modelo cargado) y debe ser un único singleton
para cumplir el requisito de performance del hot path (ver FASE 4, Cuello de Botella ONNX).
"""
from __future__ import annotations

from functools import lru_cache

from src.adapters.outbound.candle_repository import PostgresCandleRepository
from src.adapters.outbound.feature_repository import PostgresFeatureRepository
from src.adapters.outbound.model_registry_repository import PostgresModelRegistryRepository
from src.adapters.outbound.onnx_model_adapter import OnnxModelAdapter
from src.adapters.outbound.prediction_repository import PostgresPredictionRepository
from src.application.feature_service import FeatureService
from src.application.market_data_service import MarketDataService
from src.application.metrics_service import MetricsService
from src.application.prediction_service import PredictionService
from src.config.database import AsyncSessionLocal
from src.domain.ports.feature_port import FeaturePort
from src.domain.ports.market_data_port import MarketDataPort
from src.domain.ports.metrics_port import MetricsPort
from src.domain.ports.model_inference_port import ModelInferencePort
from src.domain.ports.prediction_port import PredictionPort


@lru_cache
def get_candle_repository() -> PostgresCandleRepository:
    return PostgresCandleRepository(AsyncSessionLocal)


@lru_cache
def get_feature_repository() -> PostgresFeatureRepository:
    return PostgresFeatureRepository(AsyncSessionLocal)


@lru_cache
def get_prediction_repository() -> PostgresPredictionRepository:
    return PostgresPredictionRepository(AsyncSessionLocal)


@lru_cache
def get_model_registry_repository() -> PostgresModelRegistryRepository:
    return PostgresModelRegistryRepository(AsyncSessionLocal)


@lru_cache
def get_model_inference_adapter() -> ModelInferencePort:
    """Singleton: el modelo ONNX se carga una sola vez y se reutiliza en cada request."""
    return OnnxModelAdapter()


def get_market_data_service() -> MarketDataPort:
    return MarketDataService(get_candle_repository())


def get_prediction_service() -> PredictionPort:
    return PredictionService(get_prediction_repository())


def get_metrics_service() -> MetricsPort:
    return MetricsService(get_prediction_repository(), get_model_registry_repository())


def get_feature_service() -> FeaturePort:
    return FeatureService(get_feature_repository())
