"""Tests de la inyección de dependencias (`config.dependencies`): wiring hexagonal."""
from __future__ import annotations

from src.adapters.outbound.candle_repository import PostgresCandleRepository
from src.adapters.outbound.onnx_model_adapter import OnnxModelAdapter
from src.config import dependencies


def test_repository_singletons_are_cached_across_calls():
    assert dependencies.get_candle_repository() is dependencies.get_candle_repository()
    assert dependencies.get_feature_repository() is dependencies.get_feature_repository()
    assert dependencies.get_prediction_repository() is dependencies.get_prediction_repository()
    assert (
        dependencies.get_model_registry_repository()
        is dependencies.get_model_registry_repository()
    )


def test_model_inference_adapter_is_a_singleton_onnx_adapter():
    first = dependencies.get_model_inference_adapter()
    second = dependencies.get_model_inference_adapter()

    assert first is second
    assert isinstance(first, OnnxModelAdapter)


def test_market_data_service_wraps_the_cached_candle_repository():
    service = dependencies.get_market_data_service()

    assert isinstance(service._candle_repo, PostgresCandleRepository)
    assert service._candle_repo is dependencies.get_candle_repository()


def test_prediction_service_wraps_the_cached_prediction_repository():
    service = dependencies.get_prediction_service()

    assert service._prediction_repo is dependencies.get_prediction_repository()


def test_metrics_service_wraps_both_cached_repositories():
    service = dependencies.get_metrics_service()

    assert service._prediction_repo is dependencies.get_prediction_repository()
    assert service._model_registry is dependencies.get_model_registry_repository()


def test_feature_service_wraps_the_cached_feature_repository():
    service = dependencies.get_feature_service()

    assert service._feature_repo is dependencies.get_feature_repository()
