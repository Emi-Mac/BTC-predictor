"""Tests de `MetricsService` usando puertos en memoria."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.application.metrics_service import MetricsService
from src.domain.entities.metrics import MonthlyAccuracy
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.exceptions import NotFoundError
from src.domain.ports.model_registry_port import ModelRegistryPort
from src.domain.ports.prediction_repository_port import PredictionRepositoryPort


class FakePredictionRepository(PredictionRepositoryPort):
    """Devuelve, en orden, los valores configurados para cada llamada a `accuracy_since`."""

    def __init__(self, accuracy_since_sequence, by_month) -> None:
        self._sequence = list(accuracy_since_sequence)
        self._by_month = by_month

    async def create(self, *args, **kwargs):
        raise NotImplementedError

    async def exists_for_candle(self, candle_id, model_version):
        raise NotImplementedError

    async def get_latest(self):
        raise NotImplementedError

    async def list_history(self, limit, offset):
        raise NotImplementedError

    async def resolve_for_candle(self, candle_id, actual_direction, resolved_at):
        raise NotImplementedError

    async def accuracy_since(self, since):
        return self._sequence.pop(0)

    async def accuracy_by_month(self):
        return self._by_month


class FakeModelRegistry(ModelRegistryPort):
    def __init__(self, active: ModelMetadata | None) -> None:
        self._active = active

    async def get_active(self) -> ModelMetadata | None:
        return self._active

    async def get_by_version(self, version: int) -> ModelMetadata | None:
        return self._active if self._active and self._active.version == version else None


def _model(version: int = 1) -> ModelMetadata:
    return ModelMetadata(
        version=version,
        model_type="xgboost",
        val_accuracy=0.56,
        val_f1=0.55,
        val_precision=0.57,
        val_recall=0.54,
        baseline_accuracy=0.52,
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        is_active=True,
        feature_names=["rsi_14", "macd"],
        artifact_path="/artifacts/1/model.onnx",
        scaler_path="/artifacts/1/preprocessor.json",
    )


@pytest.mark.asyncio
async def test_get_accuracy_combines_overall_30d_7d_and_by_month():
    repo = FakePredictionRepository(
        accuracy_since_sequence=[(60, 100), (20, 30), (5, 7)],
        by_month=[MonthlyAccuracy(month="2026-01", accuracy=0.55, total_predictions=100)],
    )
    service = MetricsService(repo, FakeModelRegistry(active=None))

    metrics = await service.get_accuracy()

    assert metrics.overall == pytest.approx(0.6)
    assert metrics.last_30d == pytest.approx(20 / 30)
    assert metrics.last_7d == pytest.approx(5 / 7)
    assert metrics.total_resolved == 100
    assert metrics.by_month == [
        MonthlyAccuracy(month="2026-01", accuracy=0.55, total_predictions=100)
    ]


@pytest.mark.asyncio
async def test_get_accuracy_returns_none_ratios_when_no_resolved_predictions():
    repo = FakePredictionRepository(
        accuracy_since_sequence=[(0, 0), (0, 0), (0, 0)],
        by_month=[],
    )
    service = MetricsService(repo, FakeModelRegistry(active=None))

    metrics = await service.get_accuracy()

    assert metrics.overall is None
    assert metrics.last_30d is None
    assert metrics.last_7d is None
    assert metrics.total_resolved == 0


@pytest.mark.asyncio
async def test_get_active_model_returns_metadata_when_present():
    service = MetricsService(
        FakePredictionRepository([], []), FakeModelRegistry(active=_model())
    )

    model = await service.get_active_model()

    assert model.version == 1
    assert model.is_active is True


@pytest.mark.asyncio
async def test_get_active_model_raises_not_found_when_no_active_model():
    service = MetricsService(FakePredictionRepository([], []), FakeModelRegistry(active=None))

    with pytest.raises(NotFoundError):
        await service.get_active_model()


@pytest.mark.asyncio
async def test_active_model_reports_honest_status_and_baseline_comparison():
    service = MetricsService(
        FakePredictionRepository([], []), FakeModelRegistry(active=_model()), "experimental"
    )
    model = await service.get_active_model()
    assert model.status == "experimental"
    assert model.best_baseline_accuracy == pytest.approx(0.52)
    assert model.beats_baseline is True
    assert model.evidence_url == "docs/research/experiment-log.md"


@pytest.mark.asyncio
async def test_beats_baseline_is_false_or_unknown_when_evidence_is_missing():
    worse = _model().model_copy(update={"val_accuracy": 0.51})
    unknown = _model().model_copy(update={"baseline_accuracy": None})
    for model, expected in ((worse, False), (unknown, None)):
        service = MetricsService(FakePredictionRepository([], []), FakeModelRegistry(model))
        assert (await service.get_active_model()).beats_baseline is expected


@pytest.mark.asyncio
async def test_accuracy_includes_the_trivial_baseline_of_the_same_period():
    class WithBaseline(FakePredictionRepository):
        async def majority_direction_share(self):
            return 0.54

    repo = WithBaseline([(60, 100), (20, 30), (5, 7)], [])
    metrics = await MetricsService(repo, FakeModelRegistry(active=None)).get_accuracy()
    assert metrics.baseline_overall == pytest.approx(0.54)
    plain = FakePredictionRepository([(0, 0)] * 3, [])
    assert (await MetricsService(plain, FakeModelRegistry(None)).get_accuracy()).baseline_overall is None
