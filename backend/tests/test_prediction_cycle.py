"""Tests de `run_prediction_cycle` (orquestación de inferencia) con puertos fake."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.application.prediction_cycle import run_prediction_cycle
from src.domain.entities.candle import Candle
from src.domain.entities.model_metadata import ModelMetadata
from src.domain.ports.candle_repository_port import CandleRepositoryPort
from src.domain.ports.feature_repository_port import FeatureRepositoryPort
from src.domain.ports.model_inference_port import InferenceOutcome, ModelInferencePort
from src.domain.ports.model_registry_port import ModelRegistryPort
from src.domain.ports.prediction_repository_port import PredictionRepositoryPort

SYMBOL = "BTCUSDT"
INTERVAL = "4h"


def _candle(candle_id: int, open_time: datetime, close: float) -> Candle:
    return Candle(
        id=candle_id,
        symbol=SYMBOL,
        interval=INTERVAL,
        open_time=open_time,
        close_time=open_time + timedelta(hours=4),
        open=close - 1,
        high=close + 5,
        low=close - 5,
        close=close,
        volume=100.0,
    )


def _model(
    version: int = 1, artifact_path: str | None = "/artifacts/1/model.onnx"
) -> ModelMetadata:
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
        artifact_path=artifact_path,
        scaler_path="/artifacts/1/preprocessor.json",
    )


class FakeCandleRepo(CandleRepositoryPort):
    def __init__(self, candles: list[Candle]) -> None:
        self._candles = candles

    async def list_candles(self, symbol, interval, limit, offset):
        page = self._candles[offset : offset + limit]
        return page, len(self._candles)

    async def get_latest(self, symbol, interval):
        return self._candles[0] if self._candles else None

    async def get_by_open_time(self, symbol, interval, open_time):
        raise NotImplementedError

    async def get_by_id(self, candle_id):
        raise NotImplementedError


class FakeFeatureRepo(FeatureRepositoryPort):
    def __init__(self, vector: list[float] | None) -> None:
        self._vector = vector
        self.calls: list[tuple[int, list[str]]] = []

    async def get_latest(self, symbol, interval):
        raise NotImplementedError

    async def get_vector_for_candle(self, candle_id, feature_names):
        self.calls.append((candle_id, feature_names))
        return self._vector


class FakeModelRegistry(ModelRegistryPort):
    def __init__(self, active: ModelMetadata | None) -> None:
        self._active = active

    async def get_active(self):
        return self._active

    async def get_by_version(self, version):
        raise NotImplementedError


class FakeInference(ModelInferencePort):
    def __init__(self, ready: bool = False, loaded_version: int | None = None) -> None:
        self._ready = ready
        self._loaded_version = loaded_version
        self.load_calls: list[tuple[Path, int]] = []
        self.predict_calls: list[list[float]] = []

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def loaded_version(self) -> int | None:
        return self._loaded_version

    def load(self, directory: Path, model_version: int) -> None:
        self.load_calls.append((directory, model_version))
        self._ready = True
        self._loaded_version = model_version

    def predict(self, feature_values: list[float]) -> InferenceOutcome:
        self.predict_calls.append(feature_values)
        return InferenceOutcome(direction="UP", confidence=0.65, probability_up=0.65)


class FakePredictionRepo(PredictionRepositoryPort):
    def __init__(self, existing: bool = False) -> None:
        self._existing = existing
        self.create_calls: list[dict] = []
        self.resolve_calls: list[tuple[int, str, datetime]] = []

    async def create(self, candle_id, model_version, direction, confidence, probability_up):
        self.create_calls.append(
            {
                "candle_id": candle_id,
                "model_version": model_version,
                "direction": direction,
                "confidence": confidence,
                "probability_up": probability_up,
            }
        )
        return None  # El valor de retorno no se usa en el ciclo.

    async def exists_for_candle(self, candle_id, model_version):
        return self._existing

    async def get_latest(self):
        raise NotImplementedError

    async def list_history(self, limit, offset):
        raise NotImplementedError

    async def resolve_for_candle(self, candle_id, actual_direction, resolved_at):
        self.resolve_calls.append((candle_id, actual_direction, resolved_at))
        return 1

    async def accuracy_since(self, since):
        raise NotImplementedError

    async def accuracy_by_month(self):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_skips_when_no_active_model():
    prediction_repo = FakePredictionRepo()
    await run_prediction_cycle(
        candle_repo=FakeCandleRepo([]),
        feature_repo=FakeFeatureRepo(None),
        model_registry=FakeModelRegistry(None),
        inference=FakeInference(),
        prediction_repo=prediction_repo,
        symbol=SYMBOL,
        interval=INTERVAL,
    )
    assert prediction_repo.create_calls == []


@pytest.mark.asyncio
async def test_skips_when_active_model_missing_artifact_path():
    prediction_repo = FakePredictionRepo()
    await run_prediction_cycle(
        candle_repo=FakeCandleRepo([]),
        feature_repo=FakeFeatureRepo(None),
        model_registry=FakeModelRegistry(_model(artifact_path=None)),
        inference=FakeInference(),
        prediction_repo=prediction_repo,
        symbol=SYMBOL,
        interval=INTERVAL,
    )
    assert prediction_repo.create_calls == []


@pytest.mark.asyncio
async def test_loads_model_when_inference_not_ready_or_stale_version():
    inference = FakeInference(ready=False)
    model = _model(version=3, artifact_path="/artifacts/3/model.onnx")
    await run_prediction_cycle(
        candle_repo=FakeCandleRepo([]),
        feature_repo=FakeFeatureRepo(None),
        model_registry=FakeModelRegistry(model),
        inference=inference,
        prediction_repo=FakePredictionRepo(),
        symbol=SYMBOL,
        interval=INTERVAL,
    )
    assert inference.load_calls == [(Path("/artifacts/3"), 3)]


@pytest.mark.asyncio
async def test_does_not_reload_model_when_already_loaded_with_matching_version():
    inference = FakeInference(ready=True, loaded_version=1)
    await run_prediction_cycle(
        candle_repo=FakeCandleRepo([]),
        feature_repo=FakeFeatureRepo(None),
        model_registry=FakeModelRegistry(_model(version=1)),
        inference=inference,
        prediction_repo=FakePredictionRepo(),
        symbol=SYMBOL,
        interval=INTERVAL,
    )
    assert inference.load_calls == []


@pytest.mark.asyncio
async def test_returns_early_when_no_candles():
    prediction_repo = FakePredictionRepo()
    inference = FakeInference(ready=True, loaded_version=1)
    await run_prediction_cycle(
        candle_repo=FakeCandleRepo([]),
        feature_repo=FakeFeatureRepo(None),
        model_registry=FakeModelRegistry(_model(version=1)),
        inference=inference,
        prediction_repo=prediction_repo,
        symbol=SYMBOL,
        interval=INTERVAL,
    )
    assert prediction_repo.create_calls == []
    assert inference.predict_calls == []


@pytest.mark.asyncio
async def test_emits_prediction_for_latest_candle_when_missing():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    latest = _candle(2, base + timedelta(hours=4), 110.0)
    inference = FakeInference(ready=True, loaded_version=1)
    prediction_repo = FakePredictionRepo(existing=False)
    feature_repo = FakeFeatureRepo([55.0, 0.2])

    await run_prediction_cycle(
        candle_repo=FakeCandleRepo([latest]),
        feature_repo=feature_repo,
        model_registry=FakeModelRegistry(_model(version=1)),
        inference=inference,
        prediction_repo=prediction_repo,
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    assert feature_repo.calls == [(2, ["rsi_14", "macd"])]
    assert inference.predict_calls == [[55.0, 0.2]]
    assert prediction_repo.create_calls == [
        {
            "candle_id": 2,
            "model_version": 1,
            "direction": "UP",
            "confidence": 0.65,
            "probability_up": 0.65,
        }
    ]


@pytest.mark.asyncio
async def test_skips_prediction_when_already_exists_for_candle():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    latest = _candle(2, base, 110.0)
    prediction_repo = FakePredictionRepo(existing=True)
    inference = FakeInference(ready=True, loaded_version=1)

    await run_prediction_cycle(
        candle_repo=FakeCandleRepo([latest]),
        feature_repo=FakeFeatureRepo([55.0, 0.2]),
        model_registry=FakeModelRegistry(_model(version=1)),
        inference=inference,
        prediction_repo=prediction_repo,
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    assert prediction_repo.create_calls == []
    assert inference.predict_calls == []


@pytest.mark.asyncio
async def test_skips_prediction_when_feature_vector_unavailable():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    latest = _candle(2, base, 110.0)
    prediction_repo = FakePredictionRepo(existing=False)

    await run_prediction_cycle(
        candle_repo=FakeCandleRepo([latest]),
        feature_repo=FakeFeatureRepo(None),
        model_registry=FakeModelRegistry(_model(version=1)),
        inference=FakeInference(ready=True, loaded_version=1),
        prediction_repo=prediction_repo,
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    assert prediction_repo.create_calls == []


@pytest.mark.asyncio
async def test_resolves_previous_candle_prediction_as_up_when_price_increased():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    previous = _candle(1, base, 100.0)
    latest = _candle(2, base + timedelta(hours=4), 110.0)
    prediction_repo = FakePredictionRepo(existing=True)  # Evita ruido de la emisión nueva.

    await run_prediction_cycle(
        candle_repo=FakeCandleRepo([latest, previous]),
        feature_repo=FakeFeatureRepo(None),
        model_registry=FakeModelRegistry(_model(version=1)),
        inference=FakeInference(ready=True, loaded_version=1),
        prediction_repo=prediction_repo,
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    assert prediction_repo.resolve_calls == [(1, "UP", latest.close_time)]


@pytest.mark.asyncio
async def test_resolves_previous_candle_prediction_as_down_when_price_decreased():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    previous = _candle(1, base, 100.0)
    latest = _candle(2, base + timedelta(hours=4), 90.0)
    prediction_repo = FakePredictionRepo(existing=True)

    await run_prediction_cycle(
        candle_repo=FakeCandleRepo([latest, previous]),
        feature_repo=FakeFeatureRepo(None),
        model_registry=FakeModelRegistry(_model(version=1)),
        inference=FakeInference(ready=True, loaded_version=1),
        prediction_repo=prediction_repo,
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    assert prediction_repo.resolve_calls == [(1, "DOWN", latest.close_time)]


@pytest.mark.asyncio
async def test_does_not_resolve_when_no_previous_candle():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    latest = _candle(2, base, 110.0)
    prediction_repo = FakePredictionRepo(existing=True)

    await run_prediction_cycle(
        candle_repo=FakeCandleRepo([latest]),
        feature_repo=FakeFeatureRepo(None),
        model_registry=FakeModelRegistry(_model(version=1)),
        inference=FakeInference(ready=True, loaded_version=1),
        prediction_repo=prediction_repo,
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    assert prediction_repo.resolve_calls == []
