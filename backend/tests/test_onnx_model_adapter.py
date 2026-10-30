"""Tests de `OnnxModelAdapter` con una sesión ONNX simulada (sin modelo real)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.adapters.outbound import onnx_model_adapter as module
from src.adapters.outbound.onnx_model_adapter import OnnxModelAdapter


class _FakeInput:
    name = "features"


class _FakeSession:
    """Sustituye a `onnxruntime.InferenceSession`; devuelve P(UP)=0.7 fija."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, output_names, feed):
        return [None, np.array([[0.3, 0.7]], dtype=np.float32)]


@pytest.fixture(autouse=True)
def _patch_onnx_session(monkeypatch):
    monkeypatch.setattr(module.ort, "InferenceSession", _FakeSession)


def _write_artifacts(tmp_path, family: str = "xgboost", feature_names=("a", "b")):
    (tmp_path / "preprocessor.json").write_text(
        json.dumps(
            {
                "feature_names": list(feature_names),
                "median": [0.0] * len(feature_names),
                "mean": [0.0] * len(feature_names),
                "scale": [1.0] * len(feature_names),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "calibration.json").write_text(
        json.dumps({"slope": 1.0, "intercept": 0.0}), encoding="utf-8"
    )
    (tmp_path / "inference.json").write_text(json.dumps({"family": family}), encoding="utf-8")
    (tmp_path / "model.onnx").write_bytes(b"fake-onnx-bytes")


def test_load_sets_ready_and_version(tmp_path):
    _write_artifacts(tmp_path)
    adapter = OnnxModelAdapter()
    assert adapter.is_ready is False
    assert adapter.loaded_version is None

    adapter.load(tmp_path, model_version=7)

    assert adapter.is_ready is True
    assert adapter.loaded_version == 7


def test_load_rejects_lstm_family(tmp_path):
    _write_artifacts(tmp_path, family="lstm")
    adapter = OnnxModelAdapter()

    with pytest.raises(ValueError, match="no soporta"):
        adapter.load(tmp_path, model_version=1)
    assert adapter.is_ready is False


def test_predict_returns_up_when_probability_above_half(tmp_path):
    _write_artifacts(tmp_path)
    adapter = OnnxModelAdapter()
    adapter.load(tmp_path, model_version=1)

    outcome = adapter.predict([1.0, 2.0])

    assert outcome.direction == "UP"
    assert outcome.probability_up == pytest.approx(0.7, abs=1e-6)
    assert outcome.confidence == pytest.approx(0.7, abs=1e-6)


def test_predict_raises_when_no_model_loaded():
    adapter = OnnxModelAdapter()
    with pytest.raises(RuntimeError, match="No hay un modelo"):
        adapter.predict([1.0, 2.0])


def test_predict_raises_on_wrong_vector_length(tmp_path):
    _write_artifacts(tmp_path)
    adapter = OnnxModelAdapter()
    adapter.load(tmp_path, model_version=1)

    with pytest.raises(ValueError, match="Se esperaban"):
        adapter.predict([1.0])


def test_predict_raises_on_infinite_values(tmp_path):
    _write_artifacts(tmp_path)
    adapter = OnnxModelAdapter()
    adapter.load(tmp_path, model_version=1)

    with pytest.raises(ValueError, match="infinitos"):
        adapter.predict([1.0, float("inf")])


def test_predict_imputes_nan_with_median(tmp_path):
    _write_artifacts(tmp_path)
    adapter = OnnxModelAdapter()
    adapter.load(tmp_path, model_version=1)

    # No debe lanzar: el NaN se imputa con la mediana antes de escalar.
    outcome = adapter.predict([float("nan"), 2.0])

    assert outcome.direction == "UP"
