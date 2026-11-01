"""Adaptador outbound: `ModelInferencePort` cargando un artefacto ONNX con onnxruntime.

Implementación deliberadamente independiente del paquete `ml` (dependencias pesadas como
torch/xgboost/onnxmltools no se instalan en la imagen del backend). Lee los mismos
artefactos (`model.onnx`, `preprocessor.json`, `calibration.json`, `inference.json`)
producidos por `ml.src.export.onnx_model.export_bundle`.

Soporta los modelos no secuenciales (xgboost/lightgbm/logistic). Los modelos LSTM
requieren una secuencia de 20 velas como entrada y quedan fuera del alcance de este
adaptador ligero; deben servirse desde el propio `ml-worker` si se activan en el futuro.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.domain.ports.model_inference_port import InferenceOutcome, ModelInferencePort

_UNSUPPORTED_FAMILIES = frozenset({"lstm"})


def _logit(probability: float) -> float:
    clamped = min(max(probability, 1e-15), 1 - 1e-15)
    return math.log(clamped / (1 - clamped))


class OnnxModelAdapter(ModelInferencePort):
    """Carga un único modelo ONNX como singleton y expone inferencia rápida (<50ms)."""

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._feature_names: list[str] = []
        self._median: list[float] = []
        self._mean: list[float] = []
        self._scale: list[float] = []
        self._slope: float = 1.0
        self._intercept: float = 0.0
        self._version: int | None = None

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @property
    def loaded_version(self) -> int | None:
        return self._version

    def load(self, directory: Path, model_version: int) -> None:
        preprocessor = json.loads((directory / "preprocessor.json").read_text(encoding="utf-8"))
        calibration = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
        inference = json.loads((directory / "inference.json").read_text(encoding="utf-8"))
        family = inference["family"]
        if family in _UNSUPPORTED_FAMILIES:
            msg = f"El adaptador ONNX del backend no soporta la familia de modelo: {family}"
            raise ValueError(msg)

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            str(directory / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        # Se asignan todos los atributos juntos al final para que una carga fallida nunca
        # deje el adaptador en un estado parcialmente actualizado (singleton del hot path).
        self._feature_names = list(preprocessor["feature_names"])
        self._median = list(preprocessor["median"])
        self._mean = list(preprocessor["mean"])
        self._scale = list(preprocessor["scale"])
        self._slope = float(calibration["slope"])
        self._intercept = float(calibration["intercept"])
        self._session = session
        self._version = model_version

    def predict(self, feature_values: list[float]) -> InferenceOutcome:
        if self._session is None:
            msg = "No hay un modelo ONNX cargado"
            raise RuntimeError(msg)
        if len(feature_values) != len(self._feature_names):
            msg = (
                f"Se esperaban {len(self._feature_names)} features, "
                f"se recibieron {len(feature_values)}"
            )
            raise ValueError(msg)

        raw = np.asarray(feature_values, dtype=float)
        if np.isinf(raw).any():
            msg = "Las features contienen valores infinitos"
            raise ValueError(msg)
        imputed = np.where(np.isnan(raw), np.asarray(self._median), raw)
        scaled = ((imputed - np.asarray(self._mean)) / np.asarray(self._scale)).astype(np.float32)
        scaled = scaled.reshape(1, -1)

        outputs = self._session.run(None, {self._session.get_inputs()[0].name: scaled})
        # xgboost/lightgbm/logistic (zipmap=False): outputs[1] son probabilidades por clase.
        probability_up_raw = float(np.asarray(outputs[1], dtype=float)[0, 1])

        z = max(min(self._slope * _logit(probability_up_raw) + self._intercept, 700), -700)
        probability_up = 1 / (1 + math.exp(-z))
        direction = "UP" if probability_up >= 0.5 else "DOWN"
        confidence = probability_up if direction == "UP" else 1 - probability_up
        return InferenceOutcome(
            direction=direction, confidence=confidence, probability_up=probability_up
        )
