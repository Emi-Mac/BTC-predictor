"""Exportación ONNX y una ruta de inferencia que reproduce el preprocesamiento."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxmltools
import onnxruntime as ort
import torch
from lightgbm import LGBMClassifier
from onnxmltools.convert.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.evaluation.calibration import PlattCalibration
from src.training.models import DirectionLSTM, predict_lstm
from src.training.preprocess import Preprocessor

if TYPE_CHECKING:
    from pathlib import Path

    from src.training.types import FloatArray
    from src.training.walk_forward import ModelBundle


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _raw_probabilities(
    session: ort.InferenceSession, family: str, scaled: FloatArray
) -> FloatArray:
    outputs = session.run(None, {session.get_inputs()[0].name: scaled.astype(np.float32)})
    if family in ("xgboost", "logistic", "lightgbm"):
        return np.asarray(outputs[1], dtype=float)[:, 1]
    if family == "lstm":
        logits = np.asarray(outputs[0], dtype=float).reshape(-1)
        probabilities: FloatArray = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        return probabilities
    raise ValueError(f"Familia de modelo desconocida: {family}")


def export_bundle(bundle: ModelBundle, directory: Path) -> dict[str, float]:
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "model.onnx"
    if bundle.family == "xgboost" and isinstance(bundle.model, XGBClassifier):
        model = onnxmltools.convert_xgboost(
            bundle.model,
            initial_types=[
                ("features", FloatTensorType([None, len(bundle.preprocessor.feature_names)]))
            ],
            target_opset=15,
        )
        onnx.checker.check_model(model)
        onnx.save_model(model, path)
        native = np.asarray(bundle.model.predict_proba(bundle.sample[:16]), dtype=float)[:, 1]
    elif bundle.family == "lightgbm" and isinstance(bundle.model, LGBMClassifier):
        model = onnxmltools.convert_lightgbm(
            bundle.model,
            initial_types=[
                ("features", FloatTensorType([None, len(bundle.preprocessor.feature_names)]))
            ],
            target_opset=15,
            zipmap=False,
        )
        label_batch = model.graph.output[0].type.tensor_type.shape.dim[0]
        label_batch.ClearField("dim_value")
        label_batch.dim_param = "batch"
        onnx.checker.check_model(model)
        onnx.save_model(model, path)
        native = np.asarray(bundle.model.predict_proba(bundle.sample[:16]), dtype=float)[:, 1]
    elif bundle.family == "logistic" and isinstance(bundle.model, LogisticRegression):
        model = convert_sklearn(
            bundle.model,
            initial_types=[
                ("features", SklearnFloatTensorType([None, len(bundle.preprocessor.feature_names)]))
            ],
            options={id(bundle.model): {"zipmap": False}},
            target_opset=15,
        )
        onnx.checker.check_model(model)
        onnx.save_model(model, path)
        native = bundle.model.predict_proba(bundle.sample[:16])[:, 1]
    elif bundle.family == "lstm" and isinstance(bundle.model, DirectionLSTM):
        bundle.model.eval()
        torch.onnx.export(
            bundle.model,
            (torch.from_numpy(bundle.sample[:1]),),
            str(path),
            input_names=["features"],
            output_names=["logit"],
            dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(path))
        native = predict_lstm(bundle.model, bundle.sample[:16])
    else:
        raise ValueError("El modelo no coincide con su familia")
    session = _session(path)
    converted = _raw_probabilities(session, bundle.family, bundle.sample[:16])
    error = float(np.max(np.abs(native - converted)))
    if not np.allclose(native, converted, rtol=1e-5, atol=1e-5):
        raise ValueError(f"El modelo ONNX no reproduce las probabilidades nativas: error={error}")
    (directory / "preprocessor.json").write_text(
        json.dumps(asdict(bundle.preprocessor), indent=2, allow_nan=False), encoding="utf-8"
    )
    (directory / "calibration.json").write_text(
        json.dumps(asdict(bundle.calibration), indent=2, allow_nan=False), encoding="utf-8"
    )
    (directory / "inference.json").write_text(
        json.dumps({"family": bundle.family}, indent=2), encoding="utf-8"
    )
    return {"max_probability_error": error}


class OnnxPredictor:
    """Acepta features crudas en el orden de preprocessor.json y devuelve P(UP)."""

    def __init__(self, directory: Path) -> None:
        self.preprocessor = Preprocessor(
            **json.loads((directory / "preprocessor.json").read_text(encoding="utf-8"))
        )
        self.calibration = PlattCalibration(
            **json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
        )
        self.family = json.loads((directory / "inference.json").read_text(encoding="utf-8"))[
            "family"
        ]
        self.session = _session(directory / "model.onnx")

    def predict_proba(self, raw_features: FloatArray) -> FloatArray:
        values = np.asarray(raw_features, dtype=float)
        dimensions = (
            2
            if self.family in ("xgboost", "logistic", "lightgbm")
            else 3
            if self.family == "lstm"
            else 0
        )
        if values.ndim != dimensions or values.shape[-1] != len(self.preprocessor.feature_names):
            raise ValueError("La forma de las features no coincide con el modelo")
        if self.family == "lstm" and values.shape[1] != 20:
            raise ValueError("El LSTM requiere secuencias de 20 velas")
        if not len(values) or np.isinf(values).any():
            raise ValueError("Las features están vacías o contienen infinitos")
        values = np.where(np.isnan(values), np.asarray(self.preprocessor.median), values)
        scaled = (values - np.asarray(self.preprocessor.mean)) / np.asarray(self.preprocessor.scale)
        return self.calibration.transform(_raw_probabilities(self.session, self.family, scaled))
