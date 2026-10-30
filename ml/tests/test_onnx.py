import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation.calibration import PlattCalibration
from src.export.onnx_model import OnnxPredictor, export_bundle
from src.training.models import fit_lightgbm, fit_lstm, fit_xgb, predict_lstm
from src.training.preprocess import Preprocessor
from src.training.walk_forward import ModelBundle


@pytest.mark.parametrize("family", ["xgboost", "lstm", "logistic", "lightgbm"])
def test_onnx_matches_native_and_preprocessing(family, tmp_path):
    rng = np.random.default_rng(12)
    features = ["a", "b", "c"]
    pre = Preprocessor(features, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 4.0, 3.0])
    calibrator = PlattCalibration(0.7, -0.2)
    raw = rng.normal(size=(24, 3))
    raw[0, 0] = np.nan
    scaled = pre.transform(pd.DataFrame(raw, columns=features))
    y = np.arange(24) % 2
    if family == "xgboost":
        model = fit_xgb(
            scaled,
            y,
            {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 8, "subsample": 1.0},
            42,
        )
        native = model.predict_proba(scaled)[:, 1]
        assert np.unique(native).size > 1
        sample, input_raw = scaled, raw
    elif family == "logistic":
        model = LogisticRegression(random_state=42).fit(scaled, y)
        native = model.predict_proba(scaled)[:, 1]
        sample, input_raw = scaled, raw
    elif family == "lightgbm":
        model = fit_lightgbm(
            scaled,
            y,
            {
                "max_depth": 2,
                "num_leaves": 4,
                "learning_rate": 0.1,
                "n_estimators": 8,
                "subsample": 1.0,
                "min_child_samples": 2,
            },
            42,
        )
        native = model.predict_proba(scaled)[:, 1]
        assert np.unique(native).size > 1
        sample, input_raw = scaled, raw
    else:
        sample = np.repeat(scaled[:, None, :], 20, axis=1).astype(np.float32)
        input_raw = np.repeat(raw[:, None, :], 20, axis=1)
        model = fit_lstm(sample, y, seed=42, epochs=1)
        native = predict_lstm(model, sample)
    directory = tmp_path / family
    parity = export_bundle(ModelBundle(family, model, pre, calibrator, {}, sample), directory)
    assert parity["max_probability_error"] < 1e-5
    predictor = OnnxPredictor(directory)
    actual = predictor.predict_proba(input_raw)
    np.testing.assert_allclose(actual, calibrator.transform(native), atol=1e-5)
    np.testing.assert_allclose(actual[:1], predictor.predict_proba(input_raw[:1]), atol=1e-6)
    with pytest.raises(ValueError):
        predictor.predict_proba(np.ones((2, 4)))
