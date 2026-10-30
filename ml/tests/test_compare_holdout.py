import json

import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation.calibration import PlattCalibration
from src.evaluation.metrics import evaluate
from src.export.onnx_model import OnnxPredictor, export_bundle
from src.features.price_context import PRICE_CONTEXT_COLUMNS
from src.training.compare_holdout import compare_fixed_models
from src.training.preprocess import Preprocessor
from src.training.snapshot import snapshot_hash
from src.training.walk_forward import ModelBundle
from tests.test_training import training_frame


def test_comparison_uses_frozen_models_and_snapshot(tmp_path):
    data = training_frame(600)
    for column in PRICE_CONTEXT_COLUMNS:
        data[column] = 0.0
    names = ["rsi_14", "macd"]
    pre = Preprocessor(names, [0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
    x = pre.transform(data)
    y = data["target_label"].to_numpy(dtype=int)
    current = LogisticRegression(C=1.0, random_state=1).fit(x[:300], y[:300])
    prior = LogisticRegression(C=0.01, random_state=1).fit(x[:300], y[:300])
    cal = PlattCalibration(1.0, 0.0)
    directory = tmp_path / "research"
    directory.mkdir()
    reference = tmp_path / "previous"
    export_bundle(ModelBundle("logistic", current, pre, cal, {}, x), directory / "model")
    export_bundle(ModelBundle("logistic", prior, pre, cal, {}, x), reference)

    start = data["open_time"].iloc[400]
    end = data["open_time"].iloc[500]
    (directory / "development.json").write_text(
        json.dumps({"snapshot_sha256": snapshot_hash(data)})
    )
    probability = OnnxPredictor(directory / "model").predict_proba(x[400:500])
    metrics = evaluate(y[400:500], (probability >= 0.5).astype(int), probability)
    (directory / "holdout.json").write_text(
        json.dumps(
            {
                "holdout_start": start.isoformat(),
                "holdout_end_exclusive": end.isoformat(),
                "metrics": metrics,
            }
        )
    )
    (reference / "walk_forward.json").write_text(
        json.dumps({"folds": [{"validation_end_exclusive": start.isoformat()}]})
    )
    report = compare_fixed_models(data, directory, reference)
    assert report["samples"] == 100
    assert report["selected_model"]["accuracy"] == metrics["accuracy"]
    assert len(report["selected_accuracy_wilson_95"]) == 2
    data.loc[10, "target_label"] = 1 - data.loc[10, "target_label"]
    with pytest.raises(ValueError, match="dataset cambió"):
        compare_fixed_models(data, directory, reference)
