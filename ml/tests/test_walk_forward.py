import numpy as np

from src.training.walk_forward import TrainingOptions, run_walk_forward
from tests.test_training import STEP, training_frame


def test_walk_forward_regression_on_known_signal():
    data = training_frame(3200)
    options = TrainingOptions(
        n_folds=2,
        lstm_epochs=1,
        grid=[{"max_depth": 2, "learning_rate": 0.2, "n_estimators": 15, "subsample": 1.0}],
    )
    result = run_walk_forward(data, STEP, options)
    report = result.report
    assert len(report["folds"]) == 2
    assert report["selected_family"] == "xgboost"
    assert report["final_fold_accuracy"] > report["final_fold_best_baseline"] + 0.02
    assert not report["criteria"]["passed"]  # No se promueve con solo dos folds.
    assert np.isclose(sum(report["importance"]["xgboost"].values()), 1)
    for fold in report["folds"]:
        assert fold["calibration_samples"] > 0
        assert len(fold["xgboost_grid"]) == 1
