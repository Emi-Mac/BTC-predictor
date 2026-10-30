from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibration import fit_platt
from src.evaluation.metrics import evaluate, evaluate_baselines
from src.features.compute import FEATURE_COLUMNS
from src.training.data import build_folds, inner_windows
from src.training.models import fit_lstm, predict_lstm, sequences, xgb_grid
from src.training.preprocess import fit_preprocessor

STEP = timedelta(hours=4)


def training_frame(n=8000):
    rng = np.random.default_rng(2026)
    values = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    frame["target_label"] = (values[:, 0] > 0).astype(int)
    frame["previous_direction"] = rng.integers(0, 2, n)
    frame["open_time"] = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    weekday = frame["open_time"].dt.day_of_week
    for i, day in enumerate(("mon", "tue", "wed", "thu", "fri", "sat", "sun")):
        frame[f"weekday_{day}"] = (weekday == i).astype(int)
    frame["is_weekend"] = (weekday >= 5).astype(int)
    return frame


def test_walk_forward_has_ten_complete_months_and_purged_labels():
    data = training_frame()
    folds = build_folds(data, STEP)
    assert len(folds) == 10
    for i, fold in enumerate(folds):
        train_times = data.iloc[fold.train]["open_time"]
        validation_times = data.iloc[fold.validation]["open_time"]
        assert train_times.iloc[-1] + 2 * STEP <= fold.start
        assert validation_times.iloc[0] == fold.start
        assert validation_times.iloc[-1] + STEP == fold.end
        assert fold.start - train_times.iloc[0] >= pd.Timedelta(days=365)
        if i:
            assert fold.start == folds[i - 1].end
            assert len(fold.train) > len(folds[i - 1].train)

        base, tuning, pre_cal, calibration = inner_windows(data, fold, STEP)
        tune_start = fold.start - pd.DateOffset(months=2)
        cal_start = fold.start - pd.DateOffset(months=1)
        assert (data.iloc[base]["open_time"] + 2 * STEP <= tune_start).all()
        assert (data.iloc[tuning]["open_time"] + 2 * STEP <= cal_start).all()
        assert (data.iloc[pre_cal]["open_time"] + 2 * STEP <= cal_start).all()
        assert (data.iloc[calibration]["open_time"] + 2 * STEP <= fold.start).all()
        assert not set(pre_cal).intersection(calibration)


def test_fold_rejects_missing_periods():
    data = training_frame().drop(index=300)
    with pytest.raises(ValueError, match="continuidad"):
        build_folds(data, STEP)


def test_imputer_scaler_and_feature_selection_ignore_future_values():
    data = training_frame(800)
    data.loc[10:20, "rsi_14"] = np.nan
    cutoff = data.loc[399, "open_time"] + 2 * STEP
    pre = fit_preprocessor(data, cutoff, STEP)
    changed = data.copy()
    changed.loc[400:, FEATURE_COLUMNS] = 1000000.0
    changed.loc[400:, "target_label"] = 1 - changed.loc[400:, "target_label"]
    assert fit_preprocessor(changed, cutoff, STEP) == pre
    transformed = pre.transform(data.iloc[:20])
    assert np.isfinite(transformed).all()
    assert "rsi_14" in pre.feature_names
    index = pre.feature_names.index("rsi_14")
    assert pre.median[index] == pytest.approx(data.loc[:399, "rsi_14"].median())


def test_metrics_calibration_bins_and_baselines_are_deterministic():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    probability = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1])
    metrics = evaluate(y, y, probability)
    assert sum(item["count"] for item in metrics["calibration_bins"]) == len(y)
    assert metrics["accuracy"] == 1
    assert all(0 <= metrics[key] <= 1 for key in ["f1_macro", "auc_roc", "brier"])
    with pytest.raises(ValueError, match="inválidas"):
        evaluate(y, y, np.full(len(y), 1.1))
    first = evaluate_baselines(y, y, y, y, seed=7)
    assert evaluate_baselines(y, y, y, y, seed=7) == first
    assert first["previous_direction"]["accuracy"] == 1
    calibration = fit_platt(np.linspace(0.1, 0.9, len(y)), y)
    assert np.isfinite(calibration.transform(probability)).all()


def test_conditional_baseline_uses_only_training_transition_probabilities():
    train_previous = np.array([0] * 10 + [1] * 10)
    train_y = np.array([1] * 8 + [0] * 2 + [1] * 2 + [0] * 8)
    validation_previous = np.array([0, 1, 0, 1])
    validation_y = np.array([1, 0, 1, 0])
    scores = evaluate_baselines(train_y, train_previous, validation_y, validation_previous, 42)
    assert scores["conditional_direction"]["accuracy"] == 1
    assert scores["previous_direction"]["accuracy"] == 0
    assert scores["conditional_direction"]["log_loss"] == pytest.approx(
        scores["previous_direction"]["log_loss"]
    )


def test_sequences_have_only_past_context_and_lstm_is_deterministic():
    x = np.arange(60 * 3, dtype=np.float32).reshape(60, 3) / 100
    windows, indices = sequences(x, np.array([19, 20, 50]))
    assert indices.tolist() == [19, 20, 50]
    np.testing.assert_array_equal(windows[1], x[1:21])
    targets = np.array([0, 1, 0])
    model_a = fit_lstm(windows, targets, seed=123, epochs=1)
    model_b = fit_lstm(windows, targets, seed=123, epochs=1)
    np.testing.assert_allclose(predict_lstm(model_a, windows), predict_lstm(model_b, windows))


def test_grid_explores_all_requested_parameters():
    grid = xgb_grid()
    assert len(grid) == 16
    assert all(
        len({candidate[name] for candidate in grid}) >= 2
        for name in ("max_depth", "learning_rate", "n_estimators", "subsample")
    )
