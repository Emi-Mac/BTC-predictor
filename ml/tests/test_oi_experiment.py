import numpy as np
import pandas as pd
import pytest

from src.features.funding_context import FUNDING_FEATURE_COLUMNS
from src.features.price_context import PRICE_CONTEXT_COLUMNS
from src.research import oi_experiment
from src.research.metrics_features import OI_FEATURE_COLUMNS, POSITION_FEATURE_COLUMNS
from src.research.oi_experiment import _fit_variant, inner_splits, paired_fold_comparison
from src.training.data import build_folds
from tests.test_training import STEP, training_frame


def test_nested_splits_purge_labels_and_use_three_distinct_windows():
    data = training_frame(5000)
    fold = build_folds(data, STEP, n_folds=2)[0]
    splits = inner_splits(data, fold, STEP)
    assert len(splits) == 3
    for index, (train, valid) in enumerate(splits):
        start = fold.start - pd.DateOffset(months=3 - index)
        assert (data.iloc[train]["open_time"] + 2 * STEP <= start).all()
        assert (data.iloc[valid]["open_time"] + 2 * STEP <= fold.start).all()
        assert not set(train).intersection(valid)


def test_nested_tuning_never_uses_outer_validation_labels():
    data = training_frame(5000)
    fold = build_folds(data, STEP, n_folds=2)[0]
    names = ["rsi_14", "macd"]
    first_p, first_meta = _fit_variant(data, fold, names, STEP)
    changed = data.copy()
    changed.iloc[fold.validation, changed.columns.get_loc("target_label")] = (
        1 - changed.iloc[fold.validation]["target_label"]
    )
    second_p, second_meta = _fit_variant(changed, fold, names, STEP)
    np.testing.assert_allclose(first_p, second_p)
    assert first_meta == second_meta


def test_ablation_is_development_only_and_includes_conditional_baseline(monkeypatch):
    data = training_frame(3200)
    for column in (
        PRICE_CONTEXT_COLUMNS
        + FUNDING_FEATURE_COLUMNS
        + OI_FEATURE_COLUMNS
        + POSITION_FEATURE_COLUMNS
    ):
        data[column] = 0.1
    cutoff = data["open_time"].iloc[-1] + STEP
    monkeypatch.setattr(oi_experiment, "FOLDS", 2)

    def fake_fit(train, fold, names, step):
        assert (train["open_time"] + 2 * STEP <= cutoff).all()
        return np.full(len(fold.validation), 0.55), {"feature_names": names}

    monkeypatch.setattr(oi_experiment, "_fit_variant", fake_fit)
    report, predictions = oi_experiment.run_ablation(data, cutoff, STEP)
    assert report["independent_holdout_evaluated"] is False
    assert report["model_registered"] is False
    assert "conditional_direction" in report["baseline_mean_accuracy"]
    assert set(report["paired_fold_comparisons"]) == {
        "base_oi_minus_base",
        "base_oi_position_minus_base",
        "base_minus_conditional",
    }
    assert len(report["folds"]) == 2
    assert predictions["open_time"].is_unique


def test_paired_fold_bootstrap_is_deterministic_and_validates_inputs():
    first = [0.50, 0.55, 0.60]
    second = [0.45, 0.50, 0.55]
    result = paired_fold_comparison(first, second, seed=7, draws=100)
    assert result == paired_fold_comparison(first, second, seed=7, draws=100)
    assert result["mean_accuracy_difference"] == pytest.approx(0.05)
    assert result["bootstrap_95_interval"] == pytest.approx([0.05, 0.05])
    with pytest.raises(ValueError, match="folds pareados"):
        paired_fold_comparison([0.5], [0.4])
    with pytest.raises(ValueError, match="entre 0 y 1"):
        paired_fold_comparison([0.5, 1.5], [0.4, 0.6])
