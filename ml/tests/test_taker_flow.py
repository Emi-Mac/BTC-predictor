from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation.importance import xgb_importance
from src.features.funding_context import FUNDING_FEATURE_COLUMNS
from src.features.price_context import PRICE_CONTEXT_COLUMNS
from src.features.taker_flow import TAKER_FEATURE_COLUMNS, add_taker_flow
from src.training import research, taker_ablation
from src.training.data import build_folds
from src.training.research_models import Candidate, candidates, fit_candidate
from src.training.snapshot import snapshot_hash
from tests.conftest import STEP, make_candles
from tests.test_training import training_frame


def _candles(n=60):
    candles = make_candles(n)
    weights = np.linspace(0.2, 0.8, n)
    candles["taker_buy_base_volume"] = candles["volume"] * weights
    candles["taker_buy_quote_volume"] = candles["quote_volume"] * weights
    return candles


def test_taker_ratios_weighted_windows_and_imbalance():
    candles = _candles()
    features = candles[["open_time"]]
    result = add_taker_flow(features, candles, STEP)
    i = 30
    buy = candles["taker_buy_base_volume"]
    volume = candles["volume"]
    ratio = buy.iloc[i] / volume.iloc[i]
    assert result.loc[i, "taker_buy_ratio"] == pytest.approx(ratio)
    assert result.loc[i, "taker_buy_quote_ratio"] == pytest.approx(ratio)
    assert result.loc[i, "taker_buy_imbalance"] == pytest.approx(2 * ratio - 1)
    assert result.loc[i, "taker_buy_ratio_6"] == pytest.approx(
        buy.iloc[i - 5 : i + 1].sum() / volume.iloc[i - 5 : i + 1].sum()
    )
    assert result.loc[i, "taker_buy_ratio_24"] == pytest.approx(
        buy.iloc[i - 23 : i + 1].sum() / volume.iloc[i - 23 : i + 1].sum()
    )
    assert "taker_buy_imbalance" not in TAKER_FEATURE_COLUMNS


def test_taker_features_match_each_incremental_prefix():
    candles = _candles()
    full = add_taker_flow(candles[["open_time"]], candles, STEP)
    columns = [*TAKER_FEATURE_COLUMNS, "taker_buy_imbalance"]
    for length in range(1, len(candles) + 1):
        prefix = candles.iloc[:length]
        incremental = add_taker_flow(prefix[["open_time"]], prefix, STEP)
        pd.testing.assert_series_equal(
            full.loc[length - 1, columns],
            incremental.loc[length - 1, columns],
            check_names=False,
            rtol=1e-10,
            atol=1e-10,
        )


def test_unknown_and_zero_volume_remain_missing_without_fabrication():
    candles = _candles()
    candles.loc[5, ["taker_buy_base_volume", "taker_buy_quote_volume"]] = np.nan
    candles.loc[
        20, ["volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
    ] = 0
    result = add_taker_flow(candles[["open_time"]], candles, STEP)
    assert pd.isna(result.loc[5, "taker_buy_ratio"])
    assert pd.isna(result.loc[20, "taker_buy_ratio"])
    assert pd.isna(result.loc[20, "taker_buy_quote_ratio"])
    assert pd.isna(result.loc[10, "taker_buy_ratio_6"])
    assert pd.notna(result.loc[11, "taker_buy_ratio_6"])


def test_invalid_taker_values_and_gaps_are_rejected():
    candles = _candles()
    candles.loc[5, "taker_buy_base_volume"] = candles.loc[5, "volume"] + 1
    with pytest.raises(ValueError, match="inválidos"):
        add_taker_flow(candles[["open_time"]], candles, STEP)
    candles = _candles()
    with pytest.raises(ValueError, match="consecutivas"):
        add_taker_flow(candles[["open_time"]], candles.drop(index=5), STEP)


def _training_with_taker(n=8000):
    data = training_frame(n)
    for name in PRICE_CONTEXT_COLUMNS:
        data[name] = 0.0
    for name in TAKER_FEATURE_COLUMNS:
        data[name] = np.linspace(0.2, 0.8, n)
    for name in FUNDING_FEATURE_COLUMNS:
        data[name] = np.linspace(-0.001, 0.001, n)
    return data


def test_old_snapshots_are_stable_but_taker_snapshots_detect_changes():
    data = _training_with_taker()
    old = snapshot_hash(data)
    new = snapshot_hash(data, include_taker=True)
    with_funding = snapshot_hash(data, include_funding=True)
    data.loc[10, "taker_buy_ratio"] = 0.99
    assert snapshot_hash(data) == old
    assert snapshot_hash(data, include_taker=True) != new
    assert len(candidates()) == 9
    assert len(candidates(include_taker=True)) == 15
    assert len(candidates(include_lightgbm=True)) == 13
    assert len(candidates(include_taker=True, include_lightgbm=True)) == 21
    data.loc[10, "funding_rate"] = 0.002
    assert snapshot_hash(data) == old
    assert snapshot_hash(data, include_funding=True) != with_funding
    assert len(candidates(include_funding=True)) == 13
    assert len(candidates(include_taker=True, include_funding=True)) == 23


@pytest.mark.parametrize(
    ("feature_set", "expected"),
    [
        ("expanded_taker", set(TAKER_FEATURE_COLUMNS)),
        ("expanded_taker_ratio", {"taker_buy_ratio"}),
        (
            "expanded_taker_multiscale",
            {"taker_buy_ratio", "taker_buy_ratio_6", "taker_buy_ratio_24"},
        ),
    ],
)
def test_taker_candidate_does_not_use_validation_labels(feature_set, expected):
    data = _training_with_taker(3200)
    fold = build_folds(data, STEP, n_folds=2)[0]
    candidate = Candidate("logistic", feature_set, None)
    first = fit_candidate(data, fold, STEP, candidate)
    modified = data.copy()
    modified.iloc[fold.validation, modified.columns.get_loc("target_label")] = (
        1 - modified.iloc[fold.validation]["target_label"]
    )
    second = fit_candidate(modified, fold, STEP, candidate)
    np.testing.assert_allclose(first.raw_validation, second.raw_validation)
    assert first.parameters == second.parameters
    assert set(TAKER_FEATURE_COLUMNS).intersection(first.preprocessor.feature_names) == expected


def test_ablation_is_development_only_and_does_not_read_future_labels(monkeypatch):
    data = _training_with_taker()
    cutoff = pd.Timestamp("2025-08-01T00:00:00Z")

    def fake_fit(train, fold, step, candidate, seed):
        assert train["open_time"].max() < cutoff
        assert fold.end < cutoff
        return SimpleNamespace(raw_validation=np.full(len(fold.validation), 0.55))

    monkeypatch.setattr(taker_ablation, "fit_candidate", fake_fit)
    first = taker_ablation.run_ablation(data, cutoff, STEP)
    changed = data.copy()
    mask = changed["open_time"] >= cutoff
    changed.loc[mask, "target_label"] = 1 - changed.loc[mask, "target_label"]
    second = taker_ablation.run_ablation(changed, cutoff, STEP)
    assert first == second
    assert first["independent_holdout_evaluated"] is False
    assert first["mean_accuracy_difference"] == 0
    assert len(first["folds"]) == 10


def test_research_flags_freeze_optional_taker_and_lightgbm_candidates(monkeypatch):
    data = _training_with_taker()
    start = pd.Timestamp("2025-08-01T00:00:00Z")
    end = pd.Timestamp("2025-08-10T00:00:00Z")
    candidate = Candidate("lightgbm", "expanded_taker", None)
    monkeypatch.setattr(research, "candidates", lambda include_taker, include_lightgbm: [candidate])

    def fake_fit(train, fold, step, spec, seed):
        assert train["open_time"].max() < start
        assert spec == candidate
        return SimpleNamespace(
            raw_validation=np.full(len(fold.validation), 0.55),
            platt=SimpleNamespace(transform=lambda values: values),
        )

    monkeypatch.setattr(research, "fit_candidate", fake_fit)
    report = research.develop(data, start, end, STEP, include_taker=True, include_lightgbm=True)
    assert report["include_taker"] is True
    assert report["include_lightgbm"] is True
    assert report["snapshot_sha256"] == snapshot_hash(data, include_taker=True)
    assert report["selected_candidate"].startswith(candidate.name)


def test_funding_candidate_and_ablation_do_not_use_future_labels(monkeypatch):
    data = _training_with_taker(3200)
    fold = build_folds(data, STEP, n_folds=2)[0]
    candidate = Candidate("logistic", "expanded_taker_funding", None)
    first = fit_candidate(data, fold, STEP, candidate)
    changed = data.copy()
    changed.iloc[fold.validation, changed.columns.get_loc("target_label")] = (
        1 - changed.iloc[fold.validation]["target_label"]
    )
    second = fit_candidate(changed, fold, STEP, candidate)
    np.testing.assert_allclose(first.raw_validation, second.raw_validation)
    assert all(name in first.preprocessor.feature_names for name in FUNDING_FEATURE_COLUMNS)

    large = _training_with_taker()
    cutoff = pd.Timestamp("2025-08-01T00:00:00Z")

    def fake_fit(train, validation, step, spec, seed):
        assert train["open_time"].max() < cutoff
        return SimpleNamespace(raw_validation=np.full(len(validation.validation), 0.55))

    monkeypatch.setattr(taker_ablation, "fit_candidate", fake_fit)
    report = taker_ablation.run_ablation(large, cutoff, STEP, include_funding=True)
    assert report["independent_holdout_evaluated"] is False
    assert report["include_funding"] is True
    assert report["rows_with_missing_funding_features"] == 0
    assert report["funding_development_difference"] == 0


def test_lightgbm_candidate_does_not_use_validation_labels():
    data = _training_with_taker(3200)
    fold = build_folds(data, STEP, n_folds=2)[0]
    candidate = Candidate("lightgbm", "expanded_taker", None)
    first = fit_candidate(data, fold, STEP, candidate)
    modified = data.copy()
    modified.iloc[fold.validation, modified.columns.get_loc("target_label")] = (
        1 - modified.iloc[fold.validation]["target_label"]
    )
    second = fit_candidate(modified, fold, STEP, candidate)
    np.testing.assert_allclose(first.raw_validation, second.raw_validation)
    assert first.parameters == second.parameters
    importance = xgb_importance(
        first.model, first.scaled_validation, first.preprocessor.feature_names
    )
    assert sum(importance.values()) == pytest.approx(1.0)
