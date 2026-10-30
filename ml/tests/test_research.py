from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibration import PlattCalibration
from src.features.compute import compute_features
from src.features.funding_context import FUNDING_FEATURE_COLUMNS
from src.features.price_context import PRICE_CONTEXT_COLUMNS, add_price_context
from src.features.taker_flow import TAKER_FEATURE_COLUMNS
from src.training import research
from src.training.data import build_folds
from src.training.preprocess import Preprocessor
from src.training.research_models import Candidate, fit_candidate
from tests.test_features import STEP
from tests.test_training import training_frame

pytest_plugins = ["tests.test_features"]


def test_price_context_is_causal_and_keeps_candle_alignment(market_data):
    candles, macro, fgi = market_data
    base = compute_features(candles, macro, fgi, STEP)
    full = add_price_context(base, candles, STEP)
    for length in (200, 201, 230, 300, 649):
        incremental = add_price_context(base.iloc[:length], candles.iloc[:length], STEP)
        pd.testing.assert_series_equal(
            full.loc[length - 1, PRICE_CONTEXT_COLUMNS],
            incremental.loc[length - 1, PRICE_CONTEXT_COLUMNS],
            check_names=False,
            rtol=1e-10,
            atol=1e-10,
        )
    i = 250
    assert full.loc[i, "return_1"] == pytest.approx(
        candles.loc[i, "close"] / candles.loc[i - 1, "close"] - 1
    )
    assert full.loc[i, "return_1_lag1"] == pytest.approx(
        candles.loc[i - 1, "close"] / candles.loc[i - 2, "close"] - 1
    )
    with pytest.raises(ValueError, match="consecutivas"):
        add_price_context(base, candles.drop(index=50), STEP)


def test_candidate_fit_does_not_use_validation_labels():
    data = training_frame(3200)
    fold = build_folds(data, STEP, n_folds=2)[0]
    candidate = Candidate("logistic", "existing", 540)
    first = fit_candidate(data, fold, STEP, candidate)
    modified = data.copy()
    modified.iloc[fold.validation, modified.columns.get_loc("target_label")] = (
        1 - modified.iloc[fold.validation]["target_label"]
    )
    second = fit_candidate(modified, fold, STEP, candidate)
    np.testing.assert_allclose(first.raw_validation, second.raw_validation)
    assert first.parameters == second.parameters
    assert first.preprocessor == second.preprocessor


def test_expanded_xgboost_does_not_use_validation_labels():
    data = training_frame(3200)
    for column in PRICE_CONTEXT_COLUMNS:
        data[column] = data["rsi_14"] * 0.1
    fold = build_folds(data, STEP, n_folds=2)[0]
    candidate = Candidate("xgboost", "expanded", 540)
    first = fit_candidate(data, fold, STEP, candidate)
    modified = data.copy()
    modified.iloc[fold.validation, modified.columns.get_loc("target_label")] = (
        1 - modified.iloc[fold.validation]["target_label"]
    )
    second = fit_candidate(modified, fold, STEP, candidate)
    np.testing.assert_allclose(first.raw_validation, second.raw_validation)
    assert first.parameters == second.parameters
    assert all(name in first.preprocessor.feature_names for name in PRICE_CONTEXT_COLUMNS)


def _development_data():
    data = training_frame(8000)
    for name in PRICE_CONTEXT_COLUMNS:
        data[name] = 0.0
    return data


def test_development_selection_never_receives_holdout_labels(monkeypatch):
    data = _development_data()
    start = pd.Timestamp("2025-08-01T00:00:00Z")
    end = pd.Timestamp("2025-08-10T00:00:00Z")
    candidate = Candidate("logistic", "existing", None)
    monkeypatch.setattr(research, "candidates", lambda: [candidate])

    def fake_fit(train, fold, step, spec, seed):
        assert train["open_time"].max() < start
        assert fold.end <= start
        return SimpleNamespace(
            raw_validation=np.full(len(fold.validation), 0.55),
            platt=PlattCalibration(1.0, 0.0),
        )

    monkeypatch.setattr(research, "fit_candidate", fake_fit)
    first = research.develop(data, start, end, STEP)
    modified = data.copy()
    mask = modified["open_time"] >= start
    modified.loc[mask, "target_label"] = 1 - modified.loc[mask, "target_label"]
    second = research.develop(modified, start, end, STEP)
    assert first["selected_candidate"] == second["selected_candidate"]
    assert first["rankings"] == second["rankings"]
    assert first["snapshot_sha256"] != second["snapshot_sha256"]
    assert len(first["folds"]) == 10
    assert first["folds"][-1]["end"] < start.isoformat()


async def test_holdout_rejects_changed_snapshot_before_fitting(tmp_path, monkeypatch):
    data = _development_data()
    start = pd.Timestamp("2025-08-01T00:00:00Z")
    end = pd.Timestamp("2025-08-10T00:00:00Z")
    development = {
        "status": "selection_frozen",
        "snapshot_sha256": research.snapshot_hash(data),
        "holdout_start": start.isoformat(),
        "holdout_end_exclusive": end.isoformat(),
    }
    data.loc[10, "target_label"] = 1 - data.loc[10, "target_label"]
    with pytest.raises(ValueError, match="dataset cambió"):
        await research.evaluate_holdout(None, data, development, tmp_path, STEP)
    assert not (tmp_path / "holdout.lock").exists()


@pytest.mark.parametrize(
    ("include_taker", "include_funding"),
    [(False, False), (True, False), (False, True), (True, True)],
)
async def test_selected_holdout_is_evaluated_once(
    tmp_path, monkeypatch, include_taker, include_funding
):
    data = _development_data()
    if include_taker:
        for name in TAKER_FEATURE_COLUMNS:
            data[name] = 0.5
    if include_funding:
        for name in FUNDING_FEATURE_COLUMNS:
            data[name] = 0.0001
    start = pd.Timestamp("2025-08-01T00:00:00Z")
    end = pd.Timestamp("2025-08-10T00:00:00Z")
    if include_taker and include_funding:
        feature_set = "expanded_taker_funding"
    elif include_funding:
        feature_set = "expanded_funding"
    elif include_taker:
        feature_set = "expanded_taker"
    else:
        feature_set = "existing"
    candidate = Candidate("logistic", feature_set, None)
    development = {
        "status": "selection_frozen",
        "snapshot_sha256": research.snapshot_hash(
            data, include_taker=include_taker, include_funding=include_funding
        ),
        "include_taker": include_taker,
        "include_funding": include_funding,
        "holdout_start": start.isoformat(),
        "holdout_end_exclusive": end.isoformat(),
        "selected_candidate": candidate.name + "_raw",
        "folds": [{}] * 10,
        "rankings": [{"mean_accuracy": 0.6}],
        "best_baseline_mean_accuracy": 0.5,
    }

    def fake_fit(frame, fold, step, spec, seed):
        assert spec == candidate
        assert (frame.iloc[fold.train]["open_time"] + 2 * STEP <= start).all()
        return SimpleNamespace(
            candidate=spec,
            model=SimpleNamespace(coef_=np.array([[0.3, 0.7]])),
            preprocessor=Preprocessor(["rsi_14", "macd"], [0, 0], [0, 0], [1, 1]),
            parameters={},
            platt=PlattCalibration(1.0, 0.0),
            raw_validation=np.tile([0.4, 0.6], len(fold.validation) // 2),
            scaled_validation=np.zeros((len(fold.validation), 2)),
            tuning_scores=[],
        )

    async def fake_register(*args):
        return 7

    monkeypatch.setattr(research, "fit_candidate", fake_fit)
    monkeypatch.setattr(research, "export_bundle", lambda *args: {"max_probability_error": 0.0})
    monkeypatch.setattr(research, "register_model", fake_register)
    result, version = await research.evaluate_holdout(None, data, development, tmp_path, STEP)
    assert version == 7
    assert result["predicted_up"] == result["predicted_down"] == 27
    assert (tmp_path / "holdout.json").exists()
    with pytest.raises(FileExistsError):
        await research.evaluate_holdout(None, data, development, tmp_path, STEP)


def test_prevents_overlapping_previously_evaluated_holdouts(tmp_path):
    directory = tmp_path / "research-past"
    directory.mkdir()
    (directory / "development.json").write_text(
        '{"holdout_start":"2025-08-01T00:00:00Z","holdout_end_exclusive":"2025-09-01T00:00:00Z"}'
    )
    (directory / "holdout.lock").write_text("evaluated")
    with pytest.raises(ValueError, match="se solapa"):
        research.assert_unused_holdout(
            tmp_path, pd.Timestamp("2025-08-10T00:00:00Z"), pd.Timestamp("2025-09-10T00:00:00Z")
        )
    research.assert_unused_holdout(
        tmp_path, pd.Timestamp("2025-09-01T00:00:00Z"), pd.Timestamp("2025-10-01T00:00:00Z")
    )
