import json
from math import comb

import numpy as np
import pandas as pd
import pytest

from src.evaluation.economic import (
    backtest,
    performance,
    positions_from_probabilities,
    strategy_returns,
)
from src.evaluation.recalibration import expanding_oos_calibration
from src.evaluation.robustness import (
    block_bootstrap_mean,
    cpcv_paths,
    cpcv_splits,
    deflated_sharpe_ratio,
    effective_trials,
    moving_block_indices,
    probability_of_backtest_overfitting,
    reality_check,
)
from src.evaluation.selective import coverage_curve, skill_scores, wilson_interval
from src.research import evaluation_harness as harness
from tests.test_training import STEP, training_frame

# ─── Selectiva y skill ───────────────────────────────────────────────────────


def test_coverage_curve_separates_causal_margins_from_descriptive_quantiles():
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.52, 0.51, 0.49, 0.48])
    curve = coverage_curve(y, p, margins=(0.0, 0.05), coverages=(1.0, 1 / 3))
    full, confident = curve["causal_margins"]
    assert full["samples"] == 6 and full["accuracy"] == pytest.approx(4 / 6)
    assert confident["samples"] == 2 and confident["accuracy"] == 1.0
    assert curve["descriptive_top_confidence"][1]["samples"] == 2
    assert curve["descriptive_top_confidence"][1]["accuracy"] == 1.0
    with pytest.raises(ValueError, match="margen"):
        coverage_curve(y, p, margins=(0.6,))


def test_skill_scores_are_zero_for_reference_and_positive_when_better():
    y = np.array([1, 0, 1, 1, 0, 1])
    reference = np.full(len(y), y.mean())
    assert skill_scores(y, reference, reference)["brier_skill_score"] == pytest.approx(0)
    better = np.where(y == 1, 0.8, 0.2)
    scores = skill_scores(y, better, reference)
    assert scores["brier_skill_score"] > 0 and scores["log_loss_skill_score"] > 0


def test_wilson_interval_contains_proportion():
    lower, upper = wilson_interval(55, 100)
    assert lower < 0.55 < upper
    with pytest.raises(ValueError):
        wilson_interval(5, 0)


# ─── Económico ───────────────────────────────────────────────────────────────


def test_positions_respect_margin_and_short_flag():
    p = np.array([0.7, 0.51, 0.5, 0.3])
    assert positions_from_probabilities(p, 0.02).tolist() == [1, 0, 0, -1]
    assert positions_from_probabilities(p, 0.0, allow_short=False).tolist() == [1, 1, 0, 0]


def test_strategy_returns_charge_entry_and_each_rebalance():
    pos = np.array([1.0, 1.0, -1.0, 0.0])
    ret = np.array([0.01, -0.02, -0.01, 0.05])
    net = strategy_returns(pos, ret, cost_bps=10)
    np.testing.assert_allclose(net, [0.01 - 0.001, -0.02, 0.01 - 0.002, -0.001])


def test_performance_reports_profit_factor_and_drawdown():
    result = performance(np.array([0.1, -0.05, 0.05, -0.1]), np.array([1, 1, -1, 0]))
    assert result["profit_factor"] == pytest.approx(0.15 / 0.15)
    assert result["max_drawdown"] < 0
    assert result["exposure"] == 0.75
    rows = backtest(np.array([0.6, 0.4, 0.55]), np.array([0.01, -0.01, 0.02]), (0.0,), (0.0, 7.0))
    assert len(rows) == 2 and rows[0]["total_return"] > rows[1]["total_return"]


# ─── Recalibración ───────────────────────────────────────────────────────────


def _calibration_inputs(months=6, seed=1):
    times = pd.Series(
        pd.date_range("2024-01-01", "2024-07-01", freq="4h", tz="UTC", inclusive="left")
    )
    fold = times.dt.month.to_numpy()
    starts = {m: pd.Timestamp(f"2024-{m:02d}-01", tz="UTC") for m in range(1, months + 1)}
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 0.7, len(times))
    labels = (rng.uniform(size=len(times)) < raw).astype(int)
    return times, fold, starts, raw, labels


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_oos_calibration_never_uses_current_or_unknown_labels(method):
    times, fold, starts, raw, labels = _calibration_inputs()
    first = expanding_oos_calibration(
        times, fold, starts, raw, labels, STEP, method, min_samples=300
    )
    assert np.isnan(first[fold <= 2]).all() and np.isfinite(first[fold >= 3]).all()
    changed = labels.copy()
    changed[fold == 4] = 1 - changed[fold == 4]
    # La última vela de marzo aún no tiene etiqueta conocida al inicio de abril.
    changed[np.flatnonzero(fold == 3)[-1]] ^= 1
    second = expanding_oos_calibration(
        times, fold, starts, raw, changed, STEP, method, min_samples=300
    )
    np.testing.assert_allclose(first[fold <= 4], second[fold <= 4])
    windowed = expanding_oos_calibration(
        times, fold, starts, raw, labels, STEP, method, min_samples=300, window=300
    )
    assert np.isfinite(windowed[fold >= 3]).all()


# ─── Robustez ────────────────────────────────────────────────────────────────


def test_cpcv_purges_embargoes_and_builds_complete_paths():
    groups, splits = cpcv_splits(240, 6, 2, purge=2, embargo=5)
    assert len(splits) == comb(6, 2)
    for split in splits:
        assert not set(split.train) & set(split.test)
        for g in split.test_groups:
            start, end = groups[g][0], groups[g][-1]
            assert not ((split.train >= start - 2) & (split.train <= end + 5)).any()
    paths = cpcv_paths(6, 2)
    assert len(paths) == comb(5, 1)
    for g in range(6):
        used = [path[g] for path in paths]
        assert len(set(used)) == len(used)
        assert all(g in splits[i].test_groups for i in used)


def test_block_bootstrap_and_reality_check_are_deterministic():
    rng = np.random.default_rng(0)
    assert len(moving_block_indices(100, 7, rng)) == 100
    values = rng.normal(0.1, 1, 500)
    assert block_bootstrap_mean(values, 10, 300, 3) == block_bootstrap_mean(values, 10, 300, 3)
    strong = np.column_stack([rng.normal(0.3, 1, 1000), rng.normal(0, 1, 1000)])
    null = rng.normal(0, 1, (1000, 2))
    assert reality_check(strong, 10, 300)["p_value"] < 0.05
    assert reality_check(strong, 10, 300)["best_column"] == 0
    assert reality_check(null, 10, 300)["p_value"] > 0.05


def test_pbo_low_for_dominant_candidate_and_high_for_noise():
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 1, (1600, 8))
    dominant = noise.copy()
    dominant[:, 0] += 1.0
    assert probability_of_backtest_overfitting(dominant, 8)["pbo"] == 0.0
    # Sin señal, el mejor in-sample no conserva su ventaja: PBO lejos de 0.
    result = probability_of_backtest_overfitting(noise, 8, "sharpe")
    assert result["pbo"] > 0.3
    with pytest.raises(ValueError, match="par"):
        probability_of_backtest_overfitting(noise, 7)


def test_deflated_sharpe_penalizes_more_trials():
    rng = np.random.default_rng(9)
    returns = rng.normal(0.05, 1, 2000)
    trials = rng.normal(0, 0.03, 50)
    single = deflated_sharpe_ratio(returns, trials, 1)
    many = deflated_sharpe_ratio(returns, trials, 50)
    assert many["deflated_sharpe_ratio"] < single["deflated_sharpe_ratio"]
    assert single["expected_max_sharpe_null"] == 0
    identical = np.column_stack([returns, returns, returns])
    assert effective_trials(identical) == pytest.approx(1.0)
    assert effective_trials(rng.normal(size=(5000, 4))) == pytest.approx(4, abs=0.1)


# ─── Harness ─────────────────────────────────────────────────────────────────


def harness_frame(n=4000):
    data = training_frame(n)
    rng = np.random.default_rng(11)
    extra = {
        name
        for names in harness.feature_sets().values()
        for name in names
        if name not in data.columns
    }
    for name in sorted(extra):
        data[name] = rng.normal(size=n)
    data["forward_return"] = np.where(data["target_label"] == 1, 0.01, -0.01) * rng.uniform(
        0.1, 2, n
    )
    return data


def test_monthly_walk_forward_purges_labels_by_horizon():
    data = harness_frame()
    folds = harness.monthly_walk_forward(data["open_time"], 3, horizon=2)
    for fold in folds:
        assert (data["open_time"].iloc[fold.train] + 3 * STEP <= fold.start).all()
        assert data["open_time"].iloc[fold.validation].iloc[0] == fold.start
    assert folds[1].start == folds[0].end


def test_forward_returns_must_match_labels():
    times = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    candles = pd.DataFrame({"open_time": times, "close": [100.0, 101.0, 100.0, 102.0]})
    features = pd.DataFrame({"open_time": times[:3], "target_label": [1, 0, 1]})
    attached = harness.attach_forward_returns(features, candles)
    np.testing.assert_allclose(attached["forward_return"], [0.01, 100 / 101 - 1, 0.02])
    features.loc[0, "target_label"] = 0
    with pytest.raises(ValueError, match="no coinciden"):
        harness.attach_forward_returns(features, candles)


def test_walk_forward_predictions_ignore_validation_labels():
    data = harness_frame()
    folds = harness.monthly_walk_forward(data["open_time"], 2)
    pool = [
        c for c in harness.candidates() if c.name in ("logistic_existing_C0.01", "lightgbm_base")
    ]
    first = harness.run_walk_forward(data, folds, pool)
    changed = data.copy()
    for fold in folds[-1:]:
        changed.loc[fold.validation, "target_label"] = (
            1 - changed.loc[fold.validation, "target_label"]
        )
    second = harness.run_walk_forward(changed, folds, pool)
    last = first["fold"] == folds[-1].number
    for candidate in pool:
        np.testing.assert_allclose(
            first.loc[last, candidate.name], second.loc[last, candidate.name]
        )


def test_build_report_is_serializable_and_contains_gate(monkeypatch):
    data = harness_frame(4500)
    monkeypatch.setattr(harness, "CALIBRATION_MIN_SAMPLES", 200)
    monkeypatch.setattr(harness, "CPCV_GROUPS", 4)
    monkeypatch.setattr(harness, "PBO_PARTITIONS", 4)

    def fake(candidate, train, test):
        rng = np.random.default_rng(len(train) + len(candidate.name))
        signal = test["rsi_14"].to_numpy() * 0.05
        return np.clip(0.5 + signal + rng.normal(0, 0.02, len(test)), 0.01, 0.99)

    report, predictions = harness.build_report(data, n_folds=4, predictor=fake)
    json.dumps(report, allow_nan=False)
    assert report["independent_holdout_evaluated"] is False
    assert report["walk_forward"]["folds"] == 4
    assert set(report["gate"]) >= {
        "accuracy_difference_ci_95",
        "pbo_accuracy",
        "deflated_sharpe_ratio",
    }
    assert report["cpcv"]["paths"] == comb(3, 1)
    assert report["calibration"]["evaluable_samples"] > 0
    assert predictions["open_time"].is_unique
    assert report["models"]["baseline_previous_direction"]["accuracy"] >= 0


def test_append_registry_is_append_only(tmp_path):
    harness.append_registry(tmp_path, {"run_id": "a"})
    harness.append_registry(tmp_path, {"run_id": "b"})
    lines = (tmp_path / "experiments.jsonl").read_text().splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == ["a", "b"]
