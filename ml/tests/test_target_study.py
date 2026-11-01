import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.economic import overlapping_positions
from src.evaluation.selective import causal_margin_selection
from src.research import target_study as study
from tests.test_evaluation_phase0 import harness_frame


def test_overlapping_positions_average_last_h_signals():
    signals = np.array([1.0, 1.0, -1.0, 0.0, 1.0])
    np.testing.assert_allclose(overlapping_positions(signals, 1), signals)
    np.testing.assert_allclose(overlapping_positions(signals, 2), [0.5, 1.0, 0.0, -0.5, 0.5])
    with pytest.raises(ValueError):
        overlapping_positions(signals, 0)


def test_causal_margin_selection_ignores_current_fold_and_unknown_return():
    folds = np.repeat([1, 2, 3], 50)
    rng = np.random.default_rng(3)
    p = rng.uniform(0.3, 0.7, len(folds))
    r = rng.normal(0, 0.01, len(folds))
    margins, choices = causal_margin_selection(folds, p, r, 2, (0.0, 0.1), 7.0, 40)
    assert choices[1] is None and np.isnan(margins[folds == 1]).all()
    assert choices[2] in (0.0, 0.1) and choices[3] in (0.0, 0.1)
    changed = r.copy()
    changed[49:] = rng.normal(0, 0.05, len(changed) - 49)
    _, other = causal_margin_selection(folds, p, changed, 2, (0.0, 0.1), 7.0, 40)
    assert other[2] == choices[2]
    with pytest.raises(ValueError, match="contiguo"):
        causal_margin_selection(np.array([1, 2, 1]), p[:3], r[:3], 1, (0.0,), 0.0, 2)


def _study_frame(n=4200):
    data = harness_frame(n)
    rng = np.random.default_rng(21)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n + 30)))
    candles = pd.DataFrame(
        {
            "open_time": pd.date_range(
                data["open_time"].iloc[0] - 10 * study.STEP, periods=n + 30, freq="4h"
            ),
            "close": close,
        }
    )
    aligned = candles.set_index("open_time")["close"]
    data["target_label"] = (
        aligned.shift(-1).reindex(data["open_time"]).to_numpy()
        > aligned.reindex(data["open_time"]).to_numpy()
    ).astype(int)
    data["previous_direction"] = (
        aligned.reindex(data["open_time"]).to_numpy()
        > aligned.shift(1).reindex(data["open_time"]).to_numpy()
    ).astype(int)
    return data.drop(columns=["forward_return"]), candles


def test_attach_horizons_and_development_rows_are_causal():
    features, candles = _study_frame()
    data = study.attach_horizons(features, candles)
    row = data.iloc[100]
    close = candles.set_index("open_time")["close"]
    t = row["open_time"]
    assert row["forward_return_6"] == pytest.approx(close[t + 6 * study.STEP] / close[t] - 1)
    assert row["past_return_3"] == pytest.approx(close[t] / close[t - 3 * study.STEP] - 1)
    cutoff = data["open_time"].iloc[-1] - 5 * study.STEP
    development = study.development_rows(data, cutoff)
    longest = max(study.HORIZONS)
    assert (development["open_time"] + (longest + 1) * study.STEP <= cutoff).all()
    broken = features.copy()
    broken.loc[5, "target_label"] = 1 - broken.loc[5, "target_label"]
    with pytest.raises(ValueError, match="no coinciden"):
        study.attach_horizons(broken, candles)


def test_dead_zone_excludes_small_moves_relative_to_volatility():
    frame = pd.DataFrame({"sigma": [0.01, 0.01, 0.02], "forward_return_4": [0.004, 0.006, 0.01]})
    np.testing.assert_array_equal(study.dead_zone_mask(frame, 4, 0.25), [False, True, True])
    assert study.dead_zone_mask(frame, 4, 0.0).all()


def test_walk_forward_ignores_validation_labels_and_purges_by_horizon(monkeypatch):
    features, candles = _study_frame()
    data = study.attach_horizons(features, candles)
    development = study.development_rows(data, data["open_time"].iloc[-1])
    pool = [c for c in study.configs() if c.model == "logistic_base_C0.01"]
    seen = []

    def spy(candidate, train, test):
        seen.append((train["open_time"].max(), test["open_time"].min()))
        return np.full(len(test), 0.5) + 0.01 * np.tanh(train["target_label"].mean())

    first = study.run_walk_forward(development, 2, pool, spy)
    # Orden: horizonte → fold → zona muerta (una config logística por zona).
    per_horizon = 2 * len(study.DEAD_ZONES)
    for index, (train_end, test_start) in enumerate(seen):
        horizon = study.HORIZONS[index // per_horizon]
        assert train_end + (horizon + 1) * study.STEP <= test_start
    changed = development.copy()
    validation = development["open_time"] >= first["open_time"].iloc[0]
    for h in study.HORIZONS:
        changed.loc[validation, f"forward_return_{h}"] *= -1
    second = study.run_walk_forward(changed, 2, pool, spy)
    first_fold = first["fold"] == first["fold"].min()
    for config in pool:
        np.testing.assert_allclose(
            first.loc[first_fold, config.name], second.loc[first_fold, config.name]
        )


def test_build_report_is_serializable_with_gate(monkeypatch):
    features, candles = _study_frame(5200)
    data = study.attach_horizons(features, candles)
    development = study.development_rows(data, data["open_time"].iloc[-1])
    monkeypatch.setattr(study, "MIN_SELECTION_HISTORY", 300)
    monkeypatch.setattr(study, "PBO_PARTITIONS", 4)

    def fake(candidate, train, test):
        rng = np.random.default_rng(len(train))
        return np.clip(0.5 + rng.normal(0, 0.05, len(test)), 0.01, 0.99)

    report, predictions = study.build_report(development, n_folds=4, predictor=fake)
    json.dumps(report, allow_nan=False)
    assert report["independent_holdout_evaluated"] is False
    assert len(report["configs"]) == len(study.configs())
    assert len(report["strategies"]["details"]) == 2 * len(study.configs())
    assert set(report["gate"]["reality_check_p_values"]) == set(study.BASELINES)
    assert isinstance(report["gate"]["passed"], bool)
    assert predictions["open_time"].is_unique


def test_strategy_returns_charge_funding_to_longs_and_pay_shorts():
    from src.evaluation.economic import strategy_returns

    pos = np.array([1.0, -1.0, 0.0])
    net = strategy_returns(pos, np.zeros(3), 0.0, np.full(3, 0.0001))
    np.testing.assert_allclose(net, [-0.0001, 0.0001, 0.0])


def test_simultaneous_lower_bounds_are_wider_than_single_and_detect_signal():
    from src.evaluation.robustness import block_bootstrap_mean, simultaneous_lower_bounds

    rng = np.random.default_rng(4)
    matrix = rng.normal(0, 1, (2000, 10))
    matrix[:, 3] += 0.3
    joint = simultaneous_lower_bounds(matrix, 20, draws=400)
    single = block_bootstrap_mean(matrix[:, 3], 20, draws=400)
    assert joint["lower_bounds"][3] > 0
    assert joint["lower_bounds"][3] < single["ci_95"][0]
    assert (np.delete(joint["lower_bounds"], 3) < 0).all()


def test_maximin_pbo_uses_worst_criterion():
    from src.evaluation.robustness import maximin_pbo

    rng = np.random.default_rng(8)
    stack = rng.normal(0, 1, (2, 1600, 6))
    stack[:, :, 2] += 1.0
    assert maximin_pbo(stack, 8)["pbo"] == 0.0
    # Domina en un criterio pero pierde en el otro: el maximin no debe preferirla.
    stack[0, :, 4] += 3.0
    stack[1, :, 4] -= 3.0
    assert maximin_pbo(stack, 8)["pbo"] == 0.0
