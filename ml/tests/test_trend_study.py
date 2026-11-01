import json

import numpy as np
import pandas as pd

from src.research import trend_study as trend


def _market(days=700, seed=3):
    rng = np.random.default_rng(seed)
    n = days * 6
    times = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    candles = pd.DataFrame({"open_time": times, "close": close})
    funding_times = pd.date_range(times[0], times[-1], freq="8h")
    funding = pd.DataFrame(
        {"funding_time": funding_times, "funding_rate": rng.normal(1e-4, 5e-5, len(funding_times))}
    )
    return candles, funding


def test_positions_are_causal_and_bounded():
    candles, _ = _market()
    close = candles.set_index("open_time")["close"]
    first = trend.strategy_positions(close)
    cut = close.index[2500]
    changed = close.copy()
    changed[changed.index > cut] *= np.linspace(1.0, 3.0, int((changed.index > cut).sum()))
    second = trend.strategy_positions(changed)
    pd.testing.assert_frame_equal(first.loc[:cut], second.loc[:cut])
    valid = first.dropna()
    flat = [c for c in valid if "long_flat" in c]
    assert (valid[flat] >= 0).all().all()
    assert (valid.abs() <= 1 + 1e-12).all().all()
    assert set(np.unique(valid["tsmom_30d_long_short"])) <= {-1.0, 1.0}


def test_build_report_compares_against_buy_and_hold(monkeypatch):
    candles, funding = _market()
    monkeypatch.setattr(trend, "PBO_PARTITIONS", 4)
    monkeypatch.setattr(
        trend,
        "WINDOWS",
        {
            "oos_48m": ("2022-01-01", "2022-10-01"),
            "evaluable_e08_e09": ("2022-04-01", "2022-10-01"),
        },
    )
    report = trend.build_report(candles, None, funding)
    json.dumps(report, allow_nan=False)
    window = report["windows"]["oos_48m"]
    assert len(window["strategies"]) == 5 * 2 * 2
    assert window["buy_and_hold_spot"]["exposure"] == 1.0
    assert isinstance(report["gate"]["beats_buy_and_hold_returns"], bool)
    ci = report["gate"]["best_sharpe_difference_ci_95"]
    assert ci[0] <= ci[1]
