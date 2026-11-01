import json

import numpy as np
import pandas as pd
import pytest

from src.features.long_horizon import build_long_horizon_features, price_history
from src.research import volatility_study as e12
from src.volatility import features as vf
from src.volatility.model import (
    VolatilityArtifact,
    fit_artifact,
    horizon_sigma,
    predict_log_variance,
    predict_variance,
    regime,
    standardize,
)
from tests.test_phase2_features import synthetic_inputs


def synthetic_bundle(days=900):
    features, sources, metrics, aux = synthetic_inputs(days=days)
    rng = np.random.default_rng(3)
    features["vix_close"] = rng.uniform(12, 30, len(features))
    features["volume_sma_ratio"] = rng.uniform(0.5, 1.5, len(features))
    features["funding_rate"] = 1e-4
    enriched = build_long_horizon_features(features, sources, metrics, aux)
    close = price_history(sources["candles"], aux["btc_spot_klines"])
    return features, sources, aux, close, vf.model_inputs(enriched, close)


@pytest.fixture(scope="module")
def bundle():
    return synthetic_bundle()


def test_live_columns_exclude_open_interest_and_cover_har_x():
    assert "oi_change_7d" not in vf.HAR_X_LIVE_EXTRA
    assert set(vf.HAR_X_LIVE_EXTRA) | {"oi_change_7d"} == set(vf.HAR_X_EXTRA)


def test_live_inputs_equal_research_inputs_on_full_history(bundle):
    features, sources, aux, close, research = bundle
    live = vf.live_inputs(
        features[["open_time", "vix_close", "volume_sma_ratio"]],
        close,
        sources["funding"],
        aux["deribit_dvol"],
        aux["fomc"],
    )
    columns = [*vf.HAR_COLUMNS, *vf.HAR_X_LIVE_EXTRA, "target_var_7d", "var_ewma"]
    pd.testing.assert_frame_equal(
        live[columns], research[columns], check_exact=False, rtol=0, atol=1e-12
    )


def test_live_inputs_from_trailing_windows_equal_research_inputs(bundle):
    features, sources, aux, close, research = bundle
    ordered = features.sort_values("open_time").reset_index(drop=True)
    for position in (len(ordered) // 2, len(ordered) - 1):
        window = ordered.iloc[position - vf.LIVE_WINDOW_CANDLES + 1 : position + 1]
        last = window["open_time"].iloc[-1]
        decision = last + vf.STEP
        funding = sources["funding"]
        funding = funding[pd.to_datetime(funding["funding_time"], utc=True) <= decision].tail(120)
        dvol = aux["deribit_dvol"]
        stamps = pd.to_datetime(dvol["open_time"], utc=True)
        dvol = dvol[(stamps <= decision) & (stamps > decision - pd.Timedelta(days=30))]
        row = vf.live_inputs(
            window[["open_time", "vix_close", "volume_sma_ratio"]],
            close.loc[window["open_time"].iloc[0] : last],
            funding,
            dvol,
            aux["fomc"],
        ).iloc[-1]
        columns = [*vf.HAR_COLUMNS, *vf.HAR_X_LIVE_EXTRA]
        expected = research.iloc[position][columns].to_numpy(dtype=float)
        assert row[columns].to_numpy(dtype=float) == pytest.approx(expected, abs=1e-9, nan_ok=True)


def test_live_inputs_do_not_use_future_data(bundle):
    features, sources, aux, close, research = bundle
    base = features[["open_time", "vix_close", "volume_sma_ratio"]]
    cut = len(base) - 200
    t = base["open_time"].iloc[cut]
    changed = close.copy()
    changed.loc[changed.index > t] *= 1.7
    funding = sources["funding"].copy()
    later = pd.to_datetime(funding["funding_time"], utc=True) > t
    funding.loc[later, "funding_rate"] = 0.5
    original = vf.live_inputs(base, close, sources["funding"], aux["deribit_dvol"], aux["fomc"])
    shifted = vf.live_inputs(base, changed, funding, aux["deribit_dvol"], aux["fomc"])
    columns = [*vf.HAR_COLUMNS, *vf.HAR_X_LIVE_EXTRA]
    pd.testing.assert_frame_equal(original.loc[:cut, columns], shifted.loc[:cut, columns])


@pytest.fixture(scope="module")
def development(bundle):
    data = bundle[4]
    return e12.development_rows(data, data["open_time"].iloc[-1])


def test_artifact_matches_the_research_linear_model(development):
    split = len(development) - 600
    train, test = development.iloc[: split - 50], development.iloc[split:]
    for kind, name in (("har", "har"), ("har_x_live", "har_x_live")):
        columns = tuple(
            vf.HAR_COLUMNS if kind == "har" else [*vf.HAR_COLUMNS, *vf.HAR_X_LIVE_EXTRA]
        )
        model = e12.VolModel(name, columns, "linear")
        legacy = e12.fit_predict_variance(model, train, test, "7d")
        artifact = fit_artifact(kind, train, "7d")
        assert predict_variance(artifact, test) == pytest.approx(legacy, rel=1e-4)


def test_artifact_roundtrip_validation_and_missing_value_imputation(development):
    artifact = fit_artifact("har_x_live", development, "1d")
    assert artifact.problems() == []
    assert artifact.horizon_candles == 6
    restored = VolatilityArtifact.from_json(artifact.to_json())
    assert restored == artifact
    json.loads(artifact.to_json())
    rows = development.iloc[:5].copy()
    assert predict_variance(restored, rows) == pytest.approx(predict_variance(artifact, rows))
    rows.loc[rows.index[0], "log_dvol_var"] = np.nan
    imputed = rows.copy()
    imputed.loc[imputed.index[0], "log_dvol_var"] = artifact.median[
        artifact.feature_names.index("log_dvol_var")
    ]
    assert predict_log_variance(artifact, rows) == pytest.approx(
        predict_log_variance(artifact, imputed)
    )
    rows.loc[rows.index[0], "log_dvol_var"] = np.inf
    with pytest.raises(ValueError, match="infinitos"):
        standardize(artifact, rows)


def test_artifact_problems_are_reported(development):
    artifact = fit_artifact("har", development, "7d")
    from dataclasses import replace

    broken = replace(
        artifact,
        schema_version=9,
        kind="x",
        horizon_candles=7,
        scale=[0.0, 1.0, 1.0],
        coefficients=[float("nan")],
        residual_variance=-1.0,
        regime_thresholds={"low": 2.0, "high": 1.0, "extreme": 0.5},
    )
    text = " | ".join(broken.problems())
    for fragment in (
        "schema_version",
        "kind desconocido",
        "horizonte",
        "dimensiones",
        "no finitos",
        "escalas",
        "negativa",
        "crecientes",
    ):
        assert fragment in text
    wrong_features = replace(artifact, feature_names=["a", "b", "c"])
    assert "feature_names no corresponde al kind" in wrong_features.problems()


def test_fit_rejects_bad_training_data(development):
    with pytest.raises(ValueError, match="kind desconocido"):
        fit_artifact("lightgbm", development, "7d")
    empty = development.iloc[0:0]
    with pytest.raises(ValueError, match="Train vacío"):
        fit_artifact("har", empty, "7d")


def test_regimes_follow_the_thresholds_and_sigma_scales_with_horizon(development):
    artifact = fit_artifact("har_x_live", development, "7d")
    t = artifact.regime_thresholds
    assert t["low"] < t["high"] < t["extreme"]
    assert [regime(v, t) for v in (t["low"] - 0.01, t["low"], t["high"], t["extreme"])] == [
        "low",
        "normal",
        "high",
        "extreme",
    ]
    var = np.array([0.25])
    assert horizon_sigma(var, 42)[0] == pytest.approx(np.sqrt(0.25 * 42 / 2190))
    assert horizon_sigma(var, 42)[0] == pytest.approx(horizon_sigma(var, 6)[0] * np.sqrt(7))
