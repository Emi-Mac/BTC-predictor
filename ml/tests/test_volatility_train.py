import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.research import volatility_study as e12
from src.volatility import study, train
from src.volatility.calibration import MIN_RESOLVED, WINDOW
from src.volatility.features import HORIZONS, STEP
from src.volatility.model import VolatilityArtifact
from tests.test_volatility_live_model import synthetic_bundle


@pytest.fixture(scope="module")
def bundle():
    features, sources, aux, close, data = synthetic_bundle()
    return SimpleNamespace(
        features=features, sources=sources, aux=aux, close=close, data=data, metrics=None
    )


@pytest.fixture(scope="module")
def development(bundle):
    return e12.development_rows(bundle.data, bundle.data["open_time"].iloc[-1])


@pytest.mark.parametrize("candles", [6, 42])
def test_train_builds_primary_and_fallback_with_validation_and_calibration_seed(
    bundle, development, candles, tmp_path
):
    provenance = {"study": "E-12b", "dataset_id": "synthetic"}
    artifacts = train.train_artifacts(bundle, development, candles, 6, provenance)
    assert set(artifacts) == {"primary", "fallback"}
    assert artifacts["primary"].kind == "har_x_live"
    assert artifacts["fallback"].kind == "har"
    for artifact in artifacts.values():
        assert artifact.problems() == []
        assert artifact.horizon_candles == candles
        assert artifact.trained_through == development["open_time"].max().isoformat()
        assert artifact.validation["study"] == "E-12b"
        assert artifact.validation["qlike"] > 0
        assert artifact.validation["oos_samples"] > 900
        seed = artifact.calibration_seed
        assert len(seed["z"]) == len(seed["issued_at"]) == len(seed["resolved_at"]) > 0
        gap = pd.Timestamp(seed["resolved_at"][0]) - pd.Timestamp(seed["issued_at"][0])
        assert gap == candles * STEP
        assert np.isfinite(seed["z"]).all() and min(seed["z"]) >= 0
        newest = pd.Timestamp(seed["issued_at"][-1])
        assert newest - pd.Timestamp(seed["issued_at"][0]) < WINDOW + STEP
    paths = train.write_artifacts(artifacts, tmp_path, "run")
    assert sorted(path.name for path in paths.values()) == [
        f"h{candles}-fallback-har.json",
        f"h{candles}-primary-har_x_live.json",
    ]
    assert VolatilityArtifact.load(paths["primary"]) == artifacts["primary"]
    json.loads(paths["fallback"].read_text())
    with pytest.raises(FileExistsError):
        train.write_artifacts(artifacts, tmp_path, "run")


def test_horizon_name_rejects_unknown_horizons():
    assert train.horizon_name(6) == "1d"
    assert train.horizon_name(42) == "7d"
    with pytest.raises(ValueError, match="no soportado"):
        train.horizon_name(7)


def test_study_predict_routes_naive_and_legacy_models(development):
    train_rows, test_rows = development.iloc[:2500], development.iloc[2600:2700]
    by_name = {m.name: m for m in study.pool()}
    naive = study.predict(by_name["naive"], train_rows, test_rows, "7d")
    assert naive == pytest.approx(test_rows["var_7d"].to_numpy())
    legacy = study.predict(by_name["har_x"], train_rows, test_rows, "7d")
    live = study.predict(by_name["har_x_live"], train_rows, test_rows, "7d")
    assert np.isfinite(legacy).all() and np.isfinite(live).all()
    assert not np.allclose(legacy, live)


def synthetic_predictions(n=6000, vol=0.5):
    rng = np.random.default_rng(5)
    times = pd.date_range("2022-01-01", periods=n + 60, freq="4h", tz="UTC")
    per_candle = vol / np.sqrt(2190)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, per_candle, len(times)))), index=times)
    frame = pd.DataFrame({"open_time": times[:n], "fold": 1})
    for name in HORIZONS:
        frame[f"har_x_live_{name}"] = vol**2
        frame[f"target_var_{name}"] = vol**2
    return frame, close


def test_calibration_report_reaches_nominal_coverage_with_a_calibrated_forecast():
    predictions, close = synthetic_predictions()
    result = study.calibration_report(predictions, close)
    for horizon in HORIZONS:
        entry = result[horizon]
        assert entry["evaluated"] > 3000
        assert entry["all_levels_contain_nominal"]
        assert entry["conformal"]["inside_95"]["n"] == entry["evaluated"]
        assert set(entry["by_year"]) == {"2022", "2023", "2024"}
    json.dumps(result, allow_nan=False)


def test_calibration_report_without_enough_history_is_not_evaluated():
    predictions, close = synthetic_predictions(n=MIN_RESOLVED // 2)
    result = study.calibration_report(predictions, close)
    assert result["7d"] == {"evaluated": 0, "all_levels_contain_nominal": False}


def test_study_report_gate_is_serializable_and_parity_holds(bundle, development):
    report, predictions = study.build_report(bundle, development, n_folds=6)
    json.dumps(report, allow_nan=False)
    assert report["parity"]["full_history_max_abs_diff"] < 1e-9
    assert report["parity"]["trailing_window_max_abs_diff"] < 1e-9
    assert report["parity"]["forecast_max_relative_diff"] < 1e-6
    assert report["gate"]["criterion_3_inputs_match"]
    assert report["gate"]["criterion_3_forecast_match"]
    assert report["pre_specified"]["excluded"] == ["oi_change_7d"]
    assert {f"{m.name}_{h}" for m in study.pool() for h in HORIZONS} <= set(predictions.columns)
