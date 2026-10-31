import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from src.features.funding_context import FUNDING_FEATURE_COLUMNS
from src.features.price_context import PRICE_CONTEXT_COLUMNS
from src.research import oi_experiment, oi_holdout
from src.research.metrics_features import OI_FEATURE_COLUMNS, POSITION_FEATURE_COLUMNS
from src.research.oi_experiment import variant_features
from src.research.source_snapshot import file_sha256
from src.training import run as training_run
from src.training import taker_ablation
from src.training.holdouts import assert_no_reserved_training_data, assert_unused_holdout
from tests.test_training import training_frame


@pytest.fixture
def frozen_setup(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    experiment = root / "oi-ablation-test"
    experiment.mkdir()
    dataset = root / "dataset"
    dataset.mkdir()
    (dataset / "archive-manifest.json").write_text(json.dumps({"files": []}))
    (experiment / "predictions.parquet").write_bytes(b"fixed-predictions")
    signature = {"python": "test", "files": {}, "packages": {}}
    variants = variant_features()
    models = {
        "base": {"accuracy": 0.52, "log_loss": 0.69},
        "base_oi": {"accuracy": 0.54, "log_loss": 0.68},
        "base_oi_position": {"accuracy": 0.53, "log_loss": 0.69},
    }
    report = {
        "dataset_manifest_sha256": "manifest",
        "predictions_sha256": file_sha256(experiment / "predictions.parquet"),
        "implementation": signature,
        "independent_holdout_evaluated": False,
        "model_registered": False,
        "evaluation": "exploratory_development_only",
        "development_end_exclusive": "2025-08-01T00:00:00+00:00",
        "best_baseline_per_fold_mean": 0.52,
        "models": models,
        "folds": [{"tuning": {name: {"feature_names": cols} for name, cols in variants.items()}}]
        * 10,
    }
    (experiment / "ablation.json").write_text(json.dumps(report))
    data = training_frame(9000)
    for column in (
        PRICE_CONTEXT_COLUMNS
        + FUNDING_FEATURE_COLUMNS
        + OI_FEATURE_COLUMNS
        + POSITION_FEATURE_COLUMNS
    ):
        data[column] = 0.1
    manifest = {
        "dataset_id": "test",
        "symbol": "BTCUSDT",
        "manifest_sha256": "manifest",
        "files": {"features": {"path": "features.parquet"}},
        "quality": {"candle_synthetic_rows": 0},
        "point_in_time_verified": False,
        "implementation": signature,
        "availability_assumption": {"metrics_delay_hours": 48},
    }
    monkeypatch.setattr(oi_holdout, "verify_dataset", lambda path: manifest)
    monkeypatch.setattr(oi_holdout, "implementation_signature", lambda: signature)
    monkeypatch.setattr(oi_holdout.pd, "read_parquet", lambda path: data.copy())
    return experiment, dataset, data, monkeypatch


def test_freeze_and_single_holdout_evaluation_remain_inactive(frozen_setup):
    experiment, dataset, data, monkeypatch = frozen_setup
    start = pd.Timestamp("2025-09-01T00:00:00Z")
    end = pd.Timestamp("2025-10-01T00:00:00Z")
    frozen = oi_holdout.freeze_experiment(
        experiment, dataset, start, end, datetime(2025, 7, 1, tzinfo=UTC)
    )
    assert frozen["selected_variant"] == "base_oi"
    assert frozen["point_in_time_verified"] is False
    with pytest.raises(ValueError, match="aún no ha terminado"):
        oi_holdout.evaluate_frozen(experiment, dataset, datetime(2025, 9, 2, tzinfo=UTC))
    assert not (experiment / "holdout.lock").exists()

    def fake_fit(frame, fold, names, step):
        assert (frame.iloc[fold.train]["open_time"] + 2 * step <= start).all()
        return np.full(len(fold.validation), 0.55), {"C": 0.01}

    monkeypatch.setattr(oi_holdout, "_fit_variant", fake_fit)
    result = oi_holdout.evaluate_frozen(
        experiment, dataset, (end + timedelta(hours=4)).to_pydatetime()
    )
    assert result["model_registered"] is False and result["active"] is False
    assert result["criteria"]["point_in_time_verified"] is False
    assert result["criteria"]["passed"] is False
    assert_no_reserved_training_data(experiment.parent, pd.Series([start]), timedelta(hours=4))
    with pytest.raises(FileExistsError):
        oi_holdout.evaluate_frozen(experiment, dataset, (end + timedelta(hours=4)).to_pydatetime())


def test_freeze_requires_future_nonoverlapping_period(frozen_setup):
    experiment, dataset, data, monkeypatch = frozen_setup
    start = pd.Timestamp("2025-09-01T00:00:00Z")
    end = pd.Timestamp("2025-10-01T00:00:00Z")
    with pytest.raises(ValueError, match="antes de que comience"):
        oi_holdout.freeze_experiment(
            experiment, dataset, start, end, datetime(2025, 9, 2, tzinfo=UTC)
        )
    prior = experiment.parent / "research-past"
    prior.mkdir()
    (prior / "development.json").write_text(
        json.dumps(
            {
                "holdout_start": "2025-08-15T00:00:00Z",
                "holdout_end_exclusive": "2025-09-15T00:00:00Z",
            }
        )
    )
    (prior / "holdout.lock").write_text("evaluated")
    with pytest.raises(ValueError, match="solapa"):
        oi_holdout.freeze_experiment(
            experiment, dataset, start, end, datetime(2025, 7, 1, tzinfo=UTC)
        )


def test_future_dataset_with_revised_history_is_rejected(frozen_setup):
    experiment, dataset, data, monkeypatch = frozen_setup
    start = pd.Timestamp("2025-09-01T00:00:00Z")
    end = pd.Timestamp("2025-10-01T00:00:00Z")
    oi_holdout.freeze_experiment(experiment, dataset, start, end, datetime(2025, 7, 1, tzinfo=UTC))
    data.loc[10, "rsi_14"] += 1
    with pytest.raises(ValueError, match="historial.*difiere"):
        oi_holdout.evaluate_frozen(experiment, dataset, (end + timedelta(hours=4)).to_pydatetime())
    assert not (experiment / "holdout.lock").exists()


def test_history_hash_matches_numeric_values_used_by_the_model(frozen_setup):
    _, _, data, _ = frozen_setup
    cutoff = pd.Timestamp("2025-08-01T00:00:00Z")
    integer_history = data.copy()
    integer_history["fear_greed_value"] = 50
    float_history = integer_history.copy()
    float_history["fear_greed_value"] = float_history["fear_greed_value"].astype(float)
    assert oi_holdout.history_hash(integer_history, cutoff) == oi_holdout.history_hash(
        float_history, cutoff
    )
    float_history.loc[10, "fear_greed_value"] += 1
    assert oi_holdout.history_hash(integer_history, cutoff) != oi_holdout.history_hash(
        float_history, cutoff
    )


def test_frozen_holdout_blocks_legacy_selection_and_training(frozen_setup):
    experiment, dataset, _, _ = frozen_setup
    start = pd.Timestamp("2025-09-01T00:00:00Z")
    end = pd.Timestamp("2025-10-01T00:00:00Z")
    oi_holdout.freeze_experiment(experiment, dataset, start, end, datetime(2025, 7, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="reserva existente"):
        assert_unused_holdout(experiment.parent, start, end)
    assert_unused_holdout(experiment.parent, end, end + timedelta(days=30))
    assert_no_reserved_training_data(
        experiment.parent, pd.Series([start - timedelta(hours=8)]), timedelta(hours=4)
    )
    with pytest.raises(ValueError, match="aún reservado"):
        assert_no_reserved_training_data(
            experiment.parent, pd.Series([start - timedelta(hours=4)]), timedelta(hours=4)
        )
    with pytest.raises(ValueError, match="aún reservado"):
        assert_no_reserved_training_data(experiment.parent, pd.Series([start]), timedelta(hours=4))


def test_concurrent_freezes_cannot_reserve_the_same_period(frozen_setup):
    experiment, dataset, _, _ = frozen_setup
    other = experiment.parent / "oi-ablation-other"
    shutil.copytree(experiment, other)
    start = pd.Timestamp("2025-09-01T00:00:00Z")
    end = pd.Timestamp("2025-10-01T00:00:00Z")

    def reserve(path):
        try:
            oi_holdout.freeze_experiment(
                path, dataset, start, end, datetime(2025, 7, 1, tzinfo=UTC)
            )
            return "reserved"
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, (experiment, other)))
    assert results.count("reserved") == 1
    assert sum("reserva existente" in item for item in results) == 1


def test_evaluation_rejects_changed_implementation_and_assumptions(frozen_setup):
    experiment, dataset, _, monkeypatch = frozen_setup
    start = pd.Timestamp("2025-09-01T00:00:00Z")
    end = pd.Timestamp("2025-10-01T00:00:00Z")
    oi_holdout.freeze_experiment(experiment, dataset, start, end, datetime(2025, 7, 1, tzinfo=UTC))
    now = (end + timedelta(hours=4)).to_pydatetime()
    signature = oi_holdout.implementation_signature()
    monkeypatch.setattr(oi_holdout, "implementation_signature", lambda: {"changed": True})
    with pytest.raises(ValueError, match="implementación"):
        oi_holdout.evaluate_frozen(experiment, dataset, now)
    monkeypatch.setattr(oi_holdout, "implementation_signature", lambda: signature)
    manifest = oi_holdout.verify_dataset(dataset)
    manifest["implementation"] = {"changed": True}
    with pytest.raises(ValueError, match="construir el dataset cambió"):
        oi_holdout.evaluate_frozen(experiment, dataset, now)
    manifest["implementation"] = signature
    manifest["availability_assumption"] = {"metrics_delay_hours": 0}
    with pytest.raises(ValueError, match="supuestos temporales"):
        oi_holdout.evaluate_frozen(experiment, dataset, now)
    assert not (experiment / "holdout.lock").exists()


def test_image_identity_must_be_recorded_together(frozen_setup):
    experiment, dataset, _, _ = frozen_setup
    start = pd.Timestamp("2025-09-01T00:00:00Z")
    end = pd.Timestamp("2025-10-01T00:00:00Z")
    with pytest.raises(ValueError, match="registrarse juntos"):
        oi_holdout.freeze_experiment(
            experiment, dataset, start, end, datetime(2025, 7, 1, tzinfo=UTC), runtime_image="tag"
        )
    with pytest.raises(ValueError, match="SHA-256"):
        oi_holdout.freeze_experiment(
            experiment,
            dataset,
            start,
            end,
            datetime(2025, 7, 1, tzinfo=UTC),
            runtime_image="tag",
            runtime_image_id="invalid",
        )
    assert not (experiment / "frozen.json").exists()


def test_modified_reservation_is_rejected(frozen_setup):
    experiment, dataset, _, _ = frozen_setup
    start = pd.Timestamp("2025-09-01T00:00:00Z")
    end = pd.Timestamp("2025-10-01T00:00:00Z")
    frozen = oi_holdout.freeze_experiment(
        experiment, dataset, start, end, datetime(2025, 7, 1, tzinfo=UTC)
    )
    frozen["selected_variant"] = "base"
    (experiment / "frozen.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="reserva fue modificada"):
        assert_unused_holdout(experiment.parent, start, end)


def test_real_nested_fitting_on_the_reserved_holdout(frozen_setup):
    experiment, dataset, _, _ = frozen_setup
    start = pd.Timestamp("2025-09-01T00:00:00Z")
    end = pd.Timestamp("2025-10-01T00:00:00Z")
    oi_holdout.freeze_experiment(experiment, dataset, start, end, datetime(2025, 7, 1, tzinfo=UTC))
    result = oi_holdout.evaluate_frozen(
        experiment, dataset, (end + timedelta(hours=4)).to_pydatetime()
    )
    assert result["metrics"]["samples"] == 180
    assert result["tuning"]["C"] in oi_experiment.REGULARIZATION_GRID
    assert set(result["tuning"]["inner_log_loss"]) == {"0.003", "0.01", "0.03", "0.1"}
    assert result["criteria"]["point_in_time_verified"] is False
    assert result["model_registered"] is False


def test_oi_ablation_cannot_consume_reserved_period(frozen_setup, monkeypatch):
    experiment, dataset, _, _ = frozen_setup
    start = pd.Timestamp("2025-09-01T00:00:00Z")
    end = pd.Timestamp("2025-10-01T00:00:00Z")
    oi_holdout.freeze_experiment(experiment, dataset, start, end, datetime(2025, 7, 1, tzinfo=UTC))
    monkeypatch.setattr(oi_experiment, "verify_dataset", oi_holdout.verify_dataset)
    monkeypatch.setattr(
        oi_experiment, "implementation_signature", oi_holdout.implementation_signature
    )
    monkeypatch.setattr(oi_experiment, "run_ablation", lambda *args: pytest.fail("No debe ajustar"))
    with pytest.raises(ValueError, match="aún reservado"):
        oi_experiment.execute(dataset, end, experiment.parent)


async def test_training_entrypoints_cannot_consume_reserved_period(frozen_setup, monkeypatch):
    experiment, dataset, data, _ = frozen_setup
    start = pd.Timestamp("2025-09-01T00:00:00Z")
    end = pd.Timestamp("2025-10-01T00:00:00Z")
    oi_holdout.freeze_experiment(experiment, dataset, start, end, datetime(2025, 7, 1, tzinfo=UTC))
    settings = SimpleNamespace(
        database_url="postgresql+asyncpg://example.invalid/test",
        default_interval="4h",
        btc_symbol="BTCUSDT",
        model_artifact_dir=str(experiment.parent),
    )
    engine = SimpleNamespace(dispose=AsyncMock())

    async def load_data(*args):
        return data

    monkeypatch.setattr(training_run, "get_settings", lambda: settings)
    monkeypatch.setattr(training_run, "create_async_engine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(training_run, "load_training_data", load_data)
    monkeypatch.setattr(
        training_run, "run_walk_forward", lambda *args: pytest.fail("No debe ajustar")
    )
    with pytest.raises(ValueError, match="aún reservado"):
        await training_run.execute(experiment.parent)
    engine.dispose.assert_awaited_once()

    engine.dispose.reset_mock()
    monkeypatch.setattr(taker_ablation, "get_settings", lambda: settings)
    monkeypatch.setattr(taker_ablation, "create_async_engine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(taker_ablation, "load_training_data", load_data)
    monkeypatch.setattr(
        taker_ablation, "IngestionRepository", lambda *args: SimpleNamespace(load_candles=load_data)
    )
    monkeypatch.setattr(taker_ablation, "add_price_context", lambda frame, *args: frame)
    monkeypatch.setattr(taker_ablation, "add_taker_flow", lambda frame, *args: frame)
    monkeypatch.setattr(
        taker_ablation, "run_ablation", lambda *args: pytest.fail("No debe ajustar")
    )
    with pytest.raises(ValueError, match="aún reservado"):
        await taker_ablation.execute(end, experiment.parent)
    engine.dispose.assert_awaited_once()
