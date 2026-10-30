"""Congela una variante y protege una evaluación posterior de uso único."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import get_settings
from src.evaluation.metrics import evaluate, evaluate_baselines
from src.ingestion.run import configure_logging
from src.ingestion.time_utils import floor_to_step
from src.research.dataset import verify_dataset
from src.research.oi_experiment import _fit_variant, variant_features
from src.research.provenance import audit_archive_versions, implementation_signature
from src.research.source_snapshot import file_sha256
from src.training.data import Fold
from src.training.holdouts import (
    assert_unused_holdout,
    frozen_digest,
    read_frozen,
    reservation_lock,
)

STEP = timedelta(hours=4)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Falta el archivo requerido: {path.name}")
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def history_hash(data: pd.DataFrame, cutoff: pd.Timestamp) -> str:
    feature_names = variant_features()["base_oi_position"]
    columns = [
        "open_time",
        "target_label",
        "previous_direction",
        *feature_names,
    ]
    historical = data.loc[
        (data["open_time"] < cutoff)
        & (data["open_time"] + 2 * pd.Timedelta(STEP) <= cutoff)
        & data["target_label"].notna(),
        columns,
    ]
    if (
        historical.empty
        or not historical["open_time"].is_monotonic_increasing
        or historical["open_time"].duplicated().any()
        or (historical["open_time"].diff().dropna() != pd.Timedelta(STEP)).any()
    ):
        raise ValueError("El historial del experimento está vacío o desordenado")
    historical = historical.astype(
        {
            "target_label": "int64",
            "previous_direction": "int64",
            **dict.fromkeys(feature_names, "float64"),
        }
    )
    historical["open_time"] = pd.to_datetime(historical["open_time"], utc=True).astype(
        pd.DatetimeTZDtype(unit="ns", tz="UTC")
    )
    return hashlib.sha256(
        pd.util.hash_pandas_object(historical, index=False).to_numpy().tobytes()
    ).hexdigest()


def freeze_experiment(
    experiment: Path,
    dataset: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    now: datetime | None = None,
    runtime_image: str | None = None,
    runtime_image_id: str | None = None,
) -> dict[str, Any]:
    if (runtime_image is None) != (runtime_image_id is None):
        raise ValueError("La imagen y su identificador deben registrarse juntos")
    if runtime_image is not None and (
        not runtime_image.strip()
        or runtime_image_id is None
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", runtime_image_id)
    ):
        raise ValueError("La imagen o su identificador SHA-256 son inválidos")
    if not experiment.name.startswith("oi-ablation-"):
        raise ValueError("El experimento debe estar en un directorio oi-ablation-*")
    report = _read_json(experiment / "ablation.json")
    manifest = verify_dataset(dataset)
    if report["dataset_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("El experimento no corresponde al dataset proporcionado")
    if file_sha256(experiment / "predictions.parquet") != report["predictions_sha256"]:
        raise ValueError("Las predicciones del experimento fueron modificadas")
    if report.get("implementation") != implementation_signature():
        raise ValueError("La implementación no coincide con la usada en desarrollo")
    if (
        len(report["folds"]) < 10
        or report["independent_holdout_evaluated"]
        or report["model_registered"]
        or report["evaluation"] != "exploratory_development_only"
    ):
        raise ValueError("El experimento no tiene los folds de desarrollo requeridos")
    if (
        start.tzinfo is None
        or start.utcoffset() is None
        or end.tzinfo is None
        or end.utcoffset() is None
        or end <= start
    ):
        raise ValueError("El holdout requiere límites válidos con zona horaria")
    use_current_time = now is None
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None or start <= pd.Timestamp(now):
        raise ValueError("El holdout debe reservarse antes de que comience")
    if end - start < pd.Timedelta(days=28):
        raise ValueError("El holdout debe cubrir al menos 28 días")
    if (
        floor_to_step(start.to_pydatetime(), STEP) != start
        or floor_to_step(end.to_pydatetime(), STEP) != end
    ):
        raise ValueError("Los límites deben estar alineados con velas de 4h")
    cutoff = pd.Timestamp(report["development_end_exclusive"])
    if cutoff > start:
        raise ValueError("El holdout se solapa con el desarrollo")
    if manifest["quality"]["candle_synthetic_rows"]:
        raise ValueError("El dataset de desarrollo contiene velas sintéticas")

    variants = variant_features()
    if set(report["models"]) != set(variants):
        raise ValueError("Las variantes del experimento no coinciden con la implementación")
    for fold in report["folds"]:
        for name, columns in variants.items():
            if fold["tuning"][name]["feature_names"] != columns:
                raise ValueError("Las features registradas no coinciden con la implementación")
    selected = min(
        variants,
        key=lambda name: (
            -report["models"][name]["accuracy"],
            report["models"][name]["log_loss"],
            name,
        ),
    )
    data = pd.read_parquet(dataset / manifest["files"]["features"]["path"])
    archives = _read_json(dataset / "archive-manifest.json")
    provenance = audit_archive_versions(
        archives["files"],
        timedelta(hours=manifest["availability_assumption"]["metrics_delay_hours"]),
    )
    frozen = {
        "status": "frozen",
        "created_at": now.astimezone(UTC).isoformat(),
        "selected_variant": selected,
        "selection_rule": "max development accuracy; tie: lower log loss, then name",
        "development_end_exclusive": cutoff.isoformat(),
        "holdout_start": start.isoformat(),
        "holdout_end_exclusive": end.isoformat(),
        "dataset_id": manifest["dataset_id"],
        "symbol": manifest["symbol"],
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "development_report_sha256": file_sha256(experiment / "ablation.json"),
        "development_predictions_sha256": report["predictions_sha256"],
        "development_history_sha256": history_hash(data, cutoff),
        "implementation": report["implementation"],
        "runtime_image": runtime_image,
        "runtime_image_id": runtime_image_id,
        "dataset_build_implementation_recorded": "implementation" in manifest,
        "availability_assumption": manifest["availability_assumption"],
        "archive_provenance": provenance,
        "point_in_time_verified": (
            manifest["point_in_time_verified"] and provenance["point_in_time_verified"]
        ),
    }
    with reservation_lock(experiment.parent):
        created_at = datetime.now(UTC) if use_current_time else now
        if start <= pd.Timestamp(created_at):
            raise ValueError("El holdout comenzó antes de completar la reserva")
        assert_unused_holdout(experiment.parent, start, end, exclude=experiment)
        frozen["created_at"] = created_at.astimezone(UTC).isoformat()
        frozen["frozen_sha256"] = frozen_digest(frozen)
        with (experiment / "frozen.json").open("x", encoding="utf-8") as file:
            json.dump(frozen, file, indent=2, allow_nan=False)
    return frozen


def evaluate_frozen(
    experiment: Path, future_dataset: Path, now: datetime | None = None
) -> dict[str, Any]:
    frozen = read_frozen(experiment / "frozen.json")
    report = _read_json(experiment / "ablation.json")
    if (
        frozen["status"] != "frozen"
        or file_sha256(experiment / "ablation.json") != frozen["development_report_sha256"]
    ):
        raise ValueError("El reporte de desarrollo no coincide con la selección congelada")
    if file_sha256(experiment / "predictions.parquet") != frozen["development_predictions_sha256"]:
        raise ValueError("Las predicciones de desarrollo cambiaron")
    if implementation_signature() != frozen["implementation"]:
        raise ValueError("La implementación o dependencias cambiaron desde la selección")
    start = pd.Timestamp(frozen["holdout_start"])
    end = pd.Timestamp(frozen["holdout_end_exclusive"])
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None or pd.Timestamp(now) < end + STEP:
        raise ValueError("El holdout aún no ha terminado o sus últimas etiquetas no se conocen")
    assert_unused_holdout(experiment.parent, start, end, exclude=experiment)
    manifest = verify_dataset(future_dataset)
    if manifest.get("implementation") != frozen["implementation"]:
        raise ValueError("La implementación usada para construir el dataset cambió")
    if manifest["symbol"] != frozen["symbol"]:
        raise ValueError("El símbolo del dataset no coincide con la reserva")
    if manifest["availability_assumption"] != frozen["availability_assumption"]:
        raise ValueError("Los supuestos temporales del dataset cambiaron")
    if manifest["quality"]["candle_synthetic_rows"]:
        raise ValueError("El dataset del holdout contiene velas sintéticas")
    archives = _read_json(future_dataset / "archive-manifest.json")
    future_provenance = audit_archive_versions(
        archives["files"],
        timedelta(hours=manifest["availability_assumption"]["metrics_delay_hours"]),
    )
    data = pd.read_parquet(future_dataset / manifest["files"]["features"]["path"])
    cutoff = pd.Timestamp(frozen["development_end_exclusive"])
    if history_hash(data, cutoff) != frozen["development_history_sha256"]:
        raise ValueError("El historial del nuevo dataset difiere del usado en desarrollo")
    times = pd.to_datetime(data["open_time"], utc=True)
    train = np.flatnonzero((times < start) & (times + 2 * pd.Timedelta(STEP) <= start))
    valid = np.flatnonzero((times >= start) & (times < end))
    expected = len(pd.date_range(start, end - pd.Timedelta(STEP), freq=pd.Timedelta(STEP)))
    if not len(train) or len(valid) != expected:
        raise ValueError("El nuevo dataset no cubre todas las velas y etiquetas del holdout")
    if (times.diff().dropna() != pd.Timedelta(STEP)).any():
        raise ValueError("El nuevo dataset contiene discontinuidades temporales")
    selected = frozen["selected_variant"]
    names = variant_features().get(selected)
    if names is None:
        raise ValueError("La variante congelada no existe")
    fold = Fold(1, train, valid, start, end)
    predictions_path = experiment / "holdout_predictions.parquet"
    result_path = experiment / "holdout.json"
    if predictions_path.exists() or result_path.exists():
        raise FileExistsError("Ya existen resultados para este holdout")
    with (experiment / "holdout.lock").open("x", encoding="utf-8") as file:
        file.write(f"Holdout iniciado {now.astimezone(UTC).isoformat()}\n")
    labels = data["target_label"].to_numpy(dtype=int)
    previous = data["previous_direction"].to_numpy(dtype=int)
    with threadpool_limits(limits=2):
        probability, tuning = _fit_variant(data, fold, names, STEP)
    predictions = (probability >= 0.5).astype(int)
    metrics = evaluate(labels[valid], predictions, probability)
    baselines = evaluate_baselines(
        labels[train], previous[train], labels[valid], previous[valid], 43
    )
    best_baseline = max(item["accuracy"] for item in baselines.values())
    criteria = {
        "development_above_oracle_plus_3pp": (
            report["models"][selected]["accuracy"] > report["best_baseline_per_fold_mean"] + 0.03
        ),
        "holdout_above_best_baseline_plus_2pp": metrics["accuracy"] > best_baseline + 0.02,
        "predicts_both_directions": np.unique(predictions).size == 2,
        "point_in_time_verified": frozen["point_in_time_verified"]
        and manifest["point_in_time_verified"]
        and future_provenance["point_in_time_verified"],
    }
    criteria["passed"] = all(criteria.values())
    result = {
        "selected_variant": selected,
        "holdout_start": start.isoformat(),
        "holdout_end_exclusive": end.isoformat(),
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "metrics": metrics,
        "baselines": baselines,
        "best_baseline_accuracy": best_baseline,
        "tuning": tuning,
        "archive_provenance": future_provenance,
        "criteria": criteria,
        "model_registered": False,
        "active": False,
    }
    temporary_predictions = experiment / ".holdout_predictions.parquet.part"
    pd.DataFrame(
        {
            "open_time": data["open_time"].iloc[valid].to_numpy(),
            "target_label": labels[valid],
            "probability_up": probability,
        }
    ).to_parquet(temporary_predictions, index=False)
    temporary_predictions.replace(predictions_path)
    result["predictions_sha256"] = file_sha256(predictions_path)
    temporary_result = experiment / ".holdout.json.part"
    temporary_result.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary_result.replace(result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reserva y evalúa un holdout de OI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--experiment", type=Path, required=True)
    freeze.add_argument("--dataset", type=Path, required=True)
    freeze.add_argument("--holdout-start", required=True)
    freeze.add_argument("--holdout-end", required=True)
    freeze.add_argument("--runtime-image")
    freeze.add_argument("--runtime-image-id")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--experiment", type=Path, required=True)
    evaluate_parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    if args.command == "freeze":
        result = freeze_experiment(
            args.experiment,
            args.dataset,
            pd.Timestamp(args.holdout_start),
            pd.Timestamp(args.holdout_end),
            runtime_image=args.runtime_image,
            runtime_image_id=args.runtime_image_id,
        )
    else:
        result = evaluate_frozen(args.experiment, args.dataset)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
