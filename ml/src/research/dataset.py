"""Finalización y verificación de datasets de investigación inmutables."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.config import get_settings
from src.features.compute import FEATURE_COLUMNS, compute_features
from src.features.funding_context import add_funding_context
from src.features.price_context import add_price_context
from src.features.taker_flow import add_taker_flow
from src.ingestion.run import configure_logging
from src.research.metrics_archive import METRICS_COLUMNS, parse_daily_archive
from src.research.metrics_features import (
    OI_FEATURE_COLUMNS,
    POSITION_FEATURE_COLUMNS,
    add_metrics_context,
)
from src.research.source_snapshot import SOURCE_FILES, file_sha256, load_spec

log = structlog.get_logger(__name__)


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Falta el manifiesto requerido: {path.name}")
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _safe_file(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("El manifiesto contiene una ruta fuera del dataset")
    return path


def _manifest_digest(manifest: dict[str, Any]) -> str:
    content = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def finalize_dataset(directory: Path) -> dict[str, Any]:
    from src.research.provenance import implementation_signature

    spec = load_spec(directory)
    implementation = implementation_signature()
    archives = _read_manifest(directory / "archive-manifest.json")
    sources = _read_manifest(directory / "source-manifest.json")
    start, end = date.fromisoformat(spec["start"]), date.fromisoformat(spec["end_exclusive"])
    expected_days = (end - start).days
    expected_dates = [(start + timedelta(days=index)).isoformat() for index in range(expected_days)]
    if [item["date"] for item in archives["files"]] != expected_dates:
        raise ValueError("El manifiesto de archivos no cubre exactamente el rango solicitado")
    if set(sources["files"]) != set(SOURCE_FILES):
        raise ValueError("El manifiesto de fuentes está incompleto")
    for name in SOURCE_FILES:
        item = sources["files"][name]
        path = _safe_file(directory, item["path"])
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise ValueError(f"La fuente {name} no coincide con su huella")

    frames = []
    audits = []
    for item in archives["files"]:
        path = _safe_file(directory, f"archives/{item['filename']}")
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise ValueError(f"El archivo {item['filename']} no coincide con su checksum")
        frame, audit = parse_daily_archive(
            path.read_bytes(), spec["symbol"], date.fromisoformat(item["date"])
        )
        for key in (
            "raw_rows",
            "unique_rows",
            "duplicates_removed",
            "missing_intervals",
            "null_values",
        ):
            if audit[key] != item[key]:
                raise ValueError(f"La auditoría del archivo {item['filename']} cambió")
        frames.append(frame)
        audits.append(audit)
    metrics = pd.concat(frames, ignore_index=True)
    if (
        metrics["create_time"].duplicated().any()
        or not metrics["create_time"].is_monotonic_increasing
    ):
        raise ValueError("Las métricas consolidadas tienen timestamps duplicados o desordenados")
    source_frames = {
        name: pd.read_parquet(_safe_file(directory, sources["files"][name]["path"]))
        for name in SOURCE_FILES
    }
    candles = source_frames["candles"].copy()
    candles["id"] = np.arange(1, len(candles) + 1)
    step = timedelta(hours=4)
    features = compute_features(candles, source_frames["macro"], source_frames["sentiment"], step)
    features = add_price_context(features, candles, step)
    features = add_taker_flow(features, candles, step)
    features = add_funding_context(features, source_frames["funding"], step, timedelta(hours=36))
    features = add_metrics_context(
        features,
        metrics,
        step,
        timedelta(hours=spec["assumed_availability_delay_hours"]),
        timedelta(minutes=10),
    )
    close = candles["close"]
    previous = candles[["open_time"]].copy()
    previous["previous_direction"] = (
        (close > close.shift(1)).astype("Int64").mask(close.shift(1).isna())
    )
    features = features.merge(previous, on="open_time", validate="one_to_one")
    cutoff = pd.Timestamp(datetime.combine(end, datetime.min.time(), tzinfo=UTC))
    features = features.iloc[199:].copy()
    features = features.loc[
        (features["open_time"] + 2 * pd.Timedelta(step) <= cutoff)
        & features["target_label"].notna()
        & features["previous_direction"].notna()
    ].reset_index(drop=True)
    if features.empty:
        raise ValueError("El dataset no contiene filas utilizables")
    if (features["open_time"].diff().dropna() != pd.Timedelta(step)).any():
        raise ValueError("Las filas del dataset no tienen continuidad temporal")
    present = features["oi_available_at"].notna()
    if (
        not present.any()
        or (
            features.loc[present, "oi_available_at"]
            > features.loc[present, "open_time"] + pd.Timedelta(step)
        ).any()
    ):
        raise ValueError("Las métricas de futuros no cumplen la alineación temporal")
    if not np.isfinite(
        features[FEATURE_COLUMNS].to_numpy(dtype=float)[
            ~np.isnan(features[FEATURE_COLUMNS].to_numpy(dtype=float))
        ]
    ).all():
        raise ValueError("El dataset contiene features base infinitas")
    if implementation_signature() != implementation:
        raise RuntimeError("La implementación cambió durante la construcción del dataset")

    files = {}
    for name, frame in (("metrics", metrics), ("features", features)):
        target = directory / f"{name}.parquet"
        temporary = directory / f".{name}.parquet.part"
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
        files[name] = {"path": target.name, "sha256": file_sha256(target), "rows": len(frame)}
    manifest = {
        "version": 1,
        "dataset_id": spec["dataset_id"],
        "symbol": spec["symbol"],
        "start": spec["start"],
        "end_exclusive": spec["end_exclusive"],
        "finalized_at": datetime.now(UTC).isoformat(),
        "implementation": implementation,
        "point_in_time_verified": False,
        "availability_assumption": {
            "metrics_timestamp_zone": "UTC inferred from a recent API comparison",
            "metrics_interval_end_minutes": 5,
            "metrics_delay_hours": spec["assumed_availability_delay_hours"],
            "metrics_max_staleness_minutes": 10,
            "funding_delay": "one completed 4h candle",
            "archive_revisions": "historical publication times and past revisions are unknown",
        },
        "quality": {
            "archive_days": len(audits),
            "days_with_gaps": sum(item["missing_intervals"] > 0 for item in audits),
            "missing_5m_intervals": sum(item["missing_intervals"] for item in audits),
            "duplicates_removed": sum(item["duplicates_removed"] for item in audits),
            "archive_null_values": {
                name: sum(item["null_values"][name] for item in audits) for name in METRICS_COLUMNS
            },
            "feature_rows_with_missing_oi": int(
                features[OI_FEATURE_COLUMNS].isna().any(axis=1).sum()
            ),
            "feature_rows_with_missing_positioning": int(
                features[POSITION_FEATURE_COLUMNS].isna().any(axis=1).sum()
            ),
            "feature_rows_with_missing_base": int(
                features[FEATURE_COLUMNS].isna().any(axis=1).sum()
            ),
            "candle_synthetic_rows": sources["candle_synthetic_rows"],
        },
        "source_manifest_sha256": file_sha256(directory / "source-manifest.json"),
        "archive_manifest_sha256": file_sha256(directory / "archive-manifest.json"),
        "spec_sha256": file_sha256(directory / "build-spec.json"),
        "files": files,
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    temporary = directory / ".manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "manifest.json")
    log.info("research_dataset_finalized", dataset=spec["dataset_id"], quality=manifest["quality"])
    return manifest


def verify_dataset(directory: Path, deep: bool = True) -> dict[str, Any]:
    manifest = _read_manifest(directory / "manifest.json")
    expected = manifest.pop("manifest_sha256")
    if _manifest_digest(manifest) != expected:
        raise ValueError("La huella del manifiesto del dataset no coincide")
    manifest["manifest_sha256"] = expected
    for filename, key in (
        ("build-spec.json", "spec_sha256"),
        ("source-manifest.json", "source_manifest_sha256"),
        ("archive-manifest.json", "archive_manifest_sha256"),
    ):
        if file_sha256(directory / filename) != manifest[key]:
            raise ValueError(f"La huella de {filename} no coincide")
    sources = _read_manifest(directory / "source-manifest.json")
    for item in [*sources["files"].values(), *manifest["files"].values()]:
        path = _safe_file(directory, item["path"])
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise ValueError(f"La huella del archivo {path.name} no coincide")
    if deep:
        archives = _read_manifest(directory / "archive-manifest.json")
        for item in archives["files"]:
            path = _safe_file(directory, f"archives/{item['filename']}")
            if not path.is_file() or file_sha256(path) != item["sha256"]:
                raise ValueError(f"La huella del archivo {path.name} no coincide")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Finaliza o verifica un dataset de investigación")
    parser.add_argument("command", choices=["finalize", "verify"])
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    if args.command == "finalize":
        finalize_dataset(args.directory)
    else:
        manifest = verify_dataset(args.directory)
        log.info("research_dataset_verified", dataset=manifest["dataset_id"])


if __name__ == "__main__":
    main()
