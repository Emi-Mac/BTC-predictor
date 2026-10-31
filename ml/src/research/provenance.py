"""Audita fechas de modificación sin afirmar disponibilidad histórica verificable."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import UTC, date, datetime, time, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from src.research.dataset import verify_dataset
from src.research.source_snapshot import file_sha256

IMPLEMENTATION_FILES = (
    "config.py",
    "evaluation/metrics.py",
    "features/compute.py",
    "features/funding_context.py",
    "features/labels.py",
    "features/price_context.py",
    "features/taker_flow.py",
    "ingestion/alignment.py",
    "ingestion/binance.py",
    "ingestion/cleaning.py",
    "ingestion/funding_client.py",
    "ingestion/market_data.py",
    "ingestion/time_utils.py",
    "research/dataset.py",
    "research/metrics_archive.py",
    "research/metrics_features.py",
    "research/oi_experiment.py",
    "research/oi_holdout.py",
    "research/provenance.py",
    "research/seed.py",
    "research/source_snapshot.py",
    "training/data.py",
    "training/holdouts.py",
    "training/research_models.py",
    "training/preprocess.py",
)


def implementation_signature() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "joblib",
                "threadpoolctl",
                "pyarrow",
            )
        },
        "files": {name: file_sha256(root / name) for name in IMPLEMENTATION_FILES},
    }


def audit_archive_versions(files: list[dict[str, Any]], delay: timedelta) -> dict[str, Any]:
    if delay < timedelta(0):
        raise ValueError("La demora asumida no puede ser negativa")
    later = []
    unknown = []
    for item in files:
        day = date.fromisoformat(item["date"])
        header = item.get("source_last_modified")
        if not header:
            unknown.append(day.isoformat())
            continue
        modified = parsedate_to_datetime(header)
        if modified.tzinfo is None or modified.utcoffset() is None:
            raise ValueError(f"Last-Modified no contiene zona horaria: {day}")
        last_possible_event = datetime.combine(day + timedelta(days=1), time.min, tzinfo=UTC)
        if modified.astimezone(UTC) > last_possible_event + delay:
            later.append({"date": day.isoformat(), "last_modified": modified.isoformat()})
    return {
        "files": len(files),
        "last_modified_unknown": len(unknown),
        "modified_after_last_event_plus_delay": len(later),
        "late_examples": later[:10],
        "point_in_time_verified": False,
        "interpretation": (
            "Last-Modified posterior no demuestra que los valores cambiaran, pero la versión "
            "histórica disponible no puede reconstruirse con esta información."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita procedencia de un dataset congelado")
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_dataset(args.dataset)
    archive = json.loads((args.dataset / "archive-manifest.json").read_text(encoding="utf-8"))
    report = audit_archive_versions(
        archive["files"],
        timedelta(hours=manifest["availability_assumption"]["metrics_delay_hours"]),
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
