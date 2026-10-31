"""Descarga reanudable y auditoría de archivos públicos de métricas de futuros."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import re
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import structlog
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.ingestion.http_utils import MinIntervalLimiter, is_retryable, wait_with_retry_after
from src.ingestion.run import configure_logging

log = structlog.get_logger(__name__)
ARCHIVE_BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
METRICS_COLUMNS = [
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]
MAX_ZIP_BYTES = 10_000_000
MAX_CSV_BYTES = 10_000_000


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checksum_from_text(text: str, expected_filename: str) -> str:
    parts = text.strip().split()
    if not parts or not re.fullmatch(r"[a-fA-F0-9]{64}", parts[0]):
        raise ValueError(f"Checksum inválido para {expected_filename}")
    if len(parts) > 1 and parts[1].lstrip("*") != expected_filename:
        raise ValueError(f"El checksum corresponde a otro archivo: {parts[1]}")
    return parts[0].lower()


def parse_daily_archive(data: bytes, symbol: str, day: date) -> tuple[pd.DataFrame, dict[str, Any]]:
    filename = f"{symbol}-metrics-{day.isoformat()}.csv"
    if len(data) > MAX_ZIP_BYTES:
        raise ValueError(f"Archivo ZIP demasiado grande: {day}")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != filename:
            raise ValueError(f"Contenido inesperado en el archivo de {day}")
        if members[0].file_size > MAX_CSV_BYTES:
            raise ValueError(f"CSV demasiado grande en el archivo de {day}")
        with archive.open(members[0]) as source:
            frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = {"create_time", "symbol", *METRICS_COLUMNS}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Esquema vacío o incompleto en el archivo de {day}")
    if not frame["symbol"].eq(symbol).all():
        raise ValueError(f"Símbolos inesperados en el archivo de {day}")
    if not frame["create_time"].str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all():
        raise ValueError(f"Formato de timestamp inesperado en el archivo de {day}")
    raw_rows = len(frame)
    frame = frame.drop_duplicates(subset=["create_time", "symbol", *METRICS_COLUMNS])
    duplicates_removed = raw_rows - len(frame)
    if frame["create_time"].duplicated().any():
        raise ValueError(f"Registros contradictorios para un mismo timestamp en {day}")
    frame["create_time"] = pd.to_datetime(
        frame["create_time"], format="%Y-%m-%d %H:%M:%S", utc=True, errors="raise"
    )
    if not frame["create_time"].dt.date.eq(day).all():
        raise ValueError(f"El archivo de {day} contiene fechas distintas")
    if (frame["create_time"].dt.minute % 5 != 0).any() or (
        frame["create_time"].dt.second != 0
    ).any():
        raise ValueError(f"Timestamps fuera de la cuadrícula de 5 minutos en {day}")
    for column in METRICS_COLUMNS:
        frame[column] = pd.to_numeric(frame[column].replace("", np.nan), errors="raise")
        known = frame[column].dropna()
        if not np.isfinite(known).all() or (known < 0).any():
            raise ValueError(f"Métricas no finitas o negativas en {column}, {day}")
    frame = frame.sort_values("create_time").reset_index(drop=True)
    expected = pd.date_range(day.isoformat(), periods=288, freq="5min", tz="UTC")
    missing = expected.difference(pd.DatetimeIndex(frame["create_time"]))
    audit = {
        "date": day.isoformat(),
        "raw_rows": raw_rows,
        "unique_rows": len(frame),
        "duplicates_removed": duplicates_removed,
        "missing_intervals": len(missing),
        "missing_sample": [str(value) for value in missing[:5]],
        "null_values": {name: int(frame[name].isna().sum()) for name in METRICS_COLUMNS},
    }
    return frame[["create_time", *METRICS_COLUMNS]], audit


class MetricsArchiveClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        concurrency: int = 4,
        min_interval_seconds: float = 0.1,
        max_attempts: int = 6,
    ) -> None:
        if not 1 <= concurrency <= 4 or min_interval_seconds < 0:
            raise ValueError("Concurrencia o intervalo de descarga inválidos")
        self._client = client
        self._semaphore = asyncio.Semaphore(concurrency)
        self._limiter = MinIntervalLimiter(min_interval_seconds)
        self._max_attempts = max_attempts

    async def _get(self, url: str) -> httpx.Response:
        retrying = AsyncRetrying(
            retry=retry_if_exception(is_retryable),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_with_retry_after(wait_exponential(multiplier=1, max=60)),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                await self._limiter.acquire()
                response = await self._client.get(url)
                response.raise_for_status()
                return response
        raise RuntimeError(f"No se pudo descargar {url}")  # pragma: no cover

    async def fetch_day(self, root: Path, symbol: str, day: date) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Z0-9]{1,20}", symbol):
            raise ValueError("Símbolo de archivo inválido")
        filename = f"{symbol}-metrics-{day.isoformat()}.zip"
        archive_path = root / filename
        metadata_path = root / f"{filename}.json"
        async with self._semaphore:
            if archive_path.exists() and metadata_path.exists():
                metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
                if _sha256(archive_path.read_bytes()) != metadata["sha256"]:
                    raise ValueError(f"El archivo cacheado de {day} no coincide con su checksum")
                return metadata
            checksum_response = await self._get(f"{ARCHIVE_BASE}/{symbol}/{filename}.CHECKSUM")
            expected = checksum_from_text(checksum_response.text, filename)
            if archive_path.exists():
                data = archive_path.read_bytes()
                last_modified = None
            else:
                response = await self._get(f"{ARCHIVE_BASE}/{symbol}/{filename}")
                data = response.content
                last_modified = response.headers.get("Last-Modified")
            if _sha256(data) != expected:
                raise ValueError(f"El checksum de Binance no coincide para {day}")
            _, audit = parse_daily_archive(data, symbol, day)
            metadata = {
                **audit,
                "filename": filename,
                "sha256": expected,
                "source_last_modified": last_modified,
                "retrieved_at": datetime.now(UTC).isoformat(),
            }
            if not archive_path.exists():
                temporary = root / f".{filename}.part"
                temporary.write_bytes(data)
                temporary.replace(archive_path)
            temporary_metadata = root / f".{filename}.json.part"
            temporary_metadata.write_text(
                json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
            )
            temporary_metadata.replace(metadata_path)
            return metadata

    async def fetch_range(
        self, root: Path, symbol: str, start: date, end: date
    ) -> list[dict[str, Any]]:
        if end <= start:
            raise ValueError("El rango de archivos debe ser positivo")
        root.mkdir(parents=True, exist_ok=True)
        days = [start + timedelta(days=index) for index in range((end - start).days)]
        results: list[dict[str, Any] | None] = [None] * len(days)
        completed = 0

        async def fetch(index: int, day: date) -> None:
            nonlocal completed
            results[index] = await self.fetch_day(root, symbol, day)
            completed += 1
            if completed % 50 == 0 or completed == len(days):
                log.info("metrics_archives_progress", completed=completed, total=len(days))

        async with asyncio.TaskGroup() as group:
            for index, day in enumerate(days):
                group.create_task(fetch(index, day))
        if any(result is None for result in results):
            raise RuntimeError("La descarga terminó con archivos sin registrar")
        return [result for result in results if result is not None]


async def download_dataset_archives(
    root: Path,
    dataset_id: str,
    symbol: str,
    start: date,
    end: date,
    resume: bool = False,
    seed_dataset: Path | None = None,
) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", dataset_id):
        raise ValueError("Identificador de dataset inválido")
    if end <= start or end > datetime.now(UTC).date():
        raise ValueError("El rango de fechas del dataset es inválido")
    directory = root / dataset_id
    spec = {
        "dataset_id": dataset_id,
        "symbol": symbol,
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "archive_base": ARCHIVE_BASE,
        "assumed_interval_end_minutes": 5,
        "assumed_availability_delay_hours": 48,
        "point_in_time_verified": False,
    }
    if seed_dataset is not None:
        from src.research.seed import verify_seed

        if seed_dataset.resolve() == directory.resolve():
            raise ValueError("El dataset base y el extendido deben ser distintos")
        seed_manifest = verify_seed(seed_dataset, spec)
        spec["seed_manifest_sha256"] = seed_manifest["manifest_sha256"]
    spec_path = directory / "build-spec.json"
    if directory.exists():
        if not resume or not spec_path.is_file():
            raise FileExistsError(
                "El dataset ya existe; use --resume solo para una descarga propia"
            )
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError("El rango o la configuración no coinciden con el dataset existente")
        if (directory / "manifest.json").exists():
            raise FileExistsError("El dataset ya fue finalizado y no puede modificarse")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    if seed_dataset is not None:
        from src.research.seed import copy_seed_archives

        copy_seed_archives(seed_dataset, directory / "archives")
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http:
        results = await MetricsArchiveClient(http).fetch_range(
            directory / "archives", symbol, start, end
        )
    manifest = {
        "files": results,
        "days": len(results),
        "missing_intervals": sum(item["missing_intervals"] for item in results),
        "duplicates_removed": sum(item["duplicates_removed"] for item in results),
        "days_with_gaps": sum(item["missing_intervals"] > 0 for item in results),
    }
    temporary = directory / ".archive-manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "archive-manifest.json")
    log.info(
        "metrics_archive_audited",
        directory=str(directory),
        days=manifest["days"],
        days_with_gaps=manifest["days_with_gaps"],
        missing_intervals=manifest["missing_intervals"],
        duplicates_removed=manifest["duplicates_removed"],
    )
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita archivos públicos de métricas de futuros")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-dataset", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(
        download_dataset_archives(
            Path(settings.model_artifact_dir) / "datasets",
            args.dataset_id,
            settings.btc_symbol,
            args.start,
            args.end,
            args.resume,
            args.seed_dataset,
        )
    )


if __name__ == "__main__":
    main()
