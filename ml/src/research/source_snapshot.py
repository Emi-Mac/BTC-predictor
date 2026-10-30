"""Captura fuentes de investigación sin escribir en las tablas operativas."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import get_settings
from src.features.compute import REQUIRED_MACRO
from src.ingestion.binance import BinanceKlinesClient
from src.ingestion.cleaning import clean_candles
from src.ingestion.funding_client import BinanceFundingClient
from src.ingestion.integrity import check_candle_integrity
from src.ingestion.market_data import fetch_macro_series
from src.ingestion.repository import IngestionRepository
from src.ingestion.run import configure_logging

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.config import Settings

log = structlog.get_logger(__name__)
SOURCE_FILES = ("candles", "macro", "sentiment", "funding")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_spec(directory: Path) -> dict[str, Any]:
    path = directory / "build-spec.json"
    if not path.is_file():
        raise ValueError("El directorio no contiene una especificación de dataset")
    spec: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if spec["symbol"] != "BTCUSDT":
        raise ValueError("La captura actual solo admite el símbolo BTCUSDT")
    if (directory / "manifest.json").exists():
        raise FileExistsError("El dataset está finalizado y no puede modificarse")
    return spec


async def capture_sources(
    engine: AsyncEngine, settings: Settings, directory: Path, seed_dataset: Path | None = None
) -> dict[str, Any]:
    spec = load_spec(directory)
    start = date.fromisoformat(spec["start"])
    end = date.fromisoformat(spec["end_exclusive"])
    start_time = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    end_time = datetime.combine(end, datetime.min.time(), tzinfo=UTC)
    seed_manifest = None
    if spec.get("seed_manifest_sha256"):
        if seed_dataset is None:
            raise ValueError("La captura requiere el dataset base indicado en la especificación")
        from src.research.seed import verify_seed

        seed_manifest = verify_seed(seed_dataset, spec)
        capture_start = datetime.combine(
            date.fromisoformat(seed_manifest["end_exclusive"]), datetime.min.time(), tzinfo=UTC
        )
    elif seed_dataset is not None:
        raise ValueError("La especificación no contiene un dataset base")
    else:
        capture_start = start_time
    if settings.btc_symbol != spec["symbol"] or settings.default_interval != "4h":
        raise ValueError("La captura requiere el símbolo configurado y velas de 4h")
    sources = directory / "sources"
    sources.mkdir(exist_ok=True)
    existing_manifest = directory / "source-manifest.json"
    if existing_manifest.exists():
        manifest: dict[str, Any] = json.loads(existing_manifest.read_text(encoding="utf-8"))
        for item in manifest["files"].values():
            path = directory / item["path"]
            if not path.is_file() or file_sha256(path) != item["sha256"]:
                raise ValueError(f"El archivo de fuente no coincide con su huella: {path.name}")
        return manifest

    async with (
        BinanceKlinesClient(
            base_url=settings.binance_base_url,
            rate_limit_per_min=settings.binance_rate_limit_per_min,
        ) as spot,
        BinanceFundingClient(
            base_url=settings.binance_futures_base_url,
            rate_limit_per_min=settings.funding_rate_limit_per_min,
        ) as futures,
    ):
        candles, macro, funding, sentiment = await asyncio.gather(
            spot.fetch_klines(
                spec["symbol"], "4h", capture_start, end_time - timedelta(milliseconds=1)
            ),
            fetch_macro_series(
                settings.macro_symbols, capture_start.date() - timedelta(days=45), end
            ),
            futures.fetch(
                spec["symbol"],
                capture_start - timedelta(days=7),
                end_time - timedelta(milliseconds=1),
            ),
            IngestionRepository(engine).load_fear_greed(),
        )
    if seed_manifest is not None and seed_dataset is not None:
        from src.research.seed import load_seed_sources, merge_seed_sources

        merged = merge_seed_sources(
            load_seed_sources(seed_dataset),
            {"candles": candles, "macro": macro, "sentiment": sentiment, "funding": funding},
            pd.Timestamp(capture_start),
        )
        candles, macro = merged["candles"], merged["macro"]
        sentiment, funding = merged["sentiment"], merged["funding"]
    if candles.empty:
        raise ValueError("No se descargaron velas para el dataset")
    clean, cleaning = clean_candles(candles, timedelta(hours=4), max_fill=0)
    integrity = check_candle_integrity(clean, timedelta(hours=4))
    if (
        not integrity.ok
        or integrity.completeness < 0.995
        or clean["open_time"].iloc[0] != pd.Timestamp(start_time)
        or clean["open_time"].iloc[-1] != pd.Timestamp(end_time - timedelta(hours=4))
    ):
        raise ValueError(f"Las velas no cumplen integridad o cobertura: {integrity.problems}")
    if REQUIRED_MACRO - set(macro["symbol"]):
        raise ValueError("Faltan series macro para el dataset aislado")
    for symbol in REQUIRED_MACRO:
        subset = macro.loc[macro["symbol"] == symbol]
        if (
            subset.empty
            or subset["trade_date"].min() > start
            or subset["trade_date"].max() < end - timedelta(days=7)
        ):
            raise ValueError(f"Cobertura macro insuficiente: {symbol}")
        if not np.isfinite(subset["close"].to_numpy(dtype=float)).all():
            raise ValueError(f"Valores macro inválidos: {symbol}")
    if funding.empty or funding["funding_time"].min() > pd.Timestamp(
        start_time - timedelta(days=1)
    ):
        raise ValueError("Cobertura inicial insuficiente de funding")
    if funding["funding_time"].max() < pd.Timestamp(end_time - timedelta(days=2)):
        raise ValueError("Cobertura final insuficiente de funding")
    sentiment = sentiment.loc[
        (sentiment["obs_date"] >= start - timedelta(days=45)) & (sentiment["obs_date"] < end)
    ].reset_index(drop=True)
    if (
        sentiment.empty
        or sentiment["obs_date"].min() > start - timedelta(days=7)
        or sentiment["obs_date"].max() < end - timedelta(days=2)
    ):
        raise ValueError("Cobertura insuficiente de Fear & Greed")

    funding["funding_rate"] = funding["funding_rate"].astype(float)
    funding["mark_price"] = funding["mark_price"].astype(float)
    frames = {"candles": clean, "macro": macro, "sentiment": sentiment, "funding": funding}
    files = {}
    for name, frame in frames.items():
        target = sources / f"{name}.parquet"
        temporary = sources / f".{name}.parquet.part"
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
        files[name] = {
            "path": str(target.relative_to(directory)),
            "rows": len(frame),
            "sha256": file_sha256(target),
        }
    manifest = {
        "captured_at": datetime.now(UTC).isoformat(),
        "files": files,
        "candle_synthetic_rows": cleaning.filled,
        "candle_completeness": integrity.completeness,
        "historical_publication_times_verified": False,
    }
    if seed_manifest is not None:
        manifest["seed_dataset_id"] = seed_manifest["dataset_id"]
        manifest["seed_manifest_sha256"] = seed_manifest["manifest_sha256"]
    temporary = directory / ".source-manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(existing_manifest)
    log.info("research_sources_captured", directory=str(directory), files=files)
    return manifest


async def _run(directory: Path, seed_dataset: Path | None = None) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        await capture_sources(engine, settings, directory, seed_dataset)
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Captura fuentes para un dataset aislado")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--seed-dataset", type=Path)
    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    asyncio.run(_run(args.directory, args.seed_dataset))


if __name__ == "__main__":
    main()
