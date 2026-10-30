"""CLI offline: cálculo o selección ajustada a una fecha de entrenamiento."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime

from sqlalchemy.ext.asyncio import create_async_engine

from src.config import get_settings
from src.features.pipeline import run_features
from src.features.repository import FeatureRepository
from src.features.selection import select_features
from src.ingestion.binance import interval_to_timedelta
from src.ingestion.run import configure_logging


async def execute(command: str, train_end: datetime | None) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        if command == "compute":
            await run_features(engine, settings)
        else:
            if train_end is None:
                raise ValueError("La selección requiere --train-end con zona horaria")
            data = await FeatureRepository(engine).load_for_selection(
                settings.btc_symbol, settings.default_interval
            )
            result = select_features(
                data, train_end, interval_to_timedelta(settings.default_interval)
            )
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Features de BTC (Fase 2)")
    parser.add_argument("command", choices=["compute", "select"])
    parser.add_argument("--train-end", type=datetime.fromisoformat)
    args = parser.parse_args()
    if args.command == "select" and args.train_end is None:
        parser.error("select requiere --train-end (ISO 8601 con zona horaria)")
    configure_logging(get_settings().log_level)
    asyncio.run(execute(args.command, args.train_end))


if __name__ == "__main__":
    main()
