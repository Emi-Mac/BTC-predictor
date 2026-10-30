"""CLI de ingesta: `python -m src.ingestion.run [backfill|update|check|serve]`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import warnings
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import get_settings
from src.features.pipeline import run_cycle
from src.features.repository import FeatureRepository
from src.ingestion.binance import interval_to_timedelta
from src.ingestion.freshness import funding_is_stale
from src.ingestion.funding_repository import FundingRepository
from src.ingestion.integrity import check_candle_integrity
from src.ingestion.repository import IngestionRepository
from src.ingestion.scheduler import serve

MIN_HISTORY_DAYS = 3 * 365


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # no registrar URLs de cada request
    warnings.filterwarnings("ignore", module=r"yfinance(\..*)?")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
    )


async def _run(command: str) -> int:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    repo = IngestionRepository(engine)
    try:
        if command in ("backfill", "update"):
            await run_cycle(repo, settings)
        elif command == "serve":
            await serve(repo, settings)
        elif command == "check":
            df = await repo.load_candles(settings.btc_symbol, settings.default_interval)
            report = check_candle_integrity(
                df,
                interval_to_timedelta(settings.default_interval),
                min_span_days=MIN_HISTORY_DAYS,
                as_of=datetime.now(UTC),
            )
            latest_feature = await FeatureRepository(engine).last_feature_open_time(
                settings.btc_symbol, settings.default_interval
            )
            if report.last_open_time is not None and latest_feature != report.last_open_time:
                report.problems.append("las features no están actualizadas hasta la última vela")
            latest_funding = None
            if settings.funding_enabled:
                latest_funding = await FundingRepository(engine).last_funding_time(
                    settings.btc_symbol
                )
                if funding_is_stale(
                    latest_funding,
                    datetime.now(UTC),
                    timedelta(hours=settings.funding_max_lag_hours),
                ):
                    report.problems.append("el funding no está actualizado")
            print(  # noqa: T201
                f"rows={report.rows} span_days={report.span_days:.0f} "
                f"completeness={report.completeness:.4%} gaps={len(report.gaps)} "
                f"duplicates={report.duplicates} synthetic={report.synthetic_rows} "
                f"missing_recent={report.missing_recent} "
                f"last_open={report.last_open_time} expected_open={report.expected_open_time} "
                f"last_feature={latest_feature} last_funding={latest_funding}"
            )
            for problem in report.problems:
                print(f"PROBLEMA: {problem}")  # noqa: T201
            return 0 if report.ok else 1
    finally:
        await engine.dispose()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingesta de datos BTC (Fase 1)")
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["backfill", "update", "check", "serve"],
        help="backfill/update: una pasada; check: integridad; serve: ingesta + scheduler",
    )
    args = parser.parse_args(argv)
    configure_logging(get_settings().log_level)
    return asyncio.run(_run(args.command))


if __name__ == "__main__":
    raise SystemExit(main())
