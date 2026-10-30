"""Sincroniza el histórico de funding sin volver a ejecutar todas las fuentes."""

import asyncio

from sqlalchemy.ext.asyncio import create_async_engine

from src.config import get_settings
from src.ingestion.pipeline import IngestionSummary, ingest_funding
from src.ingestion.repository import IngestionRepository
from src.ingestion.run import configure_logging


async def _run() -> None:
    settings = get_settings()
    if not settings.funding_enabled:
        raise ValueError("FUNDING_ENABLED debe estar habilitado para sincronizar funding")
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        await ingest_funding(IngestionRepository(engine), settings, IngestionSummary())
    finally:
        await engine.dispose()


def main() -> None:
    configure_logging(get_settings().log_level)
    asyncio.run(_run())


if __name__ == "__main__":
    main()
