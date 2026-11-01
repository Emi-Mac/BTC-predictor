"""Persiste el contexto de precio por vela para que el backend pueda servir modelos `expanded`.

`computed_features` solo almacena `FEATURE_COLUMNS`. Los modelos entrenados con el conjunto
`expanded` (p. ej. la versión 2 del registro) también requieren `PRICE_CONTEXT_COLUMNS`, que
se derivan de las features y las velas con `add_price_context`. Este módulo las calcula con
esa misma función y las guarda en `contextual_features` (JSONB por vela).
"""

from __future__ import annotations

import argparse
import asyncio
import math
from typing import TYPE_CHECKING, Any

import pandas as pd
import sqlalchemy as sa
import structlog
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import get_settings
from src.features.compute import FEATURE_COLUMNS
from src.features.price_context import PRICE_CONTEXT_COLUMNS, add_price_context
from src.ingestion.binance import interval_to_timedelta
from src.ingestion.repository import IngestionRepository
from src.ingestion.tables import contextual_features, ohlcv_candles

if TYPE_CHECKING:
    from datetime import timedelta

    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.config import Settings

log = structlog.get_logger(__name__)
CHUNK_SIZE = 1000


def context_records(
    features: pd.DataFrame, candles: pd.DataFrame, step: timedelta
) -> list[dict[str, Any]]:
    """Filas `{candle_id, feature_values}` solo para velas con todo el contexto finito."""
    if features.empty:
        return []
    enriched = add_price_context(features, candles, step)
    records = []
    for row in enriched[["candle_id", *PRICE_CONTEXT_COLUMNS]].to_dict("records"):
        values = {name: float(row[name]) for name in PRICE_CONTEXT_COLUMNS}
        if all(math.isfinite(value) for value in values.values()):
            records.append({"candle_id": int(row["candle_id"]), "feature_values": values})
    return records


async def _load_features(engine: AsyncEngine, symbol: str, interval: str) -> pd.DataFrame:
    async with engine.connect() as conn:
        features = await conn.run_sync(
            lambda sync: sa.Table("computed_features", sa.MetaData(), autoload_with=sync)
        )
        query = (
            sa.select(
                features.c.candle_id,
                ohlcv_candles.c.open_time,
                *(features.c[name] for name in FEATURE_COLUMNS),
            )
            .join(ohlcv_candles, features.c.candle_id == ohlcv_candles.c.id)
            .where(ohlcv_candles.c.symbol == symbol, ohlcv_candles.c.interval == interval)
            .order_by(ohlcv_candles.c.open_time)
        )
        result = await conn.execute(query)
        frame = pd.DataFrame(result.all(), columns=list(result.keys()))
    if not frame.empty:
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].astype(float)
    return frame


async def save_contextual_features(engine: AsyncEngine, settings: Settings) -> int:
    symbol, interval = settings.btc_symbol, settings.default_interval
    features = await _load_features(engine, symbol, interval)
    candles = await IngestionRepository(engine).load_candles(symbol, interval)
    records = context_records(features, candles, interval_to_timedelta(interval))
    async with engine.begin() as conn:
        for start in range(0, len(records), CHUNK_SIZE):
            statement = insert(contextual_features).values(records[start : start + CHUNK_SIZE])
            statement = statement.on_conflict_do_update(
                index_elements=["candle_id"],
                set_={
                    "feature_values": statement.excluded.feature_values,
                    "computed_at": sa.func.now(),
                },
            )
            await conn.execute(statement)
    log.info("contextual_features_saved", rows=len(records))
    return len(records)


async def _main() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        await save_contextual_features(engine, settings)
    finally:
        await engine.dispose()


def main() -> None:
    argparse.ArgumentParser(
        description="Recalcula y guarda el contexto de precio de todas las velas"
    ).parse_args()
    from src.ingestion.run import configure_logging  # evita el ciclo run → pipeline → aquí

    configure_logging(get_settings().log_level)
    asyncio.run(_main())


if __name__ == "__main__":
    main()
