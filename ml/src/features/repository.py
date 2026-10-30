"""Lectura y persistencia transaccional de features por vela."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd
import sqlalchemy as sa
import structlog
from sqlalchemy.dialects.postgresql import insert

from src.features.compute import FEATURE_COLUMNS, STORED_COLUMNS, WEEKDAYS
from src.ingestion.repository import CANDLE_FIELDS
from src.ingestion.tables import ohlcv_candles

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

log = structlog.get_logger(__name__)
CHUNK_SIZE = 400
BOOL_COLUMNS = {"is_weekend", *(f"weekday_{day}" for day in WEEKDAYS)}
INT_COLUMNS = {
    "higher_highs_20",
    "lower_lows_20",
    "fear_greed_value",
    "hour_of_day",
    "day_of_week",
    "target_label",
}


async def _features_table(conn: AsyncConnection) -> sa.Table:
    def reflect(sync_conn: sa.Connection) -> sa.Table:
        return sa.Table("computed_features", sa.MetaData(), autoload_with=sync_conn)

    return await conn.run_sync(reflect)


class FeatureRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def load_candles(self, symbol: str, interval: str) -> pd.DataFrame:
        columns = ["id", *CANDLE_FIELDS]
        query = (
            sa.select(*[ohlcv_candles.c[name] for name in columns])
            .where(ohlcv_candles.c.symbol == symbol, ohlcv_candles.c.interval == interval)
            .order_by(ohlcv_candles.c.open_time)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        frame = pd.DataFrame(rows, columns=columns)
        if not frame.empty:
            for name in (
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
            ):
                frame[name] = frame[name].astype(float)
            frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        return frame

    async def save(self, symbol: str, interval: str, features: pd.DataFrame) -> int:
        if features.empty or features[FEATURE_COLUMNS].isna().any().any():
            raise ValueError("No se pueden guardar features vacías o incompletas")
        rows: list[dict[str, Any]] = []
        for item in features[["id", *STORED_COLUMNS]].to_dict("records"):
            record: dict[str, Any] = {"candle_id": int(item["id"])}
            for name in STORED_COLUMNS:
                value = item[name]
                if pd.isna(value):
                    if name != "target_label":
                        raise ValueError(f"Feature incompleta: {name}")
                    record[name] = None
                elif name in BOOL_COLUMNS:
                    record[name] = bool(value)
                elif name in INT_COLUMNS:
                    record[name] = int(value)
                else:
                    record[name] = Decimal(str(float(value)))
            rows.append(record)

        async with self._engine.begin() as conn:
            table = await _features_table(conn)
            existing = await conn.execute(
                sa.select(table.c.candle_id)
                .join(ohlcv_candles, table.c.candle_id == ohlcv_candles.c.id)
                .where(ohlcv_candles.c.symbol == symbol, ohlcv_candles.c.interval == interval)
            )
            stale = set(existing.scalars()) - {record["candle_id"] for record in rows}
            for start in range(0, len(rows), CHUNK_SIZE):
                stmt = insert(table).values(rows[start : start + CHUNK_SIZE])
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_features_candle",
                    set_={name: stmt.excluded[name] for name in STORED_COLUMNS},
                )
                await conn.execute(stmt)
            stale_ids = list(stale)
            for start in range(0, len(stale_ids), CHUNK_SIZE):
                await conn.execute(
                    table.delete().where(
                        table.c.candle_id.in_(stale_ids[start : start + CHUNK_SIZE])
                    )
                )
        log.info("features_saved", rows=len(rows), stale_removed=len(stale))
        return len(rows)

    async def load_for_selection(self, symbol: str, interval: str) -> pd.DataFrame:
        async with self._engine.connect() as conn:
            table = await _features_table(conn)
            columns = [ohlcv_candles.c.open_time, table.c.target_label]
            columns.extend(table.c[name] for name in FEATURE_COLUMNS)
            query = (
                sa.select(*columns)
                .join(ohlcv_candles, table.c.candle_id == ohlcv_candles.c.id)
                .where(ohlcv_candles.c.symbol == symbol, ohlcv_candles.c.interval == interval)
                .order_by(ohlcv_candles.c.open_time)
            )
            result = await conn.execute(query)
            frame = pd.DataFrame(result.all(), columns=list(result.keys()))
        if not frame.empty:
            frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
            frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].astype(float)
        return frame

    async def last_feature_open_time(self, symbol: str, interval: str) -> datetime | None:
        async with self._engine.connect() as conn:
            table = await _features_table(conn)
            query = (
                sa.select(sa.func.max(ohlcv_candles.c.open_time))
                .select_from(table.join(ohlcv_candles, table.c.candle_id == ohlcv_candles.c.id))
                .where(ohlcv_candles.c.symbol == symbol, ohlcv_candles.c.interval == interval)
            )
            latest: datetime | None = (await conn.execute(query)).scalar()
        return latest
