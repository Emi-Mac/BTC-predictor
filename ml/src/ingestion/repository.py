"""Repositorio PostgreSQL del pipeline de ingesta (bulk upserts idempotentes)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd
import sqlalchemy as sa
import structlog
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.ingestion.tables import fear_greed_daily, macro_daily, ohlcv_candles

if TYPE_CHECKING:
    from datetime import date, datetime

    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

log = structlog.get_logger(__name__)

CHUNK_SIZE = 1000  # asyncpg admite máx. 32767 parámetros por statement
CANDLE_FIELDS = [
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]
TAKER_FIELDS = ("taker_buy_base_volume", "taker_buy_quote_volume")
_DECIMAL_FIELDS = {"open", "high", "low", "close", "volume", "quote_volume", *TAKER_FIELDS}
_STORAGE_QUANTUM = Decimal("0.00000001")


def _chunks(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [rows[i : i + CHUNK_SIZE] for i in range(0, len(rows), CHUNK_SIZE)]


def _dec(value: float | None) -> Decimal | None:
    if value is None or pd.isna(value):
        return None
    return Decimal(repr(float(value)))


def candles_to_rows(symbol: str, interval: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    missing = set(CANDLE_FIELDS) - set(TAKER_FIELDS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas de velas: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for rec in df.reindex(columns=CANDLE_FIELDS).to_dict("records"):
        row: dict[str, Any] = {"symbol": symbol, "interval": interval}
        for key, value in rec.items():
            name = str(key)
            if name in ("open_time", "close_time"):
                row[name] = value.to_pydatetime()
            elif name in _DECIMAL_FIELDS:
                row[name] = _dec(value)
            else:
                row[name] = None if pd.isna(value) else int(value)
        rows.append(row)
    return rows


async def _fill_missing_taker_volumes(
    conn: AsyncConnection,
    symbol: str,
    interval: str,
    rows: list[dict[str, Any]],
    require_all: bool,
) -> int:
    source: dict[datetime, dict[str, Any]] = {}
    for row in rows:
        base, quote = row[TAKER_FIELDS[0]], row[TAKER_FIELDS[1]]
        if (base is None) != (quote is None):
            raise ValueError("Los volúmenes agresores deben estar ambos presentes o ausentes")
        if base is not None:
            if row["open_time"] in source:
                raise ValueError("Hay velas duplicadas en los datos de Binance")
            source[row["open_time"]] = row
    if not source and not require_all:
        return 0

    query = sa.select(
        ohlcv_candles.c.id,
        ohlcv_candles.c.open_time,
        ohlcv_candles.c.volume,
        ohlcv_candles.c.quote_volume,
    ).where(
        ohlcv_candles.c.symbol == symbol,
        ohlcv_candles.c.interval == interval,
        sa.or_(
            ohlcv_candles.c.taker_buy_base_volume.is_(None),
            ohlcv_candles.c.taker_buy_quote_volume.is_(None),
        ),
    )
    if not require_all:
        query = query.where(ohlcv_candles.c.open_time.in_(list(source)))
    existing = (await conn.execute(query.with_for_update())).mappings().all()
    updates = []
    for candle in existing:
        incoming = source.get(candle["open_time"])
        if incoming is None:
            raise ValueError(f"Binance no devolvió la vela histórica {candle['open_time']}")
        source_volume = incoming["volume"]
        source_quote = incoming["quote_volume"]
        if source_volume is None or source_quote is None:
            raise ValueError(f"Faltan volúmenes totales para {candle['open_time']}")
        if (
            source_volume.quantize(_STORAGE_QUANTUM, rounding=ROUND_HALF_UP) != candle["volume"]
            or source_quote.quantize(_STORAGE_QUANTUM, rounding=ROUND_HALF_UP)
            != candle["quote_volume"]
        ):
            raise ValueError(
                f"Los volúmenes históricos de Binance no coinciden con la DB: {candle['open_time']}"
            )
        updates.append((candle["id"], incoming[TAKER_FIELDS[0]], incoming[TAKER_FIELDS[1]]))

    for offset in range(0, len(updates), CHUNK_SIZE):
        batch = updates[offset : offset + CHUNK_SIZE]
        values = sa.values(
            sa.column("id", sa.BigInteger()),
            sa.column("base", sa.Numeric(30, 8)),
            sa.column("quote", sa.Numeric(30, 8)),
            name="taker_input",
        ).data(batch)
        statement = (
            sa.update(ohlcv_candles)
            .where(ohlcv_candles.c.id == values.c.id)
            .where(
                ohlcv_candles.c.taker_buy_base_volume.is_(None),
                ohlcv_candles.c.taker_buy_quote_volume.is_(None),
            )
            .values(
                taker_buy_base_volume=values.c.base,
                taker_buy_quote_volume=values.c.quote,
            )
            .returning(ohlcv_candles.c.id)
        )
        updated = (await conn.execute(statement)).scalars().all()
        if len(updated) != len(batch):
            raise RuntimeError("No se pudieron actualizar todas las velas históricas")
    return len(updates)


class IngestionRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    # ── Velas ────────────────────────────────────────────────
    async def upsert_candles(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        """INSERT ... ON CONFLICT DO NOTHING. Devuelve cuántas filas nuevas se insertaron."""
        if df.empty:
            return 0
        inserted = 0
        flow_updated = 0
        async with self._engine.begin() as conn:
            for chunk in _chunks(candles_to_rows(symbol, interval, df)):
                stmt = (
                    pg_insert(ohlcv_candles)
                    .values(chunk)
                    .on_conflict_do_nothing(constraint="uq_candle_symbol_interval_time")
                )
                result = await conn.execute(stmt)
                inserted += result.rowcount or 0
                flow_updated += await _fill_missing_taker_volumes(
                    conn, symbol, interval, chunk, require_all=False
                )
        log.info(
            "candles_upserted",
            symbol=symbol,
            interval=interval,
            new=inserted,
            taker_updated=flow_updated,
            total=len(df),
        )
        return inserted

    async def missing_taker_bounds(
        self, symbol: str, interval: str
    ) -> tuple[datetime | None, datetime | None, int]:
        statement = sa.select(
            sa.func.min(ohlcv_candles.c.open_time),
            sa.func.max(ohlcv_candles.c.open_time),
            sa.func.count(),
        ).where(
            ohlcv_candles.c.symbol == symbol,
            ohlcv_candles.c.interval == interval,
            sa.or_(
                ohlcv_candles.c.taker_buy_base_volume.is_(None),
                ohlcv_candles.c.taker_buy_quote_volume.is_(None),
            ),
        )
        async with self._engine.connect() as conn:
            result = (await conn.execute(statement)).one()
        return result[0], result[1], int(result[2])

    async def backfill_taker_volumes(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        if df.empty:
            raise ValueError("Binance no devolvió velas para el backfill de volumen agresor")
        async with self._engine.begin() as conn:
            updated = await _fill_missing_taker_volumes(
                conn, symbol, interval, candles_to_rows(symbol, interval, df), require_all=True
            )
        log.info("taker_volume_backfilled", symbol=symbol, interval=interval, updated=updated)
        return updated

    async def load_candles(self, symbol: str, interval: str) -> pd.DataFrame:
        stmt = (
            sa.select(*[ohlcv_candles.c[name] for name in CANDLE_FIELDS])
            .where(ohlcv_candles.c.symbol == symbol, ohlcv_candles.c.interval == interval)
            .order_by(ohlcv_candles.c.open_time)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        df = pd.DataFrame(rows, columns=CANDLE_FIELDS)
        if df.empty:
            return df
        for col in _DECIMAL_FIELDS:
            df[col] = df[col].astype(float)
        for col in ("open_time", "close_time"):
            df[col] = pd.to_datetime(df[col], utc=True)
        df["num_trades"] = df["num_trades"].fillna(0).astype(int)
        return df

    async def last_open_time(self, symbol: str, interval: str) -> datetime | None:
        stmt = sa.select(sa.func.max(ohlcv_candles.c.open_time)).where(
            ohlcv_candles.c.symbol == symbol, ohlcv_candles.c.interval == interval
        )
        async with self._engine.connect() as conn:
            result: datetime | None = (await conn.execute(stmt)).scalar()
        return result

    async def first_open_time(self, symbol: str, interval: str) -> datetime | None:
        stmt = sa.select(sa.func.min(ohlcv_candles.c.open_time)).where(
            ohlcv_candles.c.symbol == symbol, ohlcv_candles.c.interval == interval
        )
        async with self._engine.connect() as conn:
            result: datetime | None = (await conn.execute(stmt)).scalar()
        return result

    # ── Macro ────────────────────────────────────────────────
    async def upsert_macro(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        rows = [
            {"symbol": r["symbol"], "trade_date": r["trade_date"], "close": _dec(r["close"])}
            for r in df.to_dict("records")
        ]
        async with self._engine.begin() as conn:
            for chunk in _chunks(rows):
                stmt = pg_insert(macro_daily).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol", "trade_date"], set_={"close": stmt.excluded.close}
                )
                await conn.execute(stmt)
        log.info("macro_upserted", rows=len(rows))
        return len(rows)

    async def load_macro(self) -> pd.DataFrame:
        stmt = sa.select(macro_daily.c.symbol, macro_daily.c.trade_date, macro_daily.c.close)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt.order_by(macro_daily.c.trade_date))).all()
        df = pd.DataFrame(rows, columns=["symbol", "trade_date", "close"])
        if not df.empty:
            df["close"] = df["close"].astype(float)
        return df

    async def last_macro_date(self) -> date | None:
        """Fecha más reciente común a todas las series (la más atrasada manda)."""
        sub = sa.select(sa.func.max(macro_daily.c.trade_date).label("d")).group_by(
            macro_daily.c.symbol
        )
        async with self._engine.connect() as conn:
            dates = [r[0] for r in (await conn.execute(sub)).all()]
        return min(dates) if dates else None

    # ── Fear & Greed ─────────────────────────────────────────
    async def upsert_fear_greed(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        rows = [
            {
                "obs_date": r["obs_date"],
                "value": int(r["value"]),
                "classification": r["classification"],
            }
            for r in df.to_dict("records")
        ]
        async with self._engine.begin() as conn:
            for chunk in _chunks(rows):
                stmt = pg_insert(fear_greed_daily).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["obs_date"],
                    set_={
                        "value": stmt.excluded.value,
                        "classification": stmt.excluded.classification,
                    },
                )
                await conn.execute(stmt)
        log.info("fear_greed_upserted", rows=len(rows))
        return len(rows)

    async def load_fear_greed(self) -> pd.DataFrame:
        stmt = sa.select(
            fear_greed_daily.c.obs_date, fear_greed_daily.c.value, fear_greed_daily.c.classification
        ).order_by(fear_greed_daily.c.obs_date)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return pd.DataFrame(rows, columns=["obs_date", "value", "classification"])

    async def last_fear_greed_date(self) -> date | None:
        async with self._engine.connect() as conn:
            result: date | None = (
                await conn.execute(sa.select(sa.func.max(fear_greed_daily.c.obs_date)))
            ).scalar()
        return result
