"""Definiciones Core de las tablas que usa el ML worker (espejo de las migraciones Alembic)."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

metadata = sa.MetaData()

ohlcv_candles = sa.Table(
    "ohlcv_candles",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
    sa.Column("symbol", sa.String(20), nullable=False),
    sa.Column("interval", sa.String(10), nullable=False),
    sa.Column("open_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("close_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("open", sa.Numeric(20, 8), nullable=False),
    sa.Column("high", sa.Numeric(20, 8), nullable=False),
    sa.Column("low", sa.Numeric(20, 8), nullable=False),
    sa.Column("close", sa.Numeric(20, 8), nullable=False),
    sa.Column("volume", sa.Numeric(30, 8), nullable=False),
    sa.Column("quote_volume", sa.Numeric(30, 8)),
    sa.Column("num_trades", sa.Integer),
    sa.Column("taker_buy_base_volume", sa.Numeric(30, 8)),
    sa.Column("taker_buy_quote_volume", sa.Numeric(30, 8)),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    sa.UniqueConstraint("symbol", "interval", "open_time", name="uq_candle_symbol_interval_time"),
    sa.CheckConstraint(
        "taker_buy_base_volume IS NULL OR "
        "(taker_buy_base_volume >= 0 AND taker_buy_base_volume <= volume)",
        name="ck_candle_taker_base_volume",
    ),
    sa.CheckConstraint(
        "taker_buy_quote_volume IS NULL OR "
        "(taker_buy_quote_volume >= 0 AND quote_volume IS NOT NULL "
        "AND taker_buy_quote_volume <= quote_volume)",
        name="ck_candle_taker_quote_volume",
    ),
    sa.CheckConstraint(
        "(taker_buy_base_volume IS NULL) = (taker_buy_quote_volume IS NULL)",
        name="ck_candle_taker_volume_pair",
    ),
)

macro_daily = sa.Table(
    "macro_daily",
    metadata,
    sa.Column("symbol", sa.String(20), primary_key=True),
    sa.Column("trade_date", sa.Date, primary_key=True),
    sa.Column("close", sa.Numeric(20, 6), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
)

fear_greed_daily = sa.Table(
    "fear_greed_daily",
    metadata,
    sa.Column("obs_date", sa.Date, primary_key=True),
    sa.Column("value", sa.Integer, nullable=False),
    sa.Column("classification", sa.String(30)),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
)

funding_rates = sa.Table(
    "funding_rates",
    metadata,
    sa.Column("symbol", sa.String(20), primary_key=True),
    sa.Column("funding_time", sa.DateTime(timezone=True), primary_key=True),
    sa.Column("funding_rate", sa.Numeric(20, 12), nullable=False),
    sa.Column("mark_price", sa.Numeric(20, 8)),
    sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    sa.CheckConstraint("mark_price IS NULL OR mark_price > 0", name="ck_funding_mark_price"),
)

contextual_features = sa.Table(
    "contextual_features",
    metadata,
    sa.Column(
        "candle_id",
        sa.BigInteger,
        sa.ForeignKey("ohlcv_candles.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("feature_values", JSONB, nullable=False),
    sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
)
