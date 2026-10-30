"""Añade volumen de compra agresora a las velas existentes.

Revision ID: 004_taker_buy_volume
Revises: 003_feature_engineering
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "004_taker_buy_volume"
down_revision: str | None = "003_feature_engineering"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ohlcv_candles", sa.Column("taker_buy_base_volume", sa.Numeric(30, 8), nullable=True)
    )
    op.add_column(
        "ohlcv_candles", sa.Column("taker_buy_quote_volume", sa.Numeric(30, 8), nullable=True)
    )
    op.create_check_constraint(
        "ck_candle_taker_base_volume",
        "ohlcv_candles",
        "taker_buy_base_volume IS NULL OR "
        "(taker_buy_base_volume >= 0 AND taker_buy_base_volume <= volume)",
    )
    op.create_check_constraint(
        "ck_candle_taker_quote_volume",
        "ohlcv_candles",
        "taker_buy_quote_volume IS NULL OR "
        "(taker_buy_quote_volume >= 0 AND quote_volume IS NOT NULL "
        "AND taker_buy_quote_volume <= quote_volume)",
    )
    op.create_check_constraint(
        "ck_candle_taker_volume_pair",
        "ohlcv_candles",
        "(taker_buy_base_volume IS NULL) = (taker_buy_quote_volume IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_candle_taker_volume_pair", "ohlcv_candles", type_="check")
    op.drop_constraint("ck_candle_taker_quote_volume", "ohlcv_candles", type_="check")
    op.drop_constraint("ck_candle_taker_base_volume", "ohlcv_candles", type_="check")
    op.drop_column("ohlcv_candles", "taker_buy_quote_volume")
    op.drop_column("ohlcv_candles", "taker_buy_base_volume")
