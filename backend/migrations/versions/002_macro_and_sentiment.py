"""Fase 1: series macro diarias (yfinance) y Fear & Greed Index

Revision ID: 002_macro_and_sentiment
Revises: 001_initial_schema
Create Date: 2026-10-28 23:30:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "002_macro_and_sentiment"
down_revision: str | None = "001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "macro_daily",
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("close", sa.Numeric(20, 6), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("symbol", "trade_date", name="pk_macro_daily"),
        comment="Cierres diarios de series macro (DXY, S&P500, VIX, Gold) desde yfinance",
    )

    op.create_table(
        "fear_greed_daily",
        sa.Column("obs_date", sa.Date(), nullable=False),
        sa.Column("value", sa.Integer(), nullable=False),
        sa.Column("classification", sa.String(30), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("obs_date", name="pk_fear_greed_daily"),
        sa.CheckConstraint("value BETWEEN 0 AND 100", name="ck_fear_greed_range"),
        comment="Fear & Greed Index diario (alternative.me), publicado a las 00:00 UTC",
    )


def downgrade() -> None:
    op.drop_table("fear_greed_daily")
    op.drop_table("macro_daily")
