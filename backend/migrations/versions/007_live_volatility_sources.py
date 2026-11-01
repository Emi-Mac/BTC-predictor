"""Fuentes en vivo para el pronóstico de volatilidad: DVOL, interés abierto y calendario FOMC.

`recorded_at` / `captured_at` registran cuándo se leyó el dato; no son su fecha de
disponibilidad (los datos rellenados retroactivamente usan la regla de cada fuente).

Revision ID: 007_live_volatility_sources
Revises: 006_contextual_features
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "007_live_volatility_sources"
down_revision: str | None = "006_contextual_features"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "implied_volatility",
        sa.Column("currency", sa.String(10), nullable=False),
        sa.Column("open_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("close", sa.Numeric(12, 6), nullable=False),
        sa.Column(
            "recorded_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.PrimaryKeyConstraint("currency", "open_time", name="pk_implied_volatility"),
        sa.CheckConstraint("close > 0", name="ck_implied_volatility_close"),
        comment="Índice de volatilidad implícita (DVOL de Deribit), velas horarias cerradas",
    )
    op.create_table(
        "open_interest",
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("create_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sum_open_interest", sa.Numeric(30, 8), nullable=False),
        sa.Column("sum_open_interest_value", sa.Numeric(30, 8), nullable=False),
        sa.Column(
            "recorded_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.PrimaryKeyConstraint("symbol", "create_time", name="pk_open_interest"),
        sa.CheckConstraint(
            "sum_open_interest >= 0 AND sum_open_interest_value >= 0",
            name="ck_open_interest_non_negative",
        ),
        comment="Interés abierto de futuros perpetuos de Binance (5 min)",
    )
    op.create_table(
        "event_calendar",
        sa.Column("event_type", sa.String(30), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_url", sa.Text, nullable=False),
        sa.Column(
            "captured_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.PrimaryKeyConstraint("event_type", "event_time", name="pk_event_calendar"),
        comment="Calendario de eventos programados (reuniones del FOMC)",
    )


def downgrade() -> None:
    op.drop_table("event_calendar")
    op.drop_table("open_interest")
    op.drop_table("implied_volatility")
