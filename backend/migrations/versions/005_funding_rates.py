"""Almacena eventos históricos de funding de futuros perpetuos.

Revision ID: 005_funding_rates
Revises: 004_taker_buy_volume
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "005_funding_rates"
down_revision: str | None = "004_taker_buy_volume"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "funding_rates",
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("funding_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("funding_rate", sa.Numeric(20, 12), nullable=False),
        sa.Column("mark_price", sa.Numeric(20, 8), nullable=True),
        sa.Column(
            "recorded_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.PrimaryKeyConstraint("symbol", "funding_time", name="pk_funding_rates"),
        sa.CheckConstraint("mark_price IS NULL OR mark_price > 0", name="ck_funding_mark_price"),
        comment="Eventos históricos de funding de futuros perpetuos de Binance",
    )


def downgrade() -> None:
    op.drop_table("funding_rates")
