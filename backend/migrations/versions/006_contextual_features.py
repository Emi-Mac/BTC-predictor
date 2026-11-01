"""Persiste features contextuales (JSONB) que no caben en `computed_features`.

El modelo activo puede requerir señales derivadas en lote (p. ej. el contexto de precio de
`ml/src/features/price_context.py`). El ML worker las guarda aquí por vela y el backend las
combina con `computed_features` al construir el vector de inferencia.

Revision ID: 006_contextual_features
Revises: 005_funding_rates
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "006_contextual_features"
down_revision: str | None = "005_funding_rates"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "contextual_features",
        sa.Column("candle_id", sa.BigInteger, nullable=False),
        sa.Column("feature_values", postgresql.JSONB, nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("candle_id", name="pk_contextual_features"),
        sa.ForeignKeyConstraint(
            ["candle_id"],
            ["ohlcv_candles.id"],
            name="fk_contextual_features_candle",
            ondelete="CASCADE",
        ),
        comment="Features contextuales por vela calculadas por el ML worker",
    )


def downgrade() -> None:
    op.drop_table("contextual_features")
