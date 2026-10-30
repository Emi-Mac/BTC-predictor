"""Añade las features de estructura, volumen, sentimiento y calendario.

Revision ID: 003_feature_engineering
Revises: 002_macro_and_sentiment
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "003_feature_engineering"
down_revision: str | None = "002_macro_and_sentiment"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NUMERIC_COLUMNS = {
    "vwap_daily": (20, 8),
    "accumulation_distribution": (30, 8),
    "price_ema50_distance": (20, 10),
    "ema21_slope_5": (20, 10),
    "fear_greed_momentum_7": (10, 4),
}
INTEGER_COLUMNS = ("higher_highs_20", "lower_lows_20")
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def upgrade() -> None:
    for name, (precision, scale) in NUMERIC_COLUMNS.items():
        op.add_column(
            "computed_features", sa.Column(name, sa.Numeric(precision, scale), nullable=True)
        )
    for name in INTEGER_COLUMNS:
        op.add_column("computed_features", sa.Column(name, sa.Integer(), nullable=True))
    for day in WEEKDAYS:
        op.add_column(
            "computed_features", sa.Column(f"weekday_{day}", sa.Boolean(), nullable=True)
        )


def downgrade() -> None:
    for day in reversed(WEEKDAYS):
        op.drop_column("computed_features", f"weekday_{day}")
    for name in reversed(INTEGER_COLUMNS):
        op.drop_column("computed_features", name)
    for name in reversed(NUMERIC_COLUMNS):
        op.drop_column("computed_features", name)
