"""Pronósticos de volatilidad por vela y horizonte, con su resolución posterior.

Cada fila guarda el modelo que sirvió (`model_version`), el que se pidió (`requested_model_version`,
el primario activo) y, si difieren, `fallback_reason`; además `features_as_of` con la marca de
tiempo de cada fuente exógena usada. La clave foránea compuesta impide que un pronóstico apunte a
un modelo de otro horizonte, y `ON DELETE RESTRICT` conserva el historial de evaluación.

Revision ID: 009_volatility_forecasts
Revises: 008_volatility_models
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "009_volatility_forecasts"
down_revision: str | None = "008_volatility_models"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "volatility_forecasts",
        sa.Column("id", sa.BigInteger, sa.Identity(always=False), nullable=False),
        sa.Column("candle_id", sa.BigInteger, nullable=False),
        sa.Column("horizon_candles", sa.Integer, nullable=False),
        sa.Column("model_version", sa.Integer, nullable=False),
        sa.Column("requested_model_version", sa.Integer, nullable=False),
        sa.Column("fallback_reason", sa.String(100), nullable=True),
        sa.Column("features_as_of", postgresql.JSONB, nullable=False),
        sa.Column("forecast_var_annual", sa.Numeric(20, 10), nullable=False),
        sa.Column("forecast_vol_annual", sa.Numeric(20, 10), nullable=False),
        sa.Column("k68", sa.Numeric(12, 6), nullable=True),
        sa.Column("k90", sa.Numeric(12, 6), nullable=True),
        sa.Column("k95", sa.Numeric(12, 6), nullable=True),
        sa.Column("calibration_window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("regime", sa.String(10), nullable=False),
        sa.Column("reference_price", sa.Numeric(20, 8), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.Column("realized_var_annual", sa.Numeric(20, 10), nullable=True),
        sa.Column("realized_log_return", sa.Numeric(20, 10), nullable=True),
        sa.Column("inside_68", sa.Boolean, nullable=True),
        sa.Column("inside_90", sa.Boolean, nullable=True),
        sa.Column("inside_95", sa.Boolean, nullable=True),
        sa.Column("qlike", sa.Numeric(20, 10), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_volatility_forecasts"),
        sa.ForeignKeyConstraint(
            ["candle_id"],
            ["ohlcv_candles.id"],
            name="fk_volatility_forecasts_candle",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["model_version", "horizon_candles"],
            ["volatility_models.version", "volatility_models.horizon_candles"],
            name="fk_volatility_forecasts_model",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["requested_model_version", "horizon_candles"],
            ["volatility_models.version", "volatility_models.horizon_candles"],
            name="fk_volatility_forecasts_requested_model",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "candle_id", "horizon_candles", name="uq_volatility_forecasts_candle_horizon"
        ),
        sa.CheckConstraint(
            "forecast_var_annual > 0 AND forecast_vol_annual > 0",
            name="ck_volatility_forecasts_positive",
        ),
        sa.CheckConstraint(
            "regime IN ('low', 'normal', 'high', 'extreme')", name="ck_volatility_forecasts_regime"
        ),
        sa.CheckConstraint("reference_price > 0", name="ck_volatility_forecasts_price"),
        sa.CheckConstraint(
            "(fallback_reason IS NULL) = (model_version = requested_model_version)",
            name="ck_volatility_forecasts_fallback",
        ),
        sa.CheckConstraint(
            "(k68 IS NULL) = (k90 IS NULL) AND (k90 IS NULL) = (k95 IS NULL) "
            "AND (k68 IS NULL) = (calibration_window_end IS NULL)",
            name="ck_volatility_forecasts_calibration",
        ),
        sa.CheckConstraint(
            "(resolved_at IS NULL) = (realized_var_annual IS NULL) "
            "AND (resolved_at IS NULL) = (realized_log_return IS NULL) "
            "AND (resolved_at IS NULL) = (qlike IS NULL)",
            name="ck_volatility_forecasts_resolution",
        ),
        comment="Pronósticos de volatilidad y bandas por vela y horizonte, con su resolución",
    )
    op.create_index(
        "ix_volatility_forecasts_unresolved",
        "volatility_forecasts",
        ["horizon_candles", "candle_id"],
        postgresql_where=sa.text("resolved_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_volatility_forecasts_unresolved", table_name="volatility_forecasts")
    op.drop_table("volatility_forecasts")
