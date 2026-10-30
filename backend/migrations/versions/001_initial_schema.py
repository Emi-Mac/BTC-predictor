"""Migración inicial: crea las 4 tablas del sistema

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-10-28 22:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── Tabla 1: ohlcv_candles ────────────────────────────────
    op.create_table(
        "ohlcv_candles",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("interval", sa.String(10), nullable=False),
        sa.Column("open_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("close_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(20, 8), nullable=False),
        sa.Column("high", sa.Numeric(20, 8), nullable=False),
        sa.Column("low", sa.Numeric(20, 8), nullable=False),
        sa.Column("close", sa.Numeric(20, 8), nullable=False),
        sa.Column("volume", sa.Numeric(30, 8), nullable=False),
        sa.Column("quote_volume", sa.Numeric(30, 8), nullable=True),
        sa.Column("num_trades", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "interval", "open_time", name="uq_candle_symbol_interval_time"),
        comment="Velas OHLCV de Bitcoin desde Binance Public API",
    )
    op.create_index("ix_ohlcv_candles_symbol", "ohlcv_candles", ["symbol"])
    op.create_index("ix_ohlcv_candles_interval", "ohlcv_candles", ["interval"])
    op.create_index("ix_ohlcv_candles_open_time", "ohlcv_candles", ["open_time"])
    op.create_index(
        "ix_ohlcv_candles_symbol_interval_open_time",
        "ohlcv_candles",
        ["symbol", "interval", "open_time"],
    )

    # ── Tabla 2: model_registry ───────────────────────────────
    # Creada antes que predictions_log (FK dependency)
    op.create_table(
        "model_registry",
        sa.Column("version", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("hyperparameters", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("val_accuracy", sa.Numeric(6, 4), nullable=True),
        sa.Column("val_f1", sa.Numeric(6, 4), nullable=True),
        sa.Column("val_precision", sa.Numeric(6, 4), nullable=True),
        sa.Column("val_recall", sa.Numeric(6, 4), nullable=True),
        sa.Column("baseline_accuracy", sa.Numeric(6, 4), nullable=True),
        sa.Column("artifact_path", sa.String(500), nullable=True),
        sa.Column("scaler_path", sa.String(500), nullable=True),
        sa.Column("feature_names", sa.JSON(), nullable=True),
        sa.Column(
            "trained_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("notes", sa.String(1000), nullable=True),
        sa.PrimaryKeyConstraint("version"),
        comment="Registro de versiones de modelos ML entrenados",
    )

    # ── Tabla 3: computed_features ────────────────────────────
    op.create_table(
        "computed_features",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("candle_id", sa.BigInteger(), nullable=False),
        # Técnicos de precio
        sa.Column("rsi_14", sa.Numeric(10, 4), nullable=True),
        sa.Column("macd", sa.Numeric(20, 8), nullable=True),
        sa.Column("macd_signal", sa.Numeric(20, 8), nullable=True),
        sa.Column("macd_histogram", sa.Numeric(20, 8), nullable=True),
        sa.Column("bb_upper", sa.Numeric(20, 8), nullable=True),
        sa.Column("bb_middle", sa.Numeric(20, 8), nullable=True),
        sa.Column("bb_lower", sa.Numeric(20, 8), nullable=True),
        sa.Column("bb_width", sa.Numeric(10, 6), nullable=True),
        sa.Column("ema_9", sa.Numeric(20, 8), nullable=True),
        sa.Column("ema_21", sa.Numeric(20, 8), nullable=True),
        sa.Column("ema_50", sa.Numeric(20, 8), nullable=True),
        sa.Column("sma_200", sa.Numeric(20, 8), nullable=True),
        # Volatilidad
        sa.Column("atr_14", sa.Numeric(20, 8), nullable=True),
        sa.Column("hist_volatility_20", sa.Numeric(10, 6), nullable=True),
        sa.Column("hl_range_pct", sa.Numeric(10, 6), nullable=True),
        # Volumen
        sa.Column("obv", sa.Numeric(30, 2), nullable=True),
        sa.Column("volume_sma_ratio", sa.Numeric(10, 4), nullable=True),
        # Momentum
        sa.Column("roc_5", sa.Numeric(10, 6), nullable=True),
        sa.Column("roc_10", sa.Numeric(10, 6), nullable=True),
        sa.Column("williams_r", sa.Numeric(10, 4), nullable=True),
        sa.Column("stoch_rsi", sa.Numeric(10, 4), nullable=True),
        # Macro
        sa.Column("dxy_close", sa.Numeric(10, 4), nullable=True),
        sa.Column("dxy_change_pct", sa.Numeric(10, 6), nullable=True),
        sa.Column("sp500_close", sa.Numeric(10, 2), nullable=True),
        sa.Column("sp500_change_pct", sa.Numeric(10, 6), nullable=True),
        sa.Column("vix_close", sa.Numeric(10, 4), nullable=True),
        sa.Column("gold_change_pct", sa.Numeric(10, 6), nullable=True),
        sa.Column("btc_sp500_corr_20", sa.Numeric(10, 6), nullable=True),
        # Sentimiento
        sa.Column("fear_greed_value", sa.Integer(), nullable=True),
        sa.Column("fear_greed_sma_7", sa.Numeric(10, 4), nullable=True),
        # Temporales
        sa.Column("hour_of_day", sa.Integer(), nullable=True),
        sa.Column("day_of_week", sa.Integer(), nullable=True),
        sa.Column("is_weekend", sa.Boolean(), nullable=True),
        # Target
        sa.Column("target_label", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["candle_id"], ["ohlcv_candles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("candle_id", name="uq_features_candle"),
        comment="Features calculadas para el modelo ML",
    )
    op.create_index("ix_computed_features_candle_id", "computed_features", ["candle_id"])

    # ── Tabla 4: predictions_log ──────────────────────────────
    op.create_table(
        "predictions_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("candle_id", sa.BigInteger(), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("predicted_direction", sa.String(4), nullable=False),
        sa.Column("confidence_score", sa.Numeric(6, 4), nullable=False),
        sa.Column("raw_probability_up", sa.Numeric(6, 4), nullable=True),
        sa.Column("actual_direction", sa.String(4), nullable=True),
        sa.Column("is_correct", sa.Boolean(), nullable=True),
        sa.Column(
            "predicted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["candle_id"], ["ohlcv_candles.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_version"], ["model_registry.version"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        comment="Historial de predicciones emitidas vs resultados reales",
    )
    op.create_index("ix_predictions_log_candle_id", "predictions_log", ["candle_id"])
    op.create_index("ix_predictions_log_predicted_at", "predictions_log", ["predicted_at"])


def downgrade() -> None:
    op.drop_table("predictions_log")
    op.drop_table("computed_features")
    op.drop_table("model_registry")
    op.drop_table("ohlcv_candles")

