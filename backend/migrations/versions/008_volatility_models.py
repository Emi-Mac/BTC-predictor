"""Registro de modelos de volatilidad, separado de `model_registry`.

Reutilizar `model_registry` con una columna de tarea obligaría a cambiar `predictions_log`, la
activación y el repositorio del backend direccional. Una tabla propia deja intacto ese stack y
permite restricciones estrictas: un modelo por horizonte (`horizon_candles` obligatorio) y una
única versión activa por (horizonte, rol).

Revision ID: 008_volatility_models
Revises: 007_live_volatility_sources
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "008_volatility_models"
down_revision: str | None = "007_live_volatility_sources"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "volatility_models",
        sa.Column("version", sa.Integer, sa.Identity(always=False), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("role", sa.String(10), nullable=False),
        sa.Column("horizon_candles", sa.Integer, nullable=False),
        sa.Column("artifact_path", sa.Text, nullable=False),
        sa.Column("artifact_sha256", sa.String(64), nullable=False),
        sa.Column("feature_names", postgresql.JSONB, nullable=False),
        sa.Column("validation_metrics", postgresql.JSONB, nullable=False),
        sa.Column("is_active", sa.Boolean, server_default=sa.false(), nullable=False),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False
        ),
        sa.PrimaryKeyConstraint("version", name="pk_volatility_models"),
        # Necesario para la clave foránea compuesta de `volatility_forecasts` (migración 009).
        sa.UniqueConstraint(
            "version", "horizon_candles", name="uq_volatility_models_version_horizon"
        ),
        sa.CheckConstraint("kind IN ('har_x_live', 'har')", name="ck_volatility_models_kind"),
        sa.CheckConstraint("role IN ('primary', 'fallback')", name="ck_volatility_models_role"),
        sa.CheckConstraint(
            "horizon_candles IN (6, 42)", name="ck_volatility_models_horizon"
        ),
        sa.CheckConstraint(
            "char_length(artifact_sha256) = 64", name="ck_volatility_models_sha256"
        ),
        comment="Modelos de pronóstico de volatilidad (HAR / HAR-X-live) por horizonte y rol",
    )
    op.create_index(
        "uq_volatility_models_active",
        "volatility_models",
        ["horizon_candles", "role"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )


def downgrade() -> None:
    op.drop_index("uq_volatility_models_active", table_name="volatility_models")
    op.drop_table("volatility_models")
