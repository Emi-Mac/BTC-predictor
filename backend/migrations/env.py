"""
Alembic env.py — Configuración para migraciones async con SQLAlchemy 2.0.

Soporta dos modos:
  1. offline: genera SQL sin conectar a la DB
  2. online: conecta a la DB y aplica migraciones directamente
"""
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Importar configuración de la app para leer DATABASE_URL desde .env
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config.settings import get_settings
from src.config.models import Base  # Importa todos los modelos para autogenerate

# ── Configuración de logging desde alembic.ini ───────────────
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Metadata para autogenerate ────────────────────────────────
target_metadata = Base.metadata

# ── URL de la DB desde settings (no desde alembic.ini) ────────
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url_str)


def run_migrations_offline() -> None:
    """
    Modo offline: genera SQL sin conectar a la DB.
    Útil para revisar las migraciones antes de aplicarlas.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,                 # Detecta cambios de tipo
        compare_server_default=True,       # Detecta cambios en defaults
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Modo online: conecta async y aplica migraciones."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,           # Sin pool para migraciones
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

