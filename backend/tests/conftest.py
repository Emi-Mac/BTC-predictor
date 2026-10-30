"""Fixtures compartidas para los tests del backend."""
from __future__ import annotations

import os

from dotenv import load_dotenv
import pytest

# Debe fijarse ANTES de que cualquier test importe `src.main`/`src.config.dependencies`:
# `src/config/database.py` crea el engine de SQLAlchemy al cargarse el módulo, y
# `Settings()` exige `DATABASE_URL` sin valor por defecto (correcto para producción, pero
# bloquea la colección de tests si no existe un `backend/.env` local). Primero se carga un
# `.env` real si existe (respeta la configuración del desarrollador) y sólo si sigue sin
# definirse se usa un valor de relleno. No se abre ninguna conexión real: `create_async_engine()`
# sólo valida el formato de la URL, no se conecta hasta el primer uso.
load_dotenv()
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost:5432/btcpredictor_test")


@pytest.fixture
def test_database_url() -> str:
    """URL de una base de datos PostgreSQL real para tests de integración.

    Sigue la misma convención que `ml/tests/conftest.py`: si `TEST_DATABASE_URL` no está
    definido, los tests que dependan de esta fixture se saltan (no fallan) para permitir
    ejecutar la suite localmente sin una instancia de PostgreSQL disponible.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL no definido (tests de integración con PostgreSQL)")
    return url
