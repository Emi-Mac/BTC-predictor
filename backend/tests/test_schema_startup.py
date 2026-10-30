from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src import main
from src.config.settings import Settings


def test_auto_create_schema_can_be_disabled(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/db")
    monkeypatch.setenv("BACKEND_AUTO_CREATE_SCHEMA", "false")
    assert Settings(_env_file=None).backend_auto_create_schema is False
    monkeypatch.delenv("BACKEND_AUTO_CREATE_SCHEMA")
    assert Settings(_env_file=None).backend_auto_create_schema is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("development", "enabled", "expected"),
    [(True, True, 1), (True, False, 0), (False, True, 0)],
)
async def test_lifespan_respects_schema_creation_flag(monkeypatch, development, enabled, expected):
    class Engine:
        def __init__(self):
            self.created = 0

        def begin(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def run_sync(self, fn):
            self.created += 1

        async def dispose(self):
            return None

    engine = Engine()
    monkeypatch.setattr(main, "engine", engine)
    monkeypatch.setattr(main, "check_db_connection", AsyncMock(return_value=True))
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            environment="development",
            is_development=development,
            backend_auto_create_schema=enabled,
        ),
    )
    async with main.lifespan(None):
        pass
    assert engine.created == expected
