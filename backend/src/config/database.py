"""
Configuración de la base de datos con SQLAlchemy 2.0 async.
"""
from collections.abc import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.config.settings import get_settings


class Base(DeclarativeBase):
    """Base declarativa para todos los modelos ORM."""
    pass


def create_engine() -> AsyncEngine:
    settings = get_settings()
    engine = create_async_engine(
        settings.database_url_str,
        echo=settings.is_development,          # SQL logging solo en dev
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,                    # Verifica conexiones antes de usar
        pool_recycle=3600,                     # Recicla conexiones cada 1h
    )
    return engine


# Engine y sessionmaker como singletons del módulo
engine: AsyncEngine = create_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,                    # Evita lazy-loading post-commit
    autocommit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency de FastAPI que provee una sesión de DB por request.
    Garantiza cierre y rollback en caso de error.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_connection() -> bool:
    """Health check de la conexión a PostgreSQL."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

