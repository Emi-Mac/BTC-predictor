"""
Entry point del backend FastAPI.
Configura lifespan, middleware, routers y health checks.
"""
import contextlib
from collections.abc import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from src.config.database import check_db_connection, engine
from src.config.models import Base
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Gestiona el ciclo de vida de la aplicación:
    - Startup: verifica DB y crea tablas en dev solo si está habilitado
    - Shutdown: cierra pool de conexiones
    """
    logger.info(
        "Starting BTC Predictor API",
        environment=settings.environment,
        version="0.1.0",
    )

    # Verificar conexión a DB
    db_ok = await check_db_connection()
    if not db_ok:
        logger.error("Cannot connect to PostgreSQL. Check DATABASE_URL.")
        raise RuntimeError("Database connection failed at startup")

    logger.info("Database connection OK")

    # En Docker se desactiva esta opción para que Alembic gestione el esquema.
    if settings.is_development and settings.backend_auto_create_schema:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables verified/created (development mode)")

    yield  # ← La app corre aquí

    # Shutdown
    logger.info("Shutting down BTC Predictor API")
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="BTC Direction Predictor API",
        description=(
            "API REST para predicción direccional de Bitcoin usando ML. "
            "⚠️ DISCLAIMER: Herramienta educativa, NO asesoría financiera."
        ),
        version="0.1.0",
        docs_url="/docs" if settings.is_development else None,  # Ocultar en prod
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # ── Middleware ───────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    if not settings.is_development:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

    # ── Health checks ────────────────────────────────────────
    @app.get("/health", tags=["System"], summary="Liveness check")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "btc-predictor-api"})

    @app.get("/readiness", tags=["System"], summary="Readiness check (incluye DB)")
    async def readiness() -> JSONResponse:
        db_ok = await check_db_connection()
        status = "ready" if db_ok else "not_ready"
        code = 200 if db_ok else 503
        return JSONResponse({"status": status, "database": db_ok}, status_code=code)

    # ── Routers (se añadirán en Fases 4+) ───────────────────
    # from src.adapters.inbound.candle_router import router as candle_router
    # from src.adapters.inbound.prediction_router import router as prediction_router
    # app.include_router(candle_router, prefix="/api/v1")
    # app.include_router(prediction_router, prefix="/api/v1")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.backend_reload,
        log_level=settings.log_level.lower(),
    )
