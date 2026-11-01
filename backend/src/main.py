"""
Entry point del backend FastAPI.
Configura lifespan, middleware, routers y health checks.
"""
import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.adapters.inbound import (
    candle_router,
    feature_router,
    metrics_router,
    prediction_router,
    register_error_handlers,
    risk_router,
    volatility_router,
    websocket_router,
)
from src.config.database import check_db_connection, engine
from src.config.models import Base
from src.config.scheduler import create_scheduler
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.backend_rate_limit])


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Gestiona el ciclo de vida de la aplicación:
    - Startup: verifica DB, crea tablas en dev solo si está habilitado, arranca el scheduler
    - Shutdown: detiene el scheduler y cierra el pool de conexiones
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

    scheduler = create_scheduler()
    scheduler.start()
    logger.info("Prediction scheduler started")

    yield  # ← La app corre aquí

    # Shutdown
    scheduler.shutdown(wait=False)
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

    # Rate limiting (slowapi)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # Logging estructurado de cada request
    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request_handled",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response

    # Manejo centralizado de errores de dominio → HTTP
    register_error_handlers(app)

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

    # ── Routers ──────────────────────────────────────────────
    app.include_router(candle_router)
    app.include_router(prediction_router)
    app.include_router(metrics_router)
    app.include_router(feature_router)
    app.include_router(volatility_router)
    app.include_router(risk_router)
    app.include_router(websocket_router)

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
