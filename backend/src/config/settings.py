"""
Configuración central del backend.
Usa Pydantic Settings para leer variables de entorno con validación estricta.
"""
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AnyUrl, PostgresDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Entorno ──────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    secret_key: str = "dev_secret_key_change_in_production"

    # ── Base de datos ────────────────────────────────────────
    database_url: PostgresDsn

    # ── API ──────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_reload: bool = False
    backend_auto_create_schema: bool = True
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    # ── ML ───────────────────────────────────────────────────
    model_artifact_dir: str = "./ml/artifacts"
    active_model_version: int = 1

    # ── Binance ──────────────────────────────────────────────
    binance_base_url: AnyUrl = AnyUrl("https://api.binance.com")  # type: ignore[assignment]
    btc_symbol: str = "BTCUSDT"
    default_interval: str = "4h"

    # ── yfinance ─────────────────────────────────────────────
    macro_symbols: Annotated[list[str], NoDecode] = ["DX-Y.NYB", "^GSPC", "^VIX", "GC=F"]

    # ── Fear & Greed ─────────────────────────────────────────
    fear_greed_base_url: AnyUrl = AnyUrl("https://api.alternative.me/fng/")  # type: ignore[assignment]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Permite definir CORS_ORIGINS como string separado por comas."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("macro_symbols", mode="before")
    @classmethod
    def parse_macro_symbols(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",")]
        return v

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Valida que producción tenga configuraciones seguras."""
        if self.environment == "production":
            if self.secret_key == "dev_secret_key_change_in_production":
                msg = "SECRET_KEY must be set in production"
                raise ValueError(msg)
            if self.backend_reload:
                msg = "BACKEND_RELOAD must be False in production"
                raise ValueError(msg)
        return self

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def database_url_str(self) -> str:
        return str(self.database_url)


@lru_cache
def get_settings() -> Settings:
    """Singleton de settings. Cacheado para evitar leer .env múltiples veces."""
    return Settings()
