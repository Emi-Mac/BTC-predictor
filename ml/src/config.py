"""Configuración del ML worker (misma convención de variables que el backend)."""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    database_url: str
    model_artifact_dir: str = "./artifacts"

    # Binance
    binance_base_url: str = "https://api.binance.com"
    binance_rate_limit_per_min: int = 1200
    binance_futures_base_url: str = "https://fapi.binance.com"
    funding_enabled: bool = True
    funding_rate_limit_per_min: int = Field(default=90, gt=0, le=100)
    funding_max_lag_hours: int = Field(default=24, gt=0)
    funding_feature_max_age_hours: int = Field(default=36, gt=0)
    btc_symbol: str = "BTCUSDT"
    default_interval: str = "4h"

    # Histórico: ~3.3 años para cumplir el DoD de >= 3 años de velas
    history_days: int = 1200
    # Velas que se re-piden en cada actualización incremental (idempotente)
    update_overlap_candles: int = 6
    ingestion_watchdog_seconds: int = Field(default=60, gt=0)

    # yfinance
    macro_symbols: Annotated[list[str], NoDecode] = ["DX-Y.NYB", "^GSPC", "^VIX", "GC=F"]
    # Días calendario máximos de propagación hacia adelante del valor macro (fines de semana)
    macro_max_ffill_days: int = 5

    # Fear & Greed
    fear_greed_base_url: str = "https://api.alternative.me/fng/"

    @field_validator("database_url", mode="after")
    @classmethod
    def ensure_asyncpg_driver(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @field_validator("macro_symbols", mode="before")
    @classmethod
    def parse_macro_symbols(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
