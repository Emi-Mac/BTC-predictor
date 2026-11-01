"""Configuración de las fuentes en vivo de volatilidad.

Vive aparte de `config.py`: ese archivo está en `IMPLEMENTATION_FILES` y su huella queda
congelada por los holdouts pendientes (ver `research/provenance.py`).
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VolatilitySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Apagadas por defecto: requieren la migración 007 (rama feature/prediccion-volatilidad).
    volatility_sources_enabled: bool = False
    deribit_base_url: str = "https://www.deribit.com"
    dvol_currency: str = "BTC"
    dvol_rate_limit_per_min: int = Field(default=30, gt=0, le=100)
    dvol_max_lag_hours: int = Field(default=3, gt=0)
    oi_rate_limit_per_min: int = Field(default=30, gt=0, le=100)
    oi_max_lag_hours: int = Field(default=3, gt=0)
    fomc_refresh_days: int = Field(default=7, gt=0)


@lru_cache
def get_volatility_settings() -> VolatilitySettings:
    return VolatilitySettings()
