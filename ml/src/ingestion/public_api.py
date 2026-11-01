"""Cliente HTTP de fuentes públicas: límite de tasa y reintentos con `Retry-After`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import structlog
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from src.ingestion.http_utils import (
    RETRYABLE_STATUS,
    MinIntervalLimiter,
    is_retryable,
    wait_with_retry_after,
)

if TYPE_CHECKING:
    from types import TracebackType

log = structlog.get_logger(__name__)


class PublicApiClient:
    def __init__(
        self,
        base_url: str,
        source: str,
        rate_limit_per_min: int = 30,
        max_attempts: int = 6,
        backoff_multiplier: float = 1.0,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if rate_limit_per_min <= 0 or rate_limit_per_min > 100:
            raise ValueError("El límite debe estar entre 1 y 100 solicitudes/minuto")
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=30.0, headers=headers)
        self._owns_client = client is None
        self._source = source
        self._limiter = MinIntervalLimiter(60.0 / rate_limit_per_min)
        self._max_attempts = max_attempts
        self._backoff_multiplier = backoff_multiplier

    async def __aenter__(self) -> PublicApiClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        retrying = AsyncRetrying(
            retry=retry_if_exception(is_retryable),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_with_retry_after(
                wait_exponential(multiplier=self._backoff_multiplier, max=60)
            ),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                await self._limiter.acquire()
                response = await self._client.get(path, params=params)
                if response.status_code in RETRYABLE_STATUS:
                    log.warning(
                        "public_api_retryable_status",
                        source=self._source,
                        status=response.status_code,
                        attempt=attempt.retry_state.attempt_number,
                    )
                response.raise_for_status()
                return response
        raise RuntimeError(
            f"No se pudo completar la solicitud a {self._source}"
        )  # pragma: no cover
