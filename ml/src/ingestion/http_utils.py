"""Rate limiting y reintentos comunes a los conectores públicos de Binance."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

    from tenacity import RetryCallState, wait_exponential

RETRYABLE_STATUS = {418, 429, 500, 502, 503, 504}


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


def wait_with_retry_after(base: wait_exponential) -> Callable[[RetryCallState], float]:
    def wait(retry_state: RetryCallState) -> float:
        delay = base(retry_state)
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome else None
        if isinstance(exc, httpx.HTTPStatusError):
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                return max(delay, float(retry_after))
        return delay

    return wait


class MinIntervalLimiter:
    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if delay > 0:
            await asyncio.sleep(delay)
