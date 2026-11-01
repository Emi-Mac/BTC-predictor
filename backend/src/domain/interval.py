"""Utilidad compartida para convertir intervalos de Binance ('4h', '15m') a timedelta.

Deliberadamente independiente del paquete `ml` (servicios desplegables por separado).
"""
from __future__ import annotations

import re
from datetime import timedelta

_INTERVAL_RE = re.compile(r"^(\d+)([mhdw])$")
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def interval_to_timedelta(interval: str) -> timedelta:
    """Convierte un intervalo de Binance ('4h', '15m', '1d') a timedelta."""
    match = _INTERVAL_RE.match(interval)
    if not match:
        msg = f"Intervalo no soportado: {interval!r}"
        raise ValueError(msg)
    return timedelta(seconds=int(match.group(1)) * _UNIT_SECONDS[match.group(2)])
