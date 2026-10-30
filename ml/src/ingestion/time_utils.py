"""Utilidades UTC para alinear los límites temporales de las velas."""

from datetime import UTC, datetime, timedelta


def floor_to_step(moment: datetime, step: timedelta) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None or step <= timedelta(0):
        raise ValueError("Se requiere una fecha con zona horaria y un intervalo positivo")
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    return epoch + ((moment.astimezone(UTC) - epoch) // step) * step
