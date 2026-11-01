"""Contenedor genérico de paginación compartido por varios endpoints."""
from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class PaginationMeta(BaseModel):
    """Metadata de paginación: límite, offset, total y si hay más páginas."""

    model_config = ConfigDict(frozen=True)

    limit: int
    offset: int
    total: int
    has_more: bool


class Page(BaseModel, Generic[T]):
    """Página genérica de resultados con su metadata de paginación."""

    model_config = ConfigDict(frozen=True)

    data: list[T]
    pagination: PaginationMeta


def build_pagination(limit: int, offset: int, total: int, returned: int) -> PaginationMeta:
    """Construye la metadata de paginación a partir del total y lo devuelto."""
    return PaginationMeta(
        limit=limit,
        offset=offset,
        total=total,
        has_more=(offset + returned) < total,
    )
