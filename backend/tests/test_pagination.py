"""Tests de la metadata de paginación compartida (`build_pagination`)."""
from __future__ import annotations

from src.domain.entities.pagination import build_pagination


def test_build_pagination_has_more_when_offset_plus_returned_is_below_total():
    meta = build_pagination(limit=10, offset=0, total=25, returned=10)
    assert meta.has_more is True
    assert meta.total == 25


def test_build_pagination_has_more_false_on_last_page():
    meta = build_pagination(limit=10, offset=20, total=25, returned=5)
    assert meta.has_more is False


def test_build_pagination_has_more_false_when_empty():
    meta = build_pagination(limit=10, offset=0, total=0, returned=0)
    assert meta.has_more is False
