"""Excepciones de dominio, independientes de FastAPI o SQLAlchemy.

Los adaptadores inbound (routers) son responsables de traducir estas
excepciones a códigos HTTP apropiados.
"""
from __future__ import annotations


class DomainError(Exception):
    """Base de todos los errores de dominio de la aplicación."""


class NotFoundError(DomainError):
    """El recurso solicitado no existe (se traduce a HTTP 404)."""


class ModelUnavailableError(DomainError):
    """No hay un modelo activo cargado o listo para inferencia (HTTP 503)."""


class InvalidFeatureVectorError(DomainError):
    """Las features disponibles no permiten construir un vector de inferencia válido."""
