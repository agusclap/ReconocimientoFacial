"""Pydantic models used by the backend API."""
from __future__ import annotations

from pydantic import BaseModel


class AccesoIn(BaseModel):
    dni: int
    estado: str


__all__ = ["AccesoIn"]
