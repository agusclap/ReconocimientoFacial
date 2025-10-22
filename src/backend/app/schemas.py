"""Pydantic models used by the backend API."""
from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel


class SocioIn(BaseModel):
    dni: int
    nombre: str
    apellido: str
    fecha_nacimiento: date
    telefono: Optional[str] = None
    mail: Optional[str] = None
    id_plan: int
    dias_duracion: int = 30
    embedding: Optional[List[float]] = None


class AccesoIn(BaseModel):
    dni: int
    estado: str


__all__ = ["SocioIn", "AccesoIn"]
