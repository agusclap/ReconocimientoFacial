"""ASGI entrypoint for the backend FastAPI application."""
from __future__ import annotations

try:
    from .app import create_app
except Exception:
    # Fallback cuando se ejecuta el archivo directamente (sin paquete)
    from backend.app import create_app

app = create_app()
