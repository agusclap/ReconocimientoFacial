"""ASGI entrypoint for the backend FastAPI application."""
from __future__ import annotations

from .app import create_app

app = create_app()
