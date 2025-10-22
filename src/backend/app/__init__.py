"""Application factory and FastAPI configuration for the backend service."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import register_routes


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(title="Gimnasio API", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    project_root = Path(__file__).resolve().parents[2]
    frontend_root = project_root / "frontend"
    static_dir = frontend_root / "static"

    if not static_dir.exists():
        raise RuntimeError(f"Static assets directory not found: {static_dir}")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    register_routes(app, frontend_root)

    return app


__all__ = ["create_app"]
