# src/backend/main.py
from pathlib import Path
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
import subprocess, sys, os, signal

from backend.app.routes import register_routes
from backend.app.setup_db import run_setup

_worker_proc = None  # proceso del reconocimiento

def create_app() -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    def _startup():
        global _worker_proc

        print("[startup] Inicializando base de datos…")
        run_setup()
        print("[startup] Base de datos lista ✅")

        # Montar estáticos del frontend
        FRONTEND_ROOT = Path(__file__).resolve().parents[1] / "frontend"
        static_dir = FRONTEND_ROOT / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=static_dir), name="static")
        else:
            print("[WARN] Carpeta /static no encontrada")

        # Registrar rutas del frontend + API
        register_routes(app, FRONTEND_ROOT)

        # ── Lanzar tu servicio real de reconocimiento ─────────────────────────────
        # Usaremos src/face_recognition.py (el que ya tenés con liveness + embeddings)
        project_root = Path(__file__).resolve().parents[2]          # .../ReconocimientoFacial
        recognizer = project_root / "src" / "detection" / "face_recognition_service.py"               # /src/main.py

        if not recognizer.exists():
            print(f"[WARN] No encontré el worker: {recognizer}")
        else:
            env = os.environ.copy()
            # Aseguramos que el servicio apunte al backend correcto:
            env.setdefault("API_BASE", "http://127.0.0.1:8000")
            env.setdefault("LIVENESS_CONF", "0.12")
            env.setdefault("LIVENESS_OK_STREAK", "1")
            env.setdefault("SIMILARITY_THRESHOLD", "0.38")
            env.setdefault("DISABLE_DEVICE_BLOCK", "1")
            # (si querés afinar InsightFace)
            env.setdefault("INSIGHT_DET_W", "800")
            env.setdefault("INSIGHT_DET_H", "800")

            print(f"[startup] Lanzando reconocimiento: {recognizer}")
            _worker_proc = subprocess.Popen([sys.executable, str(recognizer)], env=env)

    @app.on_event("shutdown")
    def _shutdown():
        global _worker_proc
        if _worker_proc and _worker_proc.poll() is None:
            print("[shutdown] Terminando reconocimiento…")
            _worker_proc.send_signal(signal.SIGTERM)

    return app

app = create_app()
