"""Route registration for the backend FastAPI application."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .database import db
from .schemas import AccesoIn, SocioIn
from .face import FaceNotFoundError, embedding_from_video


def register_routes(app: FastAPI, frontend_root: Path) -> None:
    """Register API and frontend routes on the provided application."""
    templates_dir = frontend_root / "templates"
    index_html = templates_dir / "index.html"
    registro_html = templates_dir / "registro.html"
    eliminar_html = templates_dir / "eliminar.html"

    for template in (index_html, registro_html, eliminar_html):
        if not template.exists():
            raise RuntimeError(f"Frontend template not found: {template}")

    router = APIRouter()

    @router.get("/", response_class=FileResponse)
    def serve_frontend() -> FileResponse:
        return FileResponse(index_html)

    @router.get("/registro", response_class=FileResponse)
    def serve_registro() -> FileResponse:
        return FileResponse(registro_html)

    @router.get("/eliminar", response_class=FileResponse)
    def serve_eliminar() -> FileResponse:
        return FileResponse(eliminar_html)

    @router.get("/planes")
    def get_planes():
        return db().query("SELECT id_plan, nombre FROM planes ORDER BY id_plan")

    @router.get("/logs")
    def get_logs():
        sql = (
            "SELECT a.id, a.dni_cliente as dni, s.nombre, s.apellido, a.estado, a.fecha_hora, "
            "(SELECT nombre FROM planes p WHERE p.id_plan = (SELECT id_plan FROM membresias m "
            "WHERE m.dni_cliente=a.dni_cliente ORDER BY m.id_membresia DESC LIMIT 1)) as plan "
            "FROM accesos a LEFT JOIN socios s ON s.dni_cliente = a.dni_cliente "
            "ORDER BY a.fecha_hora DESC LIMIT 100"
        )
        return db().query(sql)

    @router.post("/logs/acceso")
    def post_acceso(payload: AccesoIn):
        socio = db().query(
            "SELECT dni_cliente FROM socios WHERE dni_cliente=%s",
            [payload.dni],
        )
        if not socio:
            raise HTTPException(status_code=404, detail="Socio no encontrado")

        db().execute(
            "INSERT INTO accesos (dni_cliente, estado) VALUES (%s, %s)",
            [payload.dni, payload.estado],
        )
        return {"ok": True}

    @router.get("/socios")
    def list_socios(q: Optional[str] = None):
        if q:
            like = f"%{q}%"
            return db().query(
                "SELECT dni_cliente as dni, nombre, apellido FROM socios WHERE CAST(dni_cliente AS TEXT) "
                "ILIKE %s OR nombre ILIKE %s OR apellido ILIKE %s ORDER BY apellido, nombre LIMIT 50",
                [like, like, like],
            )

        return db().query(
            "SELECT dni_cliente as dni, nombre, apellido FROM socios ORDER BY apellido, nombre LIMIT 50"
        )

    @router.post("/socios")
    def create_socio(s: SocioIn):
        db().execute(
            "INSERT INTO socios (dni_cliente, nombre, apellido, fecha_nacimiento, telefono, mail) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            [s.dni, s.nombre, s.apellido, s.fecha_nacimiento, s.telefono, s.mail],
        )

        db().execute(
            "INSERT INTO membresias (dni_cliente, id_plan, fecha_inicio, fecha_fin, estado) "
            "VALUES (%s,%s,CURRENT_DATE, CURRENT_DATE + (%s || ' days')::interval, 'ACTIVA')",
            [s.dni, s.id_plan, s.dias_duracion],
        )

        if s.embedding:
            db().execute(
                "INSERT INTO rostros (dni_cliente, embedding) VALUES (%s, %s)",
                [s.dni, s.embedding],
            )

        return {"ok": True}

    @router.delete("/socios/{dni}")
    def delete_socio(dni: int):
        db().execute("DELETE FROM accesos WHERE dni_cliente=%s", [dni])
        db().execute("DELETE FROM rostros WHERE dni_cliente=%s", [dni])
        db().execute("DELETE FROM membresias WHERE dni_cliente=%s", [dni])
        db().execute("DELETE FROM socios WHERE dni_cliente=%s", [dni])
        return {"ok": True}

    @router.post("/socios/{dni}/rostro-video")
    async def upload_socio_video(dni: int, video: UploadFile = File(...)):
        socio = db().query("SELECT dni_cliente FROM socios WHERE dni_cliente=%s", [dni])
        if not socio:
            raise HTTPException(status_code=404, detail="Socio no encontrado")

        try:
            video.file.seek(0)
            embedding = embedding_from_video(
                video.file, capture_frames=15, filename=video.filename
            )
        except FaceNotFoundError as exc:  # pragma: no cover - simple mapping to HTTP error
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        db().execute("DELETE FROM rostros WHERE dni_cliente=%s", [dni])
        db().execute(
            "INSERT INTO rostros (dni_cliente, embedding) VALUES (%s, %s)",
            [dni, embedding],
        )

        return {"ok": True, "embedding": embedding, "capturas": 15}

    app.include_router(router)


__all__ = ["register_routes"]
