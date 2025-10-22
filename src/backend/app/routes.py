"""Route registration for the backend FastAPI application."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .database import db
from .face import FaceNotFoundError, embedding_from_image
from ...liveness import LivenessError, evaluate_frame_bytes
from .schemas import AccesoIn


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

    @router.post("/liveness/verify")
    async def verify_liveness(frame: UploadFile = File(...)):
        if frame.content_type and not frame.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="El archivo debe ser una imagen.")

        data = await frame.read()
        if not data:
            raise HTTPException(status_code=400, detail="No se recibió ninguna imagen para validar.")

        try:
            result = evaluate_frame_bytes(data)
            ok = True
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LivenessError as exc:
            result = exc.result
            ok = False

        return {
            "ok": ok,
            "personas": result.person_count,
            "dispositivos": result.cellphone_count,
            "motivos": result.reasons,
            "detecciones": [
                {"label": d.label, "confianza": d.confidence, "caja": d.box}
                for d in result.detections
            ],
        }

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

    def _parse_embedding(raw: Optional[str]) -> Optional[List[float]]:
        if not raw:
            return None
        values: List[float] = []
        for chunk in raw.split(","):
            text = chunk.strip()
            if not text:
                continue
            try:
                values.append(float(text))
            except ValueError as exc:  # pragma: no cover - defensive parsing
                raise HTTPException(status_code=400, detail="Embedding inválido") from exc
        if values and len(values) != 512:
            raise HTTPException(status_code=400, detail="El embedding debe tener 512 valores")
        return values or None

    @router.post("/socios")
    async def create_socio(
        dni: int = Form(...),
        nombre: str = Form(...),
        apellido: str = Form(...),
        fecha_nacimiento: date = Form(...),
        telefono: Optional[str] = Form(None),
        mail: Optional[str] = Form(None),
        id_plan: int = Form(...),
        dias_duracion: int = Form(30),
        embedding: Optional[str] = Form(None),
        foto: Optional[UploadFile] = File(None),
    ):
        vector = _parse_embedding(embedding)

        telefono = telefono or None
        mail = mail or None

        if vector is None and foto is None:
            raise HTTPException(
                status_code=400,
                detail="Debés adjuntar una foto o un embedding para registrar el rostro.",
            )

        if vector is None and foto is not None:
            image_bytes = await foto.read()
            try:
                vector = embedding_from_image(image_bytes)
            except FaceNotFoundError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        db().execute(
            "INSERT INTO socios (dni_cliente, nombre, apellido, fecha_nacimiento, telefono, mail) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            [dni, nombre, apellido, fecha_nacimiento, telefono, mail],
        )

        db().execute(
            "INSERT INTO membresias (dni_cliente, id_plan, fecha_inicio, fecha_fin, estado) "
            "VALUES (%s,%s,CURRENT_DATE, CURRENT_DATE + (%s || ' days')::interval, 'ACTIVA')",
            [dni, id_plan, dias_duracion],
        )

        if vector:
            db().execute(
                "INSERT INTO rostros (dni_cliente, embedding) VALUES (%s, %s)",
                [dni, vector],
            )

        return {"ok": True, "rostro_registrado": bool(vector)}

    @router.delete("/socios/{dni}")
    def delete_socio(dni: int):
        db().execute("DELETE FROM accesos WHERE dni_cliente=%s", [dni])
        db().execute("DELETE FROM rostros WHERE dni_cliente=%s", [dni])
        db().execute("DELETE FROM membresias WHERE dni_cliente=%s", [dni])
        db().execute("DELETE FROM socios WHERE dni_cliente=%s", [dni])
        return {"ok": True}

    app.include_router(router)


__all__ = ["register_routes"]
