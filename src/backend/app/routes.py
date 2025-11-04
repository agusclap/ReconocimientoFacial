"""Route registration for the backend FastAPI application."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles  # <- importa StaticFiles

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

    # --- Montar /static para servir JS/CSS/imagenes ---
    static_dir = frontend_root / "static"
    if not static_dir.exists():
        raise RuntimeError(f"Static dir not found: {static_dir}")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    router = APIRouter()

    # --- Páginas ---
    @router.get("/", response_class=FileResponse)
    def serve_frontend() -> FileResponse:
        return FileResponse(index_html)

    @router.get("/registro", response_class=FileResponse)
    def serve_registro() -> FileResponse:
        return FileResponse(registro_html)

    @router.get("/eliminar", response_class=FileResponse)
    def serve_eliminar() -> FileResponse:
        return FileResponse(eliminar_html)

    # --- APIs ---
    @router.get("/planes")
    def get_planes():
        return db().query("SELECT id_plan, nombre FROM planes ORDER BY id_plan")



    @router.post("/membresias/renovar")
    def renovar_membresia(dni: int, dias: int = 30):
    # cerrar membresías viejas
        db().execute(
        "UPDATE membresias SET estado='VENCIDA' WHERE dni_cliente=%s AND (fecha_fin < CURRENT_DATE OR estado <> 'ACTIVA')",
        [dni],
        )
    # crear nueva
        db().execute(
        "INSERT INTO membresias (dni_cliente, id_plan, fecha_inicio, fecha_fin, estado) "
        "SELECT %s, id_plan, CURRENT_DATE, CURRENT_DATE + (%s || ' days')::interval, 'ACTIVA' "
        "FROM membresias WHERE dni_cliente=%s ORDER BY id_membresia DESC LIMIT 1",
        [dni, dias, dni],
        )
        return {"ok": True}


    @router.get("/logs")
    def get_logs():
        sql = """
        SELECT
          a.id,
          a.dni_cliente AS dni,
          s.nombre,
          s.apellido,
          COALESCE(
            (
              SELECT p.nombre
              FROM membresias m
              JOIN planes p ON p.id_plan = m.id_plan
              WHERE m.dni_cliente = a.dni_cliente
              ORDER BY
                CASE
                  WHEN (COALESCE(m.fecha_fin, CURRENT_DATE) >= CURRENT_DATE AND m.estado = 'ACTIVA')
                  THEN 0 ELSE 1
                END,
                m.fecha_inicio DESC
              LIMIT 1
            ),
            '—'
          ) AS plan,
          UPPER(a.estado) AS estado,
          a.motivo,
          a.fecha AS fecha_hora
        FROM accesos a
        LEFT JOIN socios s ON s.dni_cliente = a.dni_cliente
        ORDER BY a.fecha DESC
        LIMIT 100
    """
        rows = db().query(sql)

        # Si db().query devuelve tuplas, mapear a diccionarios:
        if rows and not isinstance(rows[0], dict):
            mapped = []
            for (id_, dni, nombre, apellido, plan, estado, fecha) in rows:
                mapped.append({
                    "id": id_,
                    "dni": dni,
                    "nombre": nombre,
                    "apellido": apellido,
                    "plan": plan,
                    "estado": estado,
                    "fecha_hora": fecha.isoformat() if fecha else None,
                })
            return mapped

        # Si ya son dicts (RealDictCursor), devolver directo:
        return rows

    @router.post("/logs/acceso")
    def post_acceso(payload: AccesoIn):
        # 1) existe el socio?
        socio = db().query(
            "SELECT dni_cliente FROM socios WHERE dni_cliente=%s",
            [payload.dni],
        )
        if not socio:
        # también lo podés loguear como denegado si querés
            db().execute(
                "INSERT INTO accesos (dni_cliente, estado, motivo) VALUES (%s, %s, %s)",
                [payload.dni, "DENEGADO", "socio no encontrado"],
            )
            raise HTTPException(status_code=404, detail="Socio no encontrado")

        # 2) tiene una membresía vigente?
        vigente = db().query(
            """
            SELECT 1
            FROM membresias
            WHERE dni_cliente = %s
              AND estado = 'ACTIVA'
              AND (fecha_fin IS NULL OR fecha_fin >= CURRENT_DATE)
            ORDER BY fecha_inicio DESC
            LIMIT 1
            """,
            [payload.dni],
        )

        if not vigente:
        # 3) NO vigente -> registrar acceso denegado
            db().execute(
                "INSERT INTO accesos (dni_cliente, estado, motivo) VALUES (%s, %s, %s)",
                [payload.dni, "DENEGADO", "membresia vencida o impaga"],
            )
            return {"ok": False, "detail": "Membresía vencida o impaga"}

        # 4) SÍ vigente -> registrar como vino del servicio (normalmente PERMITIDO)
        db().execute(
            "INSERT INTO accesos (dni_cliente, estado) VALUES (%s, %s)",
            [payload.dni, payload.estado.upper()],
        )
        return {"ok": True}


    @router.get("/socios")
    def list_socios(q: Optional[str] = None):
        if q:
            like = f"%{q}%"
            return db().query(
                "SELECT dni_cliente AS dni, nombre, apellido "
                "FROM socios "
                "WHERE CAST(dni_cliente AS TEXT) ILIKE %s "
                "   OR nombre ILIKE %s "
                "   OR apellido ILIKE %s "
                "ORDER BY apellido, nombre "
                "LIMIT 50",
                [like, like, like],
            )

        return db().query(
            "SELECT dni_cliente AS dni, nombre, apellido "
            "FROM socios "
            "ORDER BY apellido, nombre "
            "LIMIT 50"
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
        except FaceNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        db().execute("DELETE FROM rostros WHERE dni_cliente=%s", [dni])
        db().execute(
            "INSERT INTO rostros (dni_cliente, embedding) VALUES (%s, %s)",
            [dni, embedding],
        )

        return {"ok": True, "embedding": embedding, "capturas": 15}

    app.include_router(router)


__all__ = ["register_routes"]
