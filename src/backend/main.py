from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from datetime import date, timedelta
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel
from db import db
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Gimnasio API", version="1.0")

frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------- MODELOS Pydantic ---------
class SocioIn(BaseModel):
    dni: int
    nombre: str
    apellido: str
    fecha_nacimiento: date
    telefono: Optional[str] = None
    mail: Optional[str] = None
    id_plan: int
    dias_duracion: int = 30
    embedding: Optional[List[float]] = None # opcional


class AccesoIn(BaseModel):
    dni: int
    estado: str # PERMITIDO o RECHAZADO


# --------- ENDPOINTS ---------
@app.get("/planes")
def get_planes():
    return db().query("SELECT id_plan, nombre FROM planes ORDER BY id_plan")


@app.get("/logs")
def get_logs():
    sql = (
        "SELECT a.id, a.dni_cliente as dni, s.nombre, s.apellido, a.estado, a.fecha_hora, "
        "(SELECT nombre FROM planes p WHERE p.id_plan = (SELECT id_plan FROM membresias m WHERE m.dni_cliente=a.dni_cliente ORDER BY m.id_membresia DESC LIMIT 1)) as plan "
        "FROM accesos a LEFT JOIN socios s ON s.dni_cliente = a.dni_cliente "
        "ORDER BY a.fecha_hora DESC LIMIT 100"
        )
    return db().query(sql)


@app.post("/logs/acceso")
def post_acceso(payload: AccesoIn):
    # Verificamos que exista el socio
    socio = db().query("SELECT dni_cliente FROM socios WHERE dni_cliente=%s", [payload.dni])
    if not socio:
        raise HTTPException(status_code=404, detail="Socio no encontrado")
    db().execute("INSERT INTO accesos (dni_cliente, estado) VALUES (%s, %s)", [payload.dni, payload.estado])
    return {"ok": True}


@app.get("/socios")
def list_socios(q: Optional[str] = None):
    if q:
        like = f"%{q}%"
        return db().query(
            "SELECT dni_cliente as dni, nombre, apellido FROM socios WHERE CAST(dni_cliente AS TEXT) ILIKE %s OR nombre ILIKE %s OR apellido ILIKE %s ORDER BY apellido, nombre LIMIT 50",
            [like, like, like],
        )
    return db().query("SELECT dni_cliente as dni, nombre, apellido FROM socios ORDER BY apellido, nombre LIMIT 50")


@app.post("/socios")
def create_socio(s: SocioIn):
    # 1) Insert socio
    db().execute(
        "INSERT INTO socios (dni_cliente, nombre, apellido, fecha_nacimiento, telefono, mail) VALUES (%s,%s,%s,%s,%s,%s)",
        [s.dni, s.nombre, s.apellido, s.fecha_nacimiento, s.telefono, s.mail],
        )
    # 2) Membresía
    fecha_fin = s.fecha_nacimiento # placeholder to avoid flake; not used
    db().execute(
        "INSERT INTO membresias (dni_cliente, id_plan, fecha_inicio, fecha_fin, estado) VALUES (%s,%s,CURRENT_DATE, CURRENT_DATE + (%s || ' days')::interval, 'ACTIVA')",
        [s.dni, s.id_plan, s.dias_duracion],
        )
    # 3) Rostro (opcional)
    if s.embedding:
        db().execute(
            "INSERT INTO rostros (dni_cliente, embedding) VALUES (%s, %s)",
            [s.dni, s.embedding],
        )
    return {"ok": True}


@app.delete("/socios/{dni}")
def delete_socio(dni: int):
    # Borrado en cascada manual (según tus FK actuales)
    db().execute("DELETE FROM accesos WHERE dni_cliente=%s", [dni])
    db().execute("DELETE FROM rostros WHERE dni_cliente=%s", [dni])
    db().execute("DELETE FROM membresias WHERE dni_cliente=%s", [dni])
    db().execute("DELETE FROM socios WHERE dni_cliente=%s", [dni])
    return {"ok": True}