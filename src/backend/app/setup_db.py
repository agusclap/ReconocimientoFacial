# src/backend/app/setup_db.py
from .database import db

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS socios (
    dni_cliente INTEGER PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    apellido VARCHAR(100) NOT NULL,
    fecha_nacimiento DATE NOT NULL,
    telefono VARCHAR(50),
    mail VARCHAR(255) UNIQUE
);

CREATE TABLE IF NOT EXISTS planes (
    id_plan SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion TEXT,
    precio DECIMAL(10,2) NOT NULL
);

CREATE TABLE IF NOT EXISTS membresias (
    id_membresia SERIAL PRIMARY KEY,
    dni_cliente INTEGER NOT NULL,
    id_plan INTEGER NOT NULL,
    fecha_inicio DATE NOT NULL,
    fecha_fin DATE,
    estado VARCHAR(20) DEFAULT 'ACTIVA',
    CONSTRAINT fk_membresia_socio FOREIGN KEY (dni_cliente) REFERENCES socios(dni_cliente),
    CONSTRAINT fk_membresia_plan  FOREIGN KEY (id_plan) REFERENCES planes(id_plan)
);

CREATE TABLE IF NOT EXISTS rostros (
    id SERIAL PRIMARY KEY,
    dni_cliente INTEGER NOT NULL,
    embedding VECTOR(512) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT fk_rostro_socio FOREIGN KEY (dni_cliente) REFERENCES socios(dni_cliente)
);

CREATE TABLE IF NOT EXISTS accesos (
    id SERIAL PRIMARY KEY,
    dni_cliente INTEGER NOT NULL,
    estado VARCHAR(20) NOT NULL,
    fecha TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_accesos_fecha ON accesos(fecha DESC);
CREATE INDEX IF NOT EXISTS idx_accesos_dni   ON accesos(dni_cliente);
"""

SEED_PLANES = """
INSERT INTO planes (nombre, descripcion, precio) VALUES
('Plan Mensual', 'Acceso ilimitado por 30 días', 15000.00),
('Plan Trimestral', 'Acceso ilimitado por 90 días', 40000.00),
('Plan Anual', 'Acceso ilimitado por 365 días', 120000.00)
ON CONFLICT DO NOTHING;
"""

def run_setup():
    # Ejecuta DDL y semillas idempotentes
    db().execute(DDL)
    db().execute(SEED_PLANES)
    print("Database setup completed.")