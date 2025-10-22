# Gimnasio WebApp (Frontend + Backend + Reconocimiento)

## Requisitos
- Python 3.9+
- PostgreSQL 14+ con extensión `vector`

## 1) Base de datos

```bash
psql -h localhost -U postgres -d gimnasio -f backend/schema.sql
```

## 2) Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
El backend quedará en `http://127.0.0.1:8000`.

## 3) Frontend
Abrí `frontend/index.html` en el navegador (o servilo con cualquier server estático).

## 4) Servicio de reconocimiento (opcional)

```bash
cd reconocimiento
pip install insightface onnxruntime opencv-python psycopg2-binary requests pgvector
python face_recognition_service.py
```
Cuando reconozca a alguien, enviará logs al backend y se verán en `Logs`.

## Variables de entorno útiles
```
DB_NAME=gimnasio
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
SIMILARITY_THRESHOLD=0.45
API_BASE=http://127.0.0.1:8000
```

## Notas
- La pantalla de “Ingreso autorizado” **no** aparece en la web (según tu preferencia). El reconocimiento solo actualizará logs.
- El registro permite opcionalmente pegar un embedding (512 valores). Si lo dejás vacío, el servicio de reconocimiento podrá cargarlo más tarde.
