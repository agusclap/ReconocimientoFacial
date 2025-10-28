"""
Servicio local de reconocimiento facial.
- Corre en paralelo a la web (FastAPI)
- Cuando reconoce a alguien (similaridad >= TH), envía POST /logs/acceso {dni, estado}

Depende de tu mecanismo para mapear embedding->dni. Aquí asumimos que en la BD
hay al menos un rostro por socio y se busca el más cercano con pgvector.
"""
from contextlib import closing
import os
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import psycopg2
import requests
from insightface.app import FaceAnalysis
from pgvector.psycopg2 import register_vector

from liveness.antispoof import LivenessError, LivenessResult, evaluate_frame

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
DB_NAME = os.getenv("DB_NAME", "gimnasio")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))


def _load_face_app() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _connect_db() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    register_vector(conn)
    conn.autocommit = True
    return conn


def _best_match(
    conn: psycopg2.extensions.connection, embedding: Iterable[float]
) -> Optional[Tuple[int, float]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT dni_cliente, 1-(embedding <=> %s) AS sim
            FROM rostros
            ORDER BY embedding <=> %s
            LIMIT 1
            """,
            (embedding, embedding),
        )
        row = cur.fetchone()
        if not row:
            return None
        dni, similarity = row
        return int(dni), float(similarity)


def _send_access_log(dni: int, similarity: float, estado: str) -> None:
    payload = {"dni": dni, "estado": estado, "similaridad": similarity}
    try:
        requests.post(f"{API_BASE}/logs/acceso", json=payload, timeout=2)
    except Exception as exc:  # pragma: no cover - logging side effect only
        print("Error enviando log:", exc)


def _handle_liveness(frame: np.ndarray) -> Optional[LivenessResult]:
    try:
        return evaluate_frame(frame)
    except LivenessError as err:
        print("Liveness falló:", "; ".join(err.result.reasons))
    except ValueError as err:
        print("Error procesando frame para liveness:", err)
    return None


def main() -> None:
    face_app = _load_face_app()
    with closing(_connect_db()) as conn:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara para reconocimiento facial.")

        print("Reconocimiento activo. Presioná 'q' para salir")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("No se pudo leer un frame de la cámara, se finaliza el servicio.")
                    break

                liveness_result = _handle_liveness(frame)
                if liveness_result is None:
                    continue

                faces = face_app.get(frame)
                if not faces:
                    continue

                for face in faces:
                    embedding = face.normed_embedding.astype(np.float32).tolist()
                    match = _best_match(conn, embedding)
                    if not match:
                        continue

                    dni, similarity = match
                    if similarity < SIMILARITY_THRESHOLD:
                        continue

                    print(
                        f"Ingreso PERMITIDO dni={dni} sim={similarity:.2f}"
                        f" | Liveness: {liveness_result.reasons[-1]}"
                    )
                    _send_access_log(dni, similarity, "PERMITIDO")

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
