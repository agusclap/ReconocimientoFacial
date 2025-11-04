"""
Servicio local de reconocimiento facial con liveness.
- Primero pasa liveness (YOLO/antispoof).
- Si ok, compara embeddings (pgvector).
- Si sim >= TH, envía POST /logs/acceso {dni, estado="PERMITIDO"}.
"""

import sys, os, time
from pathlib import Path

# --- asegurar que /src esté en sys.path para importar paquetes locales (p.ej. liveness) ---
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent  # /src
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

import cv2
import numpy as np
import requests
import psycopg2
from pgvector.psycopg2 import register_vector
from insightface.app import FaceAnalysis
from liveness.antispoof import LivenessDetector, LivenessError, summarise_reasons

# -------------------------
# Config
# -------------------------
API_BASE = os.getenv('API_BASE', 'http://127.0.0.1:8000')

DB_NAME = os.getenv('DB_NAME', 'proyectogpi1')
DB_USER = os.getenv('DB_USER', 'admin')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'nariga')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '15432')

SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.38'))

LIVENESS_MODEL = os.getenv(
    'LIVENESS_MODEL',
    os.path.join(os.path.dirname(__file__), '..', '..', 'yolov8n.pt')
)
LIVENESS_CONF = float(os.getenv("LIVENESS_CONF", "0.12"))
LIVENESS_IMGSZ = int(os.getenv("LIVENESS_IMGSZ", "960"))
LIVENESS_OVERLAP = float(os.getenv("LIVENESS_OVERLAP", "0.50"))
LIVENESS_OK_STREAK = int(os.getenv('LIVENESS_OK_STREAK', '1'))

INSIGHT_DET_W = int(os.getenv("INSIGHT_DET_W", "640"))
INSIGHT_DET_H = int(os.getenv("INSIGHT_DET_H", "640"))

# -------------------------
# COOL DOWN POR PERSONA
# -------------------------
COOLDOWN_PER_DNI = 30  # segundos
last_detected = {}      # dni → timestamp del último log

# -------------------------
# InsightFace
# -------------------------
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(INSIGHT_DET_W, INSIGHT_DET_H))

# -------------------------
# Liveness
# -------------------------
liveness = LivenessDetector(
    model_path=LIVENESS_MODEL,
    conf=LIVENESS_CONF,
    imgsz=LIVENESS_IMGSZ,
    containment_thresh=LIVENESS_OVERLAP,
)

# -------------------------
# Postgres + pgvector
# -------------------------
conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)
conn.autocommit = True
register_vector(conn)

# -------------------------
# Utils
# -------------------------
def to_pgvector_literal(vec) -> str:
    arr = np.asarray(vec, dtype=np.float32).ravel()
    return "[" + ",".join(f"{float(x):.6f}" for x in arr) + "]"

# -------------------------
# Loop principal
# -------------------------
cap = cv2.VideoCapture(0)
print('Reconocimiento activo. Presioná "q" para salir\n')

liveness_streak = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[warn] No se pudo leer el frame de la cámara")
            continue

        # --- Liveness ---
        try:
            live = liveness.evaluate(frame)
            liveness_streak = liveness_streak + 1 if live.ok else 0
        except LivenessError:
            liveness_streak = 0

        if liveness_streak < LIVENESS_OK_STREAK:
            cv2.imshow('Reconocimiento Facial', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # --- Reconocimiento ---
        faces = app.get(frame)
        now = time.time()

        for face in faces:
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = face.embedding
                n = np.linalg.norm(emb)
                if n > 0:
                    emb = emb / n

            v = to_pgvector_literal(emb)

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT dni_cliente,
                           1 - (embedding <=> %s::vector) AS sim
                    FROM rostros
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                    """,
                    (v, v),
                )
                row = cur.fetchone()

            if not row:
                continue

            dni, sim = row

            if sim >= SIMILARITY_THRESHOLD:
                # ---- ACA SE APLICA EL COOLDOWN POR PERSONA ----
                if dni not in last_detected or (now - last_detected[dni] >= COOLDOWN_PER_DNI):
                    try:
                        requests.post(
                            f"{API_BASE}/logs/acceso",
                            json={"dni": int(dni), "estado": "PERMITIDO"},
                            timeout=2,
                        )
                        print(f"[LOG] ✅ PERMITIDO {dni} (sim={sim:.2f})")
                    except:
                        print("[LOG ERROR] No se pudo enviar")

                    last_detected[dni] = now  # guardamos el tiempo
                else:
                    print(f"[cooldown] ⏱ {dni} todavía en cooldown ({int(COOLDOWN_PER_DNI - (now - last_detected[dni]))}s restantes)")

        cv2.imshow('Reconocimiento Facial', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
