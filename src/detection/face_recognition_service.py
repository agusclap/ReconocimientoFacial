"""
Servicio local de reconocimiento facial con liveness.
- Primero pasa liveness (YOLO/antispoof).
- Si ok, compara embeddings (pgvector).
- Si sim >= TH, envía POST /logs/acceso {dni, estado="PERMITIDO"}.
"""

import sys, os
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
# Config por variables de entorno (con defaults "más permisivos" para pruebas)
# -------------------------
API_BASE = os.getenv('API_BASE', 'http://127.0.0.1:8000')

DB_NAME = os.getenv('DB_NAME', 'proyectogpi1')
DB_USER = os.getenv('DB_USER', 'admin')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'nariga')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '15432')

# Umbral de similitud (más bajo = más permisivo)
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.38'))

# Modelo de liveness
LIVENESS_MODEL = os.getenv(
    'LIVENESS_MODEL',
    os.path.join(os.path.dirname(__file__), '..', '..', 'yolov8n.pt')
)
LIVENESS_CONF = float(os.getenv("LIVENESS_CONF", "0.12"))   # antes 0.20
LIVENESS_IMGSZ = int(os.getenv("LIVENESS_IMGSZ", "960"))
LIVENESS_OVERLAP = float(os.getenv("LIVENESS_OVERLAP", "0.50"))
LIVENESS_OK_STREAK = int(os.getenv('LIVENESS_OK_STREAK', '1'))  # antes 2

# Bloqueo por “device” (celular/monitor) solapado con la cara
DISABLE_DEVICE_BLOCK = os.getenv("DISABLE_DEVICE_BLOCK", "1") == "1"  # por defecto desactivado para pruebas
DEVICE_MIN_CONF = float(os.getenv("DEVICE_MIN_CONF", "0.40"))         # confianza mínima para considerar device
OVERLAP_FACE = float(os.getenv("OVERLAP_FACE", "0.45"))               # % de cara solapada para bloquear
OVERLAP_DEV  = float(os.getenv("OVERLAP_DEV",  "0.70"))               # % del device solapado para bloquear

# InsightFace det_size (más grande = más preciso, más consumo)
INSIGHT_DET_W = int(os.getenv("INSIGHT_DET_W", "640"))
INSIGHT_DET_H = int(os.getenv("INSIGHT_DET_H", "640"))

# Dispositivos que consideramos “riesgo de spoof”
DEVICE_LABELS = {"cell phone", "cellphone", "mobile phone", "phone", "tv", "laptop", "monitor"}

# -------------------------
# InsightFace
# -------------------------
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(INSIGHT_DET_W, INSIGHT_DET_H))

# -------------------------
# Liveness (YOLO)
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
try:
    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10;")
except Exception:
    pass

# -------------------------
# Utilidades
# -------------------------
def to_pgvector_literal(vec) -> str:
    """Convierte embedding a literal pgvector: [v1,v2,...]"""
    arr = np.asarray(vec, dtype=np.float32).ravel()
    return "[" + ",".join(f"{float(x):.6f}" for x in arr) + "]"

def area(b):
    return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

def inter(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, xB-xA) * max(0.0, yB-yA)

# -------------------------
# Loop principal
# -------------------------
cap = cv2.VideoCapture(0)
print('Reconocimiento activo. Presioná "q" para salir')
print(f'[conf] SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD}  LIVENESS_CONF={LIVENESS_CONF}  OK_STREAK={LIVENESS_OK_STREAK}')
print(f'[conf] DEVICE_BLOCK={"OFF" if DISABLE_DEVICE_BLOCK else "ON"} min_conf={DEVICE_MIN_CONF} overlap(face/dev)={OVERLAP_FACE}/{OVERLAP_DEV}')
print(f'[conf] INSIGHT det_size=({INSIGHT_DET_W},{INSIGHT_DET_H})')

liveness_streak = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[warn] No se pudo leer el frame de la cámara")
            break

        # 1) Liveness
        try:
            live = liveness.evaluate(frame)

            # Boxes de dispositivos detectados por YOLO
            device_boxes = [
                d.box for d in live.detections
                if d.label.lower() in DEVICE_LABELS and d.confidence >= DEVICE_MIN_CONF
            ]

            # Bloqueo por device (si está habilitado)
            if not DISABLE_DEVICE_BLOCK and device_boxes:
                # Mensaje info
                cv2.putText(frame, "Device detectado", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # Mensaje general de liveness
            cv2.putText(
                frame,
                summarise_reasons(live)[:70],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0) if live.ok else (0, 0, 255),
                2,
            )

            if live.ok:
                liveness_streak = min(LIVENESS_OK_STREAK, liveness_streak + 1)
            else:
                liveness_streak = 0

        except LivenessError as e:
            liveness_streak = 0
            cv2.putText(frame, str(e), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Si el liveness no juntó los frames mínimos, no reconocemos aún
        if liveness_streak < LIVENESS_OK_STREAK:
            cv2.imshow('Reconocimiento Facial', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 2) Reconocimiento facial (solo si liveness OK)
        faces = app.get(frame)

        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # --- Cruce cara vs. device: si hay gran solapamiento y el bloqueo está activo, DENEGAR ---
            overlap_fail = False
            if not DISABLE_DEVICE_BLOCK:
                face_box = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                A_face = area(face_box) or 1e-6
                for db in device_boxes:
                    A_dev = area(db) or 1e-6
                    Ai = inter(face_box, db)
                    # ahora exigimos AMBAS condiciones → menos falsos positivos
                    if (Ai / A_face) >= OVERLAP_FACE and (Ai / A_dev) >= OVERLAP_DEV:
                        overlap_fail = True
                        break

            if overlap_fail:
                cv2.putText(frame, "SPOOF: dispositivo sobre el rostro",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # No consultamos DB para este rostro
                continue

            # --- Si pasó liveness + cruce, recién ahora comparamos embeddings ---
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = face.embedding
                n = np.linalg.norm(emb)
                if n > 0:
                    emb = emb / n

            v = to_pgvector_literal(emb)  # forzamos ::vector

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

            if row:
                dni, sim = row
                # Debug acotado
                cv2.putText(frame, f"sim={sim:.2f} thr={SIMILARITY_THRESHOLD:.2f}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

                if sim >= SIMILARITY_THRESHOLD:
                    try:
                        requests.post(
                            f"{API_BASE}/logs/acceso",
                            json={"dni": int(dni), "estado": "PERMITIDO"},
                            timeout=2,
                        )
                        text, color = f"PERMITIDO dni={dni} sim={sim:.2f}", (0, 255, 0)
                    except Exception:
                        text, color = f"Log error sim={sim:.2f}", (0, 165, 255)
                else:
                    text, color = f"Desconocido sim={sim:.2f}", (0, 0, 255)
            else:
                text, color = "Sin match / BD vacía", (255, 255, 0)

            cv2.putText(frame, text, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Reconocimiento Facial', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
