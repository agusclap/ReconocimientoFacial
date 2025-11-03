"""
Servicio local de reconocimiento facial con liveness.
- Primero pasa liveness (YOLO/antispoof).
- Si ok, compara embeddings (pgvector).
- Si sim >= TH, envía POST /logs/acceso {dni, estado="PERMITIDO"}.
"""

import os
import cv2
import numpy as np
import requests
import psycopg2
from pgvector.psycopg2 import register_vector
from insightface.app import FaceAnalysis
from liveness.antispoof import LivenessDetector, LivenessError, summarise_reasons

API_BASE = os.getenv('API_BASE', 'http://127.0.0.1:8000')
DB_NAME = os.getenv('DB_NAME', 'proyectogpi1')
DB_USER = os.getenv('DB_USER', 'admin')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'nariga')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '15432')

SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.45'))
# yolov8n.pt en la raíz del proyecto por defecto
LIVENESS_MODEL = os.getenv(
    'LIVENESS_MODEL',
    os.path.join(os.path.dirname(__file__), '..', '..', 'yolov8n.pt')
)

# Dispositivos que consideramos “riesgo de spoof”
DEVICE_LABELS = {"cell phone", "cellphone", "mobile phone", "phone", "tv", "laptop", "monitor"}

# --- InsightFace ---
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# --- Liveness (YOLO) ---
liveness = LivenessDetector(
    model_path=LIVENESS_MODEL,
    conf=float(os.getenv("LIVENESS_CONF", "0.20")),
    imgsz=int(os.getenv("LIVENESS_IMGSZ", "960")),
    containment_thresh=float(os.getenv("LIVENESS_OVERLAP", "0.50")),
)

# --- Postgres + pgvector ---
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

cap = cv2.VideoCapture(0)
print('Reconocimiento activo. q para salir')

# pedir varios frames seguidos OK de liveness antes de reconocer
LIVENESS_OK_STREAK = int(os.getenv('LIVENESS_OK_STREAK', '2'))
liveness_streak = 0

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

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) Liveness
        try:
            live = liveness.evaluate(frame)

            # --- REGLA ESTRICTA: si hay un device, bloqueamos ---
            has_device = any(d.label.lower() in DEVICE_LABELS and d.confidence >= 0.15
                             for d in live.detections)
            if has_device:
                live.ok = False
                cv2.putText(frame, "DISPOSITIVO DETECTADO: BLOQUEADO",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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

        # Si el liveness no juntó los frames mínimos, no reconocemos
        if liveness_streak < LIVENESS_OK_STREAK:
            cv2.imshow('Reconocimiento Facial', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 2) Reconocimiento facial (solo si liveness OK)
        faces = app.get(frame)

        # Boxes de dispositivos detectados por YOLO (para cruzar con la cara)
        device_boxes = [d.box for d in live.detections
                        if d.label.lower() in DEVICE_LABELS and d.confidence >= 0.15]

        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # --- Cruce cara vs. device: si hay gran solapamiento, bloquear este rostro ---
            face_box = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            A_face = area(face_box) or 1e-6
            overlap_fail = False
            for db in device_boxes:
                A_dev = area(db) or 1e-6
                Ai = inter(face_box, db)
                # 30% de la cara o 50% del device → considerar spoof
                if (Ai / A_face) >= 0.30 or (Ai / A_dev) >= 0.50:
                    overlap_fail = True
                    break

            if overlap_fail:
                cv2.putText(frame, "SPOOF: DISPOSITIVO SOBRE EL ROSTRO",
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
