"""
Servicio local de reconocimiento facial.
- Corre en paralelo a la web (FastAPI)
- Cuando reconoce a alguien (similaridad >= TH), envía POST /logs/acceso {dni, estado}

Depende de tu mecanismo para mapear embedding->dni. Aquí asumimos que en la BD
hay al menos un rostro por socio y se busca el más cercano con pgvector.
"""
import os
import cv2
import numpy as np
import requests
import psycopg2
from pgvector.psycopg2 import register_vector
from insightface.app import FaceAnalysis

API_BASE = os.getenv('API_BASE', 'http://127.0.0.1:8000')
DB_NAME = os.getenv('DB_NAME', 'proyectogpi1')
DB_USER = os.getenv('DB_USER', 'admin')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'nariga')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '15432')
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.45'))

# --- Modelo InsightFace ---
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# --- Conexión PG + registro pgvector (clave para evitar numeric[]/ARRAY) ---
conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)
conn.autocommit = True              # activarlo apenas se crea la conexión
register_vector(conn)               # ADAPTADOR: list[float] -> vector

# (Opcional) mejora recall si usás índice IVFFLAT en `embedding`
try:
    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10;")
except Exception:
    pass

cap = cv2.VideoCapture(0)
print('Reconocimiento activo. q para salir')

def to_pgvector_literal(vec) -> str:
    # formato exacto que espera pgvector: [v1,v2,...]
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            # bbox para overlay
            box = face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # --- embedding normalizado (defensivo) ---
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = face.embedding
                n = np.linalg.norm(emb)
                if n > 0:
                    emb = emb / n
            # MUY IMPORTANTE: pasarlo como list[float] para que register_vector lo serialice a 'vector'
            emb_list = np.asarray(emb, dtype=np.float32).tolist()
            
            v = to_pgvector_literal(emb_list)  # sólo para debug
            # --- consulta correcta: SIN ::vector, SIN f-strings, parametrizada ---
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT dni_cliente,
                           1 - (embedding <=> %s::vector) AS sim
                    FROM rostros
                    -- si tenés una columna 'activo', podrías filtrar:
                    -- WHERE activo = TRUE
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                    """,
                    (v, v),
                )
                row = cur.fetchone()

            # Mostrar texto según resultado
            if row:
                dni, sim = row
                if sim >= SIMILARITY_THRESHOLD:
                    # registrar acceso permitido
                    try:
                        requests.post(
                            f"{API_BASE}/logs/acceso",
                            json={"dni": int(dni), "estado": "PERMITIDO"},
                            timeout=2,
                        )
                        text = f"PERMITIDO dni={dni} sim={sim:.2f}"
                        color = (0, 255, 0)
                    except Exception as e:
                        text = f"Log error sim={sim:.2f}"
                        color = (0, 165, 255)
                else:
                    text = f"Desconocido sim={sim:.2f}"
                    color = (0, 0, 255)
            else:
                text = "BD vacía"
                color = (255, 255, 0)

            cv2.putText(
                frame, text, (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

        cv2.imshow('Reconocimiento Facial', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()


