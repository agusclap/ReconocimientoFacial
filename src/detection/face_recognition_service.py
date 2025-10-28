"""
Servicio local de reconocimiento facial.
- Corre en paralelo a la web (FastAPI)
- Cuando reconoce a alguien (similaridad >= TH), envía POST /logs/acceso {dni, estado}

Depende de tu mecanismo para mapear embedding->dni. Aquí asumimos que en la BD
hay al menos un rostro por socio y se busca el más cercano con pgvector.
"""
import os
import time
import json
import cv2
import numpy as np
import requests
import psycopg2
from pgvector.psycopg2 import register_vector
from insightface.app import FaceAnalysis

API_BASE = os.getenv('API_BASE', 'http://127.0.0.1:8000')
DB_NAME = os.getenv('DB_NAME', 'gimnasio')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD','0.45'))

# Modelo InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640,640))

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
register_vector(conn)
conn.autocommit=True

cap = cv2.VideoCapture(0)
print('Reconocimiento activo. q para salir')

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        for face in faces:
            emb = face.normed_embedding.astype(np.float32).tolist()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT dni_cliente, 1-(embedding <=> %s) AS sim
                    FROM rostros
                    ORDER BY embedding <=> %s
                    LIMIT 1
                """, (emb, emb))
                row = cur.fetchone()
                if row:
                    dni, sim = row
                    if sim >= SIMILARITY_THRESHOLD:
                        # registrar acceso permitido
                        try:
                            requests.post(f"{API_BASE}/logs/acceso", json={"dni": int(dni), "estado": "PERMITIDO"}, timeout=2)
                            print(f"Ingreso PERMITIDO dni={dni} sim={sim:.2f}")
                        except Exception as e:
                            print('Error enviando log:', e)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
