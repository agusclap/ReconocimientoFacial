from typing import Mapping
import cv2
from matplotlib.pylab import Any
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import psycopg2

# --- CONFIGURACIÓN ---
DB_CONFIG: Mapping[str, Any] = {
    "dbname": "proyectogpi1",
    "user": "admin",
    "password": "nariga",
    "host": "localhost",
    "port": "15432"
}
# Umbral de similitud: Si es mayor, se considera la misma persona.
# Puedes ajustarlo. Un valor más alto es más estricto.
SIMILARITY_THRESHOLD = 0.4

# Inicializar InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Iniciar la cámara
cap = cv2.VideoCapture(0)
print("\nIniciando reconocimiento... Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    
    for face in faces:
        # Dibujar un recuadro alrededor del rostro detectado
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Obtener el embedding del rostro detectado
        current_embedding = face.normed_embedding
        
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # La búsqueda mágica con pgvector
            # El operador '<=>' calcula la distancia del coseno (0 = idénticos, 2 = opuestos)
            # 1 - distancia = similitud (1 = idénticos, -1 = opuestos)
            cursor.execute(
                """
                SELECT nombre, 1 - (embedding <=> %s::vector) AS similarity
                FROM rostros_test
                ORDER BY embedding <=> %s::vector
                LIMIT 1;
                """,
                (current_embedding.tolist(), current_embedding.tolist())
            )
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()

            if result:
                nombre, similarity = result
                if similarity > SIMILARITY_THRESHOLD:
                    texto_a_mostrar = f"{nombre} ({similarity:.2f})"
                    color = (0, 0, 255) # Verde para reconocido
                else:
                    texto_a_mostrar = f"Desconocido ({similarity:.2f})"
                    color = (0, 0, 255) # Rojo para desconocido
            else:
                texto_a_mostrar = "BD Vacia"
                color = (255, 255, 0) # Azul si no hay nadie registrado
            
            cv2.putText(frame, texto_a_mostrar, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        except Exception as e:
            print(f"Error en la base de datos: {e}")

    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
