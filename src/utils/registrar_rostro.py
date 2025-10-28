import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import psycopg2
import os

# --- CONFIGURACIÓN ---
# Conexión a la base de datos (¡ajusta estos valores!)
DB_CONFIG = {
    "dbname": "proyectogpi1",
    "user": "admin",
    "password": "nariga",
    "host": "localhost",
    "port": "15432"
}

# Inicializar InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Iniciar la cámara
cap = cv2.VideoCapture(0)

# detectar soporte GUI de OpenCV
def _has_gui():
    try:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow("test", np.zeros((2,2,3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow("test")
        return True
    except Exception:
        return False

USE_GUI = _has_gui()
if not USE_GUI:
    print("Aviso: OpenCV sin soporte GUI (imshow no disponible). El script usará modo headless (guardará preview y pedirá confirmación por terminal).")

# Pedir el nombre del usuario
nombre_usuario = input("Ingresa tu nombre para registrarte: ")

print("\nPreparado para registrar. Mira a la cámara y presiona 's' para guardar tu rostro.")
print("Presiona 'q' para salir sin guardar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if USE_GUI:
        # Mostrar instrucciones en la pantalla
        cv2.putText(frame, "Presiona 's' para guardar, 'q' para salir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Registro Facial - Presiona "s" para guardar', frame)
        key = cv2.waitKey(1) & 0xFF
    else:
        # headless: guardar preview y pedir acción por terminal
        preview_path = f"/tmp/registro_preview_{nombre_usuario}.jpg"
        cv2.imwrite(preview_path, frame)
        print(f"Preview guardado en: {preview_path}")
        print("Escribe 's' + Enter para guardar el rostro, 'q' + Enter para salir, o Enter para refrescar preview.")
        choice = input().strip().lower()
        key = ord(choice[0]) if choice else 0

    if key == ord('s'):
        # Detectar rostros en el frame actual
        faces = app.get(frame)
        if len(faces) > 0:
            # Tomamos el primer rostro detectado
            face = faces[0]
            embedding = face.normed_embedding

            try:
                # Conectar a la base de datos
                conn = psycopg2.connect(**DB_CONFIG)
                cursor = conn.cursor()

                # Convertir el embedding (numpy array) a una lista para guardarlo
                embedding_list = embedding.tolist()
                
                # Insertar en la base de datos
                cursor.execute(
                    "INSERT INTO rostros_test (nombre, embedding) VALUES (%s, %s::vector)",
                    (nombre_usuario, embedding_list)
                )
                conn.commit()
                
                print(f"\n¡Éxito! Rostro de '{nombre_usuario}' guardado en la base de datos.")
                
                cursor.close()
                conn.close()

            except Exception as e:
                print(f"Error al conectar o guardar en la base de datos: {e}")
            
            break # Salir del bucle una vez guardado
        else:
            print("No se detectó ningún rostro. Inténtalo de nuevo.")

    elif key == ord('q'):
        print("Registro cancelado.")
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
