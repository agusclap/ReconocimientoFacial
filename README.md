# 🧠 Sistema de Acceso por Reconocimiento Facial

Proyecto académico desarrollado como parte de la materia **Gestión de Proyectos Informáticos**, cuyo objetivo es implementar un sistema de **control de acceso para gimnasios** mediante reconocimiento facial en tiempo real, eliminando el uso de tarjetas o huellas.

---

## 📋 Objetivo General

Desarrollar un sistema que valide la entrada de socios registrados mediante **detección y reconocimiento facial en tiempo real**, utilizando inteligencia artificial y visión por computadora, simulando la apertura de una puerta o actuador.

---

## 🎯 Objetivos Específicos

1. Implementar un pipeline de **detección facial** con **YOLOv11**.  
2. Integrar **InsightFace** para la generación y comparación de embeddings faciales.  
3. Incluir un módulo de **registro facial** para nuevos socios.  
4. Incorporar un módulo de **verificación de liveness** (anti-spoofing) para evitar suplantaciones.  
5. Diseñar una **interfaz de acceso y panel administrador** (socios, logs, métricas).  
6. Documentar todo el proceso técnico y de gestión del proyecto.

---

## 🧩 Alcance

**Incluye:**
- Detección y verificación facial en imágenes o video.
- Registro de socios y validación en la entrada principal.
- Simulación de apertura mediante relé o LED.

**Fuera de alcance:**
- Integración con sistemas de turnos o pagos.
- Múltiples sucursales o sincronización entre sedes.

---

## 🧠 Metodología

Metodología **Ágil** con sprints semanales e iteraciones cortas.

- **Líder de Proyecto:** Eduardo Friesen  
- **Desarrolladores:** Agustín Rodeyro, Bruno Garibaldi  
- **Tester / Documentador:** Bautista Macedo  

Gestión en **Trello** y control de versiones en **GitHub**.

---

## 🗓️ Cronograma General

| Fase | Descripción | Duración |
|------|--------------|-----------|
| Fase 1 | Inicio y Planificación | Semanas 1–2 |
| Fase 2 | Diseño y Preparación | Semanas 3–4 |
| Fase 3 | Desarrollo | Semanas 5–9 |
| Fase 4 | Pruebas y Optimización | Semanas 10–11 |
| Fase 5 | Cierre y Entrega | Semana 12 |

---

## ⚙️ Tecnologías y Herramientas

### 🧠 Software Principal
- **Python 3.10+**
- **Ultralytics YOLOv11**
- **InsightFace**
- **OpenCV**
- **Torch / ONNX Runtime**
- **FastAPI / Flask** (para servidor de inferencia)
- **Figma** (UI)
- **MySQL + pgvector** (almacenamiento de embeddings)

### 🧰 Gestión
- **GitHub / GitLab** → control de versiones  
- **Trello / Jira** → planificación y tareas  
- **Google Docs** → documentación técnica  

---

## 🖥️ Estructura del Proyecto

facial-access/
│
├── src/
│ ├── detection/ # Módulo YOLOv11
│ ├── embeddings/ # Módulo InsightFace
│ ├── liveness/ # Módulo anti-spoofing
│ ├── ui/ # Interfaz de acceso / panel admin
│ ├── utils/ # Funciones auxiliares
│ └── main.py # Punto de entrada principal
│
├── data/ # Datasets y registros
├── tests/ # Pruebas unitarias e integración
├── docs/ # Documentación y reportes
├── requirements.txt # Dependencias del entorno
└── README.md


---

## 🧰 Instalación y Entorno

### 🔹 Requisitos del sistema
- Ubuntu / Windows 10+
- Python 3.10+
- (Opcional) GPU NVIDIA con CUDA instalado

### 🔹 Pasos de instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu_usuario>/<repo>.git
cd <repo>

# 2. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Ejecutar prueba
python src/main.py
