"""Console workflow to enrol new members."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import cv2
from ..embeddings import EmbeddingError, FaceEmbedder
from ..liveness import LivenessDetector, summarise_reasons


@dataclass(slots=True)
class EnrollmentSample:
    """Capture produced during the enrolment process."""

    timestamp: float
    embedding: List[float]


@dataclass(slots=True)
class EnrollmentRecord:
    """Serialisable representation of a member."""

    dni: str
    nombre: str
    samples: Sequence[EnrollmentSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dni": self.dni,
            "nombre": self.nombre,
            "samples": [
                {
                    "timestamp": sample.timestamp,
                    "embedding": sample.embedding,
                }
                for sample in self.samples
            ],
        }


def _load_existing(storage_path: Path) -> List[dict]:
    if not storage_path.exists():
        return []
    with storage_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_records(storage_path: Path, records: Sequence[dict]) -> None:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    with storage_path.open("w", encoding="utf-8") as fh:
        json.dump(list(records), fh, ensure_ascii=False, indent=2)


def register_member_cli(
    storage_path: str | Path = "data/enrollments.json",
    camera_index: int = 0,
    samples_target: int = 3,
) -> None:
    """Interactive capture flow to store embeddings for a new member."""

    storage = Path(storage_path)
    dni = input("DNI del socio: ").strip()
    nombre = input("Nombre del socio: ").strip()
    if not dni or not nombre:
        print("DNI y nombre son obligatorios.")
        return

    existing = _load_existing(storage)
    if any(record.get("dni") == dni for record in existing):
        print("Ya existe un socio registrado con ese DNI.")
        return

    print("\nPreparando captura de rostro. Presiona 'q' para cancelar.")
    detector = LivenessDetector()
    embedder = FaceEmbedder()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    samples: List[EnrollmentSample] = []

    try:
        while len(samples) < samples_target:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame de la cámara.")
                break

            result = detector.evaluate(frame)
            if not result.ok:
                print(f"Liveness falló: {summarise_reasons(result)}")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Inscripción cancelada por el usuario.")
                    return
                continue

            try:
                embedding = embedder.extract(frame)
            except EmbeddingError as err:
                print(f"No se pudo obtener el embedding: {err}")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Inscripción cancelada por el usuario.")
                    return
                continue

            samples.append(EnrollmentSample(timestamp=time.time(), embedding=embedding.as_list()))
            print(f"Muestra {len(samples)}/{samples_target} capturada correctamente.")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Inscripción cancelada por el usuario.")
                return

        if len(samples) < samples_target:
            print("No se alcanzó la cantidad mínima de muestras. Intenta nuevamente.")
            return

        record = EnrollmentRecord(dni=dni, nombre=nombre, samples=samples)
        existing.append(record.to_dict())
        _save_records(storage, existing)
        print(f"Socio {nombre} ({dni}) registrado con {len(samples)} muestras.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


__all__ = ["register_member_cli"]
