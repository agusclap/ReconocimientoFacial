"""Liveness detection utilities based on a YOLO model."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(slots=True)
class Detection:
    """Simple representation of a YOLO detection."""

    label: str
    confidence: float
    box: List[float]


@dataclass(slots=True)
class LivenessResult:
    """Aggregated information about the analysed frame."""

    ok: bool
    person_count: int
    cellphone_count: int
    reasons: List[str]
    detections: List[Detection]


class LivenessError(RuntimeError):
    """Raised when liveness checks fail for the provided frame."""

    def __init__(self, result: LivenessResult):
        super().__init__("; ".join(result.reasons))
        self.result = result



class LivenessDetector:
    """Evaluate frames to ensure that a real person is in front of the camera."""

    _MODEL_LOCK = threading.Lock()
    _MODEL: YOLO | None = None

    def __init__(self, model_path: str | Path = "yolov11m.pt",
                 conf: float = 0.20, imgsz: int = 960,
                 containment_thresh: float = 0.50) -> None:
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.containment_thresh = float(containment_thresh)

    def _ensure_model(self) -> YOLO:
        model = self.__class__._MODEL
        if model is None:
            with self.__class__._MODEL_LOCK:
                model = self.__class__._MODEL
                if model is None:
                    model = YOLO(self.model_path)
                    self.__class__._MODEL = model
        return model

    def evaluate(self, frame: np.ndarray) -> LivenessResult:
        """Run liveness detection on an OpenCV frame."""
        model = self._ensure_model()
        results = model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        if not results:
            return LivenessResult(ok=False, person_count=0, cellphone_count=0,
                                  reasons=["No se encontraron detecciones"], detections=[])

        prediction = results[0]
        names: Dict[int, str] = prediction.names if hasattr(prediction, "names") else model.names

        person_boxes, device_boxes = [], []
        detections: List[Detection] = []

        # Normalizar etiquetas
        device_labels = {"cell phone", "cellphone", "mobile phone", "phone", "tv", "laptop", "monitor"}

        for box in prediction.boxes:
            cls_tensor = getattr(box, "cls", None)
            if cls_tensor is None:
                continue

            cls_id = int(cls_tensor.item())
            raw_label = names.get(cls_id, str(cls_id))
            label = raw_label.lower()
            conf = float(box.conf.item())

            coords = getattr(box, "xyxy", None)
            xyxy = []
            if coords is not None:
                if hasattr(coords, "detach"):
                    coords = coords.detach().cpu().numpy()
                else:
                    coords = np.asarray(coords)
                xyxy = coords.ravel().tolist()

            detections.append(Detection(label=raw_label, confidence=conf, box=xyxy))

            if label == "person" and conf >= self.conf:
                person_boxes.append(xyxy)
            if label in device_labels and conf >= self.conf * 0.9:  # levemente más permisivo para devices
                device_boxes.append(xyxy)

        person_count = len(person_boxes)
        cellphone_count = len(device_boxes)

        reasons: List[str] = []
        if person_count == 0:
            reasons.append("No se detectó a ninguna persona en la escena.")
        if person_count > 1:
            reasons.append("Hay más de una persona frente a la cámara.")

        # --- Regla robusta de solapamiento (bidireccional) ---
        def area(b):
            return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

        def inter(a, b):
            xA, yA = max(a[0], b[0]), max(a[1], b[1])
            xB, yB = min(a[2], b[2]), min(a[3], b[3])
            return max(0.0, xB-xA) * max(0.0, yB-yA)

        spoof = False
        if person_boxes and device_boxes:
            for pb in person_boxes:
                Ap = area(pb) or 1e-6
                for db in device_boxes:
                    Ad = area(db) or 1e-6
                    Ai = inter(pb, db)
                    # Contención en cualquiera de los dos sentidos
                    ratio_phone_in_person = Ai / Ad
                    ratio_person_in_phone = Ai / Ap
                    if (ratio_phone_in_person >= self.containment_thresh) or \
                       (ratio_person_in_phone >= self.containment_thresh):
                        spoof = True
                        break
                if spoof:
                    break

        if cellphone_count > 0 and spoof:
            reasons.append("Se detectó un dispositivo superpuesto al rostro (posible spoof).")
        elif cellphone_count > 0:
            # si hay device pero sin superposición fuerte, aún así avisamos
            reasons.append("Se detectó un dispositivo (posible fuente de spoof).")

        ok = (len(reasons) == 0)
        if ok:
            reasons.append("Escena verificada correctamente.")

        return LivenessResult(
            ok=ok,
            person_count=person_count,
            cellphone_count=cellphone_count,
            reasons=reasons,
            detections=detections,
        )


    def evaluate_bytes(self, data: bytes) -> LivenessResult:
        """Run liveness checks on raw image bytes."""

        np_img = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("No se pudo decodificar la imagen enviada.")

        result = self.evaluate(frame)
        if not result.ok:
            raise LivenessError(result)
        return result


def summarise_reasons(result: LivenessResult) -> str:
    """Render the reasons returned by :class:`LivenessResult` as text."""

    return " | ".join(result.reasons)


__all__ = [
    "Detection",
    "LivenessDetector",
    "LivenessError",
    "LivenessResult",
    "summarise_reasons",
]
