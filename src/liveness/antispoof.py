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

    def __init__(self, model_path: str | Path = "yolov8n.pt") -> None:
        self.model_path = str(model_path)

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
        results = model.predict(frame, imgsz=640, conf=0.35, verbose=False)
        if not results:
            return LivenessResult(
                ok=False,
                person_count=0,
                cellphone_count=0,
                reasons=["No se encontraron detecciones"],
                detections=[],
            )

        prediction = results[0]
        names: Dict[int, str] = prediction.names if hasattr(prediction, "names") else model.names

        person_count = 0
        cellphone_count = 0
        detections: List[Detection] = []

        for box in prediction.boxes:
            cls_tensor = getattr(box, "cls", None)
            if cls_tensor is None:
                continue

            cls_id = int(cls_tensor.item())
            label = names.get(cls_id, str(cls_id))
            conf = float(box.conf.item())
            coords = getattr(box, "xyxy", None)
            if coords is not None:
                if hasattr(coords, "detach"):
                    coords = coords.detach().cpu().numpy()
                else:
                    coords = np.asarray(coords)
                xyxy = coords.ravel().tolist()
            else:
                xyxy = []
            detections.append(Detection(label=label, confidence=conf, box=xyxy))
            if label == "person" and conf >= 0.35:
                person_count += 1
            if label in {"cell phone", "tv", "laptop", "monitor"} and conf >= 0.3:
                cellphone_count += 1

        reasons: List[str] = []
        if person_count == 0:
            reasons.append("No se detectó a ninguna persona en la escena.")
        if person_count > 1:
            reasons.append("Hay más de una persona frente a la cámara.")
        if cellphone_count > 0:
            reasons.append("Se detectó un dispositivo que podría mostrar un rostro impostor.")

        ok = not reasons
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
