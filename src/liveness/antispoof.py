"""Liveness and anti-spoofing helpers leveraging a YOLO detector."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


_MODEL_LOCK = threading.Lock()
_MODEL: YOLO | None = None


def _load_model() -> YOLO:
    """Load and memoise the YOLO model used for liveness validation."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = YOLO("yolov8n.pt")
    return _MODEL


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


def _run_inference(frame: np.ndarray) -> LivenessResult:
    model = _load_model()
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


def evaluate_frame_bytes(data: bytes) -> LivenessResult:
    """Run liveness checks on raw image bytes.

    Args:
        data: Encoded image bytes (JPG/PNG).

    Returns:
        LivenessResult summarising the scene.

    Raises:
        ValueError: If the image cannot be decoded.
        LivenessError: If the liveness verification fails.
    """

    np_img = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("No se pudo decodificar la imagen enviada.")

    return evaluate_frame(frame)


def evaluate_frame(frame: np.ndarray) -> LivenessResult:
    """Run liveness checks on an already decoded frame.

    Args:
        frame: Image in BGR format as provided by OpenCV.

    Returns:
        LivenessResult summarising the scene.

    Raises:
        ValueError: If the frame is not a valid colour image.
        LivenessError: If the liveness verification fails.
    """

    if frame is None:
        raise ValueError("El frame proporcionado es nulo.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("El frame debe ser una imagen en formato BGR.")

    result = _run_inference(frame)
    if not result.ok:
        raise LivenessError(result)
    return result


__all__ = [
    "Detection",
    "LivenessError",
    "LivenessResult",
    "evaluate_frame",
    "evaluate_frame_bytes",
]
