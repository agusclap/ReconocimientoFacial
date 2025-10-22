"""Utilities for working with InsightFace inside the API context."""
from __future__ import annotations

import threading
from typing import List

import cv2
import numpy as np
from insightface.app import FaceAnalysis


_app_lock = threading.Lock()
_face_app: FaceAnalysis | None = None


def _load_face_app() -> FaceAnalysis:
    """Return a shared :class:`FaceAnalysis` instance.

    Loading the InsightFace models is relatively expensive, so we create
    the analyzer lazily the first time we need it and reuse it across
    requests.  The function is thread-safe to avoid double-initialisation
    when running under Uvicorn with multiple workers.
    """

    global _face_app
    if _face_app is None:
        with _app_lock:
            if _face_app is None:
                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=0, det_size=(640, 640))
                _face_app = app
    return _face_app


class FaceNotFoundError(RuntimeError):
    """Raised when a valid face cannot be extracted from an image."""


def embedding_from_image(data: bytes) -> List[float]:
    """Generate a face embedding from the provided image bytes.

    Args:
        data: Raw bytes of the uploaded image.

    Returns:
        A 512-element list of floats with the normalised embedding.

    Raises:
        FaceNotFoundError: If no face is detected in the picture or the
            image cannot be decoded.
    """

    np_img = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame is None:
        raise FaceNotFoundError("No se pudo leer la imagen enviada.")

    app = _load_face_app()
    faces = app.get(frame)
    if not faces:
        raise FaceNotFoundError("No se detectó ningún rostro en la imagen.")

    # Elegimos el rostro con mayor score de detección por si hay varios.
    best_face = max(faces, key=lambda f: f.det_score)
    embedding = best_face.normed_embedding.astype(np.float32)
    if embedding.size != 512:
        raise FaceNotFoundError("El embedding generado no tiene 512 dimensiones.")

    return embedding.tolist()


__all__ = ["embedding_from_image", "FaceNotFoundError"]
