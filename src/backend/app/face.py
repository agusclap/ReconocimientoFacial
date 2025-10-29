"""Utilities for working with InsightFace inside the API context."""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import BinaryIO, List, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis


_app_lock = threading.Lock()
_face_app: Optional[FaceAnalysis] = None


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


def _embedding_from_frame(frame: np.ndarray) -> np.ndarray:
    """Return the normalised embedding for the provided frame."""

    if frame is None:
        raise FaceNotFoundError("No se pudo leer la imagen enviada.")

    app = _load_face_app()
    faces = app.get(frame)
    if not faces:
        raise FaceNotFoundError("No se detectó ningún rostro en la imagen.")

    best_face = max(faces, key=lambda f: f.det_score)
    embedding = best_face.normed_embedding.astype(np.float32)
    if embedding.size != 512:
        raise FaceNotFoundError("El embedding generado no tiene 512 dimensiones.")

    return embedding


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

    embedding = _embedding_from_frame(frame)
    return embedding.tolist()


_ALLOWED_VIDEO_SUFFIXES = {".mp4", ".webm"}


def embedding_from_video(
    file_obj: BinaryIO,
    capture_frames: int = 15,
    filename: Optional[str] = None,
) -> List[float]:
    """Generate an InsightFace embedding from a video file.

    The function extracts up to ``capture_frames`` frames from the provided
    video, computes the facial embedding for each valid detection and
    averages the results. Temporary snapshots generated during the process
    are stored on disk and removed automatically once the embedding is
    produced.

    Args:
        file_obj: File-like object pointing to the uploaded video.
        capture_frames: Number of frames to sample from the video.

    Returns:
        A 512-element list containing the normalised embedding.

    Raises:
        FaceNotFoundError: If no valid face can be detected in the video or
            the file cannot be processed.
    """

    if capture_frames <= 0:
        raise ValueError("capture_frames must be a positive integer")

    suffix_source = filename if filename else getattr(file_obj, "name", "video.mp4")
    suffix = Path(suffix_source).suffix.lower() or ".mp4"
    if suffix not in _ALLOWED_VIDEO_SUFFIXES:
        raise FaceNotFoundError(
            "Formato de video no soportado. Subí un archivo MP4 o WebM."
        )
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_video:
        file_obj.seek(0)
        shutil.copyfileobj(file_obj, tmp_video)
        temp_video_path = Path(tmp_video.name)

    embeddings: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(temp_video_path))
    try:
        if not cap.isOpened():
            raise FaceNotFoundError("No se pudo procesar el video recibido.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, total_frames // capture_frames) if total_frames else 1

        frame_index = 0
        captured = 0
        with tempfile.TemporaryDirectory(prefix="frames_", dir=None) as frames_dir:
            frames_path = Path(frames_dir)
            while captured < capture_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % step == 0:
                    snapshot_path = frames_path / f"frame_{captured:03d}.jpg"
                    cv2.imwrite(str(snapshot_path), frame)
                    try:
                        embeddings.append(_embedding_from_frame(frame))
                    except FaceNotFoundError:
                        # Saltar fotogramas sin rostro sin interrumpir el proceso.
                        pass
                    captured += 1

                frame_index += 1

    finally:
        cap.release()
        if temp_video_path.exists():
            os.remove(temp_video_path)

    if not embeddings:
        raise FaceNotFoundError("No se detectó ningún rostro en el video.")

    averaged = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(averaged)
    if not np.isfinite(norm) or norm == 0:
        raise FaceNotFoundError("El embedding calculado es inválido.")

    normalised = (averaged / norm).astype(np.float32)
    return normalised.tolist()


__all__ = ["embedding_from_image", "embedding_from_video", "FaceNotFoundError"]
