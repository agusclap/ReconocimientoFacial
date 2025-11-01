"""Wrapper around InsightFace to obtain embeddings from frames."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np
from insightface.app import FaceAnalysis


@dataclass(slots=True)
class EmbeddingResult:
    """Container for a single facial embedding."""

    embedding: np.ndarray
    bbox: Sequence[float]
    detection_score: float

    def as_list(self) -> List[float]:
        """Return the embedding as a Python list for serialisation."""

        return self.embedding.astype(np.float32).tolist()


class EmbeddingError(RuntimeError):
    """Raised when no valid face can be extracted from the frame."""


class FaceEmbedder:
    """Compute embeddings using an InsightFace model."""

    def __init__(
        self,
        providers: Sequence[str] | None = None,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self._app = FaceAnalysis(name="buffalo_l", providers=list(providers or ["CPUExecutionProvider"]))
        self._app.prepare(ctx_id=0, det_size=det_size)

    def extract(self, frame: np.ndarray) -> EmbeddingResult:
        """Return the embedding of the most prominent face in ``frame``."""

        faces = self._app.get(frame)
        if not faces:
            raise EmbeddingError("No se detectaron rostros en la imagen.")

        # Take the face with the highest detection score.
        best_face = max(faces, key=lambda face: getattr(face, "det_score", 0.0))
        embedding = getattr(best_face, "normed_embedding", None)
        if embedding is None:
            raise EmbeddingError("No fue posible obtener el embedding del rostro.")

        bbox = getattr(best_face, "bbox", [])
        score = float(getattr(best_face, "det_score", 0.0))
        return EmbeddingResult(embedding=np.asarray(embedding, dtype=np.float32), bbox=bbox, detection_score=score)

    def extract_from_bytes(self, data: bytes) -> EmbeddingResult:
        """Decode an image from bytes and compute the embedding."""

        np_img = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("No se pudo decodificar la imagen enviada.")
        return self.extract(frame)


__all__ = ["EmbeddingError", "EmbeddingResult", "FaceEmbedder"]
