"""Utilities for obtaining facial embeddings."""

from .insightface_embedder import EmbeddingError, EmbeddingResult, FaceEmbedder

__all__ = ["EmbeddingError", "EmbeddingResult", "FaceEmbedder"]
