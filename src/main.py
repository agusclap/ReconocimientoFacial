"""Command line entry point for the facial recognition toolkit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2

from .embeddings import EmbeddingError, FaceEmbedder
from .enrollment import register_member_cli
from .liveness import LivenessDetector, summarise_reasons


def _run_pipeline(video_source: str, show_window: bool = False) -> int:
    detector = LivenessDetector()
    embedder = FaceEmbedder()

    if video_source.isdigit():
        capture = cv2.VideoCapture(int(video_source))
    else:
        capture = cv2.VideoCapture(video_source)

    if not capture.isOpened():
        print(f"No se pudo abrir la fuente de video: {video_source}")
        return 1

    print("Pipeline iniciado. Presiona 'q' para salir.")

    window_name = "Reconocimiento" if show_window else None
    if window_name:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("No se pudo leer el frame.")
                break

            result = detector.evaluate(frame)
            if not result.ok:
                print(f"Liveness: {summarise_reasons(result)}")
            else:
                try:
                    embedding = embedder.extract(frame)
                except EmbeddingError as err:
                    print(f"Embedding: {err}")
                else:
                    print(
                        "Embedding capturado | longitud=%d | score=%.3f"
                        % (len(embedding.embedding), embedding.detection_score)
                    )

            if window_name:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if window_name:
            cv2.destroyWindow(window_name)
        else:
            cv2.destroyAllWindows()

    return 0


def parse_args(args: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toolkit de reconocimiento facial")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline_parser = subparsers.add_parser("pipeline", help="Ejecuta el pipeline de liveness + embeddings")
    pipeline_parser.add_argument(
        "--video-source",
        default="0",
        help="Fuente de video. Indica el índice de la cámara o la ruta a un archivo.",
    )
    pipeline_parser.add_argument(
        "--show-window",
        action="store_true",
        help="Muestra un preview de la cámara usando OpenCV.",
    )

    enroll_parser = subparsers.add_parser("enroll", help="Inscribe un nuevo socio desde consola")
    enroll_parser.add_argument(
        "--storage",
        default="data/enrollments.json",
        help="Archivo donde se almacenarán los embeddings capturados.",
    )
    enroll_parser.add_argument(
        "--camera-index",
        default=0,
        type=int,
        help="Índice de la cámara a utilizar.",
    )
    enroll_parser.add_argument(
        "--samples",
        default=3,
        type=int,
        help="Cantidad de muestras que se deben capturar.",
    )

    return parser.parse_args(list(args))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "pipeline":
        return _run_pipeline(video_source=args.video_source, show_window=args.show_window)
    if args.command == "enroll":
        register_member_cli(storage_path=Path(args.storage), camera_index=args.camera_index, samples_target=args.samples)
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
