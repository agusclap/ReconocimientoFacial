"""Utilities for anti-spoofing and scene validation."""
from .antispoof import Detection, LivenessError, LivenessResult, evaluate_frame_bytes

__all__ = ["Detection", "LivenessError", "LivenessResult", "evaluate_frame_bytes"]
