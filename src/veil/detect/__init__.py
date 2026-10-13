"""Detection layer: wraps Presidio and detect-secrets behind one schema (ADR-0001, ADR-0003)."""

from veil.detect.base import Detector
from veil.detect.custom import PatternSpec
from veil.detect.engine import DetectionEngine, DetectorConfig
from veil.detect.merge import resolve_overlaps
from veil.detect.schema import Detection, Source

__all__ = [
    "Detection",
    "DetectionEngine",
    "Detector",
    "DetectorConfig",
    "PatternSpec",
    "Source",
    "resolve_overlaps",
]
