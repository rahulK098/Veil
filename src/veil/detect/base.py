"""Detector protocol every engine adapter implements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from veil.detect.schema import Detection


@runtime_checkable
class Detector(Protocol):
    """A detector scans text and returns zero or more detections.

    Implementations must be safe to call concurrently after construction and must not
    retain references to scanned text.
    """

    @property
    def name(self) -> str: ...

    def detect(self, text: str) -> Sequence[Detection]: ...
