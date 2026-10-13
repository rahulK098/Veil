"""Shared fixtures. The Presidio/spaCy engine is session-scoped because loading costs ~4 s."""

from __future__ import annotations

import warnings

import pytest

from veil.detect import DetectionEngine, DetectorConfig


@pytest.fixture(scope="session")
def engine() -> DetectionEngine:
    warnings.filterwarnings("ignore", category=UserWarning)
    eng = DetectionEngine(DetectorConfig())
    eng.warm_up()
    return eng
