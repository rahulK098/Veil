"""Middleware tests use a deterministic regex detector so they run without spaCy."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from veil.audit import MemoryAuditSink
from veil.detect import Detection, DetectionEngine, Source
from veil.middleware import Guard
from veil.policy import Hasher, load_policy

ROOT = Path(__file__).resolve().parents[2]

_PATTERNS: dict[str, tuple[re.Pattern[str], float, Source]] = {
    "EMAIL_ADDRESS": (re.compile(r"[\w.]+@[\w.]+\.\w+"), 1.0, Source.PRESIDIO),
    "US_SSN": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0.85, Source.PRESIDIO),
    "PERSON": (re.compile(r"Maria Gonzalez|John Smith"), 0.85, Source.PRESIDIO),
    "WEAK_PERSON": (re.compile(r"Bo"), 0.3, Source.PRESIDIO),
    "AWS_ACCESS_KEY": (re.compile(r"AKIA[0-9A-Z]{16}"), 0.9, Source.DETECT_SECRETS),
    "IP_ADDRESS": (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), 0.95, Source.PRESIDIO),
}


class FakeDetectionEngine(DetectionEngine):
    def __init__(self) -> None:  # noqa: D107 — skip parent (no model load)
        pass

    def warm_up(self) -> None:
        return None

    def detect(self, text: str) -> list[Detection]:
        found: list[Detection] = []
        for entity, (pattern, conf, source) in _PATTERNS.items():
            label = "PERSON" if entity == "WEAK_PERSON" else entity
            for m in pattern.finditer(text):
                found.append(
                    Detection(
                        entity_type=label,
                        start=m.start(),
                        end=m.end(),
                        confidence=conf,
                        source=source,
                        recognizer="fake",
                    )
                )
        return sorted(found, key=lambda d: d.start)


@pytest.fixture
def sink() -> MemoryAuditSink:
    return MemoryAuditSink()


@pytest.fixture
def guard(sink: MemoryAuditSink) -> Guard:
    policy = load_policy(ROOT / "policies" / "default.yaml")
    return Guard(
        policy, detection=FakeDetectionEngine(), audit_sink=sink, hasher=Hasher(b"test-key")
    )
