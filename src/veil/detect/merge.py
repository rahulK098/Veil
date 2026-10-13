"""Resolve overlapping detections from multiple sources into a non-overlapping set.

Presidio de-duplicates within its own results, but nothing reconciles Presidio against
detect-secrets or custom recognizers. Overlaps are resolved by confidence, then span length,
then a stable source order, so the outcome is deterministic (ADR-0003).
"""

from __future__ import annotations

from collections.abc import Iterable

from veil.detect.schema import Detection, Source

_SOURCE_PRIORITY: dict[Source, int] = {
    Source.CUSTOM: 0,
    Source.DETECT_SECRETS: 1,
    Source.PRESIDIO: 2,
}


def _rank(detection: Detection) -> tuple[float, int, int]:
    """Higher confidence wins; longer span breaks ties; then source priority."""
    return (-detection.confidence, -detection.length, _SOURCE_PRIORITY[detection.source])


def resolve_overlaps(detections: Iterable[Detection]) -> list[Detection]:
    """Return detections sorted by start with no two spans overlapping.

    Among any group of mutually overlapping detections, the best-ranked one is kept and
    every detection overlapping it is dropped. Greedy by rank, so a high-confidence
    EMAIL_ADDRESS suppresses the low-confidence URL fragments Presidio finds inside it.
    """
    ordered = sorted(detections, key=_rank)
    kept: list[Detection] = []
    for candidate in ordered:
        if any(candidate.overlaps(existing) for existing in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda d: (d.start, d.end))
