"""Composes all detectors into one engine returning a non-overlapping detection list."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field

from veil.detect.base import Detector
from veil.detect.calibrate import calibrate
from veil.detect.custom import DEFAULT_CUSTOM_SPECS, EXTENDED_RECOGNIZER_NAMES, PatternSpec
from veil.detect.merge import resolve_overlaps
from veil.detect.presidio_detector import (
    DEFAULT_EXCLUDED_ENTITIES,
    DEFAULT_MODEL,
    PresidioDetector,
    build_analyzer,
)
from veil.detect.schema import Detection
from veil.detect.secrets_detector import SecretsDetector


class DetectorConfig(BaseModel):
    """Detection-layer configuration. Loaded from the ``detection:`` section of a policy file."""

    model_config = ConfigDict(frozen=True)

    spacy_model: str = DEFAULT_MODEL
    presidio_entities: tuple[str, ...] | None = Field(
        default=None,
        description="Explicit Presidio entity list; None = all supported minus excluded",
    )
    presidio_excluded_entities: frozenset[str] = DEFAULT_EXCLUDED_ENTITIES
    enable_secrets: bool = True
    disabled_secret_types: frozenset[str] = Field(
        default=frozenset(),
        description="detect-secrets plugin names to skip, e.g. 'Base64 High Entropy String'",
    )
    custom_recognizers: tuple[PatternSpec, ...] = DEFAULT_CUSTOM_SPECS
    enable_extended_recognizers: bool = Field(
        default=True,
        description="Veil-shipped recognizers closing documented Presidio gaps "
        "(2-series Mastercard, street addresses)",
    )
    calibrate_scores: bool = Field(
        default=True,
        description="Replace Presidio's uninformative constant scores with graded ones: "
        "PERSON by span structure (ADR-0012), IP_ADDRESS by reserved range and version "
        "context (ADR-0014), so policy thresholds actually bite",
    )


class DetectionEngine:
    """Lazily builds the underlying detectors (Presidio load is ~4 s) and merges results."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()
        self._lock = threading.Lock()

    @property
    def config(self) -> DetectorConfig:
        return self._config

    @cached_property
    def detectors(self) -> tuple[Detector, ...]:
        with self._lock:
            specs = self._config.custom_recognizers
            extended = self._config.enable_extended_recognizers
            analyzer = build_analyzer(self._config.spacy_model, specs, extended=extended)
            custom_names = {spec.recognizer_name for spec in specs}
            if extended:
                custom_names |= EXTENDED_RECOGNIZER_NAMES
            presidio = PresidioDetector(
                analyzer,
                entities=self._config.presidio_entities,
                excluded_entities=self._config.presidio_excluded_entities,
                custom_recognizer_names=frozenset(custom_names),
            )
            detectors: list[Detector] = [presidio]
            if self._config.enable_secrets:
                detectors.append(SecretsDetector(disabled_types=self._config.disabled_secret_types))
            return tuple(detectors)

    def warm_up(self) -> None:
        """Force model loading now rather than on the first request."""
        _ = self.detectors
        self.detect("warm up")

    def detect(self, text: str) -> list[Detection]:
        """Run every detector and return non-overlapping detections sorted by position.

        Calibration runs *before* overlap resolution so a downgraded single-token PERSON
        loses to a confident recognizer covering the same span (e.g. STREET_ADDRESS).
        """
        if not text:
            return []
        return resolve_overlaps(self._collect(text))

    def detect_raw(self, text: str) -> list[Detection]:
        """All detections before overlap resolution — for debugging and eval analysis."""
        return sorted(self._collect(text), key=lambda d: (d.start, d.end))

    def _collect(self, text: str) -> list[Detection]:
        raw: list[Detection] = []
        for detector in self.detectors:
            raw.extend(detector.detect(text))
        return calibrate(raw, text) if self._config.calibrate_scores else raw


__all__: Sequence[str] = ["DetectionEngine", "DetectorConfig"]
