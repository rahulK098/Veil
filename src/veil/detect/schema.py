"""Unified detection schema shared by every detector source (ADR-0003)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Source(StrEnum):
    """Which engine produced a detection."""

    PRESIDIO = "presidio"
    DETECT_SECRETS = "detect-secrets"
    CUSTOM = "custom"


class Detection(BaseModel):
    """A single sensitive span found in a text.

    ``start``/``end`` are character offsets into the scanned text (half-open interval).
    ``confidence`` is normalized to ``[0.0, 1.0]`` regardless of source; sources that
    only produce binary hits are assigned a fixed confidence documented in their adapter.
    ``recognizer`` names the upstream recognizer/plugin so audit entries stay traceable.
    """

    model_config = ConfigDict(frozen=True)

    entity_type: str = Field(
        min_length=1, description="Normalized entity label, e.g. EMAIL_ADDRESS"
    )
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    confidence: float = Field(ge=0.0, le=1.0)
    source: Source
    recognizer: str = Field(min_length=1, description="Upstream recognizer or plugin name")

    @model_validator(mode="after")
    def _end_after_start(self) -> Detection:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Detection) -> bool:
        return self.start < other.end and other.start < self.end

    def span_text(self, text: str) -> str:
        """Return the raw span. Callers must never persist this value (ADR-0006)."""
        return text[self.start : self.end]
