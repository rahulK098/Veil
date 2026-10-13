"""Custom pattern recognizers for domain entities Presidio's built-ins miss (ADR-0001).

These are declared as data (``PatternSpec``) so that a deployment can add an internal
case-number or employee-ID format from config without writing Python.
"""

from __future__ import annotations

import re

from presidio_analyzer import Pattern, PatternRecognizer
from presidio_analyzer.predefined_recognizers import CreditCardRecognizer
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatternSpec(BaseModel):
    """Declarative description of a regex-based entity recognizer."""

    model_config = ConfigDict(frozen=True)

    entity_type: str = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_]*$")
    regex: str = Field(min_length=1)
    score: float = Field(default=0.7, ge=0.0, le=1.0)
    context: tuple[str, ...] = Field(
        default=(),
        description="Context words that boost confidence when found near a match",
    )

    @field_validator("regex")
    @classmethod
    def _must_compile(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regex {value!r}: {exc}") from exc
        return value

    @property
    def recognizer_name(self) -> str:
        return f"Custom{self.entity_type.title().replace('_', '')}Recognizer"

    def to_recognizer(self) -> PatternRecognizer:
        return PatternRecognizer(
            supported_entity=self.entity_type,
            name=self.recognizer_name,
            patterns=[Pattern(name=self.entity_type.lower(), regex=self.regex, score=self.score)],
            context=list(self.context),
        )


# Example specs shipped with Veil. They model the "internal identifiers" case from the plan;
# real deployments replace them with their own formats.
INTERNAL_CASE_ID = PatternSpec(
    entity_type="INTERNAL_CASE_ID",
    regex=r"\bCASE-\d{4}-\d{5,6}\b",
    score=0.85,
    context=("case", "matter", "ticket"),
)

EMPLOYEE_ID = PatternSpec(
    entity_type="EMPLOYEE_ID",
    regex=r"\bEMP-\d{6}\b",
    score=0.85,
    context=("employee", "staff", "badge"),
)

DEFAULT_CUSTOM_SPECS: tuple[PatternSpec, ...] = (INTERNAL_CASE_ID, EMPLOYEE_ID)


# --- Extended recognizers: gaps in Presidio's defaults found by the eval harness ----------
#
# Each of these closes a miss documented in docs/eval/. They are enabled by
# ``DetectorConfig.enable_extended_recognizers`` (default on); ``policies/eval-baseline.yaml``
# turns them off to reproduce the pre-fix numbers.

# Presidio's CreditCardRecognizer regex covers 1/3/4/5[0-5]/6-prefixed cards but not the
# Mastercard 2-series BIN range (2221–2720, issued since 2017). We reuse Presidio's own
# recognizer class so its Luhn validation (score -> 1.0 on pass, dropped on fail) still applies.
_SERIES2_CARD_PATTERN = Pattern(
    name="Mastercard 2-series (weak)",
    regex=r"\b2[2-7]\d{2}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    score=0.3,
)

# Presidio has no address recognizer; spaCy LOCATION covers cities, not street addresses.
# Number + capitalized words + a USPS-style street suffix + optional unit is specific enough
# to outrank a spaCy PERSON hit on the same span (0.9 > 0.85), which also removes the
# PERSON false positives that street names like "Carlos Corner" otherwise produce.
_STREET_SUFFIXES = (
    "Alley|Avenue|Ave|Bend|Boulevard|Blvd|Bridge|Brook|Burg|Bypass|Camp|Canyon|Cape|Causeway|"
    "Center|Circle|Cir|Cliff|Club|Common|Corner|Course|Court|Ct|Cove|Creek|Crescent|Crest|"
    "Crossing|Dale|Dam|Divide|Drive|Dr|Estate|Expressway|Extension|Falls|Ferry|Field|Flat|"
    "Ford|Forge|Fork|Fort|Freeway|Garden|Gateway|Glen|Green|Grove|Harbor|Haven|Heights|"
    "Highway|Hwy|Hill|Hollow|Inlet|Island|Junction|Key|Knoll|Lake|Landing|Lane|Ln|Light|Loop|"
    "Mall|Manor|Meadow|Mill|Mission|Motorway|Mount|Neck|Orchard|Oval|Park|Parkway|Pkwy|Pass|"
    "Path|Pike|Pine|Place|Pl|Plain|Plaza|Point|Port|Prairie|Radial|Ramp|Ranch|Rapid|Rest|"
    "Ridge|River|Road|Rd|Route|Row|Run|Shoal|Shore|Spring|Spur|Square|Sq|Station|Stravenue|"
    "Stream|Street|St|Summit|Terrace|Ter|Throughway|Trace|Track|Trafficway|Trail|Tunnel|"
    "Turnpike|Underpass|Union|Valley|Viaduct|View|Village|Ville|Vista|Walk|Wall|Way|Well|Wood"
)
# Presidio compiles patterns with re.IGNORECASE, so the capitalisation constraint is wrapped in
# an inline (?-i:...) group; the lookbehind keeps "v5.19.8 deprecates ..." from matching.
STREET_ADDRESS = PatternSpec(
    entity_type="STREET_ADDRESS",
    regex=(
        r"(?<![.\-])\b\d{1,6}\s+(?-i:(?:[A-Z][a-z]+\s+){1,3}(?:" + _STREET_SUFFIXES + r"))s?\b\.?"
        r"(?:\s+(?:Apt\.?|Suite|Ste\.?|Unit|#)\s*\d+[A-Za-z]?)?"
    ),
    score=0.9,
    context=("address", "ship", "deliver", "street"),
)


_SERIES2_RECOGNIZER_NAME = "Series2CreditCardRecognizer"


def extended_recognizers() -> list[PatternRecognizer]:
    """Veil-shipped recognizers that close documented Presidio gaps."""
    series2 = CreditCardRecognizer(patterns=[_SERIES2_CARD_PATTERN], name=_SERIES2_RECOGNIZER_NAME)
    return [series2, STREET_ADDRESS.to_recognizer()]


EXTENDED_RECOGNIZER_NAMES: frozenset[str] = frozenset(
    {_SERIES2_RECOGNIZER_NAME, STREET_ADDRESS.recognizer_name}
)
