"""End-to-end detection engine tests. Marked slow: they load the spaCy model."""

import pytest

from veil.detect import DetectionEngine, Source
from veil.detect.presidio_detector import PresidioDetector, build_analyzer

pytestmark = pytest.mark.slow


def _types(dets: list) -> set[str]:
    return {d.entity_type for d in dets}


def test_detects_core_pii_types(engine: DetectionEngine) -> None:
    text = (
        "Call Maria Gonzalez at 555-867-5309, SSN 536-22-8741, "
        "card 4111 1111 1111 1111, email maria.g@example.com, host 10.0.0.1, in Seattle."
    )
    found = _types(engine.detect(text))
    assert {
        "PERSON",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "EMAIL_ADDRESS",
        "IP_ADDRESS",
        "LOCATION",
    } <= found


def test_email_suppresses_url_fragments_inside_it(engine: DetectionEngine) -> None:
    text = "reach me at j.smith@example.com"
    dets = engine.detect(text)
    emails = [d for d in dets if d.entity_type == "EMAIL_ADDRESS"]
    assert len(emails) == 1
    assert text[emails[0].start : emails[0].end] == "j.smith@example.com"
    assert not any(d.entity_type == "URL" for d in dets)


def test_results_never_overlap(engine: DetectionEngine) -> None:
    text = "AKIAIOSFODNN7EXAMPLE 10.0.0.1 j.smith@example.com Maria Gonzalez 4111 1111 1111 1111"
    dets = engine.detect(text)
    for a, b in zip(dets, dets[1:], strict=False):
        assert a.end <= b.start


def test_secrets_and_pii_from_both_sources(engine: DetectionEngine) -> None:
    text = "Maria Gonzalez leaked AKIAIOSFODNN7EXAMPLE in a ticket"
    dets = engine.detect(text)
    assert {d.source for d in dets} >= {Source.PRESIDIO, Source.DETECT_SECRETS}


def test_custom_recognizer_is_wired_and_tagged_custom(engine: DetectionEngine) -> None:
    dets = engine.detect("Escalate CASE-2024-00123 to EMP-123456 today")
    custom = [d for d in dets if d.source == Source.CUSTOM]
    assert _types(custom) == {"INTERNAL_CASE_ID", "EMPLOYEE_ID"}


def test_noisy_entities_excluded_by_default(engine: DetectionEngine) -> None:
    # "SSN" alone is tagged ORGANIZATION by spaCy; we do not request that entity by default.
    dets = engine.detect("SSN")
    assert not any(d.entity_type == "ORGANIZATION" for d in dets)


def test_clean_text_yields_nothing(engine: DetectionEngine) -> None:
    text = "The quarterly review covers revenue, churn, and the roadmap for the next release."
    assert engine.detect(text) == []


def test_empty_text(engine: DetectionEngine) -> None:
    assert engine.detect("") == []


def test_presidio_detector_rejects_unknown_entity(engine: DetectionEngine) -> None:
    # Reuse the already-loaded analyzer via the first detector to avoid a second model load.
    presidio = engine.detectors[0]
    assert isinstance(presidio, PresidioDetector)
    analyzer = presidio._analyzer  # noqa: SLF001 — test needs the shared engine
    with pytest.raises(ValueError, match="unsupported Presidio entities"):
        PresidioDetector(analyzer, entities=["NOT_A_THING"])


def test_build_analyzer_registers_custom_specs() -> None:
    pytest.importorskip("spacy")
    from veil.detect.custom import EMPLOYEE_ID

    analyzer = build_analyzer(custom_specs=[EMPLOYEE_ID])
    assert "EMPLOYEE_ID" in analyzer.get_supported_entities(language="en")


@pytest.mark.parametrize(
    "number",
    ["376.808.9422", "397-231-4658", "349-138-6569", "350-035-8334"],
)
def test_undiallable_numbers_are_deliberately_not_detected(
    engine: DetectionEngine, number: str
) -> None:
    """Area codes 376/397/349/350 are not allocated in the NANP, so `phonenumbers` reports
    these invalid and Presidio (leniency=VALID) does not match them.

    This is intentional, not a gap: matching any 10-digit run is what leniency=POSSIBLE does,
    and a sweep showed that puts a phone false positive in 97 of 100 clean business documents
    (it matches "09" and "14"). See ADR-0011.
    """
    assert not [d for d in engine.detect(f"call {number} today") if d.entity_type == "PHONE_NUMBER"]


@pytest.mark.parametrize(
    "number",
    ["212-555-0147", "(415) 555-0132", "617.555.0188", "+1 206 555 0170", "312-555-0143 x82"],
)
def test_allocated_numbers_are_detected_in_every_format(
    engine: DetectionEngine, number: str
) -> None:
    assert [d for d in engine.detect(f"call {number} today") if d.entity_type == "PHONE_NUMBER"]


def test_person_calibration_grades_real_spacy_output(engine: DetectionEngine) -> None:
    """spaCy scores both of these 0.85; calibration separates them (ADR-0012)."""
    text = "The Deliverables were approved by Maria Gonzalez on Tuesday."
    people = {
        text[d.start : d.end]: d.confidence
        for d in engine.detect(text)
        if d.entity_type == "PERSON"
    }
    assert people.get("Maria Gonzalez") == 0.85
    assert people.get("Deliverables", 1.0) <= 0.4


def test_calibration_can_be_disabled(engine: DetectionEngine) -> None:
    from veil.detect import DetectorConfig

    raw = DetectionEngine(DetectorConfig(calibrate_person=False))
    # Reuse the loaded analyzer: build_analyzer costs ~4 s.
    raw.__dict__["detectors"] = engine.detectors
    text = "The Deliverables were approved."
    scores = [d.confidence for d in raw.detect(text) if d.entity_type == "PERSON"]
    assert scores == [0.85]
