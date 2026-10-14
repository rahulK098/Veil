import pytest

from veil.detect.calibrate import (
    MALFORMED_CONFIDENCE,
    SINGLE_TOKEN_CONFIDENCE,
    calibrate,
    person_confidence,
)
from veil.detect.schema import Detection, Source


@pytest.mark.parametrize(
    "value",
    ["Maria Gonzalez", "Mary Jane Watson", "Renee Taylor", "Kathleen Lee"],
)
def test_multi_token_names_keep_their_score(value: str) -> None:
    assert person_confidence(value, 0.85) == 0.85


@pytest.mark.parametrize(
    "value",
    [
        "Deliverables",  # plural noun
        "Atlas",  # product name
        "Beacon",
        "Rodriguez",  # bare surname: genuinely ambiguous, not wrong
        "Bridges-Williams",  # company; one whitespace token
    ],
)
def test_single_token_spans_are_downgraded_not_dropped(value: str) -> None:
    """Downgraded, not filtered: policy decides whether 0.4 is enough for a destination."""
    assert person_confidence(value, 0.85) == SINGLE_TOKEN_CONFIDENCE


@pytest.mark.parametrize(
    "value",
    ["Elliott\nLine 1: Atlas", "ORD-347528", "Changelog\n- Fixed", "A: B"],
)
def test_malformed_spans_score_lowest(value: str) -> None:
    assert person_confidence(value, 0.85) == MALFORMED_CONFIDENCE


def test_empty_span_is_malformed() -> None:
    assert person_confidence("", 0.85) == MALFORMED_CONFIDENCE


def test_calibration_never_raises_a_score() -> None:
    assert person_confidence("Ada", 0.2) == 0.2


def _det(entity: str, start: int, end: int, conf: float, source: Source) -> Detection:
    return Detection(
        entity_type=entity,
        start=start,
        end=end,
        confidence=conf,
        source=source,
        recognizer="r",
    )


TEXT = "Deliverables and Maria Gonzalez and EMP-123456"


def test_calibrate_only_touches_presidio_person() -> None:
    dets = [
        _det("PERSON", 0, 12, 0.85, Source.PRESIDIO),  # "Deliverables"
        _det("PERSON", 17, 31, 0.85, Source.PRESIDIO),  # "Maria Gonzalez"
        _det("EMPLOYEE_ID", 36, 46, 0.85, Source.CUSTOM),  # single token, must not change
        _det("LOCATION", 0, 12, 0.85, Source.PRESIDIO),  # not PERSON, must not change
    ]
    out = calibrate(dets, TEXT)
    assert [d.confidence for d in out] == [SINGLE_TOKEN_CONFIDENCE, 0.85, 0.85, 0.85]


def test_calibrate_preserves_everything_but_confidence() -> None:
    original = _det("PERSON", 0, 12, 0.85, Source.PRESIDIO)
    (out,) = calibrate([original], TEXT)
    assert (out.entity_type, out.start, out.end, out.source, out.recognizer) == (
        original.entity_type,
        original.start,
        original.end,
        original.source,
        original.recognizer,
    )


def test_calibrate_is_identity_for_empty_input() -> None:
    assert calibrate([], TEXT) == []
