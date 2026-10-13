import pytest
from pydantic import ValidationError

from veil.detect.schema import Detection, Source


def _det(start: int, end: int, **kw: object) -> Detection:
    base: dict[str, object] = {
        "entity_type": "EMAIL_ADDRESS",
        "confidence": 0.9,
        "source": Source.PRESIDIO,
        "recognizer": "EmailRecognizer",
    }
    return Detection(start=start, end=end, **{**base, **kw})


def test_detection_is_frozen() -> None:
    det = _det(0, 5)
    with pytest.raises(ValidationError):
        det.start = 3  # type: ignore[misc]


def test_rejects_end_not_after_start() -> None:
    with pytest.raises(ValidationError):
        _det(5, 5)
    with pytest.raises(ValidationError):
        _det(6, 5)


def test_rejects_confidence_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        _det(0, 1, confidence=1.5)


def test_overlaps_uses_half_open_intervals() -> None:
    assert _det(0, 5).overlaps(_det(4, 8))
    assert not _det(0, 5).overlaps(_det(5, 8))
    assert _det(2, 3).overlaps(_det(0, 10))


def test_span_text_and_length() -> None:
    det = _det(5, 10)
    assert det.length == 5
    assert det.span_text("mail hello world") == "hello"


def test_serializes_source_as_plain_string() -> None:
    assert _det(0, 1).model_dump()["source"] == "presidio"
