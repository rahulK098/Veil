import pytest
from pydantic import ValidationError

from veil.detect.custom import (
    EMPLOYEE_ID,
    EXTENDED_RECOGNIZER_NAMES,
    INTERNAL_CASE_ID,
    STREET_ADDRESS,
    PatternSpec,
    extended_recognizers,
)


def test_spec_rejects_invalid_regex() -> None:
    with pytest.raises(ValidationError, match="invalid regex"):
        PatternSpec(entity_type="X", regex="(")


def test_spec_requires_upper_snake_entity_type() -> None:
    with pytest.raises(ValidationError):
        PatternSpec(entity_type="caseId", regex=r"\d+")


def test_to_recognizer_produces_presidio_recognizer() -> None:
    rec = INTERNAL_CASE_ID.to_recognizer()
    assert rec.supported_entities == ["INTERNAL_CASE_ID"]
    results = rec.analyze("re: CASE-2024-00123 please", entities=["INTERNAL_CASE_ID"])
    assert len(results) == 1
    assert (results[0].start, results[0].end) == (4, 19)
    assert results[0].score == pytest.approx(0.85)


def test_employee_id_does_not_match_partial_token() -> None:
    rec = EMPLOYEE_ID.to_recognizer()
    assert (
        rec.analyze("EMP-12345 (too short) and EMP-1234567 (too long)", entities=["EMPLOYEE_ID"])
        == []
    )
    assert len(rec.analyze("badge EMP-123456", entities=["EMPLOYEE_ID"])) == 1


# --- extended recognizers (documented Presidio gaps) ----------


def _spans(rec, text: str, entity: str) -> list[str]:  # type: ignore[no-untyped-def]
    return [text[r.start : r.end] for r in rec.analyze(text, entities=[entity])]


@pytest.mark.parametrize(
    "address",
    [
        "66983 Carlos Corner Apt. 207",
        "12 Main Street",
        "4501 Lakeview Drive Suite 300",
        "8 Elm St.",
        "0434 Laura Ferry",
    ],
)
def test_street_address_matches_common_forms(address: str) -> None:
    rec = STREET_ADDRESS.to_recognizer()
    assert _spans(rec, f"Ship to {address}, thanks.", "STREET_ADDRESS") == [address]


@pytest.mark.parametrize(
    "text",
    [
        "Version v5.19.8 deprecates the legacy REST v1 endpoints.",  # IGNORECASE trap
        "as of February 02, 2027 between Bridges-Williams and the Client",
        "increase volume to 500 GB before the load test",
        "45 stories pointed",
    ],
)
def test_street_address_ignores_versions_dates_and_prose(text: str) -> None:
    rec = STREET_ADDRESS.to_recognizer()
    assert _spans(rec, text, "STREET_ADDRESS") == []


def test_series2_mastercard_recognizer_keeps_luhn_validation() -> None:
    series2 = next(r for r in extended_recognizers() if r.name == "Series2CreditCardRecognizer")
    valid = "2243466741680975"  # Luhn-valid, 2-series prefix Presidio's default regex misses
    invalid = "2243466741680976"
    assert _spans(series2, f"card {valid}", "CREDIT_CARD") == [valid]
    assert _spans(series2, f"card {invalid}", "CREDIT_CARD") == []
    assert _spans(series2, "card 2277 1110 0342 1754", "CREDIT_CARD") == ["2277 1110 0342 1754"]


def test_extended_recognizer_names_match_instances() -> None:
    assert {str(r.name) for r in extended_recognizers()} == EXTENDED_RECOGNIZER_NAMES
