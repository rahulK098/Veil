import pytest

from veil.detect.calibrate import (
    MALFORMED_CONFIDENCE,
    NEVER_PERSONAL_IP_CONFIDENCE,
    SINGLE_TOKEN_CONFIDENCE,
    VERSION_LIKE_IP_CONFIDENCE,
    calibrate,
    ip_confidence,
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


# --- IP_ADDRESS (ADR-0014) ----------


@pytest.mark.parametrize(
    ("value", "preceding"),
    [
        ("10.0.0.14", "level=info user=x ip="),
        ("192.168.1.50", "the host at "),
        ("8.8.8.8", "resolver "),
    ],
)
def test_real_ip_addresses_keep_their_score(value: str, preceding: str) -> None:
    assert ip_confidence(value, preceding, 0.6) == 0.6


@pytest.mark.parametrize(
    ("value", "preceding"),
    [
        ("10.2.14.3", "upgraded the agent to version "),
        ("2.0.1.4", "release "),
        ("3.1.0.9", "bumped to v"),
    ],
)
def test_version_strings_are_downgraded(value: str, preceding: str) -> None:
    assert ip_confidence(value, preceding, 0.6) == VERSION_LIKE_IP_CONFIDENCE


@pytest.mark.parametrize(
    "value", ["0.0.0.0", "255.255.255.255", "127.0.0.1", "127.1.2.3", "0.1.2.3"]
)
def test_reserved_addresses_identify_nobody(value: str) -> None:
    """Unspecified, broadcast and loopback cannot identify a machine, so carry no PII."""
    assert ip_confidence(value, "bind to ", 0.6) == NEVER_PERSONAL_IP_CONFIDENCE


@pytest.mark.parametrize(
    "preceding",
    [
        "version 10.2.14.3 and host ",  # keyword present, but not adjacent
        "pinned the kubernetes client to ",
        "we shipped version 4 last quarter. the server at ",
    ],
)
def test_distant_version_words_do_not_suppress_a_real_address(preceding: str) -> None:
    """Only an adjacent keyword counts. Wrongly downgrading a real IP leaks it; wrongly
    keeping a version string merely over-redacts, so the window errs toward keeping."""
    assert ip_confidence("10.0.0.14", preceding, 0.6) == 0.6


def test_ip_calibration_never_raises_a_score() -> None:
    assert ip_confidence("10.2.14.3", "version ", 0.2) == 0.2


def test_calibrate_dispatches_on_entity_type() -> None:
    text = "version 10.2.14.3 and host 10.0.0.14"
    dets = [
        _det("IP_ADDRESS", 8, 17, 0.6, Source.PRESIDIO),
        _det("IP_ADDRESS", 27, 36, 0.6, Source.PRESIDIO),
    ]
    assert [d.confidence for d in calibrate(dets, text)] == [VERSION_LIKE_IP_CONFIDENCE, 0.6]


def test_calibrate_leaves_other_entities_untouched() -> None:
    text = "card 4111 1111 1111 1111 here"
    dets = [_det("CREDIT_CARD", 5, 24, 1.0, Source.PRESIDIO)]
    assert [d.confidence for d in calibrate(dets, text)] == [1.0]
