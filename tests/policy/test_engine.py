"""PolicyEngine tests use hand-built detections so they run without the spaCy model."""

from veil.detect import Detection, Source
from veil.policy import Action, Destination, Hasher, PolicyEngine, policy_from_dict

POLICY = policy_from_dict(
    {
        "version": 1,
        "name": "t",
        "defaults": {"action": "redact", "min_confidence": 0.5},
        "entities": {
            "PERSON": {"action": "redact", "min_confidence": 0.7},
            "EMAIL_ADDRESS": {"action": "mask", "min_confidence": 0.6},
            "IP_ADDRESS": {"action": "hash", "min_confidence": 0.6},
            "US_SSN": {"action": "block", "min_confidence": 0.0},
            "LOCATION": {"action": "allow", "min_confidence": 0.5},
        },
        "destinations": {"model": {"entities": {"PERSON": {"action": "allow"}}}},
    }
)
ENGINE = PolicyEngine(POLICY, hasher=Hasher(b"test-key"))

TEXT = "Maria Gonzalez <maria@example.com> from Seattle, host 10.0.0.1"


def _det(entity: str, start: int, end: int, conf: float) -> Detection:
    return Detection(
        entity_type=entity,
        start=start,
        end=end,
        confidence=conf,
        source=Source.PRESIDIO,
        recognizer="r",
    )


PERSON = _det("PERSON", 0, 14, 0.85)
EMAIL = _det("EMAIL_ADDRESS", 16, 33, 1.0)
LOCATION = _det("LOCATION", 40, 47, 0.85)
IP = _det("IP_ADDRESS", 54, 62, 0.95)


def test_applies_mixed_actions_right_to_left_preserving_offsets() -> None:
    result = ENGINE.apply(TEXT, [PERSON, EMAIL, LOCATION, IP], Destination.LOGS)
    assert not result.blocked
    ip_token = Hasher(b"test-key").token("10.0.0.1", "IP_ADDRESS")
    assert result.text == f"<PERSON> <m****@example.com> from Seattle, host {ip_token}"


def test_destination_override_allows_person_for_model() -> None:
    result = ENGINE.apply(TEXT, [PERSON], Destination.MODEL)
    assert result.text == TEXT
    (decision,) = result.decisions
    assert decision.action == Action.ALLOW
    assert decision.applied
    assert decision.rule.action_rule == "destinations.model.entities.PERSON"


def test_below_threshold_detection_is_recorded_but_not_applied() -> None:
    weak_person = _det("PERSON", 0, 14, 0.4)
    result = ENGINE.apply(TEXT, [weak_person], Destination.LOGS)
    assert result.text == TEXT
    (decision,) = result.decisions
    assert not decision.applied
    assert decision.replacement is None
    assert decision.rule.min_confidence == 0.7


def test_block_returns_original_text_and_flags() -> None:
    text = "ssn 536-22-8741 for Maria Gonzalez"
    ssn = _det("US_SSN", 4, 15, 0.5)
    person = _det("PERSON", 20, 34, 0.85)
    result = ENGINE.apply(text, [ssn, person], Destination.LOGS)
    assert result.blocked
    assert result.text == text  # nothing partially scrubbed
    assert [d.detection.entity_type for d in result.blocking_decisions] == ["US_SSN"]


def test_block_fires_even_at_zero_confidence() -> None:
    ssn = _det("US_SSN", 4, 15, 0.01)
    result = ENGINE.apply("ssn 536-22-8741", [ssn], Destination.MODEL)
    assert result.blocked


def test_no_detections_is_identity() -> None:
    result = ENGINE.apply(TEXT, [], Destination.MODEL)
    assert result.text == TEXT
    assert result.decisions == ()
    assert not result.blocked


def test_hash_is_consistent_across_calls() -> None:
    a = ENGINE.apply(TEXT, [IP], Destination.LOGS).text
    b = ENGINE.apply(TEXT, [IP], Destination.LOGS).text
    assert a == b


def test_decisions_serialize_to_json() -> None:
    result = ENGINE.apply(TEXT, [PERSON, IP], Destination.LOGS)
    payload = result.model_dump(mode="json")
    assert payload["decisions"][0]["rule"]["action_rule"] == "entities.PERSON"
