from veil.policy import Action, Destination, policy_from_dict, resolve_rule

POLICY = policy_from_dict(
    {
        "version": 1,
        "name": "t",
        "defaults": {"action": "redact", "min_confidence": 0.5},
        "entities": {
            "PERSON": {"action": "redact", "min_confidence": 0.7, "reason": "base"},
            "US_SSN": {"action": "block", "min_confidence": 0.0},
        },
        "destinations": {
            "model": {"entities": {"PERSON": {"action": "allow"}}},
            "logs": {
                "defaults": {"min_confidence": 0.3},
                "entities": {"PERSON": {"action": "hash", "reason": "correlate"}},
            },
        },
    }
)


def test_unknown_entity_falls_back_to_defaults() -> None:
    rule = resolve_rule(POLICY, "IP_ADDRESS", Destination.MODEL)
    assert rule.action == Action.REDACT
    assert rule.min_confidence == 0.5
    assert rule.action_rule == "defaults"
    assert rule.threshold_rule == "defaults"


def test_entity_rule_applies_when_destination_has_no_override() -> None:
    rule = resolve_rule(POLICY, "PERSON", Destination.VECTOR_STORE)
    assert rule.action == Action.REDACT
    assert rule.min_confidence == 0.7
    assert rule.action_rule == "entities.PERSON"
    assert rule.threshold_rule == "entities.PERSON"
    assert rule.reason == "base"


def test_destination_entity_override_changes_only_action() -> None:
    rule = resolve_rule(POLICY, "PERSON", Destination.MODEL)
    assert rule.action == Action.ALLOW
    assert rule.action_rule == "destinations.model.entities.PERSON"
    # threshold inherited from the base entity rule, and the audit path says so
    assert rule.min_confidence == 0.7
    assert rule.threshold_rule == "entities.PERSON"


def test_destination_defaults_lower_threshold_for_all_entities() -> None:
    rule = resolve_rule(POLICY, "IP_ADDRESS", Destination.LOGS)
    assert rule.min_confidence == 0.3
    assert rule.threshold_rule == "destinations.logs.defaults"
    assert rule.action == Action.REDACT
    assert rule.action_rule == "defaults"


def test_destination_entity_beats_base_entity_for_action() -> None:
    rule = resolve_rule(POLICY, "PERSON", Destination.LOGS)
    assert rule.action == Action.HASH
    assert rule.action_rule == "destinations.logs.entities.PERSON"
    assert rule.reason == "correlate"


def test_entity_specific_threshold_beats_destination_wide_default() -> None:
    """Specificity first: logs.defaults (0.3) must not loosen entities.PERSON (0.7)."""
    rule = resolve_rule(POLICY, "PERSON", Destination.LOGS)
    assert rule.min_confidence == 0.7
    assert rule.threshold_rule == "entities.PERSON"


def test_block_rule_is_inherited_by_every_destination_with_its_threshold() -> None:
    for dest in Destination:
        rule = resolve_rule(POLICY, "US_SSN", dest)
        assert rule.action == Action.BLOCK, dest
        assert rule.min_confidence == 0.0, dest
        assert rule.threshold_rule == "entities.US_SSN", dest
