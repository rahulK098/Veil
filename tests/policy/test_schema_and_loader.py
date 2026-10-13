from pathlib import Path

import pytest

from veil.policy import (
    Action,
    Destination,
    Policy,
    PolicyLoadError,
    load_policy,
    policy_from_dict,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / "policies" / "default.yaml"


def test_default_policy_file_loads() -> None:
    policy = load_policy(DEFAULT_POLICY)
    assert policy.name == "default"
    assert policy.entities["US_SSN"].action == Action.BLOCK
    assert Destination.LOGS in policy.destinations
    assert len(policy.detection.custom_recognizers) == 2


def test_missing_file_raises_load_error(tmp_path: Path) -> None:
    with pytest.raises(PolicyLoadError, match="not found"):
        load_policy(tmp_path / "nope.yaml")


def test_invalid_yaml_raises_load_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: [unclosed", encoding="utf-8")
    with pytest.raises(PolicyLoadError, match="invalid YAML"):
        load_policy(bad)


def test_non_mapping_top_level_rejected() -> None:
    with pytest.raises(PolicyLoadError, match="mapping"):
        policy_from_dict(["not", "a", "dict"])


def test_unknown_field_rejected() -> None:
    with pytest.raises(PolicyLoadError):
        policy_from_dict({"version": 1, "name": "x", "entities": {"PERSON": {"acton": "redact"}}})


def test_entity_keys_must_be_upper_snake() -> None:
    with pytest.raises(PolicyLoadError, match="UPPER_SNAKE_CASE"):
        policy_from_dict({"version": 1, "name": "x", "entities": {"person": {"action": "redact"}}})


def test_unknown_destination_rejected() -> None:
    with pytest.raises(PolicyLoadError):
        policy_from_dict({"version": 1, "name": "x", "destinations": {"analytics": {}}})


def test_unknown_action_rejected() -> None:
    with pytest.raises(PolicyLoadError):
        policy_from_dict({"version": 1, "name": "x", "entities": {"PERSON": {"action": "nuke"}}})


def test_redact_template_must_reference_entity() -> None:
    with pytest.raises(PolicyLoadError, match="redact_template"):
        policy_from_dict({"version": 1, "name": "x", "redact_template": "[REDACTED]"})


def test_minimal_policy_uses_defaults() -> None:
    policy = Policy(version=1, name="minimal")
    assert policy.defaults.action == Action.REDACT
    assert policy.defaults.min_confidence == 0.5
    assert policy.entities == {}
