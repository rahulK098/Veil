"""Policy engine: entity -> action rules with thresholds and per-destination overrides."""

from veil.policy.actions import Hasher
from veil.policy.engine import Decision, PolicyEngine, ScrubResult
from veil.policy.loader import PolicyLoadError, load_policy, policy_from_dict
from veil.policy.resolve import ResolvedRule, resolve_rule
from veil.policy.schema import (
    Action,
    Defaults,
    Destination,
    DestinationPolicy,
    MaskOptions,
    Policy,
    Rule,
)

__all__ = [
    "Action",
    "Decision",
    "Defaults",
    "Destination",
    "DestinationPolicy",
    "Hasher",
    "MaskOptions",
    "Policy",
    "PolicyEngine",
    "PolicyLoadError",
    "ResolvedRule",
    "Rule",
    "ScrubResult",
    "load_policy",
    "policy_from_dict",
    "resolve_rule",
]
