"""Resolve the effective rule for an (entity_type, destination) pair.

Precedence is **specificity first, then destination scope** (ADR-0004):

1. ``destinations.<dest>.entities.<ENTITY>``   — specific entity, specific destination
2. ``entities.<ENTITY>``                        — specific entity, any destination
3. ``destinations.<dest>.defaults``             — any entity, specific destination
4. ``defaults``                                 — any entity, any destination

A destination-wide default therefore never loosens an entity-specific rule: ``US_SSN: block
at 0.0`` stays that way in ``logs`` even if ``logs.defaults.min_confidence`` is 0.3.

Each field (action, threshold, mask, reason) is resolved independently, and the *path* of the
rule that supplied it is recorded so an audit entry can cite exactly which line of policy fired.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from veil.policy.schema import Action, Destination, MaskOptions, Policy, Rule


class ResolvedRule(BaseModel):
    """The fully-specified rule that applies to one entity at one destination."""

    model_config = ConfigDict(frozen=True)

    entity_type: str
    destination: Destination
    action: Action
    action_rule: str
    min_confidence: float
    threshold_rule: str
    mask: MaskOptions
    reason: str | None = None


def resolve_rule(policy: Policy, entity_type: str, destination: Destination) -> ResolvedRule:
    layers = _layers(policy, entity_type, destination)

    action, action_rule = _first(layers, "action", policy.defaults.action, "defaults")
    threshold, threshold_rule = _first(
        layers, "min_confidence", policy.defaults.min_confidence, "defaults"
    )
    mask, _ = _first(layers, "mask", policy.defaults.mask, "defaults")
    reason, _ = _first(layers, "reason", None, "defaults")

    return ResolvedRule(
        entity_type=entity_type,
        destination=destination,
        action=action,
        action_rule=action_rule,
        min_confidence=threshold,
        threshold_rule=threshold_rule,
        mask=mask,
        reason=reason,
    )


def _layers(policy: Policy, entity_type: str, destination: Destination) -> list[tuple[str, Rule]]:
    """Partial rules in precedence order, paired with their policy path."""
    layers: list[tuple[str, Rule]] = []
    dest_policy = policy.destinations.get(destination)
    dest_prefix = f"destinations.{destination.value}"

    if dest_policy is not None and entity_type in dest_policy.entities:
        layers.append((f"{dest_prefix}.entities.{entity_type}", dest_policy.entities[entity_type]))
    if entity_type in policy.entities:
        layers.append((f"entities.{entity_type}", policy.entities[entity_type]))
    if dest_policy is not None:
        layers.append((f"{dest_prefix}.defaults", dest_policy.defaults))
    return layers


def _first[T](
    layers: list[tuple[str, Rule]], field: str, fallback: T, fallback_path: str
) -> tuple[T, str]:
    for path, rule in layers:
        value = getattr(rule, field)
        if value is not None:
            return value, path
    return fallback, fallback_path
