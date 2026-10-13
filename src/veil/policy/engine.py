"""Policy engine: turns detections into decisions and applies them to text."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from veil.detect.schema import Detection
from veil.policy.actions import Hasher, mask, redact
from veil.policy.resolve import ResolvedRule, resolve_rule
from veil.policy.schema import Action, Destination, Policy


class Decision(BaseModel):
    """One detection evaluated against policy for one destination.

    ``applied`` is False when confidence fell below the resolved threshold; the detection is
    still recorded (and audited) but the text is left untouched.
    """

    model_config = ConfigDict(frozen=True)

    detection: Detection
    destination: Destination
    action: Action
    applied: bool
    rule: ResolvedRule
    replacement: str | None = Field(
        default=None, description="Text substituted for the span, if any"
    )

    @property
    def blocks(self) -> bool:
        return self.applied and self.action == Action.BLOCK


class ScrubResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    destination: Destination
    decisions: tuple[Decision, ...]
    blocked: bool

    @property
    def blocking_decisions(self) -> tuple[Decision, ...]:
        return tuple(d for d in self.decisions if d.blocks)

    @property
    def applied_decisions(self) -> tuple[Decision, ...]:
        return tuple(d for d in self.decisions if d.applied)


class PolicyEngine:
    """Stateless evaluator over an immutable ``Policy``."""

    def __init__(self, policy: Policy, hasher: Hasher | None = None) -> None:
        self._policy = policy
        self._hasher = hasher or Hasher.from_env(policy.hash_secret_env, policy.hash_token_length)

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def hasher(self) -> Hasher:
        return self._hasher

    def decide(
        self, text: str, detections: Sequence[Detection], destination: Destination
    ) -> list[Decision]:
        return [self._decide_one(text, det, destination) for det in detections]

    def apply(
        self, text: str, detections: Sequence[Detection], destination: Destination
    ) -> ScrubResult:
        """Evaluate every detection and return the transformed text.

        If any applied decision is ``block`` the original text is returned unchanged with
        ``blocked=True``; callers must not forward it.
        """
        decisions = self.decide(text, detections, destination)
        if any(d.blocks for d in decisions):
            return ScrubResult(
                text=text, destination=destination, decisions=tuple(decisions), blocked=True
            )
        return ScrubResult(
            text=_substitute(text, decisions),
            destination=destination,
            decisions=tuple(decisions),
            blocked=False,
        )

    def _decide_one(self, text: str, det: Detection, destination: Destination) -> Decision:
        rule = resolve_rule(self._policy, det.entity_type, destination)
        applied = det.confidence >= rule.min_confidence
        replacement = self._replacement(text, det, rule) if applied else None
        return Decision(
            detection=det,
            destination=destination,
            action=rule.action,
            applied=applied,
            rule=rule,
            replacement=replacement,
        )

    def _replacement(self, text: str, det: Detection, rule: ResolvedRule) -> str | None:
        value = det.span_text(text)
        match rule.action:
            case Action.REDACT:
                return redact(det.entity_type, self._policy.redact_template)
            case Action.MASK:
                return mask(value, det.entity_type, rule.mask)
            case Action.HASH:
                return self._hasher.token(value, det.entity_type)
            case Action.ALLOW | Action.BLOCK:
                return None


def _substitute(text: str, decisions: Sequence[Decision]) -> str:
    """Apply replacements right-to-left so earlier offsets stay valid."""
    edits = sorted(
        (d for d in decisions if d.applied and d.replacement is not None),
        key=lambda d: d.detection.start,
        reverse=True,
    )
    out = text
    for d in edits:
        out = f"{out[: d.detection.start]}{d.replacement}{out[d.detection.end :]}"
    return out
