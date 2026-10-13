"""Exceptions raised by the middleware."""

from __future__ import annotations

from veil.audit.schema import OperationRecord
from veil.policy.engine import ScrubResult


class BlockedError(RuntimeError):
    """Raised when policy blocks an inbound prompt or outbound completion.

    Carries the full ``ScrubResult`` and the audit ``OperationRecord`` so the caller can
    show a precise reason ("US_SSN blocked by entities.US_SSN") without re-scanning.
    The original text is intentionally *not* included in ``str(self)``.
    """

    def __init__(self, result: ScrubResult, operation: OperationRecord) -> None:
        self.result = result
        self.operation = operation
        reasons = ", ".join(
            f"{d.detection.entity_type} (rule {d.rule.action_rule}, "
            f"confidence {d.detection.confidence:.2f})"
            for d in result.blocking_decisions
        )
        super().__init__(
            f"blocked at destination '{result.destination.value}': {reasons} "
            f"[audit operation {operation.operation_id}]"
        )

    @property
    def entity_types(self) -> tuple[str, ...]:
        return tuple(d.detection.entity_type for d in self.result.blocking_decisions)
