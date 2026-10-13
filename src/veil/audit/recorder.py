"""Turns a ``ScrubResult`` into audit records and writes them to a sink."""

from __future__ import annotations

from veil.audit.schema import DetectionRecord, OperationRecord, Outcome
from veil.audit.sink import AuditSink
from veil.policy.actions import Hasher
from veil.policy.engine import ScrubResult


class AuditRecorder:
    def __init__(self, sink: AuditSink, hasher: Hasher, policy_name: str) -> None:
        self._sink = sink
        self._hasher = hasher
        self._policy_name = policy_name

    @property
    def sink(self) -> AuditSink:
        return self._sink

    def record(
        self,
        original_text: str,
        result: ScrubResult,
        *,
        outcome: Outcome,
        latency_ms: float,
        request_id: str | None = None,
    ) -> OperationRecord:
        operation = OperationRecord(
            request_id=request_id,
            destination=result.destination,
            outcome=outcome,
            policy_name=self._policy_name,
            text_length=len(original_text),
            detection_count=len(result.decisions),
            applied_count=len(result.applied_decisions),
            latency_ms=latency_ms,
        )
        detections = [
            DetectionRecord(
                operation_id=operation.operation_id,
                destination=result.destination,
                entity_type=d.detection.entity_type,
                source=d.detection.source,
                recognizer=d.detection.recognizer,
                start=d.detection.start,
                end=d.detection.end,
                confidence=d.detection.confidence,
                threshold=d.rule.min_confidence,
                action=d.action,
                applied=d.applied,
                action_rule=d.rule.action_rule,
                threshold_rule=d.rule.threshold_rule,
                reason=d.rule.reason,
                span_hash=self._hasher.digest(d.detection.span_text(original_text)),
            )
            for d in result.decisions
        ]
        self._sink.record(operation, detections)
        return operation
