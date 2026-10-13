"""Audit record schema (ADR-0006).

Two record types:

* ``OperationRecord`` — one per guarded operation (a prompt, a completion, a log write):
  what destination, what outcome, how long the scan took.
* ``DetectionRecord`` — one per detection evaluated in that operation, whether or not the
  action was applied. Stores a keyed hash of the span, never the span itself.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from veil.detect.schema import Source
from veil.policy.schema import Action, Destination


class Outcome(StrEnum):
    PASSED = "passed"  # no applied decisions; text unchanged
    SCRUBBED = "scrubbed"  # at least one replacement applied
    BLOCKED = "blocked"  # operation rejected (model/response path raised)
    DROPPED = "dropped"  # write suppressed (logs/vector_store path)


def _now() -> datetime:
    return datetime.now(tz=UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class OperationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation_id: str = Field(default_factory=new_id)
    request_id: str | None = Field(
        default=None, description="Caller-supplied correlation id spanning inbound/outbound"
    )
    timestamp: datetime = Field(default_factory=_now)
    destination: Destination
    outcome: Outcome
    policy_name: str
    text_length: int = Field(ge=0)
    detection_count: int = Field(ge=0)
    applied_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)


class DetectionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    detection_id: str = Field(default_factory=new_id)
    operation_id: str
    timestamp: datetime = Field(default_factory=_now)
    destination: Destination
    entity_type: str
    source: Source
    recognizer: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    confidence: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    action: Action
    applied: bool
    action_rule: str = Field(description="Policy path that decided the action")
    threshold_rule: str = Field(description="Policy path that decided the threshold")
    reason: str | None = None
    span_hash: str = Field(
        min_length=64,
        max_length=64,
        description="HMAC-SHA256 of the original span (ADR-0007); never the raw value",
    )
