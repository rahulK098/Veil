"""Veil — policy-driven PII/secrets scrubber for LLM inputs, outputs, and logs.

Typical use::

    import veil

    guard = veil.Guard.from_policy("policies/default.yaml", audit_db="audit.db")

    @veil.redacted_call(guard, adapter=veil.ChatMessagesAdapter())
    def chat(*, messages): ...

    safe_line = guard.for_logs(f"user said: {prompt}")
"""

from veil.audit import AuditRecorder, MemoryAuditSink, Outcome, SQLiteAuditSink
from veil.detect import Detection, DetectionEngine, DetectorConfig, Source
from veil.middleware import (
    BlockedError,
    ChatMessagesAdapter,
    Guard,
    GuardSession,
    TextAdapter,
    VeilLogFilter,
    redacted_call,
)
from veil.policy import Action, Destination, Policy, PolicyEngine, ScrubResult, load_policy

__all__ = [
    "Action",
    "AuditRecorder",
    "BlockedError",
    "ChatMessagesAdapter",
    "Destination",
    "Detection",
    "DetectionEngine",
    "DetectorConfig",
    "Guard",
    "GuardSession",
    "MemoryAuditSink",
    "Outcome",
    "Policy",
    "PolicyEngine",
    "SQLiteAuditSink",
    "ScrubResult",
    "Source",
    "TextAdapter",
    "VeilLogFilter",
    "load_policy",
    "redacted_call",
]
__version__ = "0.1.0"
