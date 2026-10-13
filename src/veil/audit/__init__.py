"""Structured audit log: every detection, every decision, traceable to a policy rule."""

from veil.audit.recorder import AuditRecorder
from veil.audit.schema import DetectionRecord, OperationRecord, Outcome
from veil.audit.sink import AuditSink, MemoryAuditSink, SQLiteAuditSink, to_jsonl

__all__ = [
    "AuditRecorder",
    "AuditSink",
    "DetectionRecord",
    "MemoryAuditSink",
    "OperationRecord",
    "Outcome",
    "SQLiteAuditSink",
    "to_jsonl",
]
