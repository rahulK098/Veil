"""Audit sinks: where records go. SQLite for real use, in-memory for tests."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from veil.audit.schema import DetectionRecord, OperationRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS operations (
    operation_id    TEXT PRIMARY KEY,
    request_id      TEXT,
    timestamp       TEXT NOT NULL,
    destination     TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    policy_name     TEXT NOT NULL,
    text_length     INTEGER NOT NULL,
    detection_count INTEGER NOT NULL,
    applied_count   INTEGER NOT NULL,
    latency_ms      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_operations_request ON operations(request_id);
CREATE INDEX IF NOT EXISTS ix_operations_time ON operations(timestamp);

CREATE TABLE IF NOT EXISTS detections (
    detection_id   TEXT PRIMARY KEY,
    operation_id   TEXT NOT NULL REFERENCES operations(operation_id),
    timestamp      TEXT NOT NULL,
    destination    TEXT NOT NULL,
    entity_type    TEXT NOT NULL,
    source         TEXT NOT NULL,
    recognizer     TEXT NOT NULL,
    start          INTEGER NOT NULL,
    "end"          INTEGER NOT NULL,
    confidence     REAL NOT NULL,
    threshold      REAL NOT NULL,
    action         TEXT NOT NULL,
    applied        INTEGER NOT NULL,
    action_rule    TEXT NOT NULL,
    threshold_rule TEXT NOT NULL,
    reason         TEXT,
    span_hash      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_detections_operation ON detections(operation_id);
CREATE INDEX IF NOT EXISTS ix_detections_entity ON detections(entity_type, action);
CREATE INDEX IF NOT EXISTS ix_detections_hash ON detections(span_hash);
"""

_INSERT_OPERATION = """
INSERT INTO operations VALUES (
    :operation_id, :request_id, :timestamp, :destination, :outcome, :policy_name,
    :text_length, :detection_count, :applied_count, :latency_ms
)
"""

_INSERT_DETECTION = """
INSERT INTO detections VALUES (
    :detection_id, :operation_id, :timestamp, :destination, :entity_type, :source,
    :recognizer, :start, :end, :confidence, :threshold, :action, :applied,
    :action_rule, :threshold_rule, :reason, :span_hash
)
"""


@runtime_checkable
class AuditSink(Protocol):
    def record(self, operation: OperationRecord, detections: Sequence[DetectionRecord]) -> None: ...

    def close(self) -> None: ...


class MemoryAuditSink:
    """Keeps records in lists. For tests and for callers that ship them elsewhere."""

    def __init__(self) -> None:
        self.operations: list[OperationRecord] = []
        self.detections: list[DetectionRecord] = []
        self._lock = threading.Lock()

    def record(self, operation: OperationRecord, detections: Sequence[DetectionRecord]) -> None:
        with self._lock:
            self.operations.append(operation)
            self.detections.extend(detections)

    def close(self) -> None:
        return None


class SQLiteAuditSink:
    """Append-only SQLite store. Safe for multi-threaded use within one process."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._path

    def record(self, operation: OperationRecord, detections: Sequence[DetectionRecord]) -> None:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute(_INSERT_OPERATION, _row(operation))
                self._conn.executemany(_INSERT_DETECTION, [_row(d) for d in detections])
                self._conn.execute("COMMIT")
            except sqlite3.Error:
                self._conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- query helpers used by the CLI / tests -----------------------------------------------

    def operations_for_request(self, request_id: str) -> list[OperationRecord]:
        rows = self._conn.execute(
            "SELECT * FROM operations WHERE request_id = ? ORDER BY timestamp", (request_id,)
        )
        return [OperationRecord.model_validate(_dict(rows, r)) for r in rows.fetchall()]

    def detections_for_operation(self, operation_id: str) -> list[DetectionRecord]:
        rows = self._conn.execute(
            'SELECT * FROM detections WHERE operation_id = ? ORDER BY start, "end"',
            (operation_id,),
        )
        return [DetectionRecord.model_validate(_dict(rows, r)) for r in rows.fetchall()]

    def explain(self, entity_type: str, applied: bool | None = None) -> list[DetectionRecord]:
        """All detection records for one entity type, newest first — the reviewer's query."""
        sql = "SELECT * FROM detections WHERE entity_type = ?"
        params: list[object] = [entity_type]
        if applied is not None:
            sql += " AND applied = ?"
            params.append(int(applied))
        sql += " ORDER BY timestamp DESC"
        rows = self._conn.execute(sql, params)
        return [DetectionRecord.model_validate(_dict(rows, r)) for r in rows.fetchall()]


def _row(record: OperationRecord | DetectionRecord) -> dict[str, object]:
    data = record.model_dump(mode="json")
    if "applied" in data:
        data["applied"] = int(bool(data["applied"]))
    return data


def _dict(cursor: sqlite3.Cursor, row: tuple[object, ...]) -> dict[str, object]:
    columns = [c[0] for c in cursor.description]
    data = dict(zip(columns, row, strict=True))
    if "applied" in data:
        data["applied"] = bool(data["applied"])
    return data


def to_jsonl(operation: OperationRecord, detections: Sequence[DetectionRecord]) -> str:
    """Serialize one operation and its detections as JSON lines, for shipping to a SIEM."""
    lines = [json.dumps({"kind": "operation", **operation.model_dump(mode="json")})]
    lines.extend(json.dumps({"kind": "detection", **d.model_dump(mode="json")}) for d in detections)
    return "\n".join(lines)
