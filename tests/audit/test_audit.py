import json
from pathlib import Path

import pytest

from veil.audit import (
    AuditRecorder,
    MemoryAuditSink,
    Outcome,
    SQLiteAuditSink,
    to_jsonl,
)
from veil.detect import Detection, Source
from veil.policy import Destination, Hasher, PolicyEngine, policy_from_dict

POLICY = policy_from_dict(
    {
        "version": 1,
        "name": "audit-test",
        "entities": {
            "PERSON": {"action": "redact", "min_confidence": 0.7, "reason": "why"},
            "US_SSN": {"action": "block", "min_confidence": 0.0},
        },
        "destinations": {"model": {"entities": {"PERSON": {"action": "allow"}}}},
    }
)
HASHER = Hasher(b"audit-key")
ENGINE = PolicyEngine(POLICY, hasher=HASHER)
TEXT = "Maria Gonzalez ssn 536-22-8741"
PERSON = Detection(
    entity_type="PERSON",
    start=0,
    end=14,
    confidence=0.85,
    source=Source.PRESIDIO,
    recognizer="Spacy",
)
WEAK_PERSON = PERSON.model_copy(update={"confidence": 0.4})
SSN = Detection(
    entity_type="US_SSN",
    start=19,
    end=30,
    confidence=0.5,
    source=Source.PRESIDIO,
    recognizer="UsSsn",
)


def test_memory_sink_records_operation_and_detections() -> None:
    sink = MemoryAuditSink()
    recorder = AuditRecorder(sink, HASHER, POLICY.name)
    result = ENGINE.apply(TEXT, [PERSON, WEAK_PERSON], Destination.LOGS)

    op = recorder.record(TEXT, result, outcome=Outcome.SCRUBBED, latency_ms=1.5, request_id="r1")

    assert sink.operations == [op]
    assert op.detection_count == 2 and op.applied_count == 1
    assert op.policy_name == "audit-test"
    applied, skipped = sink.detections
    assert applied.applied and applied.action_rule == "entities.PERSON"
    assert not skipped.applied and skipped.threshold == 0.7
    assert applied.reason == "why"


def test_span_hash_is_keyed_hmac_not_raw_value() -> None:
    sink = MemoryAuditSink()
    recorder = AuditRecorder(sink, HASHER, POLICY.name)
    result = ENGINE.apply(TEXT, [PERSON], Destination.LOGS)
    recorder.record(TEXT, result, outcome=Outcome.SCRUBBED, latency_ms=0)
    (rec,) = sink.detections
    assert rec.span_hash == HASHER.digest("Maria Gonzalez")
    assert "Maria" not in rec.model_dump_json()


def test_allowed_detection_is_still_audited_with_rule_path() -> None:
    """A reviewer must be able to see *why* PERSON went to the model untouched."""
    sink = MemoryAuditSink()
    recorder = AuditRecorder(sink, HASHER, POLICY.name)
    result = ENGINE.apply(TEXT, [PERSON], Destination.MODEL)
    recorder.record(TEXT, result, outcome=Outcome.PASSED, latency_ms=0)
    (rec,) = sink.detections
    assert rec.action == "allow"
    assert rec.action_rule == "destinations.model.entities.PERSON"
    assert rec.threshold_rule == "entities.PERSON"


def test_sqlite_sink_round_trips_and_never_stores_raw_span(tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    sink = SQLiteAuditSink(db)
    recorder = AuditRecorder(sink, HASHER, POLICY.name)
    result = ENGINE.apply(TEXT, [PERSON, SSN], Destination.MODEL)
    op = recorder.record(TEXT, result, outcome=Outcome.BLOCKED, latency_ms=2.25, request_id="req-9")

    ops = sink.operations_for_request("req-9")
    assert [o.operation_id for o in ops] == [op.operation_id]
    assert ops[0].outcome == Outcome.BLOCKED
    assert ops[0].latency_ms == 2.25

    dets = sink.detections_for_operation(op.operation_id)
    assert [d.entity_type for d in dets] == ["PERSON", "US_SSN"]
    assert dets[1].action == "block" and dets[1].applied

    ssn_events = sink.explain("US_SSN", applied=True)
    assert len(ssn_events) == 1 and ssn_events[0].action_rule == "entities.US_SSN"
    sink.close()

    raw = db.read_bytes()
    assert b"536-22-8741" not in raw
    assert b"Maria" not in raw


def test_sqlite_sink_is_append_only_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    for i in range(2):
        sink = SQLiteAuditSink(db)
        recorder = AuditRecorder(sink, HASHER, POLICY.name)
        result = ENGINE.apply(TEXT, [PERSON], Destination.LOGS)
        recorder.record(TEXT, result, outcome=Outcome.SCRUBBED, latency_ms=0, request_id=f"r{i}")
        sink.close()
    sink = SQLiteAuditSink(db)
    assert len(sink.explain("PERSON")) == 2
    sink.close()


def test_to_jsonl_emits_one_line_per_record() -> None:
    sink = MemoryAuditSink()
    recorder = AuditRecorder(sink, HASHER, POLICY.name)
    result = ENGINE.apply(TEXT, [PERSON, SSN], Destination.LOGS)
    op = recorder.record(TEXT, result, outcome=Outcome.DROPPED, latency_ms=0)
    lines = to_jsonl(op, sink.detections).splitlines()
    assert len(lines) == 3
    kinds = [json.loads(line)["kind"] for line in lines]
    assert kinds == ["operation", "detection", "detection"]


def test_memory_sink_close_is_noop() -> None:
    assert MemoryAuditSink().close() is None


@pytest.mark.parametrize("outcome", list(Outcome))
def test_outcomes_serialize(outcome: Outcome) -> None:
    assert Outcome(outcome.value) is outcome
