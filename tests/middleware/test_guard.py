import pytest

from veil.audit import MemoryAuditSink, Outcome
from veil.middleware import BlockedError, Guard
from veil.policy import Destination, Hasher

PROMPT = "Hi, I'm Maria Gonzalez (maria@example.com). My SSN is 536-22-8741."
SAFE_PROMPT = "Hi, I'm Maria Gonzalez (maria@example.com), can you help with my order?"


def test_inbound_allows_person_and_email_for_model(guard: Guard, sink: MemoryAuditSink) -> None:
    out = guard.inbound(SAFE_PROMPT, request_id="r1")
    assert out == SAFE_PROMPT
    (op,) = sink.operations
    assert op.outcome == Outcome.PASSED
    assert op.destination == Destination.MODEL
    assert op.request_id == "r1"
    assert {d.entity_type for d in sink.detections} == {"PERSON", "EMAIL_ADDRESS"}
    assert all(d.action == "allow" and d.applied for d in sink.detections)


def test_inbound_blocks_ssn_with_traceable_reason(guard: Guard, sink: MemoryAuditSink) -> None:
    with pytest.raises(BlockedError) as excinfo:
        guard.inbound(PROMPT, request_id="r2")
    err = excinfo.value
    assert err.entity_types == ("US_SSN",)
    assert "entities.US_SSN" in str(err)
    assert "536-22-8741" not in str(err)  # never leak the value in the exception
    (op,) = sink.operations
    assert op.outcome == Outcome.BLOCKED
    assert err.operation.operation_id == op.operation_id
    ssn = next(d for d in sink.detections if d.entity_type == "US_SSN")
    assert ssn.action == "block" and ssn.applied and ssn.action_rule == "entities.US_SSN"


def test_outbound_blocks_hallucinated_ssn(guard: Guard, sink: MemoryAuditSink) -> None:
    completion = "Sure! Your SSN on file is 536-22-8741."
    with pytest.raises(BlockedError):
        guard.outbound(completion)
    assert sink.operations[0].destination == Destination.RESPONSE
    assert sink.operations[0].outcome == Outcome.BLOCKED


def test_outbound_masks_email_because_response_lacks_override(guard: Guard) -> None:
    out = guard.outbound("Contact maria@example.com for details.")
    assert out == "Contact m****@example.com for details."


def test_for_logs_hashes_person_and_email(guard: Guard, sink: MemoryAuditSink) -> None:
    line = guard.for_logs(SAFE_PROMPT, request_id="r3")
    assert line is not None
    h = Hasher(b"test-key")
    assert h.token("Maria Gonzalez", "PERSON") in line
    assert h.token("maria@example.com", "EMAIL_ADDRESS") in line
    assert "Maria" not in line and "maria@" not in line
    assert sink.operations[0].outcome == Outcome.SCRUBBED


def test_for_logs_drops_line_instead_of_raising_on_block(
    guard: Guard, sink: MemoryAuditSink
) -> None:
    assert guard.for_logs(PROMPT) is None
    assert sink.operations[0].outcome == Outcome.DROPPED


def test_for_vector_store_redacts(guard: Guard) -> None:
    out = guard.for_vector_store(SAFE_PROMPT)
    assert out == "Hi, I'm <PERSON> (<EMAIL_ADDRESS>), can you help with my order?"


def test_for_storage_rejects_non_storage_destination(guard: Guard) -> None:
    with pytest.raises(ValueError, match="storage destination"):
        guard.for_storage("x", Destination.MODEL)


def test_below_threshold_person_left_alone_but_audited(guard: Guard, sink: MemoryAuditSink) -> None:
    out = guard.for_vector_store("Bo is here")  # fake engine tags "Bo" as PERSON @ 0.3
    assert out == "Bo is here"
    (rec,) = sink.detections
    assert rec.entity_type == "PERSON" and not rec.applied and rec.threshold == 0.5


def test_session_binds_request_id_across_surfaces(guard: Guard, sink: MemoryAuditSink) -> None:
    with guard.session("req-42") as g:
        g.inbound(SAFE_PROMPT)
        g.outbound("Happy to help!")
        g.for_logs(SAFE_PROMPT)
    assert [op.request_id for op in sink.operations] == ["req-42"] * 3
    assert [op.destination for op in sink.operations] == [
        Destination.MODEL,
        Destination.RESPONSE,
        Destination.LOGS,
    ]


def test_scan_does_not_audit(guard: Guard, sink: MemoryAuditSink) -> None:
    result = guard.scan(PROMPT, Destination.MODEL)
    assert result.blocked
    assert sink.operations == []


def test_latency_is_recorded(guard: Guard, sink: MemoryAuditSink) -> None:
    guard.inbound(SAFE_PROMPT)
    assert sink.operations[0].latency_ms >= 0.0


def test_from_policy_with_sqlite(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from tests.middleware.conftest import ROOT, FakeDetectionEngine

    monkeypatch.setenv("VEIL_HASH_SECRET", "k")
    g = Guard.from_policy(
        ROOT / "policies" / "default.yaml",
        audit_db=tmp_path / "a.db",
        detection=FakeDetectionEngine(),
    )
    g.inbound(SAFE_PROMPT, request_id="x")
    from veil.audit import SQLiteAuditSink

    assert isinstance(g.audit.sink, SQLiteAuditSink)
    assert len(g.audit.sink.operations_for_request("x")) == 1
    g.close()
