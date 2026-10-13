"""Streaming scrub tests (ADR-0010).

Most use the fake regex engine from conftest, so they run without spaCy. Chunk sizes are
deliberately small and awkward — splitting entities mid-token is the whole point.
"""

import asyncio
from collections.abc import AsyncIterator, Iterable

import pytest

from veil.audit import MemoryAuditSink, Outcome
from veil.middleware import BlockedError, Guard, StreamConfig, StreamMode
from veil.policy import Destination, Hasher, policy_from_dict
from veil.policy.resolve import can_block

SSN_TEXT = "Your file shows 536-22-8741 on record."


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


async def _aiter(items: Iterable[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


# --- mode selection ----------


def test_defaults_to_buffered_when_policy_can_block(guard: Guard) -> None:
    """The shipped policy blocks SSNs, so incremental emission cannot honour it."""
    assert can_block(guard.policy, Destination.RESPONSE)
    assert guard.stream_config().mode is StreamMode.BUFFERED


def test_defaults_to_incremental_when_no_block_is_reachable(sink: MemoryAuditSink) -> None:
    from tests.middleware.conftest import FakeDetectionEngine

    policy = policy_from_dict(
        {"version": 1, "name": "soft", "entities": {"PERSON": {"action": "redact"}}}
    )
    soft = Guard(policy, detection=FakeDetectionEngine(), audit_sink=sink, hasher=Hasher(b"k"))
    assert soft.stream_config().mode is StreamMode.INCREMENTAL


def test_mode_can_be_overridden_explicitly(guard: Guard) -> None:
    assert guard.stream_config(StreamMode.INCREMENTAL).mode is StreamMode.INCREMENTAL


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="tail_margin"):
        StreamConfig(tail_margin=0)
    with pytest.raises(ValueError, match="min_flush"):
        StreamConfig(min_flush=0)


# --- the straddling case ----------


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 13, 40])
def test_entity_split_across_chunks_is_still_caught(guard: Guard, size: int) -> None:
    """An SSN cut at any offset must still block — this is why chunks are buffered."""
    with pytest.raises(BlockedError) as excinfo:
        list(guard.outbound_stream(_chunks(SSN_TEXT, size)))
    assert excinfo.value.entity_types == ("US_SSN",)


@pytest.mark.parametrize("size", [1, 3, 7, 50])
def test_incremental_output_matches_non_streaming_output(guard: Guard, size: int) -> None:
    """Whatever the chunking, the concatenated stream equals the one-shot scrub."""
    text = (
        "Maria Gonzalez wrote in from 10.0.0.1 about the invoice, and asked that we reply to "
        "maria@example.com as soon as the billing team has reviewed the account in detail."
    )
    expected = guard.outbound(text)
    streamed = "".join(guard.outbound_stream(_chunks(text, size), mode=StreamMode.INCREMENTAL))
    assert streamed == expected


def test_buffered_mode_emits_nothing_until_the_end(guard: Guard) -> None:
    text = "Maria Gonzalez says hello " * 10
    pieces = list(guard.outbound_stream(_chunks(text, 10), mode=StreamMode.BUFFERED))
    assert len(pieces) == 1
    assert pieces[0] == guard.outbound(text)


def test_incremental_emits_before_the_stream_ends(guard: Guard) -> None:
    """The point of incremental mode: output starts before the last chunk arrives."""
    text = "The quarterly report is attached and covers revenue, churn and retention. " * 4
    emitted: list[str] = []
    stream = guard.outbound_stream(_chunks(text, 20), mode=StreamMode.INCREMENTAL)
    for piece in stream:
        emitted.append(piece)
    assert len(emitted) > 1, "expected more than one emission"
    assert "".join(emitted) == text


# --- boundaries and edges ----------


def test_empty_stream(guard: Guard, sink: MemoryAuditSink) -> None:
    assert list(guard.outbound_stream([])) == []
    assert sink.operations[0].text_length == 0
    assert sink.operations[0].outcome == Outcome.PASSED


def test_stream_of_empty_chunks(guard: Guard) -> None:
    assert list(guard.outbound_stream(["", "", ""])) == []


def test_entity_at_the_very_end_is_scrubbed(guard: Guard) -> None:
    text = "please contact maria@example.com"
    out = "".join(guard.outbound_stream(_chunks(text, 4), mode=StreamMode.INCREMENTAL))
    assert out == "please contact m****@example.com"


def test_no_entities_passes_through_unchanged(guard: Guard) -> None:
    text = "The release notes are published and the migration guide is linked from the index."
    out = "".join(guard.outbound_stream(_chunks(text, 6), mode=StreamMode.INCREMENTAL))
    assert out == text


def test_single_chunk_whole_completion(guard: Guard) -> None:
    out = "".join(guard.outbound_stream(["Contact maria@example.com today"]))
    assert out == "Contact m****@example.com today"


# --- audit ----------


def test_one_audit_operation_per_stream_not_per_chunk(guard: Guard, sink: MemoryAuditSink) -> None:
    text = "Maria Gonzalez and maria@example.com appear here. " * 3
    list(guard.outbound_stream(_chunks(text, 5), request_id="s1"))
    assert len(sink.operations) == 1
    op = sink.operations[0]
    assert op.request_id == "s1"
    assert op.destination == Destination.RESPONSE
    assert op.text_length == len(text)


def test_audit_span_hashes_match_the_real_values(guard: Guard, sink: MemoryAuditSink) -> None:
    """Offsets are rebased to the whole completion, so hashes must match the true spans."""
    text = "Ask Maria Gonzalez to confirm. " * 4 + "Then email maria@example.com for the receipt."
    list(guard.outbound_stream(_chunks(text, 7), mode=StreamMode.INCREMENTAL))
    hasher = Hasher(b"test-key")
    emails = [d for d in sink.detections if d.entity_type == "EMAIL_ADDRESS"]
    assert emails, "expected the email to be audited"
    assert emails[0].span_hash == hasher.digest("maria@example.com")
    assert all(
        d.span_hash == hasher.digest("Maria Gonzalez")
        for d in sink.detections
        if d.entity_type == "PERSON"
    )


def test_blocked_stream_is_audited_as_blocked(guard: Guard, sink: MemoryAuditSink) -> None:
    with pytest.raises(BlockedError):
        list(guard.outbound_stream(_chunks(SSN_TEXT, 4), request_id="s2"))
    assert sink.operations[0].outcome == Outcome.BLOCKED
    assert sink.operations[0].request_id == "s2"
    assert any(d.entity_type == "US_SSN" and d.applied for d in sink.detections)


def test_blocked_mid_stream_audit_hashes_the_right_span(
    guard: Guard, sink: MemoryAuditSink
) -> None:
    """A block found after some text was emitted must still hash the correct SSN span."""
    lead = "Thanks for waiting while we looked up the account details for you just now. "
    text = lead + "The number on file is 536-22-8741."
    with pytest.raises(BlockedError):
        list(guard.outbound_stream(_chunks(text, 9), mode=StreamMode.INCREMENTAL))
    ssn = next(d for d in sink.detections if d.entity_type == "US_SSN")
    assert ssn.span_hash == Hasher(b"test-key").digest("536-22-8741")
    assert text[ssn.start : ssn.end] == "536-22-8741"


# --- async ----------


def test_async_stream_matches_sync(guard: Guard) -> None:
    text = "Maria Gonzalez asked about maria@example.com and the invoice total for last month."

    async def run() -> str:
        return "".join(
            [
                piece
                async for piece in guard.aoutbound_stream(
                    _aiter(_chunks(text, 6)), mode=StreamMode.INCREMENTAL
                )
            ]
        )

    assert asyncio.run(run()) == guard.outbound(text)


def test_async_stream_blocks(guard: Guard, sink: MemoryAuditSink) -> None:
    async def run() -> list[str]:
        return [piece async for piece in guard.aoutbound_stream(_aiter(_chunks(SSN_TEXT, 3)))]

    with pytest.raises(BlockedError):
        asyncio.run(run())
    assert sink.operations[0].outcome == Outcome.BLOCKED
