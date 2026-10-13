import asyncio
import logging
from types import SimpleNamespace

import pytest

from veil.audit import MemoryAuditSink, Outcome
from veil.middleware import (
    BlockedError,
    ChatMessagesAdapter,
    Guard,
    TextAdapter,
    VeilLogFilter,
    redacted_call,
)
from veil.policy import Destination

SAFE = "Maria Gonzalez asks about maria@example.com"
SSN = "my ssn is 536-22-8741"


# --- TextAdapter ----------


def test_text_adapter_default_wraps_positional_str(guard: Guard, sink: MemoryAuditSink) -> None:
    @redacted_call(guard)
    def llm(prompt: str) -> str:
        assert prompt == SAFE  # model path allows PERSON + EMAIL
        return "Reach me at maria@example.com"

    assert llm(SAFE) == "Reach me at m****@example.com"
    assert [op.destination for op in sink.operations] == [Destination.MODEL, Destination.RESPONSE]


def test_text_adapter_keyword_argument(guard: Guard) -> None:
    @redacted_call(guard, adapter=TextAdapter("prompt"))
    def llm(*, prompt: str, model: str) -> str:
        return f"{model}: ok"

    assert llm(prompt=SAFE, model="m") == "m: ok"


def test_text_adapter_rejects_non_str(guard: Guard) -> None:
    @redacted_call(guard)
    def llm(prompt: object) -> str:
        return "x"

    with pytest.raises(TypeError, match="expected str"):
        llm(123)


def test_decorator_blocks_before_calling_the_model(guard: Guard) -> None:
    called = False

    @redacted_call(guard)
    def llm(prompt: str) -> str:
        nonlocal called
        called = True
        return "never"

    with pytest.raises(BlockedError):
        llm(SSN)
    assert not called


def test_decorator_blocks_outbound_completion(guard: Guard, sink: MemoryAuditSink) -> None:
    @redacted_call(guard)
    def llm(prompt: str) -> str:
        return f"Your record shows {SSN}"

    with pytest.raises(BlockedError):
        llm("what is on file?")
    assert sink.operations[-1].outcome == Outcome.BLOCKED
    assert sink.operations[-1].destination == Destination.RESPONSE


def test_decorator_request_id_correlates_both_records(guard: Guard, sink: MemoryAuditSink) -> None:
    @redacted_call(guard, request_id=lambda prompt, rid: rid)
    def llm(prompt: str, rid: str) -> str:
        return "ok"

    llm(SAFE, "req-7")
    assert [op.request_id for op in sink.operations] == ["req-7", "req-7"]


def test_decorator_supports_async(guard: Guard) -> None:
    @redacted_call(guard)
    async def llm(prompt: str) -> str:
        await asyncio.sleep(0)
        return "Contact maria@example.com"

    assert asyncio.run(llm(SAFE)) == "Contact m****@example.com"


def test_decorator_non_str_result_passes_through(guard: Guard) -> None:
    @redacted_call(guard)
    def llm(prompt: str) -> dict[str, int]:
        return {"tokens": 3}

    assert llm(SAFE) == {"tokens": 3}


# --- ChatMessagesAdapter ----------


def _openai_like(*contents: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=c)) for c in contents]
    )


def _anthropic_like(*texts: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=t) for t in texts])


def test_chat_adapter_scrubs_each_message_and_openai_response(guard: Guard) -> None:
    seen: list[dict[str, str]] = []

    @redacted_call(guard, adapter=ChatMessagesAdapter())
    def chat(*, model: str, messages: list[dict[str, str]]) -> SimpleNamespace:
        seen.extend(messages)
        return _openai_like("Email is maria@example.com", "second choice 10.0.0.1")

    original = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": SAFE},
    ]
    result = chat(model="gpt", messages=original)
    assert seen[1]["content"] == SAFE  # allowed at model destination
    assert original[1]["content"] == SAFE  # caller's list not mutated
    assert result.choices[0].message.content == "Email is m****@example.com"
    assert result.choices[1].message.content.startswith("second choice <IP_ADDRESS:")


def test_chat_adapter_handles_anthropic_shape(guard: Guard) -> None:
    @redacted_call(guard, adapter=ChatMessagesAdapter())
    def chat(*, messages: list[dict[str, str]]) -> SimpleNamespace:
        return _anthropic_like("Sure, maria@example.com it is.")

    result = chat(messages=[{"role": "user", "content": "hi"}])
    assert result.content[0].text == "Sure, m****@example.com it is."


def test_chat_adapter_leaves_original_response_intact(guard: Guard) -> None:
    response = _openai_like("maria@example.com")

    @redacted_call(guard, adapter=ChatMessagesAdapter())
    def chat(*, messages: list[dict[str, str]]) -> SimpleNamespace:
        return response

    scrubbed = chat(messages=[{"role": "user", "content": "hi"}])
    assert scrubbed is not response
    assert response.choices[0].message.content == "maria@example.com"


def test_chat_adapter_skips_non_string_content(guard: Guard) -> None:
    @redacted_call(guard, adapter=ChatMessagesAdapter())
    def chat(*, messages: list[dict[str, object]]) -> str:
        assert isinstance(messages[0]["content"], list)
        return "ok"

    assert chat(messages=[{"role": "user", "content": [{"type": "image"}]}]) == "ok"


def test_chat_adapter_requires_list(guard: Guard) -> None:
    @redacted_call(guard, adapter=ChatMessagesAdapter())
    def chat(*, messages: object) -> str:
        return "ok"

    with pytest.raises(TypeError, match="expected a list"):
        chat(messages="not a list")


def test_chat_adapter_plain_string_result(guard: Guard) -> None:
    @redacted_call(guard, adapter=ChatMessagesAdapter())
    def chat(*, messages: list[dict[str, str]]) -> str:
        return "maria@example.com"

    assert chat(messages=[{"role": "user", "content": "hi"}]) == "m****@example.com"


# --- VeilLogFilter ----------


def test_log_filter_scrubs_formatted_message(
    guard: Guard, sink: MemoryAuditSink, caplog: pytest.LogCaptureFixture
) -> None:
    logger = logging.getLogger("veil.test.filter")
    logger.addFilter(VeilLogFilter(guard))
    try:
        with caplog.at_level(logging.INFO, logger="veil.test.filter"):
            logger.info("user %s wrote in", "Maria Gonzalez", extra={"request_id": "r9"})
    finally:
        logger.filters.clear()
    assert "Maria Gonzalez" not in caplog.text
    assert "<PERSON:" in caplog.text
    assert sink.operations[0].destination == Destination.LOGS
    assert sink.operations[0].request_id == "r9"


def test_log_filter_drops_blocked_record(
    guard: Guard, sink: MemoryAuditSink, caplog: pytest.LogCaptureFixture
) -> None:
    logger = logging.getLogger("veil.test.filter2")
    logger.addFilter(VeilLogFilter(guard))
    try:
        with caplog.at_level(logging.INFO, logger="veil.test.filter2"):
            logger.info("customer ssn %s", "536-22-8741")
    finally:
        logger.filters.clear()
    assert caplog.records == []
    assert sink.operations[0].outcome == Outcome.DROPPED
