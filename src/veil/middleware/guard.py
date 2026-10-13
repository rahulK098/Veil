"""The ``Guard``: one object wiring detection, policy and audit for a deployment (ADR-0005).

Three surfaces, three code paths:

* ``inbound(text)``      — prompt about to go to the model. Blocks raise ``BlockedError``.
* ``outbound(text)``     — completion about to be returned. Blocks raise ``BlockedError``.
* ``for_storage(text)``  — text about to be written to logs / a vector store. Blocks do
  **not** raise; the write is dropped (``None`` returned) and audited as ``dropped``, because a
  log line must never take down the user's request.

``outbound_stream`` / ``aoutbound_stream`` cover the same ground as ``outbound`` for a
completion that arrives in chunks (ADR-0010).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from veil.audit.recorder import AuditRecorder
from veil.audit.schema import Outcome
from veil.audit.sink import AuditSink, MemoryAuditSink, SQLiteAuditSink
from veil.detect.engine import DetectionEngine
from veil.middleware.errors import BlockedError
from veil.middleware.streaming import (
    StreamBlockedError,
    StreamConfig,
    StreamMode,
    StreamScrubber,
)
from veil.policy.actions import Hasher
from veil.policy.engine import PolicyEngine, ScrubResult
from veil.policy.loader import load_policy
from veil.policy.resolve import can_block
from veil.policy.schema import Destination, Policy

_STORAGE_DESTINATIONS = frozenset({Destination.LOGS, Destination.VECTOR_STORE})


class Guard:
    def __init__(
        self,
        policy: Policy,
        *,
        detection: DetectionEngine | None = None,
        audit_sink: AuditSink | None = None,
        hasher: Hasher | None = None,
    ) -> None:
        self._policy = policy
        self._detection = detection or DetectionEngine(policy.detection)
        self._policy_engine = PolicyEngine(policy, hasher=hasher)
        self._audit = AuditRecorder(
            audit_sink or MemoryAuditSink(), self._policy_engine.hasher, policy.name
        )

    @classmethod
    def from_policy(
        cls,
        policy_path: str | Path,
        *,
        audit_db: str | Path | None = None,
        detection: DetectionEngine | None = None,
        hasher: Hasher | None = None,
    ) -> Guard:
        sink: AuditSink = SQLiteAuditSink(audit_db) if audit_db else MemoryAuditSink()
        return cls(load_policy(policy_path), detection=detection, audit_sink=sink, hasher=hasher)

    # --- accessors ----------

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def audit(self) -> AuditRecorder:
        return self._audit

    @property
    def detection(self) -> DetectionEngine:
        return self._detection

    @property
    def hasher(self) -> Hasher:
        return self._policy_engine.hasher

    def warm_up(self) -> None:
        self._detection.warm_up()

    def close(self) -> None:
        self._audit.sink.close()

    # --- the three surfaces ----------

    def inbound(self, prompt: str, *, request_id: str | None = None) -> str:
        """Scrub a prompt before it reaches the model. Raises ``BlockedError``."""
        return self._raise_if_blocked(prompt, Destination.MODEL, request_id)

    def outbound(self, completion: str, *, request_id: str | None = None) -> str:
        """Scrub a completion before it is returned to the caller. Raises ``BlockedError``."""
        return self._raise_if_blocked(completion, Destination.RESPONSE, request_id)

    def for_storage(
        self,
        text: str,
        destination: Destination = Destination.LOGS,
        *,
        request_id: str | None = None,
    ) -> str | None:
        """Scrub text before writing it to logs or a vector store.

        Returns the scrubbed text, or ``None`` if policy blocked it — the caller must skip
        the write. Never raises for policy reasons.
        """
        if destination not in _STORAGE_DESTINATIONS:
            raise ValueError(
                f"for_storage expects a storage destination, got {destination.value!r}; "
                "use inbound()/outbound() for model/response"
            )
        result, latency = self._scan(text, destination)
        if result.blocked:
            self._audit.record(
                text, result, outcome=Outcome.DROPPED, latency_ms=latency, request_id=request_id
            )
            return None
        self._audit.record(
            text, result, outcome=_outcome(result), latency_ms=latency, request_id=request_id
        )
        return result.text

    def for_logs(self, text: str, *, request_id: str | None = None) -> str | None:
        return self.for_storage(text, Destination.LOGS, request_id=request_id)

    def for_vector_store(self, text: str, *, request_id: str | None = None) -> str | None:
        return self.for_storage(text, Destination.VECTOR_STORE, request_id=request_id)

    def scan(self, text: str, destination: Destination) -> ScrubResult:
        """Evaluate without auditing or raising. For evals and dry runs only."""
        result, _ = self._scan(text, destination)
        return result

    # --- streaming ----------

    def stream_config(self, mode: StreamMode | None = None, **overrides: int) -> StreamConfig:
        """Resolve the streaming mode for the response destination.

        With ``mode=None`` (the default) ``BUFFERED`` is chosen whenever the policy could
        block, because incremental emission cannot honour a block — earlier chunks are already
        gone. Pass ``mode`` explicitly to override, accepting the weaker guarantee.
        """
        if mode is None:
            mode = (
                StreamMode.BUFFERED
                if can_block(self._policy, Destination.RESPONSE)
                else StreamMode.INCREMENTAL
            )
        return StreamConfig(mode=mode, **overrides)

    def outbound_stream(
        self,
        chunks: Iterable[str],
        *,
        request_id: str | None = None,
        mode: StreamMode | None = None,
        config: StreamConfig | None = None,
    ) -> Iterator[str]:
        """Scrub a streamed completion, yielding text that is safe to emit.

        Raises ``BlockedError`` the moment a block fires. Chunks already yielded are, by the
        nature of streaming, already with the caller — see ADR-0010. One audit operation is
        written for the whole stream, not one per chunk.
        """
        scrubber = StreamScrubber(
            scan=lambda text: self._policy_engine.apply(
                text, self._detection.detect(text), Destination.RESPONSE
            ),
            config=config or self.stream_config(mode),
        )
        started = time.perf_counter()
        try:
            for chunk in chunks:
                emitted = scrubber.feed(chunk)
                if emitted:
                    yield emitted
            tail = scrubber.finish()
            if tail:
                yield tail
        except StreamBlockedError as blocked:
            raise self._stream_blocked(scrubber, blocked, started, request_id) from None
        self._record_stream(scrubber, started, request_id)

    async def aoutbound_stream(
        self,
        chunks: AsyncIterable[str],
        *,
        request_id: str | None = None,
        mode: StreamMode | None = None,
        config: StreamConfig | None = None,
    ) -> AsyncIterator[str]:
        """``outbound_stream`` for an async iterator of chunks."""
        scrubber = StreamScrubber(
            scan=lambda text: self._policy_engine.apply(
                text, self._detection.detect(text), Destination.RESPONSE
            ),
            config=config or self.stream_config(mode),
        )
        started = time.perf_counter()
        try:
            async for chunk in chunks:
                emitted = scrubber.feed(chunk)
                if emitted:
                    yield emitted
            tail = scrubber.finish()
            if tail:
                yield tail
        except StreamBlockedError as blocked:
            raise self._stream_blocked(scrubber, blocked, started, request_id) from None
        self._record_stream(scrubber, started, request_id)

    # --- session ----------

    @contextmanager
    def session(self, request_id: str | None = None) -> Iterator[GuardSession]:
        """``with guard.session("req-1") as g:`` binds one request id to every call inside."""
        yield GuardSession(self, request_id)

    # --- internals ----------

    def _scan(self, text: str, destination: Destination) -> tuple[ScrubResult, float]:
        started = time.perf_counter()
        detections = self._detection.detect(text)
        result = self._policy_engine.apply(text, detections, destination)
        return result, (time.perf_counter() - started) * 1000.0

    def _record_stream(
        self, scrubber: StreamScrubber, started: float, request_id: str | None
    ) -> None:
        """One audit operation for the whole stream, not one per chunk."""
        decisions = scrubber.decisions
        result = ScrubResult(
            text="",
            destination=Destination.RESPONSE,
            decisions=decisions,
            blocked=False,
        )
        self._audit.record(
            scrubber.raw_text,
            result,
            outcome=_outcome(result),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            request_id=request_id,
        )

    def _stream_blocked(
        self,
        scrubber: StreamScrubber,
        blocked: StreamBlockedError,
        started: float,
        request_id: str | None,
    ) -> BlockedError:
        operation = self._audit.record(
            scrubber.raw_text,
            blocked.result,
            outcome=Outcome.BLOCKED,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            request_id=request_id,
        )
        return BlockedError(blocked.result, operation)

    def _raise_if_blocked(self, text: str, destination: Destination, request_id: str | None) -> str:
        result, latency = self._scan(text, destination)
        if result.blocked:
            operation = self._audit.record(
                text, result, outcome=Outcome.BLOCKED, latency_ms=latency, request_id=request_id
            )
            raise BlockedError(result, operation)
        self._audit.record(
            text, result, outcome=_outcome(result), latency_ms=latency, request_id=request_id
        )
        return result.text


class GuardSession:
    """A ``Guard`` view with a fixed ``request_id`` so inbound/outbound/log events correlate."""

    def __init__(self, guard: Guard, request_id: str | None) -> None:
        self._guard = guard
        self.request_id = request_id

    def inbound(self, prompt: str) -> str:
        return self._guard.inbound(prompt, request_id=self.request_id)

    def outbound(self, completion: str) -> str:
        return self._guard.outbound(completion, request_id=self.request_id)

    def for_logs(self, text: str) -> str | None:
        return self._guard.for_logs(text, request_id=self.request_id)

    def for_vector_store(self, text: str) -> str | None:
        return self._guard.for_vector_store(text, request_id=self.request_id)


def _outcome(result: ScrubResult) -> Outcome:
    changed = any(d.replacement is not None for d in result.applied_decisions)
    return Outcome.SCRUBBED if changed else Outcome.PASSED
