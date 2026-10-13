"""Scrubbing a completion that arrives in chunks (ADR-0010).

The hard part is that **emitted text cannot be recalled**. Two consequences shape everything
here:

1. An entity can straddle a chunk boundary (``"...SSN 536-"`` then ``"22-8741..."``). Scanning
   each chunk alone finds nothing, so chunks are accumulated and the *buffer* is scanned.
2. By the time a ``block`` fires mid-stream, earlier chunks are already with the caller. No
   amount of cleverness undoes that.

Hence two modes, and a default that refuses to half-promise:

* ``BUFFERED`` — accumulate the whole completion, scrub once, emit at the end. Every guarantee
  of the non-streaming path holds; time-to-first-token does not.
* ``INCREMENTAL`` — emit early, holding back a tail margin. Guarantee is narrower and stated
  precisely in ``StreamScrubber``.

``Guard.outbound_stream`` picks ``BUFFERED`` automatically whenever the policy could block at
the destination, so the weaker mode is always an explicit choice.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from veil.detect.schema import Detection
from veil.policy.engine import Decision, ScrubResult

# Covers every entity the shipped policy acts on except PRIVATE_KEY and long JWTs; see the
# guarantee in StreamScrubber and the limitation noted in ADR-0010.
DEFAULT_TAIL_MARGIN = 96

# Scanning costs ~10 ms, and token-by-token chunks are ~4 characters. Without this, a 400-token
# completion would trigger 400 scans. Growth of this many characters is required before the
# buffer is rescanned.
DEFAULT_MIN_FLUSH = 64


class StreamMode(StrEnum):
    BUFFERED = "buffered"
    INCREMENTAL = "incremental"


@dataclass(frozen=True)
class StreamConfig:
    mode: StreamMode = StreamMode.BUFFERED
    tail_margin: int = DEFAULT_TAIL_MARGIN
    min_flush: int = DEFAULT_MIN_FLUSH

    def __post_init__(self) -> None:
        if self.tail_margin < 1:
            raise ValueError("tail_margin must be >= 1")
        if self.min_flush < 1:
            raise ValueError("min_flush must be >= 1")


class StreamBlockedError(Exception):
    """Internal signal: policy blocked mid-stream. ``Guard`` converts this to ``BlockedError``."""

    def __init__(self, result: ScrubResult) -> None:
        super().__init__("blocked mid-stream")
        self.result = result


ScanFn = Callable[[str], ScrubResult]


@dataclass
class StreamScrubber:
    """Buffers chunks and releases scrubbed text that is safe to emit.

    **Guarantee (INCREMENTAL mode):** no entity up to ``tail_margin`` characters long is ever
    partially emitted. Text is only released up to ``len(buffer) - tail_margin``, pulled back
    to a whitespace boundary and further back past the start of any detection that extends
    beyond the cut. An entity *longer* than ``tail_margin`` may have a prefix emitted before it
    is complete enough to detect — which is precisely why a policy that can block does not use
    this mode.

    ``scan`` takes raw text and returns a ``ScrubResult``; the scrubber never touches the
    detection or policy engines itself, which keeps it unit-testable without spaCy.
    """

    scan: ScanFn
    config: StreamConfig = field(default_factory=StreamConfig)

    _buffer: str = field(default="", init=False)
    _raw: str = field(default="", init=False)
    _consumed: int = field(default=0, init=False)
    _since_scan: int = field(default=0, init=False)
    _decisions: list[Decision] = field(default_factory=list, init=False)

    @property
    def decisions(self) -> tuple[Decision, ...]:
        """Every decision so far, with offsets absolute to the whole completion."""
        return tuple(self._decisions)

    @property
    def raw_text(self) -> str:
        """Everything fed in so far, unscrubbed.

        Retained for the duration of the stream so the audit log can hash each span against
        the text it actually came from. The non-streaming path holds the whole completion
        too, so this is no worse; it is discarded with the scrubber.
        """
        return self._raw

    def feed(self, chunk: str) -> str:
        """Accept a chunk; return text that is safe to emit now (often ``""``)."""
        self._buffer += chunk
        self._raw += chunk
        self._since_scan += len(chunk)

        if self.config.mode is StreamMode.BUFFERED:
            return ""
        if self._since_scan < self.config.min_flush:
            return ""
        if len(self._buffer) <= self.config.tail_margin:
            return ""

        self._since_scan = 0
        return self._release()

    def finish(self) -> str:
        """Flush everything remaining, scrubbed. Safe to call on an empty stream."""
        if not self._buffer:
            return ""
        result = self.scan(self._buffer)
        self._raise_if_blocked(result)
        self._record(result.decisions, offset=self._consumed)
        self._consumed += len(self._buffer)
        self._buffer = ""
        return result.text

    # --- internals ----------

    def _release(self) -> str:
        result = self.scan(self._buffer)
        self._raise_if_blocked(result)

        limit = self._emit_limit([d.detection for d in result.decisions])
        if limit <= 0:
            return ""

        prefix = self._buffer[:limit]
        kept = [d for d in result.decisions if d.detection.end <= limit]
        emitted = _substitute(prefix, kept)

        self._record(kept, offset=self._consumed)
        self._consumed += limit
        self._buffer = self._buffer[limit:]
        return emitted

    def _emit_limit(self, detections: Sequence[Detection]) -> int:
        """Furthest offset in the buffer that is safe to release."""
        limit = len(self._buffer) - self.config.tail_margin

        # Never cut through a detection we already know about.
        for det in detections:
            if det.start < limit < det.end:
                limit = det.start

        # Never cut mid-word: the trailing word may be the start of an entity still arriving.
        boundary = _last_boundary(self._buffer, limit)
        return max(boundary, 0)

    def _raise_if_blocked(self, result: ScrubResult) -> None:
        """Re-raise with offsets absolute to the whole completion.

        The scan ran against the retained buffer, so its offsets are buffer-relative. The
        audit log hashes each span out of ``raw_text``; handing it buffer offsets would hash
        the wrong characters.
        """
        if not result.blocked:
            return
        raise StreamBlockedError(
            result.model_copy(
                update={
                    "text": self._raw,
                    "decisions": _rebase(result.decisions, self._consumed),
                }
            )
        )

    def _record(self, decisions: Sequence[Decision], offset: int) -> None:
        """Store decisions with offsets rebased onto the whole completion, for the audit log."""
        self._decisions.extend(_rebase(decisions, offset))


def _rebase(decisions: Sequence[Decision], offset: int) -> tuple[Decision, ...]:
    """Shift decision offsets from buffer-relative to completion-absolute."""
    if offset == 0:
        return tuple(decisions)
    shifted: list[Decision] = []
    for decision in decisions:
        detection = decision.detection
        shifted.append(
            decision.model_copy(
                update={
                    "detection": detection.model_copy(
                        update={
                            "start": detection.start + offset,
                            "end": detection.end + offset,
                        }
                    )
                }
            )
        )
    return tuple(shifted)


def _last_boundary(text: str, limit: int) -> int:
    """Index just after the last whitespace at or before ``limit``; 0 if there is none."""
    if limit <= 0:
        return 0
    for index in range(min(limit, len(text)) - 1, -1, -1):
        if text[index].isspace():
            return index + 1
    return 0


def _substitute(text: str, decisions: Sequence[Decision]) -> str:
    """Apply replacements right-to-left so earlier offsets stay valid."""
    edits = sorted(
        (d for d in decisions if d.applied and d.replacement is not None),
        key=lambda d: d.detection.start,
        reverse=True,
    )
    out = text
    for d in edits:
        out = f"{out[: d.detection.start]}{d.replacement}{out[d.detection.end :]}"
    return out
