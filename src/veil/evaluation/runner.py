"""Runs the ship-gate evaluation end to end and produces a structured result."""

from __future__ import annotations

import statistics
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from veil.audit.sink import MemoryAuditSink
from veil.evaluation.dataset import Example
from veil.evaluation.metrics import (
    CleanCorpusMetrics,
    EntityMetrics,
    HardCaseMetrics,
    Miss,
    aggregate,
    score_example,
)
from veil.middleware.errors import BlockedError
from veil.middleware.guard import Guard
from veil.policy.schema import Destination

_LATENCY_DESTINATIONS = (Destination.MODEL, Destination.LOGS)


class LatencyStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    destination: str
    samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_text_length: float


class BlockDemo(BaseModel):
    """One concrete blocked request with its audit trail, for the report."""

    model_config = ConfigDict(frozen=True)

    example_id: str
    destination: str
    error_message: str
    operation_id: str
    outcome: str
    detections: list[dict[str, object]]


class EvalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    policy_name: str
    labelled_examples: int
    clean_documents: int
    entity_metrics: list[EntityMetrics]
    misses: list[Miss]
    clean: CleanCorpusMetrics
    hard: HardCaseMetrics | None = None
    latency: list[LatencyStats]
    block_demo: BlockDemo | None
    blocked_at_model: dict[str, int] = Field(
        description="How many labelled examples the model destination blocked, by entity"
    )


def evaluate_hard(
    guard: Guard,
    positives: Sequence[Example],
    negatives: Sequence[Example],
) -> HardCaseMetrics:
    """Score the curated hard set (ADR-0013)."""
    found = 0
    spans = 0
    missed: Counter[str] = Counter()
    miss_examples: list[Miss] = []
    for ex in positives:
        spans += len(ex.spans)
        _, misses = score_example(ex, guard.detection.detect(ex.text))
        for miss in misses:
            if miss.kind == "fn":
                missed[miss.entity_type] += 1
                miss_examples.append(miss)
        found += len(ex.spans) - sum(1 for m in misses if m.kind == "fn")

    with_detection = 0
    by_entity: Counter[str] = Counter()
    blocked: Counter[str] = Counter()
    block_ids: list[str] = []
    for ex in negatives:
        detections = guard.detection.detect(ex.text)
        if detections:
            with_detection += 1
        by_entity.update(d.entity_type for d in detections)
        blocks_anywhere = False
        for dest in Destination:
            if guard.scan(ex.text, dest).blocked:
                blocked[dest.value] += 1
                blocks_anywhere = True
        if blocks_anywhere:
            block_ids.append(ex.id)

    return HardCaseMetrics(
        positives=len(positives),
        positive_spans=spans,
        positives_found=found,
        missed_by_entity=dict(sorted(missed.items())),
        miss_examples=tuple(miss_examples),
        negatives=len(negatives),
        negatives_with_detection=with_detection,
        negative_detections_by_entity=dict(sorted(by_entity.items())),
        negatives_blocked=dict(blocked),
        block_examples=tuple(block_ids),
    )


def evaluate(
    guard: Guard,
    labelled: Sequence[Example],
    clean: Sequence[Example],
    hard_positives: Sequence[Example] = (),
    hard_negatives: Sequence[Example] = (),
) -> EvalResult:
    guard.warm_up()
    per_example: list[dict[str, list[int]]] = []
    misses: list[Miss] = []
    for ex in labelled:
        detections = guard.detection.detect(ex.text)
        counts, ex_misses = score_example(ex, detections)
        per_example.append(counts)
        misses.extend(ex_misses)

    return EvalResult(
        policy_name=guard.policy.name,
        labelled_examples=len(labelled),
        clean_documents=len(clean),
        entity_metrics=aggregate(per_example),
        misses=misses,
        clean=_clean_corpus(guard, clean),
        hard=(
            evaluate_hard(guard, hard_positives, hard_negatives)
            if hard_positives or hard_negatives
            else None
        ),
        latency=[_latency(guard, labelled, dest) for dest in _LATENCY_DESTINATIONS],
        block_demo=_block_demo(guard, labelled),
        blocked_at_model=_blocked_at_model(guard, labelled),
    )


def _clean_corpus(guard: Guard, clean: Sequence[Example]) -> CleanCorpusMetrics:
    by_entity: Counter[str] = Counter()
    with_any = 0
    modified: Counter[str] = Counter()
    blocked: Counter[str] = Counter()
    for ex in clean:
        detections = guard.detection.detect(ex.text)
        if detections:
            with_any += 1
        by_entity.update(d.entity_type for d in detections)
        for dest in Destination:
            result = guard.scan(ex.text, dest)
            if result.blocked:
                blocked[dest.value] += 1
            elif result.text != ex.text:
                modified[dest.value] += 1
    return CleanCorpusMetrics(
        documents=len(clean),
        documents_with_any_detection=with_any,
        detections_by_entity=dict(sorted(by_entity.items())),
        documents_modified=dict(modified),
        documents_blocked=dict(blocked),
    )


def _latency(guard: Guard, examples: Sequence[Example], destination: Destination) -> LatencyStats:
    """Time the full guarded path (detect + policy + audit) per example."""
    sink = MemoryAuditSink()
    timed = Guard(guard.policy, detection=guard.detection, audit_sink=sink, hasher=guard.hasher)
    samples: list[float] = []
    for ex in examples:
        started = time.perf_counter()
        try:
            if destination is Destination.MODEL:
                timed.inbound(ex.text)
            else:
                timed.for_storage(ex.text, destination)
        except BlockedError:
            pass
        samples.append((time.perf_counter() - started) * 1000.0)
    if len(samples) >= 2:
        quantiles = statistics.quantiles(samples, n=100, method="inclusive")
        p95, p99 = quantiles[94], quantiles[98]
    else:  # quantiles need two points; a single sample is its own percentile
        p95 = p99 = samples[0] if samples else 0.0
    return LatencyStats(
        destination=destination.value,
        samples=len(samples),
        p50_ms=round(statistics.median(samples), 2) if samples else 0.0,
        p95_ms=round(p95, 2),
        p99_ms=round(p99, 2),
        max_ms=round(max(samples), 2) if samples else 0.0,
        mean_text_length=round(statistics.fmean(len(ex.text) for ex in examples), 1)
        if examples
        else 0.0,
    )


def _block_demo(guard: Guard, examples: Sequence[Example]) -> BlockDemo | None:
    """Find an outbound completion (the model echoing an SSN) that policy blocks outright."""
    sink = MemoryAuditSink()
    demo_guard = Guard(
        guard.policy, detection=guard.detection, audit_sink=sink, hasher=guard.hasher
    )
    for ex in examples:
        if not any(s.entity_type == "US_SSN" for s in ex.spans):
            continue
        completion = f"Certainly. Based on your records: {ex.text}"
        try:
            demo_guard.outbound(completion, request_id=f"demo-{ex.id}")
        except BlockedError as err:
            return BlockDemo(
                example_id=ex.id,
                destination=Destination.RESPONSE.value,
                error_message=str(err),
                operation_id=err.operation.operation_id,
                outcome=err.operation.outcome.value,
                detections=[
                    d.model_dump(mode="json", exclude={"detection_id", "operation_id", "timestamp"})
                    for d in sink.detections
                    if d.operation_id == err.operation.operation_id
                ],
            )
    return None


def _blocked_at_model(guard: Guard, examples: Sequence[Example]) -> dict[str, int]:
    blocked: Counter[str] = Counter()
    for ex in examples:
        result = guard.scan(ex.text, Destination.MODEL)
        blocked.update({d.detection.entity_type for d in result.blocking_decisions})
    return dict(sorted(blocked.items()))
