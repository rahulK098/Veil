"""Span-level precision/recall per entity type, and clean-corpus false-positive rates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from veil.detect.schema import Detection
from veil.evaluation.dataset import Example, GoldSpan

MATCH_IOU = 0.5


class EntityMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_type: str
    support: int = Field(description="Gold spans of this type")
    tp: int
    fp: int
    fn: int
    exact: int = Field(description="TPs whose offsets match the gold span exactly")

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def exact_rate(self) -> float:
        return self.exact / self.tp if self.tp else 0.0


class Miss(BaseModel):
    model_config = ConfigDict(frozen=True)

    example_id: str
    entity_type: str
    span_text: str
    kind: str = Field(description="fn (missed gold) | fp (spurious detection)")
    predicted_as: str | None = None


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union else 0.0


def score_example(
    example: Example, detections: Sequence[Detection]
) -> tuple[dict[str, list[int]], list[Miss]]:
    """Return per-entity [tp, fp, fn, exact] counts and the individual misses for one example.

    A detection matches a gold span when the entity types agree and IoU >= MATCH_IOU. Each gold
    span can be matched at most once; unmatched detections are false positives.
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    misses: list[Miss] = []
    unmatched_gold: set[GoldSpan] = set(example.spans)
    used: set[int] = set()

    for gold in example.spans:
        best_index, best_iou = -1, 0.0
        for index, det in enumerate(detections):
            if index in used or det.entity_type != gold.entity_type:
                continue
            iou = _iou(gold.start, gold.end, det.start, det.end)
            if iou > best_iou:
                best_index, best_iou = index, iou
        if best_iou >= MATCH_IOU:
            used.add(best_index)
            unmatched_gold.discard(gold)
            counts[gold.entity_type][0] += 1
            det = detections[best_index]
            if (det.start, det.end) == (gold.start, gold.end):
                counts[gold.entity_type][3] += 1

    for gold in unmatched_gold:
        counts[gold.entity_type][2] += 1
        overlapping = [
            d.entity_type
            for d in detections
            if _iou(gold.start, gold.end, d.start, d.end) > 0 and d.entity_type != gold.entity_type
        ]
        misses.append(
            Miss(
                example_id=example.id,
                entity_type=gold.entity_type,
                span_text=example.span_text(gold),
                kind="fn",
                predicted_as=overlapping[0] if overlapping else None,
            )
        )

    for index, det in enumerate(detections):
        if index in used:
            continue
        # A detection overlapping a gold span of a *different* type is a type confusion; it is
        # both an FN above and an FP here, which is the honest accounting.
        counts[det.entity_type][1] += 1
        misses.append(
            Miss(
                example_id=example.id,
                entity_type=det.entity_type,
                span_text=det.span_text(example.text),
                kind="fp",
            )
        )
    return counts, misses


def aggregate(per_example: Iterable[dict[str, list[int]]]) -> list[EntityMetrics]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for counts in per_example:
        for entity, (tp, fp, fn, exact) in counts.items():
            totals[entity][0] += tp
            totals[entity][1] += fp
            totals[entity][2] += fn
            totals[entity][3] += exact
    return sorted(
        (
            EntityMetrics(entity_type=e, support=tp + fn, tp=tp, fp=fp, fn=fn, exact=exact)
            for e, (tp, fp, fn, exact) in totals.items()
        ),
        key=lambda m: m.entity_type,
    )


class CleanCorpusMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents: int
    documents_with_any_detection: int
    detections_by_entity: dict[str, int]
    documents_modified: dict[str, int] = Field(
        description="Per destination: documents whose scrubbed text differs from the original"
    )
    documents_blocked: dict[str, int]

    @property
    def detection_fp_rate(self) -> float:
        return self.documents_with_any_detection / self.documents if self.documents else 0.0

    def modified_rate(self, destination: str) -> float:
        return (
            self.documents_modified.get(destination, 0) / self.documents if self.documents else 0.0
        )
