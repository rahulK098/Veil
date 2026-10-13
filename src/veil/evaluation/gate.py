"""Regression floors: compare an evaluation report against committed thresholds.

Kept separate from ``runner`` because this is a *policy about acceptable quality*, not a
measurement. ``eval/gate.py`` is the CLI over this module.
"""

from __future__ import annotations

from typing import Any


def _precision(m: dict[str, Any]) -> float:
    denom = m["tp"] + m["fp"]
    return m["tp"] / denom if denom else 0.0


def _recall(m: dict[str, Any]) -> float:
    denom = m["tp"] + m["fn"]
    return m["tp"] / denom if denom else 0.0


def check(report: dict[str, Any], thresholds: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return ``(violations, notes)``.

    A violation fails the build. A note flags something a human should look at — currently
    entities that carry gold spans but have no floor, so new detections cannot slip in
    unmeasured.
    """
    violations: list[str] = []
    notes: list[str] = []

    examples = report["labelled_examples"]
    minimum = thresholds["min_labelled_examples"]
    if examples < minimum:
        violations.append(f"labelled examples: {examples} < {minimum}")

    documents = report["clean_documents"]
    minimum = thresholds["min_clean_documents"]
    if documents < minimum:
        violations.append(f"clean documents: {documents} < {minimum}")

    if thresholds.get("require_block_demo") and not report.get("block_demo"):
        violations.append("no block was demonstrated (block_demo is null)")

    metrics = {m["entity_type"]: m for m in report["entity_metrics"]}
    floors: dict[str, dict[str, float]] = thresholds.get("entities", {})

    for entity, floor in sorted(floors.items()):
        m = metrics.get(entity)
        if m is None:
            violations.append(f"{entity}: gated but absent from the report")
            continue
        support = m["tp"] + m["fn"]
        if support < floor["min_support"]:
            violations.append(
                f"{entity}: support {support} < {floor['min_support']} "
                "(dataset changed; re-baseline the floors deliberately)"
            )
        precision, recall = _precision(m), _recall(m)
        if precision < floor["precision"]:
            violations.append(f"{entity}: precision {precision:.3f} < {floor['precision']:.2f}")
        if recall < floor["recall"]:
            violations.append(f"{entity}: recall {recall:.3f} < {floor['recall']:.2f}")

    for entity, m in sorted(metrics.items()):
        if entity not in floors and m["tp"] + m["fn"] > 0:
            notes.append(
                f"{entity}: ungated ({m['tp'] + m['fn']} gold spans) — add a floor or ignore"
            )

    modified = report["clean"]["documents_modified"]
    limits = thresholds.get("clean_corpus", {}).get("max_documents_modified", {})
    for destination, limit in limits.items():
        actual = modified.get(destination, 0)
        if actual > limit:
            violations.append(
                f"clean corpus: {actual} documents modified at '{destination}' > {limit}"
            )

    return violations, notes
