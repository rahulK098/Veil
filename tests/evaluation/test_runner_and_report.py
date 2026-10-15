"""Runs the full evaluation pipeline against the fake (regex) detection engine."""

import pytest

from veil.evaluation import Example, GoldSpan, evaluate, render_markdown
from veil.middleware import Guard

LABELLED = [
    Example(
        id="a",
        text="Hi Maria Gonzalez, your SSN 536-22-8741 is on file.",
        spans=(
            GoldSpan(entity_type="PERSON", start=3, end=17),
            GoldSpan(entity_type="US_SSN", start=28, end=39),
        ),
        category="pii",
    ),
    Example(
        id="b",
        text="key AKIAIOSFODNN7EXAMPLE leaked from 10.0.0.1",
        spans=(
            GoldSpan(entity_type="AWS_ACCESS_KEY", start=4, end=24),
            GoldSpan(entity_type="IP_ADDRESS", start=37, end=45),
        ),
        category="secrets",
    ),
    Example(
        id="c",
        text="John Smith wrote a note",
        spans=(GoldSpan(entity_type="PERSON", start=0, end=10),),
        category="pii",
    ),
]
CLEAN = [
    Example(
        id="c1", text="Quarterly revenue grew 12% year over year.", category="clean", kind="email"
    ),
    Example(
        id="c2", text="Bo approved the budget.", category="clean", kind="memo"
    ),  # fake PERSON @0.3
]


def test_evaluate_end_to_end(guard: Guard) -> None:
    result = evaluate(guard, LABELLED, CLEAN)

    by_entity = {m.entity_type: m for m in result.entity_metrics}
    assert by_entity["PERSON"].recall == 1.0 and by_entity["PERSON"].precision == 1.0
    assert by_entity["US_SSN"].tp == 1 and by_entity["AWS_ACCESS_KEY"].tp == 1
    assert result.misses == []

    assert result.clean.documents == 2
    assert result.clean.documents_with_any_detection == 1  # "Bo" @ 0.3, below every threshold
    assert result.clean.documents_modified == {}
    assert result.clean.documents_blocked == {}

    assert {lat.destination for lat in result.latency} == {"model", "logs"}
    assert all(lat.samples == 3 and lat.p95_ms >= lat.p50_ms for lat in result.latency)

    assert result.block_demo is not None
    assert result.block_demo.example_id == "a"
    assert result.block_demo.outcome == "blocked"
    assert any(row["entity_type"] == "US_SSN" for row in result.block_demo.detections)
    assert result.blocked_at_model == {"AWS_ACCESS_KEY": 1, "US_SSN": 1}


def test_render_markdown_contains_every_section(guard: Guard) -> None:
    result = evaluate(guard, LABELLED, CLEAN)
    md = render_markdown(result, title="T")
    for heading in (
        "# T",
        "## Precision / recall per entity type",
        "## Clean-corpus false positives",
        "## Latency",
        "## Blocks at the `model` destination",
        "## Demonstrated block",
        "## Misses",
        "## Spurious detections",
    ):
        assert heading in md
    assert "| `US_SSN` | 1 |" in md
    assert "BlockedError:" in md
    assert "536-22-8741" not in md  # the raw value never reaches the report


def test_render_markdown_with_no_block_and_misses(guard: Guard) -> None:
    only_clean = [Example(id="x", text="Maria Gonzalez called", spans=(), category="pii")]
    result = evaluate(guard, only_clean, [])
    md = render_markdown(result)
    assert "_No block demonstrated" in md
    assert "`PERSON`: 1. e.g. `Maria Gonzalez`" in md  # spurious detection listed
    assert "_none_" in md  # no false negatives


# --- hard set (ADR-0013) ----------

HARD_POS = [
    Example(
        id="hp-a",
        text="contact Maria Gonzalez at maria@example.com",
        spans=(
            GoldSpan(entity_type="PERSON", start=8, end=22),
            GoldSpan(entity_type="EMAIL_ADDRESS", start=26, end=43),
        ),
        category="hard_pii",
    ),
    Example(
        id="hp-b",
        text="ssn 5 3 6 2 2 8 7 4 1 spaced out",  # fake engine cannot find this
        spans=(GoldSpan(entity_type="US_SSN", start=4, end=21),),
        category="hard_pii",
    ),
]
HARD_NEG = [
    Example(id="hn-a", text="version 10.0.0.1 shipped", category="hard_negative"),
    Example(id="hn-b", text="order ORD-12345 dispatched", category="hard_negative"),
    Example(id="hn-c", text="ssn-shaped ledger code 536-22-8741", category="hard_negative"),
]


def test_evaluate_scores_the_hard_set(guard: Guard) -> None:
    result = evaluate(guard, LABELLED, CLEAN, hard_positives=HARD_POS, hard_negatives=HARD_NEG)
    hard = result.hard
    assert hard is not None
    assert hard.positive_spans == 3
    assert hard.positives_found == 2  # the spaced SSN is missed
    assert hard.missed_by_entity == {"US_SSN": 1}
    assert hard.positive_recall == pytest.approx(2 / 3)

    assert hard.negatives == 3
    assert hard.negatives_with_detection == 2  # version string -> IP, ledger code -> SSN
    assert "IP_ADDRESS" in hard.negative_detections_by_entity

    # The severe metric: PII-free text that policy would refuse outright.
    assert hard.negatives_blocked["model"] == 1
    assert hard.block_examples == ("hn-c",)
    assert hard.block_rate("model") == pytest.approx(1 / 3)


def test_hard_section_is_rendered(guard: Guard) -> None:
    result = evaluate(guard, LABELLED, CLEAN, hard_positives=HARD_POS, hard_negatives=HARD_NEG)
    md = render_markdown(result)
    assert "## Hard cases (curated)" in md
    assert "### Legitimate text that would be refused outright" in md
    assert "`hn-c`" in md
    assert "536-22-8741" not in md  # raw values never reach the report


def test_hard_section_absent_when_no_hard_data(guard: Guard) -> None:
    result = evaluate(guard, LABELLED, CLEAN)
    assert result.hard is None
    assert "## Hard cases" not in render_markdown(result)


def test_hard_section_handles_a_clean_sweep(guard: Guard) -> None:
    """No misses and no spurious detections must still render."""
    result = evaluate(
        guard,
        LABELLED,
        CLEAN,
        hard_positives=[HARD_POS[0]],
        hard_negatives=[Example(id="hn-x", text="nothing to see", category="hard_negative")],
    )
    assert result.hard is not None and result.hard.positive_recall == 1.0
    assert "_Nothing missed._" in render_markdown(result)
