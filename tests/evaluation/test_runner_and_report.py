"""Runs the full evaluation pipeline against the fake (regex) detection engine."""

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
