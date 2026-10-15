"""The curated hard set must stay well-formed, and its gate must actually fail (ADR-0013)."""

from typing import Any

import pytest

from veil.evaluation import generate_hard_negatives, generate_hard_positives
from veil.evaluation.gate import check


def test_hard_positive_gold_values_are_unambiguous() -> None:
    """Offsets are derived from a substring search, so each value must occur exactly once."""
    for ex in generate_hard_positives():
        for span in ex.spans:
            value = ex.span_text(span)
            assert ex.text.count(value) == 1, (ex.id, value)


def test_hard_positive_spans_do_not_overlap() -> None:
    for ex in generate_hard_positives():
        for a, b in zip(ex.spans, ex.spans[1:], strict=False):
            assert a.end <= b.start, ex.id


def test_hard_positives_cover_the_intended_difficulty_classes() -> None:
    ids = {ex.id for ex in generate_hard_positives()}
    assert {"hp-lowercase", "hp-json", "hp-url-query", "hp-spaced-ssn", "hp-uk-phone"} <= ids
    assert all(ex.spans for ex in generate_hard_positives())


def test_hard_negatives_carry_no_gold_spans() -> None:
    negatives = generate_hard_negatives()
    assert len(negatives) >= 20
    assert all(ex.spans == () and ex.category == "hard_negative" for ex in negatives)


def test_ids_are_unique_across_both_halves() -> None:
    ids = [ex.id for ex in (*generate_hard_positives(), *generate_hard_negatives())]
    assert len(ids) == len(set(ids))


# --- the gate ----------

_BASE: dict[str, Any] = {
    "labelled_examples": 160,
    "clean_documents": 100,
    "block_demo": {"id": "a"},
    "entity_metrics": [],
    "clean": {"documents_modified": {}},
    "hard": {
        "positive_spans": 26,
        "positives_found": 18,
        "negatives": 20,
        "negatives_with_detection": 16,
        "negatives_blocked": {"model": 2},
    },
}
_FLOORS: dict[str, Any] = {
    "min_labelled_examples": 150,
    "min_clean_documents": 100,
    "hard": {
        "min_positive_recall": 0.65,
        "max_negative_fp_rate": 0.85,
        "max_blocked": {"model": 2},
    },
}


def test_gate_passes_at_the_measured_level() -> None:
    assert check(_BASE, _FLOORS)[0] == []


def test_gate_fails_when_hard_recall_drops() -> None:
    report = {**_BASE, "hard": {**_BASE["hard"], "positives_found": 10}}
    assert any("hard positives: recall" in v for v in check(report, _FLOORS)[0])


def test_gate_fails_when_hard_false_positives_rise() -> None:
    report = {**_BASE, "hard": {**_BASE["hard"], "negatives_with_detection": 20}}
    assert any("hard negatives: fp rate" in v for v in check(report, _FLOORS)[0])


def test_gate_fails_when_more_pii_free_text_would_be_blocked() -> None:
    """The severe regression: refusing a legitimate request."""
    report = {**_BASE, "hard": {**_BASE["hard"], "negatives_blocked": {"model": 5}}}
    violations = check(report, _FLOORS)[0]
    assert any("5 PII-free documents blocked at 'model' > 2" in v for v in violations)


def test_gate_fails_when_hard_set_is_gated_but_missing() -> None:
    report = {k: v for k, v in _BASE.items() if k != "hard"}
    assert "hard set is gated but absent from the report" in check(report, _FLOORS)[0]


@pytest.mark.parametrize("key", ["positive_spans", "negatives"])
def test_gate_tolerates_an_empty_hard_set(key: str) -> None:
    report = {**_BASE, "hard": {**_BASE["hard"], key: 0}}
    check(report, _FLOORS)  # must not raise ZeroDivisionError
