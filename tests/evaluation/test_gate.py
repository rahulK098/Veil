"""The regression gate must actually fail — these pin each way it can."""

from pathlib import Path
from typing import Any

import yaml

from veil.evaluation.gate import check

ROOT = Path(__file__).resolve().parents[2]


def _report(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "labelled_examples": 160,
        "clean_documents": 100,
        "block_demo": {"example_id": "a"},
        "entity_metrics": [
            {"entity_type": "US_SSN", "tp": 26, "fp": 0, "fn": 0},
            {"entity_type": "PERSON", "tp": 76, "fp": 1, "fn": 2},
        ],
        "clean": {"documents_modified": {"model": 1, "logs": 35}},
    }
    return {**base, **overrides}


THRESHOLDS: dict[str, Any] = {
    "min_labelled_examples": 150,
    "min_clean_documents": 100,
    "require_block_demo": True,
    "entities": {
        "US_SSN": {"min_support": 26, "precision": 1.0, "recall": 1.0},
        "PERSON": {"min_support": 78, "precision": 0.95, "recall": 0.95},
    },
    "clean_corpus": {"max_documents_modified": {"model": 3}},
}


def test_passes_when_every_floor_is_met() -> None:
    violations, notes = check(_report(), THRESHOLDS)
    assert violations == []
    assert notes == []


def test_fails_on_precision_drop() -> None:
    report = _report(
        entity_metrics=[
            {"entity_type": "US_SSN", "tp": 26, "fp": 0, "fn": 0},
            {"entity_type": "PERSON", "tp": 76, "fp": 20, "fn": 2},  # precision 0.79
        ]
    )
    (violation,) = check(report, THRESHOLDS)[0]
    assert violation == "PERSON: precision 0.792 < 0.95"


def test_fails_on_recall_drop() -> None:
    report = _report(
        entity_metrics=[
            {"entity_type": "US_SSN", "tp": 13, "fp": 0, "fn": 13},  # recall 0.5
            {"entity_type": "PERSON", "tp": 76, "fp": 1, "fn": 2},
        ]
    )
    violations = check(report, THRESHOLDS)[0]
    assert any("US_SSN: recall 0.500 < 1.00" in v for v in violations)


def test_fails_when_a_gated_entity_disappears() -> None:
    report = _report(entity_metrics=[{"entity_type": "PERSON", "tp": 76, "fp": 1, "fn": 2}])
    violations = check(report, THRESHOLDS)[0]
    assert "US_SSN: gated but absent from the report" in violations


def test_fails_when_dataset_shrinks() -> None:
    violations = check(_report(labelled_examples=20), THRESHOLDS)[0]
    assert "labelled examples: 20 < 150" in violations


def test_fails_when_support_shrinks_so_floors_cannot_be_gamed() -> None:
    """Deleting hard examples would otherwise raise precision/recall for free."""
    report = _report(
        entity_metrics=[
            {"entity_type": "US_SSN", "tp": 2, "fp": 0, "fn": 0},  # perfect, but only 2 spans
            {"entity_type": "PERSON", "tp": 76, "fp": 1, "fn": 2},
        ]
    )
    violations = check(report, THRESHOLDS)[0]
    assert any("US_SSN: support 2 < 26" in v for v in violations)


def test_fails_when_no_block_is_demonstrated() -> None:
    violations = check(_report(block_demo=None), THRESHOLDS)[0]
    assert "no block was demonstrated (block_demo is null)" in violations


def test_fails_on_clean_corpus_over_redaction() -> None:
    report = _report(clean={"documents_modified": {"model": 40}})
    violations = check(report, THRESHOLDS)[0]
    assert "clean corpus: 40 documents modified at 'model' > 3" in violations


def test_ungated_entity_is_noted_not_failed() -> None:
    report = _report(
        entity_metrics=[
            *_report()["entity_metrics"],
            {"entity_type": "NEW_THING", "tp": 5, "fp": 0, "fn": 1},
        ]
    )
    violations, notes = check(report, THRESHOLDS)
    assert violations == []
    assert any("NEW_THING: ungated (6 gold spans)" in n for n in notes)


def test_zero_support_entity_is_neither_gated_nor_noted() -> None:
    report = _report(
        entity_metrics=[
            *_report()["entity_metrics"],
            {"entity_type": "NOISE", "tp": 0, "fp": 3, "fn": 0},
        ]
    )
    assert check(report, THRESHOLDS)[1] == []


def test_shipped_thresholds_file_is_valid_and_covers_the_ship_gate() -> None:
    thresholds = yaml.safe_load((ROOT / "eval" / "thresholds.yaml").read_text(encoding="utf-8"))
    assert thresholds["min_labelled_examples"] >= 150  # the plan's ship gate
    assert thresholds["require_block_demo"] is True
    assert {"US_SSN", "CREDIT_CARD", "AWS_ACCESS_KEY"} <= set(thresholds["entities"])
    for entity, floor in thresholds["entities"].items():
        assert {"min_support", "precision", "recall"} <= set(floor), entity
