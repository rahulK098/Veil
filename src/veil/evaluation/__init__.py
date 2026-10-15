"""Ship-gate evaluation: labelled synthetic data, per-entity P/R, clean-corpus FP, latency."""

from veil.evaluation.dataset import (
    Example,
    GoldSpan,
    generate_clean,
    generate_labelled,
    read_jsonl,
    write_jsonl,
)
from veil.evaluation.hard import generate_hard_negatives, generate_hard_positives
from veil.evaluation.metrics import (
    CleanCorpusMetrics,
    EntityMetrics,
    HardCaseMetrics,
    Miss,
    aggregate,
    score_example,
)
from veil.evaluation.report import render_markdown
from veil.evaluation.runner import EvalResult, evaluate, evaluate_hard

__all__ = [
    "CleanCorpusMetrics",
    "EntityMetrics",
    "HardCaseMetrics",
    "EvalResult",
    "Example",
    "GoldSpan",
    "Miss",
    "aggregate",
    "evaluate",
    "evaluate_hard",
    "generate_hard_negatives",
    "generate_hard_positives",
    "generate_clean",
    "generate_labelled",
    "read_jsonl",
    "render_markdown",
    "score_example",
    "write_jsonl",
]
