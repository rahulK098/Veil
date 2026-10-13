"""Ship-gate evaluation: labelled synthetic data, per-entity P/R, clean-corpus FP, latency."""

from veil.evaluation.dataset import (
    Example,
    GoldSpan,
    generate_clean,
    generate_labelled,
    read_jsonl,
    write_jsonl,
)
from veil.evaluation.metrics import (
    CleanCorpusMetrics,
    EntityMetrics,
    Miss,
    aggregate,
    score_example,
)
from veil.evaluation.report import render_markdown
from veil.evaluation.runner import EvalResult, evaluate

__all__ = [
    "CleanCorpusMetrics",
    "EntityMetrics",
    "EvalResult",
    "Example",
    "GoldSpan",
    "Miss",
    "aggregate",
    "evaluate",
    "generate_clean",
    "generate_labelled",
    "read_jsonl",
    "render_markdown",
    "score_example",
    "write_jsonl",
]
