"""Run the ship-gate evaluation and write reports.

    uv run python eval/run.py [--policy policies/default.yaml] [--tag baseline]

Outputs:
  eval/reports/<tag>.json      machine-readable result
  docs/eval/<tag>.md           human-readable report
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

from veil import Guard
from veil.evaluation import (
    evaluate,
    generate_clean,
    generate_labelled,
    read_jsonl,
    render_markdown,
    write_jsonl,
)
from veil.policy import Hasher

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "eval" / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(ROOT / "policies" / "default.yaml"))
    parser.add_argument("--tag", default="report")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.WARNING)

    # Datasets are deterministic from the seed and are *not* committed: they contain
    # realistic-looking synthetic credentials that trip secret scanners (GitHub push protection
    # blocked the first push on a fake sk_live_ key). Generate on demand instead.
    if not (DATA_DIR / "labelled.jsonl").exists() or not (DATA_DIR / "clean.jsonl").exists():
        write_jsonl(generate_labelled(), DATA_DIR / "labelled.jsonl")
        write_jsonl(generate_clean(), DATA_DIR / "clean.jsonl")
    labelled = read_jsonl(DATA_DIR / "labelled.jsonl")
    clean = read_jsonl(DATA_DIR / "clean.jsonl")
    guard = Guard.from_policy(args.policy, hasher=Hasher(b"eval-only-key"))

    result = evaluate(guard, labelled, clean)

    json_path = ROOT / "eval" / "reports" / f"{args.tag}.json"
    md_path = ROOT / "docs" / "eval" / f"{args.tag}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown(result, title=f"Veil evaluation — {args.tag}"), encoding="utf-8"
    )

    print(f"policy: {result.policy_name}")
    print(f"{'entity':<22}{'P':>6}{'R':>6}{'F1':>6}{'n':>5}")
    for m in result.entity_metrics:
        row = f"{m.entity_type:<22}{m.precision:>6.2f}{m.recall:>6.2f}{m.f1:>6.2f}"
        print(f"{row}{m.support:>5}")
    clean = result.clean
    print(f"clean docs with any detection: {clean.documents_with_any_detection}/{clean.documents}")
    for lat in result.latency:
        print(f"latency {lat.destination}: p50 {lat.p50_ms} ms, p95 {lat.p95_ms} ms")
    print(f"block demo: {'yes' if result.block_demo else 'NO'}")
    print(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
