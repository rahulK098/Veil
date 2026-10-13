"""Generate the labelled and clean evaluation datasets into eval/data/.

    uv run python eval/generate.py [--seed N]

Datasets are deterministic from the seed and are git-ignored (they contain synthetic
credentials that trip secret scanners); `eval/run.py` generates them on demand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from veil.evaluation import generate_clean, generate_labelled, write_jsonl

DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    labelled = generate_labelled(seed=args.seed)
    clean = generate_clean(seed=args.seed)
    write_jsonl(labelled, DATA_DIR / "labelled.jsonl")
    write_jsonl(clean, DATA_DIR / "clean.jsonl")
    spans = sum(len(ex.spans) for ex in labelled)
    print(
        f"wrote {len(labelled)} labelled examples ({spans} gold spans) and {len(clean)} clean docs"
    )


if __name__ == "__main__":
    main()
