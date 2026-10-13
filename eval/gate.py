"""Fail the build if an evaluation run regressed below the floors in thresholds.yaml.

    uv run python eval/gate.py [--report eval/reports/default.json]
                               [--thresholds eval/thresholds.yaml]

Exits 0 when every floor is met, 1 otherwise, printing one line per violation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from veil.evaluation.gate import check

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(ROOT / "eval" / "reports" / "default.json"))
    parser.add_argument("--thresholds", default=str(ROOT / "eval" / "thresholds.yaml"))
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"gate: report not found: {report_path} (run eval/run.py first)", file=sys.stderr)
        return 1

    report = json.loads(report_path.read_text(encoding="utf-8"))
    thresholds = yaml.safe_load(Path(args.thresholds).read_text(encoding="utf-8"))

    violations, notes = check(report, thresholds)

    for note in notes:
        print(f"note: {note}")
    if violations:
        print(f"\nFAIL: {len(violations)} regression(s) against {args.thresholds}\n")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print(f"\nPASS: all floors met ({len(thresholds.get('entities', {}))} entities gated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
