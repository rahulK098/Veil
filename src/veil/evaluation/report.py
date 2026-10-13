"""Render an ``EvalResult`` as Markdown for docs/eval/."""

from __future__ import annotations

from collections import Counter

from veil.evaluation.runner import EvalResult


def render_markdown(result: EvalResult, *, title: str = "Veil evaluation report") -> str:
    lines: list[str] = [
        f"# {title}",
        "",
        f"Generated {result.generated_at:%Y-%m-%d %H:%M UTC} · policy `{result.policy_name}` · "
        f"{result.labelled_examples} labelled examples · {result.clean_documents} clean documents",
        "",
        "Numbers below are produced by `eval/run.py`; do not hand-edit.",
        "",
        "## Precision / recall per entity type",
        "",
        "Span match = same entity type and IoU ≥ 0.5. "
        "`exact` = fraction of matches with identical offsets.",
        "",
        "| Entity | Support | TP | FP | FN | Precision | Recall | F1 | Exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in result.entity_metrics:
        lines.append(
            f"| `{m.entity_type}` | {m.support} | {m.tp} | {m.fp} | {m.fn} | "
            f"{m.precision:.2f} | {m.recall:.2f} | {m.f1:.2f} | {m.exact_rate:.2f} |"
        )

    lines += ["", "## Clean-corpus false positives", ""]
    c = result.clean
    lines.append(
        f"- Documents with **any** detection: {c.documents_with_any_detection}/{c.documents} "
        f"({c.detection_fp_rate:.1%})"
    )
    if c.detections_by_entity:
        lines.append(
            "- Detections by entity: "
            + ", ".join(f"`{k}` ×{v}" for k, v in c.detections_by_entity.items())
        )
    lines += ["", "| Destination | Documents modified | Documents blocked |", "|---|---:|---:|"]
    for dest in ("model", "response", "logs", "vector_store"):
        lines.append(
            f"| `{dest}` | {c.documents_modified.get(dest, 0)} ({c.modified_rate(dest):.1%}) | "
            f"{c.documents_blocked.get(dest, 0)} |"
        )

    lines += ["", "## Latency (full guarded path: detect + policy + audit)", ""]
    lines += [
        "| Destination | Samples | Mean text length | p50 ms | p95 ms | p99 ms | max ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lat in result.latency:
        lines.append(
            f"| `{lat.destination}` | {lat.samples} | {lat.mean_text_length} | {lat.p50_ms} | "
            f"{lat.p95_ms} | {lat.p99_ms} | {lat.max_ms} |"
        )

    lines += ["", "## Blocks at the `model` destination", ""]
    if result.blocked_at_model:
        lines.append("| Entity | Examples blocked |")
        lines.append("|---|---:|")
        lines += [f"| `{k}` | {v} |" for k, v in result.blocked_at_model.items()]
    else:
        lines.append("_none_")

    lines += ["", "## Demonstrated block (outbound completion echoing an SSN)", ""]
    if result.block_demo:
        d = result.block_demo
        lines += [
            f"Example `{d.example_id}`, destination `{d.destination}`, audit operation "
            f"`{d.operation_id}` (outcome `{d.outcome}`).",
            "",
            "```",
            f"BlockedError: {d.error_message}",
            "```",
            "",
            "Audit detection rows for that operation:",
            "",
            "| Entity | Confidence | Threshold | Action | Applied | Action rule | Threshold rule |",
            "|---|---:|---:|---|---|---|---|",
        ]
        for row in d.detections:
            lines.append(
                f"| `{row['entity_type']}` | {float(str(row['confidence'])):.2f} | "
                f"{float(str(row['threshold'])):.2f} | {row['action']} | {row['applied']} | "
                f"`{row['action_rule']}` | `{row['threshold_rule']}` |"
            )
    else:
        lines.append("_No block demonstrated — investigate._")

    lines += ["", "## Misses (false negatives) by entity", ""]
    fn_counter = Counter(m.entity_type for m in result.misses if m.kind == "fn")
    if fn_counter:
        for entity, n in fn_counter.most_common():
            examples = [m for m in result.misses if m.kind == "fn" and m.entity_type == entity][:3]
            sample = "; ".join(
                f"`{m.span_text[:40]}`" + (f" → {m.predicted_as}" if m.predicted_as else "")
                for m in examples
            )
            lines.append(f"- `{entity}`: {n} missed. e.g. {sample}")
    else:
        lines.append("_none_")

    lines += ["", "## Spurious detections (false positives) on labelled data", ""]
    fp_counter = Counter(m.entity_type for m in result.misses if m.kind == "fp")
    if fp_counter:
        for entity, n in fp_counter.most_common():
            examples = [m for m in result.misses if m.kind == "fp" and m.entity_type == entity][:3]
            sample = "; ".join(f"`{m.span_text[:40]}`" for m in examples)
            lines.append(f"- `{entity}`: {n}. e.g. {sample}")
    else:
        lines.append("_none_")

    return "\n".join(lines) + "\n"
