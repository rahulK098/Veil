<p align="center">
  <img src="logo.png" alt="Veil — Shielding Your AI. Protecting Your Data." width="720">
</p>

<p align="center">
  Policy-driven PII/secrets scrubber that sits in front of every LLM call and every log write.
</p>

<p align="center">
  <a href="https://github.com/rahulK098/Veil/actions/workflows/ci.yml">
    <img src="https://github.com/rahulK098/Veil/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
</p>

---

LLM applications have **two** data-governance surfaces: what goes *to* the model, and what gets
*written down about* the interaction (logs, traces, vector stores). Veil scans both directions,
applies a per-destination policy, and writes an audit trail that lets a reviewer answer
"why was this SSN allowed through?" from the log alone.

Veil does **not** reinvent detection. It wraps [Microsoft Presidio](https://github.com/microsoft/presidio)
(PII, spaCy NER) and [detect-secrets](https://github.com/Yelp/detect-secrets) (credentials),
and adds the parts those tools don't provide: the policy engine, bidirectional middleware,
audit log, and an evaluation harness that proves the numbers.

## Quick start

```bash
uv sync --all-groups
uv run python -m spacy download en_core_web_lg   # NER model (~600 MB)
export VEIL_HASH_SECRET=...                      # key for deterministic hash tokens
uv run pytest
```

```python
import veil

guard = veil.Guard.from_policy("policies/default.yaml", audit_db="audit.db")
guard.warm_up()  # loads spaCy once (~4 s)

# 1. Wrap an LLM call — inputs scrubbed before the provider sees them, outputs after.
@veil.redacted_call(guard, adapter=veil.ChatMessagesAdapter())
def chat(*, messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages)

# 2. Or call the surfaces directly.
with guard.session(request_id="req-42") as g:
    prompt = g.inbound("Hi, I'm Maria Gonzalez, card 4111 1111 1111 1111")
    # -> raises veil.BlockedError: blocked at destination 'model': CREDIT_CARD (rule entities.CREDIT_CARD ...)

    line = g.for_logs("user Maria Gonzalez (maria@example.com) asked about billing")
    # -> "user <PERSON:9f3a1c…> (<EMAIL_ADDRESS:b07e…>) asked about billing"

# 3. Every log line, automatically.
import logging
logging.getLogger().addFilter(veil.VeilLogFilter(guard))
```

The same `PERSON` is **allowed** on the way to the model (the assistant needs the name),
**hashed** in logs (correlate without storing), and **redacted** in the vector store — one
YAML file decides, and every decision is written to the audit log with the rule that made it.

## How it works

| Surface | Method | On policy `block` |
|---|---|---|
| Prompt → model | `guard.inbound(text)` | raises `BlockedError` |
| Completion → caller | `guard.outbound(text)` | raises `BlockedError` |
| Streamed completion | `guard.outbound_stream(chunks)` / `aoutbound_stream` | raises `BlockedError` |
| Anything → logs / vector DB | `guard.for_logs(text)` / `for_vector_store(text)` | returns `None`; write is dropped |

Policy (`policies/default.yaml`) maps entity → `redact` / `mask` / `hash` / `block` / `allow`
with a per-entity confidence threshold, and lets each destination (`model`, `response`,
`logs`, `vector_store`) override any of it. Precedence is specificity-first, so a
destination-wide default can never loosen an entity-specific rule.

The audit log (SQLite) stores one row per guarded call and one per detection — including
detections that were allowed or fell below threshold — with the policy path that decided the
action and an HMAC of the span. Raw values are never stored.

### Streaming

```python
for piece in guard.outbound_stream(provider_chunks):
    print(piece, end="", flush=True)
```

Chunks are buffered before scanning, because an entity can straddle a boundary — `"SSN 536-"`
then `"22-8741"` is invisible to anything scanning chunks in isolation.

Streaming cannot honour a `block` the way `outbound()` can: once a chunk is emitted it cannot
be recalled. Veil does not paper over that. `outbound_stream` picks its mode from the policy:

| Mode | Behaviour | Guarantee |
|---|---|---|
| `BUFFERED` *(auto when the policy can block)* | accumulate, scrub once, emit at the end | identical to `outbound()` |
| `INCREMENTAL` *(explicit opt-in)* | emit early, holding back a 96-char tail | no entity up to the tail margin is ever partially emitted |

Because `policies/default.yaml` blocks SSNs, cards and credentials, it buffers by default; you
must pass `mode=StreamMode.INCREMENTAL` to trade that guarantee for time-to-first-token. See
[ADR-0010](docs/adr/0010-streaming-completions.md).

## Evaluation (ship gate)

160 labelled synthetic examples (302 spans, Faker-generated, never real data) and 100 clean
business documents. Reproduce with `uv run python eval/run.py`.

| Entity | P | R | Entity | P | R |
|---|---:|---:|---|---:|---:|
| `US_SSN` | 1.00 | 1.00 | `AWS_ACCESS_KEY` | 1.00 | 1.00 |
| `CREDIT_CARD` | 1.00 | 1.00 | `GITHUB_TOKEN` | 1.00 | 1.00 |
| `EMAIL_ADDRESS` | 1.00 | 1.00 | `STRIPE_ACCESS_KEY` | 1.00 | 1.00 |
| `PERSON` | 0.97 | 0.96 | `PRIVATE_KEY` | 1.00 | 1.00 |
| `STREET_ADDRESS` | 1.00 | 0.93 | `JWT` | 1.00 | 1.00 |
| `PHONE_NUMBER` | 1.00 | 1.00 | `SECRET_KEYWORD` | 1.00 | 1.00 |

- **Clean corpus:** 1 % of documents modified at the `model` destination, 16 % at `logs`
  (down from 35 % — see finding 5). The remaining 16 are not errors: 15 are real city names
  the policy chooses to redact in logs, plus one phone false positive.
- **Latency** (full path, detect + policy + audit, ~74-char inputs): **p50 ~10 ms, p95 ~15 ms**
  on a developer laptop. Run-to-run variance is real (an earlier run measured p50 7.2 ms);
  the claim is single-digit-to-low-teens milliseconds, not a benchmark.
- **Demonstrated block:** an outbound completion echoing an SSN is rejected with a
  `BlockedError` naming the rule, and the audit rows show why.
- **Three Presidio gaps found by the eval and closed** with recognizers in `veil.detect.custom`:
  1. Mastercard 2-series BIN range (2221–2720) missing from Presidio's card regex — recall 0.74 → 1.00.
  2. `UK_NHS` is a bare mod-11 checksum that scores 1.0 on some US phone numbers; it was
     *blocking* requests. Excluded from the default detection set (opt-in for UK health data).
  3. No street-address recognizer; addresses were also being mis-tagged `PERSON`. Added one —
     recall 0 → 0.93, `PERSON` precision 0.93 → 0.99.
  4. detect-secrets' JWT pattern ends in a lazy quantifier, so spans stopped after the payload
     and left signatures unredacted — caught by the `exact` column, not by P/R.
  5. spaCy emits a **constant** 0.85 for every `PERSON`, making policy thresholds inert for
     the entity that most needs them. Veil grades the score by span structure (≥2 capitalised
     tokens vs. one), cutting clean-document over-redaction at `logs` from 35 % to 16 % with
     no loss of recall. See [ADR-0012](docs/adr/0012-person-confidence-calibration.md).

Weakest categories are `PERSON` (spaCy tags capitalised nouns like "Deliverables") and
`STREET_ADDRESS` recall. `PHONE_NUMBER` previously read 0.76; a leniency sweep showed the
misses were all **undiallable** numbers from unallocated area codes that the dataset had
invented — see [ADR-0011](docs/adr/0011-realistic-phone-numbers-in-eval.md) for why that was a
dataset defect rather than a detector gap, and why catching them would put a false phone
number in 97 of 100 clean documents.

With 16 of 18 entities at 1.00, this corpus is close to saturated — a good *regression*
harness, not a difficulty benchmark. So there is a second, deliberately hard one
([ADR-0013](docs/adr/0013-hard-evaluation-set.md)), reported separately:

| Hard set (42 curated documents) | Result |
|---|---|
| Real PII in awkward shapes (JSON, URLs, obfuscation, international formats) | **69 % recall** (18/26 spans) |
| PII-free text that *looks* like PII (semver, UUIDs, git SHAs, order numbers) | **80 %** produce a false detection |
| **PII-free text that would be refused outright** | **10 %** (2/20) |

That last row is the number precision and recall cannot show you: `cost centre 536-22-8741`
and `line items 4111 1111 1111 1111 units` are ordinary business sentences, and under the
shipped policy a user sending either gets their request **blocked**. Some of this is
irreducible — a ledger code can be character-identical to an SSN — which is an argument for
tuning `block` rules per destination rather than for a cleverer detector.

Easy PII is solved; hard PII is not. Both numbers are published. Full reports in `docs/eval/`.

## Project layout

```
src/veil/
  detect/       Presidio + detect-secrets adapters, unified Detection schema, custom recognizers
  policy/       YAML schema, rule resolution, redact/mask/hash actions, PolicyEngine
  middleware/   Guard, @redacted_call, call adapters, VeilLogFilter
  audit/        OperationRecord/DetectionRecord, SQLite + memory sinks
  evaluation/   dataset builder, span metrics, runner, report
policies/       default.yaml, eval-baseline.yaml
eval/           generate.py, run.py  (datasets are generated on demand, not committed)
tests/          147 tests, 97 % coverage; `-m "not slow"` skips the spaCy-loading ones
docs/           ADRs 0001–0008, architecture, policy & audit reference, eval reports  (untracked)
```

## Development

```bash
uv sync --all-groups          # installs the spaCy model too (locked dev dependency)
uv run ruff check src tests eval && uv run ruff format --check src tests eval
uv run mypy src               # --strict
uv run pytest --cov=veil      # 80 % coverage floor enforced

uv run python eval/run.py     # regenerate the evaluation reports
uv run python eval/gate.py    # fail if any metric regressed below eval/thresholds.yaml
```

CI runs exactly these on every push and pull request. The eval job is a real gate: if a change
drops a per-entity precision/recall below its floor, shrinks the dataset, stops demonstrating a
block, or starts redacting clean business text, the build fails. Latency is reported but not
gated — see [ADR-0009](docs/adr/0009-ci-and-regression-gates.md).

## Limitations

- English only; spaCy `en_core_web_lg`. Swap in `en_core_web_trf` via `detection.spacy_model`
  for better NER precision at ~5–10× the latency.
- Streaming completions are not scrubbed (spans can straddle chunks).
- detect-secrets confidences are assigned per plugin class, not measured.

## License

MIT
