"""A deliberately hard evaluation set (ADR-0013).

The main corpus is saturated — 16 of 18 entities score 1.00 — so it can confirm that nothing
broke but cannot show whether detection got better. These cases are curated rather than
generated, because difficulty here comes from specific linguistic and structural phenomena
(a name that is also a country, PII inside a URL query string, an identifier shaped like an
SSN) that a template generator cannot invent.

Two halves, reported separately:

* **Hard positives** — real PII that is awkward to find. Measures recall under pressure.
* **Hard negatives** — text with *no* PII that looks like PII. Measures the cost of
  over-triggering, including the case nobody usually counts: a legitimate request refused
  because an order number resembles a government ID.
"""

from __future__ import annotations

from typing import Final

from veil.evaluation.dataset import Example, GoldSpan

# (id, text, [(entity_type, substring)]) — offsets are derived from the substring, which a
# test asserts occurs exactly once so the span is unambiguous.
_POSITIVES: Final[tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...]] = (
    # -- names that resist NER --------------------------------------------------------------
    ("hp-lowercase", "hi its maria gonzalez here, can you help", (("PERSON", "maria gonzalez"),)),
    (
        "hp-signoff",
        "sounds good, will confirm tomorrow — Jordan Ellis",
        (("PERSON", "Jordan Ellis"),),
    ),
    ("hp-title", "Please ask Dr. Okonkwo to review it.", (("PERSON", "Okonkwo"),)),
    ("hp-in-code", 'user = Person(name="Maria Gonzalez")', (("PERSON", "Maria Gonzalez"),)),
    ("hp-possessive", "that was Priya Raman's call to make", (("PERSON", "Priya Raman"),)),
    (
        "hp-two-people",
        "loop in Ana Silva and Tom Becker",
        (("PERSON", "Ana Silva"), ("PERSON", "Tom Becker")),
    ),
    # -- PII inside structure ---------------------------------------------------------------
    (
        "hp-json",
        '{"user":{"email":"j.smith@example.com","id":42}}',
        (("EMAIL_ADDRESS", "j.smith@example.com"),),
    ),
    (
        "hp-url-query",
        "open https://app.example.com/u?email=j.smith%40example.com",
        (("EMAIL_ADDRESS", "j.smith%40example.com"),),
    ),
    (
        "hp-markdown",
        "| Name | Email |\n| Ada Lovelace | ada@example.com |",
        (("PERSON", "Ada Lovelace"), ("EMAIL_ADDRESS", "ada@example.com")),
    ),
    (
        "hp-yaml",
        "aws:\n  key: AKIAIOSFODNN7EXAMPLE\n",
        (("AWS_ACCESS_KEY", "AKIAIOSFODNN7EXAMPLE"),),
    ),
    (
        "hp-csv",
        "name,phone\nRosa Mendes,212-555-0147",
        (("PERSON", "Rosa Mendes"), ("PHONE_NUMBER", "212-555-0147")),
    ),
    (
        "hp-log-line",
        'level=info user="Kai Tanaka" ip=10.0.0.14',
        (("PERSON", "Kai Tanaka"), ("IP_ADDRESS", "10.0.0.14")),
    ),
    # -- obfuscation ------------------------------------------------------------------------
    (
        "hp-spelled-email",
        "reach me at j dot smith at example dot com",
        (("EMAIL_ADDRESS", "j dot smith at example dot com"),),
    ),
    ("hp-spaced-ssn", "ssn is 5 3 6 2 2 8 7 4 1", (("US_SSN", "5 3 6 2 2 8 7 4 1"),)),
    (
        "hp-spaced-card",
        "card 4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1",
        (("CREDIT_CARD", "4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"),),
    ),
    ("hp-ssn-nolabel", "the number is 536 22 8741 for the file", (("US_SSN", "536 22 8741"),)),
    # -- international ----------------------------------------------------------------------
    ("hp-uk-phone", "call +44 20 7946 0958 tomorrow", (("PHONE_NUMBER", "+44 20 7946 0958"),)),
    ("hp-in-phone", "ring +91 98765 43210 please", (("PHONE_NUMBER", "+91 98765 43210"),)),
    (
        "hp-uk-address",
        "ship to 221B Baker Street, London NW1 6XE",
        (("STREET_ADDRESS", "221B Baker Street"),),
    ),
    (
        "hp-de-address",
        "deliver to Hauptstrasse 14, 10115 Berlin",
        (("STREET_ADDRESS", "Hauptstrasse 14"),),
    ),
    # -- split / multi-line -----------------------------------------------------------------
    (
        "hp-wrapped-email",
        "write to j.smith@\nexample.com when ready",
        (("EMAIL_ADDRESS", "j.smith@\nexample.com"),),
    ),
    (
        "hp-card-dashes",
        "card 4111-1111-1111-1111 on file",
        (("CREDIT_CARD", "4111-1111-1111-1111"),),
    ),
)

# Text containing no PII at all. Every detection here is a false positive; some are severe,
# because the shipped policy blocks US_SSN and CREDIT_CARD outright.
_NEGATIVES: Final[tuple[tuple[str, str], ...]] = (
    ("hn-semver-4", "upgraded the agent to version 10.2.14.3 last night"),
    ("hn-semver-pin", "pinned the kubernetes client to 1.28.4.2 for the rollout"),
    ("hn-bind-addr", "bind the service to 0.0.0.0 on port 8080"),
    ("hn-localhost", "the health check hits 127.0.0.1 every ten seconds"),
    ("hn-uuid", "trace id 550e8400-e29b-41d4-a716-446655440000 in the logs"),
    ("hn-git-sha", "reverted at commit 9fceb02d0ae598e95dc970b74767f19372d61af8"),
    ("hn-sha256", "checksum e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("hn-isbn", "the handbook ISBN 978-3-16-148410-0 is on the shelf"),
    ("hn-order-id", "order ORD-347528 shipped from the depot on Friday"),
    ("hn-tracking", "tracking 1Z999AA10123456784 was delivered Tuesday"),
    ("hn-timestamp", "the job ran at 2026-10-15 03:22:41 and finished clean"),
    ("hn-mac", "nic 00:1A:2B:3C:4D:5E dropped packets overnight"),
    ("hn-cost-centre", "cost centre 536-22-8741 in the quarterly ledger"),
    ("hn-part-numbers", "line items 4111 1111 1111 1111 units shipped this week"),
    ("hn-asset-tag", "asset tag 206-555-0170 is on the rack label"),
    ("hn-products", "Atlas, Beacon and Relay all shipped this quarter"),
    ("hn-ports", "forward 8080 to 9090 and leave 3000 for the dev server"),
    ("hn-invoice", "invoice INV-55821 and PO-3842-74 are both settled"),
    ("hn-latency", "p50 was 9.8 ms and p99 reached 16.6 ms under load"),
    ("hn-coords", "the warehouse sits at 47.6062, -122.3321 roughly"),
)


def generate_hard_positives() -> list[Example]:
    """Real PII in awkward shapes. Gold spans located by unique substring."""
    out: list[Example] = []
    for example_id, text, items in _POSITIVES:
        spans = []
        for entity_type, value in items:
            start = text.index(value)
            spans.append(GoldSpan(entity_type=entity_type, start=start, end=start + len(value)))
        out.append(
            Example(
                id=example_id,
                text=text,
                spans=tuple(sorted(spans, key=lambda s: s.start)),
                category="hard_pii",
            )
        )
    return out


def generate_hard_negatives() -> list[Example]:
    """Text with no PII that is shaped like PII."""
    return [
        Example(id=example_id, text=text, spans=(), category="hard_negative")
        for example_id, text in _NEGATIVES
    ]
