"""Synthetic, labelled evaluation data (ADR-0008).

Every example carries exact gold spans. Text is assembled from fragments so offsets are
computed, never guessed. All values come from Faker or are generated token shapes — never
real customer data.
"""

from __future__ import annotations

import base64
import json
import random
import string
from collections.abc import Iterator, Sequence
from pathlib import Path

from faker import Faker
from pydantic import BaseModel, ConfigDict, Field


class GoldSpan(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_type: str
    start: int
    end: int


class Example(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    spans: tuple[GoldSpan, ...] = ()
    category: str = Field(description="pii | secrets | mixed | clean")
    kind: str = Field(default="", description="Sub-type, e.g. 'contract' for clean docs")

    def span_text(self, span: GoldSpan) -> str:
        return self.text[span.start : span.end]


def write_jsonl(examples: Sequence[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[Example]:
    with path.open(encoding="utf-8") as fh:
        return [Example.model_validate_json(line) for line in fh if line.strip()]


# --- text builder ----------


class _Builder:
    """Concatenates fragments, recording spans for the ones that are entities."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._spans: list[GoldSpan] = []
        self._length = 0

    def text(self, fragment: str) -> _Builder:
        self._parts.append(fragment)
        self._length += len(fragment)
        return self

    def entity(self, entity_type: str, value: str) -> _Builder:
        self._spans.append(
            GoldSpan(entity_type=entity_type, start=self._length, end=self._length + len(value))
        )
        return self.text(value)

    def build(self, id_: str, category: str, kind: str = "") -> Example:
        return Example(
            id=id_,
            text="".join(self._parts),
            spans=tuple(self._spans),
            category=category,
            kind=kind,
        )


# --- value generators ----------

_REAL_CITIES = (
    "Seattle",
    "Denver",
    "Austin",
    "Chicago",
    "Boston",
    "Atlanta",
    "Portland",
    "Phoenix",
    "Toronto",
    "Vancouver",
    "London",
    "Manchester",
    "Dublin",
    "Berlin",
    "Munich",
    "Amsterdam",
    "Madrid",
    "Lisbon",
    "Sydney",
    "Melbourne",
    "Singapore",
    "Mumbai",
    "Bangalore",
    "Tokyo",
)


class _Values:
    def __init__(self, seed: int) -> None:
        self.fake = Faker("en_US")
        self.fake.seed_instance(seed)
        self.rng = random.Random(seed)

    def name(self) -> str:
        return self.fake.name()

    def email(self) -> str:
        return self.fake.email()

    def phone(self) -> str:
        area, mid, last = (
            self.fake.numerify("%##"),
            self.fake.numerify("###"),
            self.fake.numerify("####"),
        )
        fmt = self.rng.choice(
            [
                f"{area}-{mid}-{last}",
                f"({area}) {mid}-{last}",
                f"{area}.{mid}.{last}",
                f"+1 {area} {mid} {last}",
                f"{area}-{mid}-{last} x{self.fake.numerify('###')}",  # extension: harder
            ]
        )
        return fmt

    def ssn(self) -> str:
        return self.fake.ssn()

    def credit_card(self) -> str:
        digits = self.fake.credit_card_number(card_type=self.rng.choice(["visa", "mastercard"]))
        if self.rng.random() < 0.5:
            return " ".join(digits[i : i + 4] for i in range(0, len(digits), 4))
        return digits

    def ipv4(self) -> str:
        return self.fake.ipv4()

    def city(self) -> str:
        # Faker's city() invents names ("Ryanport") that no NER model has seen; real cities
        # are the fair test for LOCATION, and are not personal data on their own.
        return self.rng.choice(_REAL_CITIES)

    def street_address(self) -> str:
        return self.fake.street_address()

    def aws_key(self) -> str:
        return "AKIA" + self._chars(16, string.ascii_uppercase + string.digits)

    def github_token(self) -> str:
        return "ghp_" + self._chars(36, string.ascii_letters + string.digits)

    def stripe_key(self) -> str:
        return "sk_live_" + self._chars(24, string.ascii_letters + string.digits)

    def openai_key(self) -> str:
        return (
            "sk-"
            + self._chars(20, string.ascii_letters + string.digits)
            + "T3BlbkFJ"
            + self._chars(20, string.ascii_letters + string.digits)
        )

    def slack_token(self) -> str:
        return f"xoxb-{self.fake.numerify('############')}-{self.fake.numerify('#############')}-{self._chars(24, string.ascii_lowercase + string.digits)}"

    def jwt(self) -> str:
        def b64(obj: dict[str, object]) -> str:
            raw = json.dumps(obj, separators=(",", ":")).encode()
            return base64.urlsafe_b64encode(raw).decode().rstrip("=")

        header = b64({"alg": "HS256", "typ": "JWT"})
        payload = b64({"sub": self.fake.numerify("########"), "iat": 1700000000})
        sig = self._chars(43, string.ascii_letters + string.digits + "-_")
        return f"{header}.{payload}.{sig}"

    def private_key(self) -> str:
        body = "\n".join(
            self._chars(64, string.ascii_letters + string.digits + "+/") for _ in range(3)
        )
        return f"-----BEGIN RSA PRIVATE KEY-----\n{body}\n-----END RSA PRIVATE KEY-----"

    def password(self) -> str:
        return self._chars(16, string.ascii_letters + string.digits + "!#$%")

    def case_id(self) -> str:
        return f"CASE-{self.fake.numerify('####')}-{self.fake.numerify('#####')}"

    def employee_id(self) -> str:
        return f"EMP-{self.fake.numerify('######')}"

    def _chars(self, n: int, alphabet: str) -> str:
        return "".join(self.rng.choice(alphabet) for _ in range(n))


# --- PII / secrets templates ----------


def _pii_examples(v: _Values, count: int) -> Iterator[Example]:
    templates = [
        lambda b: (
            b.text("Please call ")
            .entity("PERSON", v.name())
            .text(" at ")
            .entity("PHONE_NUMBER", v.phone())
            .text(" about the invoice.")
        ),
        lambda b: (
            b.text("Customer ")
            .entity("PERSON", v.name())
            .text(" (")
            .entity("EMAIL_ADDRESS", v.email())
            .text(") asked for a refund.")
        ),
        lambda b: b.text("Verify identity: SSN ").entity("US_SSN", v.ssn()).text(", DOB on file."),
        lambda b: (
            b.text("Charge card ")
            .entity("CREDIT_CARD", v.credit_card())
            .text(" for the remaining balance.")
        ),
        lambda b: (
            b.text("Ship the replacement to ")
            .entity("STREET_ADDRESS", v.street_address())
            .text(", ")
            .entity("LOCATION", v.city())
            .text(".")
        ),
        lambda b: (
            b.text("The request originated from ")
            .entity("IP_ADDRESS", v.ipv4())
            .text(" at 09:14 UTC.")
        ),
        lambda b: (
            b.text("Escalation for ")
            .entity("PERSON", v.name())
            .text(": phone ")
            .entity("PHONE_NUMBER", v.phone())
            .text(", email ")
            .entity("EMAIL_ADDRESS", v.email())
            .text(".")
        ),
        lambda b: (
            b.text("Applicant ")
            .entity("PERSON", v.name())
            .text(" listed social security number ")
            .entity("US_SSN", v.ssn())
            .text(" on the form.")
        ),
        lambda b: (
            b.text("Forward the contract to ")
            .entity("EMAIL_ADDRESS", v.email())
            .text(" and cc legal.")
        ),
        lambda b: (
            b.text("Meeting with ")
            .entity("PERSON", v.name())
            .text(" in ")
            .entity("LOCATION", v.city())
            .text(" next Tuesday.")
        ),
        lambda b: (
            b.text("Payment failed for card ending in ")
            .entity("CREDIT_CARD", v.credit_card())
            .text("; customer ")
            .entity("PERSON", v.name())
            .text(" notified.")
        ),
        lambda b: (
            b.text("Reach ")
            .entity("PERSON", v.name())
            .text(" on ")
            .entity("PHONE_NUMBER", v.phone())
            .text(" or ")
            .entity("EMAIL_ADDRESS", v.email())
            .text(" before Friday.")
        ),
        lambda b: (
            b.text("Login from ")
            .entity("IP_ADDRESS", v.ipv4())
            .text(" flagged for ")
            .entity("PERSON", v.name())
            .text(".")
        ),
        lambda b: (
            b.text("Tax form shows SSN: ")
            .entity("US_SSN", v.ssn())
            .text(" and address ")
            .entity("STREET_ADDRESS", v.street_address())
            .text(".")
        ),
        lambda b: (
            b.text("Internal ticket ")
            .entity("INTERNAL_CASE_ID", v.case_id())
            .text(" opened by ")
            .entity("EMPLOYEE_ID", v.employee_id())
            .text(" for ")
            .entity("PERSON", v.name())
            .text(".")
        ),
    ]
    for i in range(count):
        yield templates[i % len(templates)](_Builder()).build(f"pii-{i:03d}", "pii")


def _secret_examples(v: _Values, count: int) -> Iterator[Example]:
    templates = [
        lambda b: b.text("export AWS_ACCESS_KEY_ID=").entity("AWS_ACCESS_KEY", v.aws_key()),
        lambda b: b.text("Use this PAT for CI: ").entity("GITHUB_TOKEN", v.github_token()),
        lambda b: (
            b.text("STRIPE_SECRET=").entity("STRIPE_ACCESS_KEY", v.stripe_key()).text(" # prod")
        ),
        lambda b: b.text("OPENAI_API_KEY=").entity("OPENAI_TOKEN", v.openai_key()),
        lambda b: b.text("slack bot: ").entity("SLACK_TOKEN", v.slack_token()),
        lambda b: b.text("Authorization: Bearer ").entity("JWT", v.jwt()),
        lambda b: b.text("deploy key:\n").entity("PRIVATE_KEY", v.private_key()).text("\n"),
        lambda b: b.text('db_password = "').entity("SECRET_KEYWORD", v.password()).text('"'),
        lambda b: (
            b.text("The engineer pasted ")
            .entity("AWS_ACCESS_KEY", v.aws_key())
            .text(" into the chat by mistake.")
        ),
        lambda b: (
            b.text("Rotate ")
            .entity("GITHUB_TOKEN", v.github_token())
            .text(" and ")
            .entity("STRIPE_ACCESS_KEY", v.stripe_key())
            .text(" immediately.")
        ),
    ]
    for i in range(count):
        yield templates[i % len(templates)](_Builder()).build(f"secret-{i:03d}", "secrets")


def _mixed_examples(v: _Values, count: int) -> Iterator[Example]:
    templates = [
        lambda b: (
            b.entity("PERSON", v.name())
            .text(" shared ")
            .entity("AWS_ACCESS_KEY", v.aws_key())
            .text(" from ")
            .entity("IP_ADDRESS", v.ipv4())
            .text(".")
        ),
        lambda b: (
            b.text("Support transcript: ")
            .entity("PERSON", v.name())
            .text(", ")
            .entity("EMAIL_ADDRESS", v.email())
            .text(", card ")
            .entity("CREDIT_CARD", v.credit_card())
            .text(", token ")
            .entity("GITHUB_TOKEN", v.github_token())
            .text(".")
        ),
        lambda b: (
            b.text("Ticket ")
            .entity("INTERNAL_CASE_ID", v.case_id())
            .text(": ")
            .entity("PERSON", v.name())
            .text(" reports SSN ")
            .entity("US_SSN", v.ssn())
            .text(" exposed in logs at ")
            .entity("IP_ADDRESS", v.ipv4())
            .text(".")
        ),
    ]
    for i in range(count):
        yield templates[i % len(templates)](_Builder()).build(f"mixed-{i:03d}", "mixed")


def generate_labelled(
    seed: int = 20260919, pii: int = 105, secrets: int = 40, mixed: int = 15
) -> list[Example]:
    v = _Values(seed)
    return [*_pii_examples(v, pii), *_secret_examples(v, secrets), *_mixed_examples(v, mixed)]


# --- clean corpus ----------

_CLEAN_TEMPLATES: dict[str, list[str]] = {
    "contract": [
        "This Master Services Agreement is entered into as of {date} between {company} and the Client. "
        "Either party may terminate with {days} days' written notice. Fees are payable net {net} in USD.",
        "Section {sec}. Confidentiality. Each party shall protect the other's Confidential Information "
        "with at least the degree of care it uses for its own, and no less than reasonable care, for {years} years.",
        "The Supplier warrants that the Deliverables will conform to the Specification for {days} days "
        "following acceptance. Liability is capped at {amount} or the fees paid in the preceding {months} months.",
    ],
    "email": [
        "Hi team, the {product} release {version} is scheduled for {date}. Please freeze main by end of day "
        "and move any open work to the next sprint. Thanks!",
        "Reminder: the quarterly review deck is due {date}. Focus on churn, ARR growth ({pct}%), and the "
        "roadmap for {product}. Keep it to ten slides.",
        "Following up on order {order}: the shipment left the {city} warehouse on {date} and should arrive "
        "within {days} business days. Tracking is available in the portal.",
    ],
    "release_notes": [
        "{product} {version} — Changelog\n- Fixed a regression in the export pipeline (issue #{issue}).\n"
        "- Upgraded Kubernetes client to {version2}.\n- p95 latency improved by {pct}% on the search endpoint.",
        "Version {version} deprecates the legacy REST v1 endpoints. Migrate to v2 before {date}. "
        "See the migration guide for the mapping of status codes and pagination parameters.",
    ],
    "meeting_notes": [
        "Standup {date}: backlog grooming complete; {n} stories pointed. Blocker: staging database at "
        "{pct}% disk. Action: increase volume to {size} GB before the load test.",
        "Retro themes: better on-call handoffs, fewer flaky tests, clearer ownership for {product}. "
        "Decided to trial a {days}-day bug bash each quarter.",
    ],
    "invoice": [
        "Invoice {invoice} — {company}\nLine 1: {product} subscription, {n} seats @ {amount_small} = {amount}\n"
        "Due {date}. Pay by wire or card via the billing portal. PO reference {po}.",
    ],
    "policy": [
        "Expense policy: meals while travelling are reimbursable up to {amount_small} per day. Submit receipts "
        "within {days} days. Alcohol and upgrades are not covered unless pre-approved.",
        "Access to production requires a change ticket, a reviewer, and a rollback plan. Emergency changes must be "
        "documented within {hours} hours. Credentials are issued via the vault, never shared in chat.",
    ],
}


def generate_clean(seed: int = 20260919, count: int = 100) -> list[Example]:
    v = _Values(seed + 1)
    kinds = list(_CLEAN_TEMPLATES)
    out: list[Example] = []
    for i in range(count):
        kind = kinds[i % len(kinds)]
        template = v.rng.choice(_CLEAN_TEMPLATES[kind])
        text = template.format(
            date=v.fake.date_between("-1y", "+1y").strftime("%B %d, %Y"),
            company=v.fake.company(),
            days=v.rng.choice([14, 30, 45, 60, 90]),
            net=v.rng.choice([15, 30, 45]),
            sec=v.rng.randint(2, 14),
            years=v.rng.choice([2, 3, 5]),
            months=v.rng.choice([6, 12]),
            amount=f"${v.rng.randint(5, 250) * 1000:,}",
            amount_small=f"${v.rng.randint(20, 120)}",
            product=v.rng.choice(["Atlas", "Beacon", "Ledger", "Orbit", "Relay", "Sentinel"]),
            version=f"v{v.rng.randint(1, 9)}.{v.rng.randint(0, 20)}.{v.rng.randint(0, 9)}",
            version2=f"{v.rng.randint(1, 3)}.{v.rng.randint(0, 30)}.{v.rng.randint(0, 9)}",
            pct=v.rng.randint(3, 48),
            order=f"ORD-{v.fake.numerify('######')}",
            invoice=f"INV-{v.fake.numerify('#####')}",
            po=f"PO-{v.fake.numerify('####')}-{v.fake.numerify('##')}",
            issue=v.rng.randint(100, 9999),
            n=v.rng.randint(3, 60),
            size=v.rng.choice([200, 500, 1000]),
            hours=v.rng.choice([24, 48]),
            city=v.rng.choice(["Reno", "Columbus", "Memphis", "Rotterdam", "Leeds"]),
        )
        out.append(Example(id=f"clean-{i:03d}", text=text, spans=(), category="clean", kind=kind))
    return out
