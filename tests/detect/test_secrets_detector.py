"""detect-secrets adapter tests — no spaCy model needed, so these run fast."""

import pytest

from veil.detect.schema import Source
from veil.detect.secrets_detector import (
    CONFIDENCE_ENTROPY,
    CONFIDENCE_KEYWORD,
    CONFIDENCE_REGEX_TOKEN,
    SecretsDetector,
    normalize_secret_type,
)

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
GITHUB_PAT = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"


@pytest.fixture(scope="module")
def detector() -> SecretsDetector:
    return SecretsDetector()


def _by_type(dets: list, entity: str) -> list:
    return [d for d in dets if d.entity_type == entity]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AWS Access Key", "AWS_ACCESS_KEY"),
        ("Public IP (ipv4)", "IP_ADDRESS"),
        ("Base64 High Entropy String", "HIGH_ENTROPY_STRING"),
        ("Hex High Entropy String", "HIGH_ENTROPY_STRING"),
        ("Secret Keyword", "SECRET_KEYWORD"),
        ("JSON Web Token", "JWT"),
        ("OpenAI Token", "OPENAI_TOKEN"),
    ],
)
def test_normalize_secret_type(raw: str, expected: str) -> None:
    assert normalize_secret_type(raw) == expected


def test_aws_key_span_is_exact(detector: SecretsDetector) -> None:
    text = f"export AWS_ACCESS_KEY_ID={AWS_KEY}"
    hits = _by_type(list(detector.detect(text)), "AWS_ACCESS_KEY")
    assert len(hits) == 1
    hit = hits[0]
    assert text[hit.start : hit.end] == AWS_KEY
    assert hit.confidence == CONFIDENCE_REGEX_TOKEN
    assert hit.source == Source.DETECT_SECRETS
    assert hit.recognizer == "AWSKeyDetector"


def test_github_token_span_covers_whole_token_not_captured_prefix(
    detector: SecretsDetector,
) -> None:
    """Upstream returns secret_value='ghp' for this plugin; we must still redact the full token."""
    text = f"token: {GITHUB_PAT} ok"
    hits = _by_type(list(detector.detect(text)), "GITHUB_TOKEN")
    assert len(hits) == 1
    assert text[hits[0].start : hits[0].end] == GITHUB_PAT


def test_low_entropy_words_are_not_flagged(detector: SecretsDetector) -> None:
    """scan_line's eager mode returns every token; the adapter must re-apply the entropy limit."""
    text = "please send the quarterly report and the token summary to finance"
    assert list(detector.detect(text)) == []


def test_high_entropy_string_is_flagged_with_entropy_confidence(
    detector: SecretsDetector,
) -> None:
    blob = "Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5MEFCQ0RFRkdISUpLTE1OTw=="
    hits = _by_type(list(detector.detect(f"blob={blob}")), "HIGH_ENTROPY_STRING")
    assert len(hits) >= 1
    assert all(h.confidence == CONFIDENCE_ENTROPY for h in hits)


def test_keyword_secret_redacts_only_the_value(detector: SecretsDetector) -> None:
    text = 'db_password = "Xk9#mP2$vL7qR!wZ"'
    hits = _by_type(list(detector.detect(text)), "SECRET_KEYWORD")
    assert len(hits) == 1
    assert text[hits[0].start : hits[0].end] == "Xk9#mP2$vL7qR!wZ"
    assert hits[0].confidence == CONFIDENCE_KEYWORD


def test_private_key_span_extends_to_end_marker(detector: SecretsDetector) -> None:
    text = (
        "config:\n"
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGy0AH+D6f4yz6pM\n"
        "-----END RSA PRIVATE KEY-----\n"
        "after"
    )
    hits = _by_type(list(detector.detect(text)), "PRIVATE_KEY")
    assert len(hits) == 1
    span = text[hits[0].start : hits[0].end]
    assert span.startswith("-----BEGIN RSA PRIVATE KEY-----")
    assert span.endswith("-----END RSA PRIVATE KEY-----")
    assert "after" not in span


def test_multiline_offsets_are_absolute(detector: SecretsDetector) -> None:
    text = f"line one\nline two\nkey={AWS_KEY}\n"
    hits = _by_type(list(detector.detect(text)), "AWS_ACCESS_KEY")
    assert len(hits) == 1
    assert text[hits[0].start : hits[0].end] == AWS_KEY


def test_disabled_type_is_skipped() -> None:
    det = SecretsDetector(disabled_types=frozenset({"AWS Access Key"}))
    assert _by_type(list(det.detect(f"k={AWS_KEY}")), "AWS_ACCESS_KEY") == []


def test_empty_text(detector: SecretsDetector) -> None:
    assert list(detector.detect("")) == []
