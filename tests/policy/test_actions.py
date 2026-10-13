import logging

import pytest

from veil.policy.actions import Hasher, mask, redact
from veil.policy.schema import MaskOptions


def test_redact_uses_template() -> None:
    assert redact("EMAIL_ADDRESS", "<{entity}>") == "<EMAIL_ADDRESS>"
    assert redact("EMAIL_ADDRESS", "[{entity}]") == "[EMAIL_ADDRESS]"


def test_mask_email_keeps_first_char_and_domain() -> None:
    assert mask("john@example.com", "EMAIL_ADDRESS", MaskOptions()) == "j***@example.com"


def test_mask_single_char_local_part_is_fully_masked() -> None:
    # keeping the prefix would reveal the entire local part
    assert mask("j@example.com", "EMAIL_ADDRESS", MaskOptions()) == "*@example.com"


def test_mask_phone_keeps_last_four_and_separators() -> None:
    assert mask("555-867-5309", "PHONE_NUMBER", MaskOptions()) == "***-***-5309"


def test_mask_credit_card_keeps_last_four() -> None:
    assert mask("4111 1111 1111 1111", "CREDIT_CARD", MaskOptions()) == "**** **** **** 1111"


def test_mask_generic_keeps_first_char() -> None:
    assert mask("Seattle", "LOCATION", MaskOptions()) == "S******"


def test_mask_explicit_options_override_entity_defaults() -> None:
    opts = MaskOptions(char="#", keep_prefix=2, keep_suffix=2)
    assert mask("john@example.com", "EMAIL_ADDRESS", opts) == "jo############om"


def test_mask_never_reveals_whole_value_when_keeps_exceed_length() -> None:
    opts = MaskOptions(keep_prefix=5, keep_suffix=5)
    assert mask("short", "PERSON", opts) == "*****"


def test_hasher_is_deterministic_per_key() -> None:
    h = Hasher(b"k1")
    assert h.token("Maria", "PERSON") == h.token("Maria", "PERSON")
    assert h.token("Maria", "PERSON") != h.token("Marla", "PERSON")
    assert h.token("Maria", "PERSON") != Hasher(b"k2").token("Maria", "PERSON")


def test_hasher_token_format_and_length() -> None:
    token = Hasher(b"k", token_length=8).token("x", "IP_ADDRESS")
    assert token.startswith("<IP_ADDRESS:") and token.endswith(">")
    assert len(token) == len("<IP_ADDRESS:>") + 8


def test_hasher_rejects_empty_secret() -> None:
    with pytest.raises(ValueError):
        Hasher(b"")


def test_hasher_from_env_uses_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEIL_TEST_KEY", "s3cret")
    assert Hasher.from_env("VEIL_TEST_KEY").digest("a") == Hasher(b"s3cret").digest("a")


def test_hasher_from_env_warns_and_uses_ephemeral_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("VEIL_TEST_KEY", raising=False)
    with caplog.at_level(logging.WARNING):
        a = Hasher.from_env("VEIL_TEST_KEY")
        b = Hasher.from_env("VEIL_TEST_KEY")
    assert "VEIL_TEST_KEY is not set" in caplog.text
    assert a.digest("x") != b.digest("x")
