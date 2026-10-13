"""Text transformations for the redact / mask / hash actions."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets

from veil.policy.schema import MaskOptions

logger = logging.getLogger(__name__)


def redact(entity_type: str, template: str) -> str:
    return template.format(entity=entity_type)


def mask(value: str, entity_type: str, options: MaskOptions) -> str:
    """Partially hide ``value``.

    Explicit ``keep_prefix``/``keep_suffix`` options win. Otherwise entity-aware defaults
    apply: emails keep the first character and the domain (``j***@example.com``); phone and
    card numbers keep the last four digits; everything else keeps the first character.
    """
    if options.keep_prefix or options.keep_suffix:
        return _mask_edges(value, options.char, options.keep_prefix, options.keep_suffix)
    if entity_type == "EMAIL_ADDRESS" and "@" in value:
        local, _, domain = value.partition("@")
        return f"{_mask_edges(local, options.char, 1, 0)}@{domain}"
    if entity_type in {"PHONE_NUMBER", "CREDIT_CARD", "US_BANK_NUMBER", "IBAN_CODE"}:
        return _mask_digits_keep_last(value, options.char, 4)
    return _mask_edges(value, options.char, 1, 0)


def _mask_edges(value: str, char: str, keep_prefix: int, keep_suffix: int) -> str:
    if keep_prefix + keep_suffix >= len(value):
        # Nothing would be hidden; mask everything instead of leaking the value.
        return char * len(value)
    middle = len(value) - keep_prefix - keep_suffix
    suffix = value[len(value) - keep_suffix :] if keep_suffix else ""
    return f"{value[:keep_prefix]}{char * middle}{suffix}"


def _mask_digits_keep_last(value: str, char: str, keep: int) -> str:
    """Mask digits except the last ``keep``; preserve separators so the shape is recognizable."""
    digit_positions = [i for i, c in enumerate(value) if c.isdigit()]
    to_mask = set(digit_positions[:-keep]) if keep else set(digit_positions)
    return "".join(char if i in to_mask else c for i, c in enumerate(value))


class Hasher:
    """Deterministic, keyed hashing for the ``hash`` action (ADR-0007).

    The same raw value always maps to the same token within one key, so a name that appears
    in turn 1 and turn 5 of a conversation is consistently ``<PERSON:3f9a…>``. Without the
    key the token cannot be reversed or matched against a dictionary of guesses.
    """

    def __init__(self, secret: bytes, token_length: int = 12) -> None:
        if not secret:
            raise ValueError("hash secret must not be empty")
        self._secret = secret
        self._token_length = token_length

    @classmethod
    def from_env(cls, env_var: str, token_length: int = 12) -> Hasher:
        raw = os.environ.get(env_var)
        if raw:
            return cls(raw.encode("utf-8"), token_length)
        logger.warning(
            "%s is not set; using an ephemeral per-process hash key. Hash tokens will not be "
            "stable across restarts.",
            env_var,
        )
        return cls(secrets.token_bytes(32), token_length)

    def digest(self, value: str) -> str:
        """Full hex HMAC of ``value`` — used for audit span hashes."""
        return hmac.new(self._secret, value.encode("utf-8"), hashlib.sha256).hexdigest()

    def token(self, value: str, entity_type: str) -> str:
        return f"<{entity_type}:{self.digest(value)[: self._token_length]}>"
