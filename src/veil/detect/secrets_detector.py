"""detect-secrets adapter: credential detection normalized to character spans (ADR-0001).

detect-secrets is built for scanning files, not strings, so two upstream behaviours have
to be corrected here (see ADR-0003, "detect-secrets quirks"):

1. ``PotentialSecret`` carries no character offsets, and for regex plugins whose pattern
   has a capture group ``secret_value`` is only the captured group (e.g. ``"ghp"`` for a
   GitHub token). Spans are therefore re-derived from ``regex.finditer`` on the full match.
2. ``scan_line`` runs entropy plugins in *eager* mode and intentionally skips the entropy
   limit so callers can see near-misses. We re-apply ``plugin.entropy_limit`` ourselves.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator, Sequence
from typing import Final

from detect_secrets.core.scan import scan_line
from detect_secrets.plugins.base import RegexBasedDetector
from detect_secrets.plugins.high_entropy_strings import HighEntropyStringsPlugin
from detect_secrets.settings import default_settings, get_plugins

from veil.detect.schema import Detection, Source

# detect-secrets types are binary hits; these confidences reflect how format-specific the
# plugin is. Provider-prefixed tokens (AKIA…, ghp_…, sk_live_…) are near-certain; a keyword
# assignment or high-entropy blob is a weaker signal that policy may want to treat differently.
CONFIDENCE_REGEX_TOKEN: Final = 0.9
CONFIDENCE_KEYWORD: Final = 0.7
CONFIDENCE_ENTROPY: Final = 0.6

# Presidio already emits IP_ADDRESS with a validated recognizer; map detect-secrets' IPv4
# plugin onto the same label so the overlap resolver de-duplicates them.
_TYPE_ALIASES: Final[dict[str, str]] = {
    "Public IP (ipv4)": "IP_ADDRESS",
    "Base64 High Entropy String": "HIGH_ENTROPY_STRING",
    "Hex High Entropy String": "HIGH_ENTROPY_STRING",
    "Secret Keyword": "SECRET_KEYWORD",
    "Private Key": "PRIVATE_KEY",
    "JSON Web Token": "JWT",
}

# Plugins whose regex match includes a non-secret prefix (``password = ``, ``https://user:``).
# For these we redact only the secret value; for everything else the whole match is the token.
_VALUE_ONLY_TYPES: Final[frozenset[str]] = frozenset({"Secret Keyword", "Basic Auth Credentials"})

_PRIVATE_KEY_END: Final = re.compile(r"-----END [A-Z ]*PRIVATE KEY( BLOCK)?-----")

# detect-secrets keeps plugin/filter configuration in process-global settings; guard scans.
_SCAN_LOCK: Final = threading.Lock()


def normalize_secret_type(secret_type: str) -> str:
    """``"AWS Access Key"`` -> ``"AWS_ACCESS_KEY"``; aliases collapse near-duplicates."""
    if secret_type in _TYPE_ALIASES:
        return _TYPE_ALIASES[secret_type]
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", secret_type).strip("_")
    return cleaned.upper()


class SecretsDetector:
    """Runs every detect-secrets plugin over a string and returns offset-accurate detections."""

    def __init__(self, *, disabled_types: frozenset[str] = frozenset()) -> None:
        self._disabled_types = disabled_types

    @property
    def name(self) -> str:
        return "detect-secrets"

    def detect(self, text: str) -> Sequence[Detection]:
        if not text:
            return []
        detections: list[Detection] = []
        with _SCAN_LOCK, default_settings():
            # scan_line resets the plugin cache; resolve instances inside the same settings
            # context so entropy limits match what scan_line used.
            plugins = {p.secret_type: p for p in get_plugins()}
            for line_start, line in _lines_with_offsets(text):
                for secret in scan_line(line):
                    if secret.type in self._disabled_types:
                        continue
                    plugin = plugins.get(secret.type)
                    if plugin is None:
                        continue
                    value = secret.secret_value
                    if value is None or not _passes_entropy_limit(plugin, value):
                        continue
                    for start, end in _locate(plugin, secret.type, value, line):
                        detections.append(
                            Detection(
                                entity_type=normalize_secret_type(secret.type),
                                start=line_start + start,
                                end=line_start + end,
                                confidence=_confidence_for(plugin),
                                source=Source.DETECT_SECRETS,
                                recognizer=type(plugin).__name__,
                            )
                        )
        return _dedupe(_extend_private_keys(detections, text))


def _lines_with_offsets(text: str) -> Iterator[tuple[int, str]]:
    offset = 0
    for line in text.splitlines(keepends=True):
        yield offset, line.rstrip("\r\n")
        offset += len(line)


def _passes_entropy_limit(plugin: object, value: str) -> bool:
    if isinstance(plugin, HighEntropyStringsPlugin):
        return plugin.calculate_shannon_entropy(value) > plugin.entropy_limit
    return True


def _confidence_for(plugin: object) -> float:
    if isinstance(plugin, HighEntropyStringsPlugin):
        return CONFIDENCE_ENTROPY
    if getattr(plugin, "secret_type", "") == "Secret Keyword":
        return CONFIDENCE_KEYWORD
    return CONFIDENCE_REGEX_TOKEN


def _locate(plugin: object, secret_type: str, value: str, line: str) -> list[tuple[int, int]]:
    """Find character spans for a secret. Regex plugins use their own patterns so the span
    covers the *whole* token even when ``value`` is a truncated capture group."""
    if isinstance(plugin, HighEntropyStringsPlugin) or not isinstance(plugin, RegexBasedDetector):
        return _find_all(line, value)

    spans: list[tuple[int, int]] = []
    for regex in plugin.denylist:
        for match in regex.finditer(line):
            whole = match.group(0)
            if value not in whole:
                continue
            if secret_type in _VALUE_ONLY_TYPES:
                inner = whole.find(value)
                spans.append((match.start() + inner, match.start() + inner + len(value)))
            else:
                spans.append(match.span())
    return spans or _find_all(line, value)


def _find_all(line: str, value: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    if not value:
        return spans
    index = line.find(value)
    while index != -1:
        spans.append((index, index + len(value)))
        index = line.find(value, index + len(value))
    return spans


def _extend_private_keys(detections: list[Detection], text: str) -> list[Detection]:
    """detect-secrets only matches ``BEGIN … PRIVATE KEY`` (without the leading dashes).
    Extend the span back over the ``-----`` prefix and forward to the matching END marker
    (or end of text) so the whole PEM block is governed, not just the header."""
    extended: list[Detection] = []
    for det in detections:
        if det.entity_type != "PRIVATE_KEY":
            extended.append(det)
            continue
        new_start = det.start
        while new_start > 0 and text[new_start - 1] == "-":
            new_start -= 1
        end_match = _PRIVATE_KEY_END.search(text, det.end)
        new_end = end_match.end() if end_match else len(text)
        extended.append(det.model_copy(update={"start": new_start, "end": max(new_end, det.end)}))
    return extended


def _dedupe(detections: list[Detection]) -> list[Detection]:
    seen: set[tuple[str, int, int]] = set()
    unique: list[Detection] = []
    for det in detections:
        key = (det.entity_type, det.start, det.end)
        if key in seen:
            continue
        seen.add(key)
        unique.append(det)
    return sorted(unique, key=lambda d: (d.start, d.end))
