"""Re-score spaCy ``PERSON`` detections by structural evidence (ADR-0012).

spaCy's NER emits a flat 0.85 for every ``PERSON``, whether the span is ``Maria Gonzalez`` or
``Deliverables``. That is not a confidence — it is a constant, and it makes the policy
engine's per-entity thresholds inert for the one entity that most needs them.

Measured on the 100 clean business documents: every one of the 78 real names in the labelled
set is two or three capitalised tokens, while 33 of 39 false positives are a *single* token
(company names, product names, plural nouns like ``Deliverables``). Token structure separates
them almost perfectly, so it is used to turn the constant into a graded score.

This deliberately does **not** filter. A single-token span keeps a reduced score and reaches
the policy engine, where a deployment that wants bare surnames caught in logs can lower the
threshold for that destination. Detection reports evidence; policy decides.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Final

from veil.detect.schema import Detection, Source

PERSON: Final = "PERSON"
IP_ADDRESS: Final = "IP_ADDRESS"

# A span of two or more capitalised tokens: the shape of a real name. Keeps spaCy's score.
# A single token: ambiguous between surname, company, product or plural noun. Reduced to just
# below the shipped policy's log/vector-store threshold, so it is audited but not acted on
# unless a deployment opts in.
SINGLE_TOKEN_CONFIDENCE: Final = 0.4

# A span containing digits, newlines or sentence punctuation is a mis-aligned span rather than
# a name, e.g. "Elliott\nLine 1: Atlas". Scored low enough that no sane threshold applies it.
MALFORMED_CONFIDENCE: Final = 0.15

_MALFORMED_CHARS: Final = frozenset('\n\r\t:;,/\\|()[]{}<>@#$%&*=+"')


def _is_malformed(value: str) -> bool:
    return any(c.isdigit() for c in value) or bool(_MALFORMED_CHARS & set(value))


def _capitalised_tokens(value: str) -> int:
    return sum(1 for token in value.split() if token[:1].isupper())


def person_confidence(value: str, original: float) -> float:
    """Score a ``PERSON`` span from its surface structure."""
    if not value or _is_malformed(value):
        return MALFORMED_CONFIDENCE
    if _capitalised_tokens(value) >= 2:
        return original
    return min(original, SINGLE_TOKEN_CONFIDENCE)


# --- IP_ADDRESS ----------
#
# A four-part dotted number is syntactically a valid IPv4 whether it addresses a host or
# names a release: "version 10.2.14.3" parses exactly like "host 10.2.14.3". Presidio scores
# both 0.6. Nothing in the string separates them, so the surrounding words are the only
# evidence available — and reserved addresses are never personal data regardless of context.

# Unspecified, broadcast, loopback, and "this network". Identifying a machine these name is
# impossible, so they carry no personal information.
_NEVER_PERSONAL_IPS: Final = frozenset({"0.0.0.0", "255.255.255.255"})
_NEVER_PERSONAL_PREFIXES: Final = ("127.", "0.")

# Version vocabulary only. Deliberately not fitted to the eval corpus: no "client to" or
# "agent to", which appear in the hard negatives but are not evidence of anything.
_VERSION_CONTEXT: Final = re.compile(
    r"\b(?:version|versions|release|released|build|builds|rev|revision|tag|tagged|"
    r"upgrade|upgraded|downgrade|downgraded|bump|bumped|pin|pinned|semver|patch|v)\b",
    re.IGNORECASE,
)
# Deliberately tight: only an immediately adjacent keyword counts ("to version 10.2.14.3",
# "bumped to v3.1.0.9"). A wider window caught cases like
# "pinned the kubernetes client to 1.28.4.2", but it also downgraded a genuine address in
# "version 10.2.14.3 and host 10.0.0.14" — and wrongly downgrading a real IP leaks it, while
# wrongly keeping a version string only over-redacts. The asymmetry decides the direction.
_CONTEXT_WINDOW: Final = 12

VERSION_LIKE_IP_CONFIDENCE: Final = 0.3
NEVER_PERSONAL_IP_CONFIDENCE: Final = 0.1


def ip_confidence(value: str, preceding: str, original: float) -> float:
    """Score an ``IP_ADDRESS`` span using reserved ranges and preceding version vocabulary."""
    if value in _NEVER_PERSONAL_IPS or value.startswith(_NEVER_PERSONAL_PREFIXES):
        return min(original, NEVER_PERSONAL_IP_CONFIDENCE)
    if _VERSION_CONTEXT.search(preceding[-_CONTEXT_WINDOW:]):
        return min(original, VERSION_LIKE_IP_CONFIDENCE)
    return original


def calibrate(detections: Iterable[Detection], text: str) -> list[Detection]:
    """Return detections with spaCy ``PERSON`` scores replaced by graded ones.

    Only ``PERSON`` from the NER model is touched: pattern recognizers and detect-secrets
    carry meaningful scores of their own (ADR-0003).
    """
    out: list[Detection] = []
    for det in detections:
        if det.source is not Source.PRESIDIO:
            out.append(det)
            continue
        if det.entity_type == PERSON:
            score = person_confidence(det.span_text(text), det.confidence)
        elif det.entity_type == IP_ADDRESS:
            score = ip_confidence(det.span_text(text), text[: det.start], det.confidence)
        else:
            out.append(det)
            continue
        out.append(det if score == det.confidence else det.model_copy(update={"confidence": score}))
    return out


__all__: Sequence[str] = [
    "MALFORMED_CONFIDENCE",
    "SINGLE_TOKEN_CONFIDENCE",
    "calibrate",
    "person_confidence",
]
