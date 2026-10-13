"""Policy configuration schema (ADR-0004).

A policy maps ``entity_type -> action`` with per-entity confidence thresholds, and lets a
*destination* (where the text is about to go) override any of that. Rules are partial:
a destination override may change only the action and inherit the threshold, or vice versa.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from veil.detect.engine import DetectorConfig

ENTITY_KEY_PATTERN = r"^[A-Z][A-Z0-9_]*$"


class Action(StrEnum):
    """What to do with a detected span whose confidence clears the threshold."""

    REDACT = "redact"  # replace with a placeholder such as <EMAIL_ADDRESS>
    MASK = "mask"  # partially hide, e.g. j***@example.com
    HASH = "hash"  # replace with a deterministic HMAC token (ADR-0007)
    BLOCK = "block"  # reject the whole operation
    ALLOW = "allow"  # leave the text untouched; still audited


class Destination(StrEnum):
    """Where the scanned text is about to be sent. Each may carry its own overrides."""

    MODEL = "model"  # inbound prompt to the LLM provider
    RESPONSE = "response"  # outbound completion returned to the caller
    LOGS = "logs"  # application logs, traces, observability
    VECTOR_STORE = "vector_store"  # embeddings / retrieval index


class MaskOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    char: str = Field(default="*", min_length=1, max_length=1)
    keep_prefix: int = Field(default=0, ge=0)
    keep_suffix: int = Field(default=0, ge=0)


class Rule(BaseModel):
    """A partial rule. ``None`` means "inherit from the next level up"."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Action | None = None
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    mask: MaskOptions | None = None
    reason: str | None = Field(
        default=None, description="Free-text rationale, surfaced in the audit log"
    )


class Defaults(BaseModel):
    """Fully-specified fallback used when no entity rule matches."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Action = Action.REDACT
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    mask: MaskOptions = MaskOptions()


class DestinationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    defaults: Rule = Rule()
    entities: dict[str, Rule] = Field(default_factory=dict)

    @field_validator("entities")
    @classmethod
    def _entity_keys(cls, value: dict[str, Rule]) -> dict[str, Rule]:
        return _validate_entity_keys(value)


class Policy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1, le=1)
    name: str = Field(min_length=1)
    description: str | None = None
    redact_template: str = Field(
        default="<{entity}>",
        description="Placeholder for the redact action; {entity} is the entity type",
    )
    hash_secret_env: str = Field(
        default="VEIL_HASH_SECRET",
        description="Environment variable holding the HMAC key for the hash action",
    )
    hash_token_length: int = Field(default=12, ge=8, le=64)
    defaults: Defaults = Defaults()
    entities: dict[str, Rule] = Field(default_factory=dict)
    destinations: dict[Destination, DestinationPolicy] = Field(default_factory=dict)
    detection: DetectorConfig = DetectorConfig()

    @field_validator("entities")
    @classmethod
    def _entity_keys(cls, value: dict[str, Rule]) -> dict[str, Rule]:
        return _validate_entity_keys(value)

    @field_validator("redact_template")
    @classmethod
    def _template_has_entity(cls, value: str) -> str:
        if "{entity}" not in value:
            raise ValueError("redact_template must contain '{entity}'")
        return value


def _validate_entity_keys(value: dict[str, Rule]) -> dict[str, Rule]:
    import re

    bad = [k for k in value if not re.match(ENTITY_KEY_PATTERN, k)]
    if bad:
        raise ValueError(f"entity keys must be UPPER_SNAKE_CASE, got {bad}")
    return value
