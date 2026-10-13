"""Load and validate a policy from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from veil.policy.schema import Policy


class PolicyLoadError(ValueError):
    """Raised when a policy file is missing, malformed, or fails schema validation."""


def load_policy(path: str | Path) -> Policy:
    file = Path(path)
    if not file.is_file():
        raise PolicyLoadError(f"policy file not found: {file}")
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyLoadError(f"{file}: invalid YAML: {exc}") from exc
    return policy_from_dict(raw, source=str(file))


def policy_from_dict(raw: Any, *, source: str = "<dict>") -> Policy:
    if not isinstance(raw, dict):
        raise PolicyLoadError(f"{source}: policy must be a mapping at the top level")
    try:
        return Policy.model_validate(raw)
    except ValidationError as exc:
        raise PolicyLoadError(f"{source}: {exc}") from exc
