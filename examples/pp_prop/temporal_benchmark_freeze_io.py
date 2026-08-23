"""Strict JSON input, hashing, and atomic output for selection freezing."""

from __future__ import annotations

import hashlib
import msgspec_json
import pathlib
from collections.abc import Mapping
from typing import Any, NoReturn


class FreezeArtifactError(ValueError):
    """Indicate malformed or mutually inconsistent selection evidence."""


def _reject_constant(value: str) -> NoReturn:
    raise FreezeArtifactError(f"artifact contains non-finite JSON constant {value}")


def load_artifact(path: pathlib.Path) -> dict[str, Any]:
    """Load one strict JSON object without accepting NaN or infinity."""
    try:
        value = msgspec_json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, msgspec_json.JSONDecodeError) as error:
        raise FreezeArtifactError(f"cannot read artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise FreezeArtifactError(f"artifact {path} must contain a JSON object")
    return value


def artifact_reference(
    path: pathlib.Path, document: Mapping[str, object]
) -> dict[str, object]:
    """Identify an exact input by semantic header and byte-level digest."""
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise FreezeArtifactError(f"cannot hash artifact {path}: {error}") from error
    return {
        "filename": path.name,
        "sha256": digest,
        "kind": document.get("kind"),
        "schema_version": document.get("schema_version"),
    }


def write_artifact(path: pathlib.Path, document: Mapping[str, object]) -> None:
    """Atomically write strict, stable JSON suitable for a committed artifact."""
    serialized = msgspec_json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)
