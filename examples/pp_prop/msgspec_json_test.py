"""Tests for the repository's msgspec-backed JSON compatibility helpers."""

from __future__ import annotations

import math
from io import StringIO

import msgspec
import pytest

import msgspec_json


def test_dumps_preserves_canonical_and_pretty_json_shapes() -> None:
    value = {"z": 1, "a": [True, None]}

    assert msgspec_json.dumps(value, sort_keys=True, separators=(",", ":")) == (
        '{"a":[true,null],"z":1}'
    )
    assert msgspec_json.dumps(
        value,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    ) == '{\n  "a": [\n    true,\n    null\n  ],\n  "z": 1\n}'


def test_dumps_rejects_non_finite_values_when_strict() -> None:
    assert msgspec_json.dumps({"loss": math.nan}) == '{"loss": NaN}'

    with pytest.raises(ValueError, match="JSON|range|compliant"):
        msgspec_json.dumps({"loss": math.nan}, allow_nan=False)


def test_load_dump_and_encoder_use_msgspec_and_support_text_and_bytes() -> None:
    value = {"answer": 42}
    stream = StringIO()
    msgspec_json.dump(value, stream)

    assert msgspec_json.loads(stream.getvalue()) == value
    assert msgspec_json.loads(stream.getvalue().encode()) == value
    assert msgspec_json.load(StringIO(stream.getvalue())) == value
    assert list(msgspec_json.JSONEncoder(sort_keys=True).iterencode(value)) == [
        '{"answer": 42}'
    ]


def test_decode_errors_are_msgspec_decode_errors() -> None:
    with pytest.raises(msgspec.DecodeError):
        msgspec_json.loads(b"{broken")
    assert msgspec_json.JSONDecodeError is msgspec.DecodeError


def test_loads_preserves_duplicate_key_rejection_hook() -> None:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        values: dict[str, object] = {}
        for key, value in pairs:
            if key in values:
                raise ValueError(f"duplicate JSON key {key!r}")
            values[key] = value
        return values

    with pytest.raises(ValueError, match="duplicate JSON key"):
        msgspec_json.loads(
            b'{"a": 1, "a": 2}',
            object_pairs_hook=reject_duplicates,
        )
