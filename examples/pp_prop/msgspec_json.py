"""Small ``msgspec``-backed compatibility helpers for JSON artifacts.

The Example 21 and benchmark tools exchange ordinary JSON documents.  This
module keeps their existing text-oriented call sites while making
``msgspec.json`` the only JSON implementation used by the repository's
example tooling.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any, TextIO

import msgspec


JSONDecodeError = msgspec.DecodeError


class _JsonScanError(Exception):
    """Internal marker for malformed input during duplicate-key scanning."""


def _skip_json_whitespace(data: bytes, position: int) -> int:
    while position < len(data) and data[position] in b" \t\r\n":
        position += 1
    return position


def _scan_json_string(data: bytes, position: int) -> tuple[str, int]:
    if position >= len(data) or data[position] != ord('"'):
        raise _JsonScanError("Expected JSON string. Fix the input condition named in the error, then rerun the operation.")
    end = position + 1
    while end < len(data):
        if data[end] == ord('\\'):
            end += 2
            continue
        if data[end] == ord('"'):
            try:
                value = msgspec.json.decode(data[position : end + 1])
            except msgspec.DecodeError as error:
                raise _JsonScanError from error
            if not isinstance(value, str):
                raise _JsonScanError("JSON object key was not text. Fix the input condition named in the error, then rerun the operation.")
            return value, end + 1
        end += 1
    raise _JsonScanError("Unterminated JSON string. Fix the input condition named in the error, then rerun the operation.")


def _scan_json_value(
    data: bytes,
    position: int,
    object_pairs_hook: Callable[[list[tuple[str, Any]]], Any],
) -> int:
    position = _skip_json_whitespace(data, position)
    if position >= len(data):
        raise _JsonScanError("Missing JSON value. Provide the missing item named in the message.")
    token = data[position]
    if token == ord('"'):
        return _scan_json_string(data, position)[1]
    if token == ord('{'):
        return _scan_json_object(data, position, object_pairs_hook)
    if token == ord('['):
        position += 1
        position = _skip_json_whitespace(data, position)
        if position < len(data) and data[position] == ord(']'):
            return position + 1
        while True:
            position = _scan_json_value(data, position, object_pairs_hook)
            position = _skip_json_whitespace(data, position)
            if position >= len(data):
                raise _JsonScanError("Unterminated JSON array. Fix the input condition named in the error, then rerun the operation.")
            if data[position] == ord(']'):
                return position + 1
            if data[position] != ord(','):
                raise _JsonScanError("Expected JSON array separator. Fix the input condition named in the error, then rerun the operation.")
            position += 1
    start = position
    while position < len(data) and data[position] not in b" \t\r\n,]}":
        position += 1
    if position == start:
        raise _JsonScanError("Missing JSON scalar. Provide the missing item named in the message.")
    return position


def _scan_json_object(
    data: bytes,
    position: int,
    object_pairs_hook: Callable[[list[tuple[str, Any]]], Any],
) -> int:
    position = _skip_json_whitespace(data, position + 1)
    keys: list[str] = []
    if position < len(data) and data[position] == ord('}'):
        object_pairs_hook([])
        return position + 1
    while True:
        key, position = _scan_json_string(data, position)
        keys.append(key)
        position = _skip_json_whitespace(data, position)
        if position >= len(data) or data[position] != ord(':'):
            raise _JsonScanError("Expected JSON object separator. Fix the input condition named in the error, then rerun the operation.")
        position = _scan_json_value(data, position + 1, object_pairs_hook)
        position = _skip_json_whitespace(data, position)
        if position >= len(data):
            raise _JsonScanError("Unterminated JSON object. Fix the input condition named in the error, then rerun the operation.")
        if data[position] == ord('}'):
            object_pairs_hook([(item, None) for item in keys])
            return position + 1
        if data[position] != ord(','):
            raise _JsonScanError("Expected JSON object separator. Fix the input condition named in the error, then rerun the operation.")
        position = _skip_json_whitespace(data, position + 1)


def _check_duplicate_keys(
    value: str | bytes | bytearray,
    object_pairs_hook: Callable[[list[tuple[str, Any]]], Any],
) -> None:
    data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    try:
        position = _scan_json_value(data, 0, object_pairs_hook)
        if _skip_json_whitespace(data, position) != len(data):
            raise _JsonScanError("Trailing JSON data. Fix the input condition named in the error, then rerun the operation.")
    except _JsonScanError:
        # Let msgspec produce the authoritative malformed-input exception.
        return


def _check_json_constants(
    value: str | bytes | bytearray,
    parse_constant: Callable[[str], Any],
) -> None:
    """Invoke a legacy constant hook for non-standard numeric tokens."""

    data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    position = 0
    in_string = False
    while position < len(data):
        token = data[position]
        if token == ord('"'):
            in_string = not in_string
            position += 1
            continue
        if in_string:
            if token == ord('\\'):
                position += 2
            else:
                position += 1
            continue
        if token in (ord('N'), ord('I'), ord('-')):
            end = position
            while end < len(data) and data[end] not in b" \t\r\n,]}":
                end += 1
            candidate = data[position:end].decode("ascii", errors="ignore")
            if candidate in {"NaN", "Infinity", "-Infinity"}:
                parse_constant(candidate)
            position = end
            continue
        position += 1


def _reject_non_finite(value: object) -> None:
    """Reject non-finite real values before msgspec maps them to ``null``."""

    if isinstance(value, Real) and not isinstance(value, (bool, int)):
        if not math.isfinite(value):
            raise ValueError("Out of range float values are not JSON compliant. Fix the input condition named in the error, then rerun the operation.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(key)
            _reject_non_finite(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)


def _normalise_non_finite(
    value: object,
    replacements: list[tuple[bytes, bytes]],
) -> object:
    if isinstance(value, Real) and not isinstance(value, (bool, int)):
        if math.isfinite(value):
            return value
        token = "NaN" if math.isnan(value) else "-Infinity" if value < 0 else "Infinity"
        marker = f"__braintrace_nonfinite_{len(replacements)}__"
        replacements.append(
            (msgspec.json.encode(marker), token.encode("ascii"))
        )
        return marker
    if isinstance(value, Mapping):
        return {
            key: _normalise_non_finite(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalise_non_finite(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalise_non_finite(item, replacements) for item in value)
    return value


def _restore_non_finite(
    encoded: bytes,
    replacements: list[tuple[bytes, bytes]],
) -> bytes:
    for marker, token in replacements:
        encoded = encoded.replace(marker, token)
    return encoded


def _encoded(
    value: object,
    *,
    allow_nan: bool,
    sort_keys: bool,
) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    if not allow_nan:
        _reject_non_finite(value)
        normalised = value
        replacements: list[tuple[bytes, bytes]] = []
    else:
        replacements = []
        normalised = _normalise_non_finite(value, replacements)
    return (
        msgspec.json.encode(normalised, order="sorted" if sort_keys else None),
        replacements,
    )


def dumps(
    value: object,
    *,
    allow_nan: bool = True,
    indent: int | None = None,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
) -> str:
    """Serialize a JSON-compatible value as UTF-8 text.

    Parameters
    ----------
    value
        JSON-compatible value to serialize.
    allow_nan
        If false, reject non-finite real values like the standard library
        encoder's strict mode.
    Indent
        Optional indentation width.  ``None`` preserves the standard
        library's compact-with-separators default.
    sort_keys
        Sort mapping keys deterministically when true.
    Separators
        Optional comma/colon separators.  The compact ``(',', ':')`` form is
        supported for canonical fingerprints; the other repository call sites
        use the normal JSON separators.

    Returns
    -------
    str
        UTF-8 JSON text.
    """

    encoded, replacements = _encoded(
        value, allow_nan=allow_nan, sort_keys=sort_keys
    )
    if separators == (",", ":"):
        return _restore_non_finite(encoded, replacements).decode("utf-8")
    if indent is None:
        # Indent=0 is msgspec's compact representation with standard JSON
        # spaces after commas and colons, matching the standard encoder default.
        encoded = msgspec.json.format(encoded, indent=0)
    else:
        encoded = msgspec.json.format(encoded, indent=indent)
    return _restore_non_finite(encoded, replacements).decode("utf-8")


def _encode_bytes(
    value: object,
    *,
    allow_nan: bool = True,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
) -> bytes:
    """Serialize a JSON-compatible value directly to UTF-8 bytes."""

    encoded, replacements = _encoded(
        value, allow_nan=allow_nan, sort_keys=sort_keys
    )
    if separators == (",", ":"):
        return _restore_non_finite(encoded, replacements)
    return _restore_non_finite(msgspec.json.format(encoded, indent=0), replacements)


def loads(
    value: str | bytes | bytearray,
    *,
    parse_constant: Callable[[str], Any] | None = None,
    object_pairs_hook: Callable[[list[tuple[str, Any]]], Any] | None = None,
    strict: bool = True,
) -> Any:
    """Deserialize a JSON document from text or bytes.

    ``parse_constant`` is accepted for compatibility with older call sites.
    ``msgspec`` rejects non-standard constants before a callback could map
    them, which preserves the strict behavior required by these tools.
    ``object_pairs_hook`` is used to retain duplicate-key rejection for the
    one strict artifact loader that relies on it.
    """

    if parse_constant is not None:
        _check_json_constants(value, parse_constant)
    if object_pairs_hook is not None:
        _check_duplicate_keys(value, object_pairs_hook)

    return msgspec.json.decode(value, strict=strict)


def load(stream: TextIO) -> Any:
    """Deserialize one JSON document from a text or binary stream."""

    return msgspec.json.decode(stream.read())


def dump(
    value: object,
    stream: TextIO,
    *,
    allow_nan: bool = True,
    indent: int | None = None,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
) -> None:
    """Serialize one JSON document to a text stream."""

    stream.write(
        dumps(
            value,
            allow_nan=allow_nan,
            indent=indent,
            sort_keys=sort_keys,
            separators=separators,
        )
    )


class JSONEncoder:
    """Minimal ``JSONEncoder`` surface for the legacy streaming call site."""

    def __init__(
        self,
        *,
        allow_nan: bool = True,
        indent: int | None = None,
        sort_keys: bool = False,
        separators: tuple[str, str] | None = None,
    ) -> None:
        self._allow_nan = allow_nan
        self._indent = indent
        self._sort_keys = sort_keys
        self._separators = separators

    def encode(self, value: object) -> str:
        """Encode one value as text."""

        if self._indent is None:
            return _encode_bytes(
                value,
                allow_nan=self._allow_nan,
                sort_keys=self._sort_keys,
                separators=self._separators,
            ).decode("utf-8")
        return dumps(
            value,
            allow_nan=self._allow_nan,
            indent=self._indent,
            sort_keys=self._sort_keys,
            separators=self._separators,
        )

    def iterencode(self, value: object):
        """Yield the msgspec-encoded document as one UTF-8 text chunk."""

        yield self.encode(value)
