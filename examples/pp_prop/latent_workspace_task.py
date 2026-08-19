"""Standard ARC data, provenance, augmentation, and row-event encoding.

This module deliberately contains no model code.  It turns ordinary ARC tasks
into an immutable host representation and then into bounded, target-free row
events that a recurrent spiking model can consume.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, TypeAlias

import brainstate
import numpy as np
from numpy.typing import NDArray


MAX_GRID_SIZE = 30
COLOR_COUNT = 10

SourceRole: TypeAlias = Literal["train", "tuning", "evaluation", "fixture"]
SourceFormat: TypeAlias = Literal["auto", "task_json", "collection_json", "jsonl"]
FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int32]
BoolArray: TypeAlias = NDArray[np.bool_]

_SOURCE_ROLES = frozenset(("train", "tuning", "evaluation", "fixture"))
_SOURCE_FORMATS = frozenset(("auto", "task_json", "collection_json", "jsonl"))
_EVALUATION_ONLY_SOURCES = frozenset(("arc-agi-1 evaluation", "arc-task-gen"))


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _normalise_grid(value: object, *, field: str) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, ArcGrid):
        return value.cells
    if isinstance(value, np.ndarray):
        if value.ndim != 2:
            raise ValueError(f"{field} must be a rectangular two-dimensional grid")
        rows: object = value.tolist()
    else:
        rows = value
    if not _is_sequence(rows) or len(rows) == 0:
        raise ValueError(f"{field} must contain at least one row")
    if len(rows) > MAX_GRID_SIZE:
        raise ValueError(
            f"{field} height {len(rows)} exceeds ARC maximum {MAX_GRID_SIZE}"
        )

    normalised: list[tuple[int, ...]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(rows):
        if not _is_sequence(raw_row) or len(raw_row) == 0:
            raise ValueError(f"{field}[{row_index}] must be a non-empty row")
        if width is None:
            width = len(raw_row)
            if width > MAX_GRID_SIZE:
                raise ValueError(
                    f"{field} width {width} exceeds ARC maximum {MAX_GRID_SIZE}"
                )
        elif len(raw_row) != width:
            raise ValueError(
                f"{field} is ragged: row {row_index} has width {len(raw_row)}, "
                f"expected {width}"
            )

        row: list[int] = []
        for column_index, cell in enumerate(raw_row):
            if isinstance(cell, (bool, np.bool_)) or not isinstance(
                cell, (int, np.integer)
            ):
                raise ValueError(
                    f"{field}[{row_index}][{column_index}] must be an integer ARC color"
                )
            color = int(cell)
            if not 0 <= color < COLOR_COUNT:
                raise ValueError(
                    f"{field}[{row_index}][{column_index}] color {color} is "
                    "outside 0..9"
                )
            row.append(color)
        normalised.append(tuple(row))
    return tuple(normalised)


@dataclass(frozen=True, slots=True)
class ArcGrid:
    """An immutable validated ARC color grid.

    Parameters
    ----------
    cells
        Rectangular rows containing integer colors from 0 through 9.

    Attributes
    ----------
    cells
        Tuple-backed rectangular grid.
    """

    cells: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cells", _normalise_grid(self.cells, field="ArcGrid.cells")
        )

    @property
    def height(self) -> int:
        """Return the number of rows."""

        return len(self.cells)

    @property
    def width(self) -> int:
        """Return the number of columns."""

        return len(self.cells[0])

    def as_array(self) -> IntArray:
        """Return a writable ``int32`` copy of the grid.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(height, width)``.
        """

        return np.asarray(self.cells, dtype=np.int32)

    def to_list(self) -> list[list[int]]:
        """Return the ordinary JSON-compatible ARC representation.

        Returns
        -------
        list of list of int
            Mutable rows suitable for ``json.dump``.
        """

        return [list(row) for row in self.cells]


@dataclass(frozen=True, slots=True)
class ArcPair:
    """One ARC input and its optional output.

    Parameters
    ----------
    input
        Input grid.
    output
        Output grid, or ``None`` for an unscored test query.

    Attributes
    ----------
    input
        Validated input grid.
    output
        Validated output grid when available.
    """

    input: ArcGrid
    output: ArcGrid | None

    def __post_init__(self) -> None:
        if not isinstance(self.input, ArcGrid):
            object.__setattr__(self, "input", ArcGrid(self.input))
        if self.output is not None and not isinstance(self.output, ArcGrid):
            object.__setattr__(self, "output", ArcGrid(self.output))


@dataclass(frozen=True, slots=True)
class ArcTask:
    """An immutable standard ARC task.

    Parameters
    ----------
    train
        One or more demonstration input/output pairs.
    test
        One or more query pairs.  Query outputs may be absent for inference.
    task_id
        Optional human-facing identifier.  It is never encoded for the model.

    Attributes
    ----------
    train
        Demonstration pairs in task order.
    test
        Test queries in task order.
    task_id
        Optional source identifier excluded from fingerprints and model input.
    """

    train: tuple[ArcPair, ...]
    test: tuple[ArcPair, ...]
    task_id: str | None = None

    def __post_init__(self) -> None:
        train = tuple(self.train)
        test = tuple(self.test)
        if not train:
            raise ValueError("ArcTask.train must contain at least one demonstration")
        if not test:
            raise ValueError("ArcTask.test must contain at least one query")
        if any(not isinstance(pair, ArcPair) for pair in train):
            raise ValueError("ArcTask.train entries must be ArcPair instances")
        if any(not isinstance(pair, ArcPair) for pair in test):
            raise ValueError("ArcTask.test entries must be ArcPair instances")
        for index, pair in enumerate(train):
            if pair.output is None:
                raise ValueError(f"ArcTask.train[{index}].output is required")
        if self.task_id is not None and (
            not isinstance(self.task_id, str) or not self.task_id.strip()
        ):
            raise ValueError("ArcTask.task_id must be a non-empty string or None")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "test", test)


@dataclass(frozen=True, slots=True)
class ArcQueryEpisode:
    """One test query paired with the complete task demonstrations.

    Attributes
    ----------
    task_index
        Stable index of the parent task in the evaluated collection.
    query_index
        Stable index of this query inside ``ArcTask.test``.
    task_id
        Optional human-facing task identifier.
    task_fingerprint
        Content fingerprint of the complete parent task.
    demonstrations
        Complete, ordered demonstration set shared by every task query.
    query_input
        Held-out query input grid.
    target
        Query output when available for scoring.
    """

    task_index: int
    query_index: int
    task_id: str | None
    task_fingerprint: str
    demonstrations: tuple[ArcPair, ...]
    query_input: ArcGrid
    target: ArcGrid | None


def arc_task_from_mapping(
    payload: Mapping[str, object],
    *,
    task_id: str | None = None,
    require_test_outputs: bool = False,
) -> ArcTask:
    """Parse one ordinary ARC task mapping.

    Parameters
    ----------
    payload
        Mapping with ``train`` and ``test`` sequences.
    task_id
        Optional identifier used only in diagnostics.
    require_test_outputs
        If true, reject unscored test entries.

    Returns
    -------
    ArcTask
        Validated immutable task.

    Raises
    ------
    ValueError
        If a task, pair, or grid violates the ARC contract.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(f"task {task_id or '<unknown>'} must be a JSON object")
    label = f"task {task_id or '<unknown>'}"
    train_payload = payload.get("train")
    test_payload = payload.get("test")
    if not _is_sequence(train_payload) or not train_payload:
        raise ValueError(f"{label}.train must contain at least one pair")
    if not _is_sequence(test_payload) or not test_payload:
        raise ValueError(f"{label}.test must contain at least one query")

    train = tuple(
        _pair_from_mapping(pair, field=f"{label}.train[{index}]", require_output=True)
        for index, pair in enumerate(train_payload)
    )
    test = tuple(
        _pair_from_mapping(
            pair,
            field=f"{label}.test[{index}]",
            require_output=require_test_outputs,
        )
        for index, pair in enumerate(test_payload)
    )
    return ArcTask(train=train, test=test, task_id=task_id)


def _pair_from_mapping(payload: object, *, field: str, require_output: bool) -> ArcPair:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    if "input" not in payload:
        raise ValueError(f"{field}.input is required")
    if require_output and "output" not in payload:
        raise ValueError(f"{field}.output is required")
    try:
        input_grid = ArcGrid(_normalise_grid(payload["input"], field=f"{field}.input"))
        output_grid = (
            ArcGrid(_normalise_grid(payload["output"], field=f"{field}.output"))
            if "output" in payload and payload["output"] is not None
            else None
        )
    except (TypeError, KeyError) as error:
        raise ValueError(f"{field} is malformed: {error}") from error
    if require_output and output_grid is None:
        raise ValueError(f"{field}.output is required")
    return ArcPair(input=input_grid, output=output_grid)


def arc_task_to_mapping(
    task: ArcTask, *, include_test_outputs: bool = True
) -> dict[str, list[dict[str, object]]]:
    """Convert a task to the canonical JSON-compatible ARC shape.

    Parameters
    ----------
    task
        Task to serialize.
    include_test_outputs
        Whether available test targets should be included.

    Returns
    -------
    dict
        Mapping containing only normalized ``train`` and ``test`` content.
    """

    def pair_mapping(pair: ArcPair, *, include_output: bool) -> dict[str, object]:
        result: dict[str, object] = {"input": pair.input.to_list()}
        if include_output and pair.output is not None:
            result["output"] = pair.output.to_list()
        return result

    return {
        "train": [pair_mapping(pair, include_output=True) for pair in task.train],
        "test": [
            pair_mapping(pair, include_output=include_test_outputs)
            for pair in task.test
        ],
    }


def load_arc_task(
    path: str | os.PathLike[str], *, require_test_outputs: bool = False
) -> ArcTask:
    """Load one standard per-task ARC JSON file.

    Parameters
    ----------
    path
        JSON file containing one ``train``/``test`` task.
    require_test_outputs
        If true, reject test entries without outputs.

    Returns
    -------
    ArcTask
        Parsed task whose identifier is the file stem.
    """

    task_path = Path(path)
    try:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load ARC task {task_path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"ARC task file {task_path} must contain one JSON object")
    return arc_task_from_mapping(
        payload,
        task_id=task_path.stem,
        require_test_outputs=require_test_outputs,
    )


def canonical_task_fingerprint(
    task: ArcTask, *, include_test_outputs: bool = True
) -> str:
    """Hash normalized task content independently of names and provenance.

    Parameters
    ----------
    task
        Task whose ordered train/test content should be hashed.
    include_test_outputs
        Whether available held-out outputs contribute to the content hash.

    Returns
    -------
    str
        Lowercase SHA-256 hexadecimal digest.
    """

    canonical = json.dumps(
        arc_task_to_mapping(task, include_test_outputs=include_test_outputs),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def query_episodes(
    task: ArcTask, *, task_index: int = 0
) -> tuple[ArcQueryEpisode, ...]:
    """Expand all test queries while retaining their shared task identity.

    Parameters
    ----------
    task
        Parent ARC task.
    task_index
        Stable collection index used by strict task aggregation.

    Returns
    -------
    tuple of ArcQueryEpisode
        One episode per query with the complete demonstration tuple.
    """

    if isinstance(task_index, bool) or not isinstance(task_index, (int, np.integer)):
        raise ValueError("task_index must be a non-negative integer")
    task_index = int(task_index)
    if task_index < 0:
        raise ValueError("task_index must be a non-negative integer")
    fingerprint = canonical_task_fingerprint(task)
    return tuple(
        ArcQueryEpisode(
            task_index=task_index,
            query_index=query_index,
            task_id=task.task_id,
            task_fingerprint=fingerprint,
            demonstrations=task.train,
            query_input=pair.input,
            target=pair.output,
        )
        for query_index, pair in enumerate(task.test)
    )


def leave_one_demonstration_out_episodes(
    task: ArcTask, *, task_index: int = 0
) -> tuple[ArcQueryEpisode, ...]:
    """Construct supervised episodes by holding out each demonstration.

    Each episode retains the parent task fingerprint and original held-out
    index.  The held-out input becomes the query, its output remains in the
    out-of-band ``target`` field, and all other demonstrations remain in their
    original order.

    Parameters
    ----------
    task
        Parent ARC task with at least two demonstrations.
    task_index
        Stable collection index used by strict task aggregation.

    Returns
    -------
    tuple of ArcQueryEpisode
        One supervised episode per demonstration.

    Raises
    ------
    ValueError
        If ``task_index`` is invalid or the task has fewer than two
        demonstrations.
    """

    if isinstance(task_index, bool) or not isinstance(task_index, (int, np.integer)):
        raise ValueError("task_index must be a non-negative integer")
    task_index = int(task_index)
    if task_index < 0:
        raise ValueError("task_index must be a non-negative integer")
    if len(task.train) < 2:
        raise ValueError(
            "leave-one-demonstration-out episodes require at least two demonstrations"
        )

    fingerprint = canonical_task_fingerprint(task)
    return tuple(
        ArcQueryEpisode(
            task_index=task_index,
            query_index=held_out_index,
            task_id=task.task_id,
            task_fingerprint=fingerprint,
            demonstrations=(
                task.train[:held_out_index] + task.train[held_out_index + 1 :]
            ),
            query_input=held_out.input,
            target=held_out.output,
        )
        for held_out_index, held_out in enumerate(task.train)
    )


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """Declaration of one locally available ARC corpus.

    Parameters
    ----------
    name
        Declared public corpus name.
    role
        One of ``train``, ``tuning``, ``evaluation``, or ``fixture``.
    version
        Operator-declared dataset version or immutable revision.
    path
        Local file or directory.
    license_reference
        Non-empty license URL/name or authoritative dataset reference.
    format
        ``task_json``, ``collection_json``, ``jsonl``, or automatic detection.
    exclude_fingerprints
        Explicit task fingerprints to omit from a training/tuning source. Each
        declaration must match and is recorded in the resolved manifest.

    Attributes
    ----------
    name, role, version, path, license_reference, format, exclude_fingerprints
        Auditable source declaration fields.
    """

    name: str
    role: SourceRole
    version: str
    path: str
    license_reference: str
    format: SourceFormat = "auto"
    exclude_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "license_reference"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"DatasetSource.{field_name} must be non-empty")
            object.__setattr__(self, field_name, value.strip())
        if not isinstance(self.path, (str, os.PathLike)) or not os.fspath(self.path):
            raise ValueError("DatasetSource.path must be non-empty")
        object.__setattr__(self, "path", os.fspath(self.path))
        if self.role not in _SOURCE_ROLES:
            raise ValueError(
                f"DatasetSource.role must be one of {sorted(_SOURCE_ROLES)}"
            )
        if self.format not in _SOURCE_FORMATS:
            raise ValueError(
                f"DatasetSource.format must be one of {sorted(_SOURCE_FORMATS)}"
            )
        fingerprints: list[str] = []
        for fingerprint in self.exclude_fingerprints:
            if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                raise ValueError(
                    "DatasetSource.exclude_fingerprints entries must be SHA-256 hex digests"
                )
            normalized = fingerprint.casefold()
            if any(character not in "0123456789abcdef" for character in normalized):
                raise ValueError(
                    "DatasetSource.exclude_fingerprints entries must be SHA-256 hex digests"
                )
            fingerprints.append(normalized)
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("DatasetSource.exclude_fingerprints must be unique")
        if fingerprints and self.role not in {"train", "tuning"}:
            raise ValueError(
                "DatasetSource.exclude_fingerprints are allowed only for train/tuning roles"
            )
        object.__setattr__(self, "exclude_fingerprints", tuple(fingerprints))
        folded_name = self.name.casefold()
        if folded_name in _EVALUATION_ONLY_SOURCES and self.role != "evaluation":
            raise ValueError(f"{self.name} is evaluation-only, not role {self.role}")
        if "private" in folded_name and (
            "paper" in folded_name or "bdh" in folded_name
        ):
            raise ValueError(
                "the paper's private data is not an available dataset source"
            )


@dataclass(frozen=True, slots=True)
class SourceFileHash:
    """Hash evidence for one source file.

    Attributes
    ----------
    path
        Path relative to the declared source root.
    sha256
        Lowercase file digest.
    size_bytes
        Exact file size.
    """

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class RejectedTask:
    """One rejected corpus item and its validation reason.

    Attributes
    ----------
    origin
        File and collection/line position.
    reason
        Human-readable validation failure.
    """

    origin: str
    reason: str


@dataclass(frozen=True, slots=True)
class ExcludedTask:
    """One explicitly excluded fitting task.

    Attributes
    ----------
    origin
        File and collection/line position from which the task was loaded.
    task_id
        Logical task identifier supplied by the corpus adapter.
    fingerprint
        Canonical task content fingerprint matched by the declaration.
    """

    origin: str
    task_id: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """Resolved evidence for one loaded source.

    Attributes
    ----------
    source
        Original auditable declaration.
    resolved_path
        Absolute local source path, or ``<embedded>`` for fixtures.
    files
        Hashes of every parsed source file.
    parsed_task_count
        Valid tasks before within-source deduplication.
    valid_task_count
        Unique accepted tasks returned to the caller.
    rejected
        Invalid file/items and their reasons.
    duplicate_fingerprints
        Fingerprints removed as within-source duplicates.
    task_fingerprints
        Unique accepted content fingerprints.
    exclusions
        Every explicit source-declaration exclusion that matched a valid task.
    plumbing_only
        True only for embedded smoke fixtures.
    private_paper_data_available
        Always false: the paper's private corpus was unavailable.
    private_training_recipe_available
        Always false: the paper's private recipe was unavailable.
    """

    source: DatasetSource
    resolved_path: str
    files: tuple[SourceFileHash, ...]
    parsed_task_count: int
    valid_task_count: int
    rejected: tuple[RejectedTask, ...]
    duplicate_fingerprints: tuple[str, ...]
    task_fingerprints: tuple[str, ...]
    exclusions: tuple[ExcludedTask, ...] = ()
    plumbing_only: bool = False
    private_paper_data_available: bool = False
    private_training_recipe_available: bool = False

    def __post_init__(self) -> None:
        if self.private_paper_data_available or self.private_training_recipe_available:
            raise ValueError(
                "the paper's private data and training recipe are unavailable"
            )

    @property
    def rejected_task_count(self) -> int:
        """Return the number of rejected source items."""

        return len(self.rejected)

    @property
    def duplicate_task_count(self) -> int:
        """Return the number of removed duplicate tasks."""

        return len(self.duplicate_fingerprints)

    @property
    def excluded_task_count(self) -> int:
        """Return the number of explicitly excluded valid tasks."""

        return len(self.exclusions)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible evidence mapping.

        Returns
        -------
        dict
            Complete declaration, hash, count, rejection, and claim evidence.
        """

        return {
            "source": {
                "name": self.source.name,
                "role": self.source.role,
                "version": self.source.version,
                "path": self.source.path,
                "license_reference": self.source.license_reference,
                "format": self.source.format,
                "exclude_fingerprints": list(self.source.exclude_fingerprints),
            },
            "resolved_path": self.resolved_path,
            "files": [
                {
                    "path": item.path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.files
            ],
            "parsed_task_count": self.parsed_task_count,
            "valid_task_count": self.valid_task_count,
            "rejected_task_count": self.rejected_task_count,
            "rejected": [
                {"origin": item.origin, "reason": item.reason} for item in self.rejected
            ],
            "duplicate_task_count": self.duplicate_task_count,
            "duplicate_fingerprints": list(self.duplicate_fingerprints),
            "task_fingerprints": list(self.task_fingerprints),
            "excluded_task_count": self.excluded_task_count,
            "exclusions": [
                {
                    "origin": item.origin,
                    "task_id": item.task_id,
                    "fingerprint": item.fingerprint,
                }
                for item in self.exclusions
            ],
            "plumbing_only": self.plumbing_only,
            "private_paper_data_available": self.private_paper_data_available,
            "private_training_recipe_available": self.private_training_recipe_available,
        }


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """Unique valid tasks and the manifest proving how they were loaded.

    Attributes
    ----------
    tasks
        Unique accepted ARC tasks.
    manifest
        Resolved source evidence and rejection accounting.
    """

    tasks: tuple[ArcTask, ...]
    manifest: SourceManifest


def _source_files(root: Path, source_format: SourceFormat) -> tuple[Path, ...]:
    if root.is_file():
        files = (root,)
    elif root.is_dir():
        suffixes = (
            {".json", ".jsonl", ".ndjson"}
            if source_format == "auto"
            else {".jsonl", ".ndjson"}
            if source_format == "jsonl"
            else {".json"}
        )
        files = tuple(
            sorted(
                (
                    path
                    for path in root.rglob("*")
                    if path.suffix.casefold() in suffixes
                ),
                key=lambda path: path.as_posix(),
            )
        )
    else:
        raise ValueError(f"dataset path does not exist: {root}")
    if not files:
        raise ValueError(f"dataset path contains no supported files: {root}")
    return files


def _hash_file(path: Path, *, root: Path) -> SourceFileHash:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    relative = path.name if root.is_file() else path.relative_to(root).as_posix()
    return SourceFileHash(
        path=relative, sha256=digest.hexdigest(), size_bytes=path.stat().st_size
    )


def _looks_like_task(payload: object) -> bool:
    return isinstance(payload, Mapping) and "train" in payload and "test" in payload


def _unwrap_task(payload: object, fallback_id: str) -> tuple[str, object]:
    if isinstance(payload, Mapping) and "task" in payload:
        task_id = payload.get("id", payload.get("task_id", fallback_id))
        return str(task_id), payload["task"]
    if isinstance(payload, Mapping):
        task_id = payload.get("id", payload.get("task_id", fallback_id))
        return str(task_id), payload
    return fallback_id, payload


def _collection_entries(payload: object, stem: str) -> list[tuple[str, object]]:
    if _looks_like_task(payload):
        return [(stem, payload)]
    if isinstance(payload, Mapping) and "tasks" in payload:
        payload = payload["tasks"]
    if _is_sequence(payload):
        return [
            _unwrap_task(item, f"{stem}:{index}") for index, item in enumerate(payload)
        ]
    if isinstance(payload, Mapping):
        return [_unwrap_task(item, str(task_id)) for task_id, item in payload.items()]
    raise ValueError("collection must be a task, sequence of tasks, or task mapping")


def _file_entries(
    path: Path, source_format: SourceFormat
) -> list[tuple[str, object, str]]:
    effective = source_format
    if effective == "auto":
        effective = (
            "jsonl"
            if path.suffix.casefold() in {".jsonl", ".ndjson"}
            else "collection_json"
        )
    if effective == "jsonl":
        entries: list[tuple[str, object, str]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError(str(error)) from error
        for line_index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                origin = f"{path.name}:line {line_index}"
                entries.append((f"{path.stem}:{line_index}", error, origin))
                continue
            task_id, task_payload = _unwrap_task(payload, f"{path.stem}:{line_index}")
            origin = f"{path.name}:line {line_index}:{task_id}"
            entries.append((task_id, task_payload, origin))
        return entries
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(str(error)) from error
    if effective == "task_json":
        return [(path.stem, payload, f"{path.name}:{path.stem}")]
    return [
        (task_id, task_payload, f"{path.name}:{task_id}")
        for task_id, task_payload in _collection_entries(payload, path.stem)
    ]


def load_dataset_source(
    source: DatasetSource, *, require_test_outputs: bool | None = None
) -> LoadedDataset:
    """Load, validate, hash, and deduplicate one declared corpus.

    Parameters
    ----------
    source
        Manifest-backed local dataset declaration.
    require_test_outputs
        Override whether every query needs a target.  By default evaluation and
        fixture sources require outputs while training/tuning sources do not.

    Returns
    -------
    LoadedDataset
        Unique tasks plus complete provenance and rejection accounting.

    Raises
    ------
    ValueError
        If the source path is absent, contains no supported files, or contains
        no valid unique tasks.
    """

    root = Path(source.path).expanduser().resolve()
    files = _source_files(root, source.format)
    if require_test_outputs is None:
        require_test_outputs = source.role in {"evaluation", "fixture"}

    file_hashes = tuple(_hash_file(path, root=root) for path in files)
    parsed: list[tuple[ArcTask, str]] = []
    rejected: list[RejectedTask] = []
    for path in files:
        try:
            entries = _file_entries(path, source.format)
        except ValueError as error:
            rejected.append(RejectedTask(origin=path.name, reason=str(error)))
            continue
        for item_index, (item_id, payload, origin) in enumerate(entries):
            if isinstance(payload, Exception):
                rejected.append(RejectedTask(origin=origin, reason=str(payload)))
                continue
            try:
                if not isinstance(payload, Mapping):
                    raise ValueError("task payload must be a JSON object")
                parsed.append(
                    (
                        arc_task_from_mapping(
                            payload,
                            task_id=str(item_id or f"{path.stem}:{item_index}"),
                            require_test_outputs=require_test_outputs,
                        ),
                        origin,
                    )
                )
            except ValueError as error:
                rejected.append(RejectedTask(origin=origin, reason=str(error)))

    unique: list[ArcTask] = []
    fingerprints: list[str] = []
    duplicates: list[str] = []
    exclusions: list[ExcludedTask] = []
    matched_exclusions: set[str] = set()
    seen: set[str] = set()
    for task, origin in parsed:
        fingerprint = canonical_task_fingerprint(task)
        if fingerprint in source.exclude_fingerprints:
            matched_exclusions.add(fingerprint)
            exclusions.append(
                ExcludedTask(
                    origin=origin,
                    task_id=task.task_id or "<unknown>",
                    fingerprint=fingerprint,
                )
            )
            continue
        if fingerprint in seen:
            duplicates.append(fingerprint)
            continue
        seen.add(fingerprint)
        fingerprints.append(fingerprint)
        unique.append(task)
    unmatched = set(source.exclude_fingerprints) - matched_exclusions
    if unmatched:
        raise ValueError(
            f"DatasetSource.exclude_fingerprints did not match source {source.name}: "
            f"{sorted(unmatched)}"
        )
    if not unique:
        summary = "; ".join(item.reason for item in rejected[:3])
        raise ValueError(
            f"dataset source {source.name} has no valid unique tasks: {summary}"
        )

    manifest = SourceManifest(
        source=source,
        resolved_path=str(root),
        files=file_hashes,
        parsed_task_count=len(parsed),
        valid_task_count=len(unique),
        rejected=tuple(rejected),
        duplicate_fingerprints=tuple(duplicates),
        task_fingerprints=tuple(fingerprints),
        exclusions=tuple(exclusions),
    )
    return LoadedDataset(tasks=tuple(unique), manifest=manifest)


@dataclass(frozen=True, slots=True)
class SplitOverlap:
    """One forbidden training/tuning versus evaluation overlap.

    Attributes
    ----------
    fingerprint
        Canonical task digest shared across the split boundary.
    fitting_sources
        Training or tuning source names and roles.
    evaluation_sources
        Evaluation source names and roles.
    """

    fingerprint: str
    fitting_sources: tuple[str, ...]
    evaluation_sources: tuple[str, ...]


class SplitLeakageError(ValueError):
    """Error raised when evaluation content appears in a fitting split.

    Parameters
    ----------
    overlaps
        One or more forbidden fitting/evaluation content overlaps.

    Attributes
    ----------
    overlaps
        Immutable overlap evidence included in the error message.
    """

    def __init__(self, overlaps: Sequence[SplitOverlap]):
        self.overlaps = tuple(overlaps)
        details = "; ".join(
            f"{item.fingerprint}: fitting={list(item.fitting_sources)}, "
            f"evaluation={list(item.evaluation_sources)}"
            for item in self.overlaps
        )
        super().__init__(f"ARC train/evaluation fingerprint leakage: {details}")


def detect_split_leakage(
    manifests: Iterable[SourceManifest],
) -> tuple[SplitOverlap, ...]:
    """Find canonical content shared by fitting and evaluation roles.

    Parameters
    ----------
    manifests
        Resolved source manifests to compare.

    Returns
    -------
    tuple of SplitOverlap
        Deterministically ordered forbidden overlaps.
    """

    fitting: dict[str, set[str]] = {}
    evaluation: dict[str, set[str]] = {}
    for manifest in manifests:
        label = f"{manifest.source.name} ({manifest.source.role})"
        target = (
            fitting
            if manifest.source.role in {"train", "tuning"}
            else evaluation
            if manifest.source.role == "evaluation"
            else None
        )
        if target is None:
            continue
        for fingerprint in manifest.task_fingerprints:
            target.setdefault(fingerprint, set()).add(label)
    return tuple(
        SplitOverlap(
            fingerprint=fingerprint,
            fitting_sources=tuple(sorted(fitting[fingerprint])),
            evaluation_sources=tuple(sorted(evaluation[fingerprint])),
        )
        for fingerprint in sorted(fitting.keys() & evaluation.keys())
    )


def assert_no_evaluation_leakage(manifests: Iterable[SourceManifest]) -> None:
    """Abort when a training/tuning task also occurs in evaluation.

    Parameters
    ----------
    manifests
        Resolved source manifests to compare before optimization.

    Raises
    ------
    SplitLeakageError
        If one or more canonical fingerprints cross the protected boundary.
    """

    overlaps = detect_split_leakage(manifests)
    if overlaps:
        raise SplitLeakageError(overlaps)


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Training-only ARC augmentation switches.

    Attributes
    ----------
    permute_colors
        Permute foreground colors 1 through 9 while retaining background 0.
    dihedral
        Draw one of the eight square-grid dihedral transforms.
    shuffle_demonstrations
        Permute demonstration order without changing query order.
    """

    permute_colors: bool = True
    dihedral: bool = True
    shuffle_demonstrations: bool = True


@dataclass(frozen=True, slots=True)
class TaskAugmentation:
    """An augmented task and the exact sampled transformation.

    Attributes
    ----------
    task
        New immutable task.  The source task is never mutated.
    color_map
        Ten-element source-to-destination color mapping.
    dihedral_index
        Transform index 0 through 7.
    demonstration_order
        Source indices in the resulting demonstration order.
    """

    task: ArcTask
    color_map: tuple[int, ...]
    dihedral_index: int
    demonstration_order: tuple[int, ...]


def _transform_grid(
    grid: ArcGrid, color_map: NDArray[np.int32], dihedral_index: int
) -> ArcGrid:
    array = color_map[grid.as_array()]
    if dihedral_index >= 4:
        array = np.fliplr(array)
    array = np.rot90(array, k=dihedral_index % 4)
    return ArcGrid(tuple(tuple(int(cell) for cell in row) for row in array.tolist()))


def draw_training_augmentation(
    task: ArcTask,
    rng: brainstate.random.RandomState,
    *,
    role: SourceRole = "train",
    config: AugmentationConfig = AugmentationConfig(),
) -> TaskAugmentation:
    """Draw and apply a relation-preserving training augmentation.

    Parameters
    ----------
    task
        Immutable source task.
    rng
        BrainState random stream used for every random choice.
    role
        Must be ``train``; evaluation and fixture tasks fail closed.
    config
        Enabled color, dihedral, and demonstration-order transforms.

    Returns
    -------
    TaskAugmentation
        New task plus exact sampled transformation evidence.
    """

    if role != "train":
        raise ValueError(f"augmentation is training-only, not role {role}")
    if not isinstance(config, AugmentationConfig):
        raise ValueError("config must be an AugmentationConfig")

    color_map = np.arange(COLOR_COUNT, dtype=np.int32)
    if config.permute_colors:
        color_map[1:] = np.asarray(
            rng.permutation(np.arange(1, COLOR_COUNT, dtype=np.int32)),
            dtype=np.int32,
        )
    dihedral_index = int(np.asarray(rng.randint(8))) if config.dihedral else 0
    order = (
        tuple(int(index) for index in np.asarray(rng.permutation(len(task.train))))
        if config.shuffle_demonstrations
        else tuple(range(len(task.train)))
    )

    def transform_pair(pair: ArcPair) -> ArcPair:
        return ArcPair(
            input=_transform_grid(pair.input, color_map, dihedral_index),
            output=(
                _transform_grid(pair.output, color_map, dihedral_index)
                if pair.output is not None
                else None
            ),
        )

    augmented = ArcTask(
        train=tuple(transform_pair(task.train[index]) for index in order),
        test=tuple(transform_pair(pair) for pair in task.test),
        task_id=task.task_id,
    )
    return TaskAugmentation(
        task=augmented,
        color_map=tuple(int(value) for value in color_map),
        dihedral_index=dihedral_index,
        demonstration_order=order,
    )


def augment_training_task(
    task: ArcTask,
    rng: brainstate.random.RandomState,
    *,
    role: SourceRole = "train",
    config: AugmentationConfig = AugmentationConfig(),
) -> ArcTask:
    """Return only the task produced by ``draw_training_augmentation``.

    Parameters
    ----------
    task, rng, role, config
        See :func:`draw_training_augmentation`.

    Returns
    -------
    ArcTask
        Newly augmented immutable training task.
    """

    return draw_training_augmentation(task, rng, role=role, config=config).task


@dataclass(frozen=True, slots=True)
class RowEventConfig:
    """Static capacity and feature layout for lossless row events.

    Parameters
    ----------
    max_demonstrations
        Maximum number of demonstration pairs in one episode.
    max_grid_size
        Maximum height/width.  Standard ARC supports at most 30.
    color_count
        ARC color vocabulary size; must be 10.

    Attributes
    ----------
    max_demonstrations, max_grid_size, color_count
        Static representation capacities.
    """

    max_demonstrations: int = 10
    max_grid_size: int = MAX_GRID_SIZE
    color_count: int = COLOR_COUNT

    def __post_init__(self) -> None:
        for name in ("max_demonstrations", "max_grid_size", "color_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"RowEventConfig.{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.max_demonstrations < 1:
            raise ValueError("RowEventConfig.max_demonstrations must be positive")
        if not 1 <= self.max_grid_size <= MAX_GRID_SIZE:
            raise ValueError("RowEventConfig.max_grid_size must be in 1..30")
        if self.color_count != COLOR_COUNT:
            raise ValueError("RowEventConfig.color_count must be 10 for ARC")

    @property
    def max_events(self) -> int:
        """Return the maximum packed demonstration/query row count."""

        return (self.max_demonstrations + 1) * self.max_grid_size

    @property
    def valid_slice(self) -> slice:
        """Return the event-valid scalar slice."""

        return slice(0, 1)

    @property
    def phase_slice(self) -> slice:
        """Return the ``(demonstration, query)`` one-hot phase slice."""

        return slice(1, 3)

    @property
    def demonstration_slice(self) -> slice:
        """Return the demonstration-index one-hot slice."""

        return slice(3, 3 + self.max_demonstrations)

    @property
    def side_valid_slice(self) -> slice:
        """Return the ``(input, output)`` row-valid flag slice."""

        start = self.demonstration_slice.stop
        return slice(start, start + 2)

    @property
    def normalized_slice(self) -> slice:
        """Return normalized row/input/output dimension scalars."""

        start = self.side_valid_slice.stop
        return slice(start, start + 5)

    @property
    def row_index_slice(self) -> slice:
        """Return the row-position one-hot slice."""

        start = self.normalized_slice.stop
        return slice(start, start + self.max_grid_size)

    def _dimension_slice(self, offset: int) -> slice:
        start = self.row_index_slice.stop + offset * self.max_grid_size
        return slice(start, start + self.max_grid_size)

    @property
    def input_height_slice(self) -> slice:
        """Return the input-height one-hot slice."""

        return self._dimension_slice(0)

    @property
    def input_width_slice(self) -> slice:
        """Return the input-width one-hot slice."""

        return self._dimension_slice(1)

    @property
    def output_height_slice(self) -> slice:
        """Return the demonstration-output-height one-hot slice."""

        return self._dimension_slice(2)

    @property
    def output_width_slice(self) -> slice:
        """Return the demonstration-output-width one-hot slice."""

        return self._dimension_slice(3)

    @property
    def input_mask_slice(self) -> slice:
        """Return the 30-column input validity mask slice."""

        start = self.output_width_slice.stop
        return slice(start, start + self.max_grid_size)

    @property
    def output_mask_slice(self) -> slice:
        """Return the 30-column demonstration-output mask slice."""

        start = self.input_mask_slice.stop
        return slice(start, start + self.max_grid_size)

    @property
    def input_color_slice(self) -> slice:
        """Return position-specific input color one-hot features."""

        start = self.output_mask_slice.stop
        return slice(start, start + self.max_grid_size * self.color_count)

    @property
    def output_color_slice(self) -> slice:
        """Return position-specific demonstration-output color features."""

        start = self.input_color_slice.stop
        return slice(start, start + self.max_grid_size * self.color_count)

    @property
    def input_width(self) -> int:
        """Return the complete row-event feature width."""

        return self.output_color_slice.stop


@dataclass(frozen=True, slots=True)
class AssociativeMemoryFeatureIndices:
    """Matched row-event features for associative-memory keys and values.

    Parameters
    ----------
    key_indices
        Ordered input-side and shared row feature indices.
    value_indices
        Ordered output-side and shared row feature indices.

    Attributes
    ----------
    key_indices, value_indices
        Unique, non-negative index tuples with the same feature width.
    """

    key_indices: tuple[int, ...]
    value_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized: dict[str, tuple[int, ...]] = {}
        for name in ("key_indices", "value_indices"):
            raw_indices = getattr(self, name)
            if not raw_indices:
                raise ValueError(
                    "AssociativeMemoryFeatureIndices tuples must be non-empty"
                )
            if any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                for index in raw_indices
            ):
                raise ValueError(
                    "AssociativeMemoryFeatureIndices entries must be integers"
                )
            indices = tuple(int(index) for index in raw_indices)
            if any(index < 0 for index in indices):
                raise ValueError(
                    "AssociativeMemoryFeatureIndices entries must be non-negative"
                )
            if len(indices) != len(set(indices)):
                raise ValueError(
                    f"AssociativeMemoryFeatureIndices.{name} must be unique"
                )
            normalized[name] = indices
        if len(normalized["key_indices"]) != len(normalized["value_indices"]):
            raise ValueError(
                "AssociativeMemoryFeatureIndices tuples must have the same width"
            )
        object.__setattr__(self, "key_indices", normalized["key_indices"])
        object.__setattr__(self, "value_indices", normalized["value_indices"])


def associative_memory_feature_indices(
    config: RowEventConfig = RowEventConfig(),
) -> AssociativeMemoryFeatureIndices:
    """Return matched key/value row features for associative memory.

    The key contains input-side validity, the shared normalized row scalar,
    normalized input dimensions, row position, input dimension one-hots, input
    mask, and input colors.  The value mirrors that contract for the output
    side.  Event validity, phase, and demonstration identity are deliberately
    excluded so memory content represents row bindings rather than episode
    layout.

    Parameters
    ----------
    config
        Validated row-event layout whose slice boundaries define the indices.

    Returns
    -------
    AssociativeMemoryFeatureIndices
        Immutable matched feature-index tuples in semantic concatenation order.
    """

    def expand(feature_slice: slice) -> tuple[int, ...]:
        return tuple(range(feature_slice.start, feature_slice.stop))

    normalized_start = config.normalized_slice.start
    key_indices = (
        (
            config.side_valid_slice.start,
            normalized_start,
            normalized_start + 1,
            normalized_start + 2,
        )
        + expand(config.row_index_slice)
        + expand(config.input_height_slice)
        + expand(config.input_width_slice)
        + expand(config.input_mask_slice)
        + expand(config.input_color_slice)
    )
    value_indices = (
        (
            config.side_valid_slice.start + 1,
            normalized_start,
            normalized_start + 3,
            normalized_start + 4,
        )
        + expand(config.row_index_slice)
        + expand(config.output_height_slice)
        + expand(config.output_width_slice)
        + expand(config.output_mask_slice)
        + expand(config.output_color_slice)
    )
    return AssociativeMemoryFeatureIndices(key_indices, value_indices)


@dataclass(frozen=True, slots=True)
class DecodedQueryContext:
    """Row-event round-trip result without a held-out target.

    Attributes
    ----------
    demonstrations
        Recovered ordered input/output pairs.
    query_input
        Recovered query grid.
    """

    demonstrations: tuple[ArcPair, ...]
    query_input: ArcGrid


@dataclass(frozen=True, slots=True)
class EncodedQueryEpisode:
    """Static row-event tensor and non-input query metadata.

    Attributes
    ----------
    events
        Read-only ``float32`` array with shape ``(max_events, input_width)``.
    valid_event_count
        Number of non-padding row events across fixed demonstration/query blocks.
    query_start, query_stop
        Half-open query row span inside ``events``.
    demonstration_spans
        Fixed-size half-open row block for each ordered demonstration.
    task_index, query_index
        Stable indices used to reconstruct strict task metrics.
    task_id, task_fingerprint
        Host metadata that is not present in ``events``.
    target
        Held-out target kept outside model input, when available.
    """

    events: FloatArray
    valid_event_count: int
    query_start: int
    query_stop: int
    demonstration_spans: tuple[tuple[int, int], ...]
    task_index: int
    query_index: int
    task_id: str | None
    task_fingerprint: str
    target: ArcGrid | None

    def __post_init__(self) -> None:
        events = np.array(self.events, dtype=np.float32, copy=True)
        events.setflags(write=False)
        object.__setattr__(self, "events", events)


@dataclass(frozen=True, slots=True)
class GridTarget:
    """Padded output target kept separate from model input.

    Attributes
    ----------
    colors
        Read-only ``int32`` grid padded to the configured maximum.
    valid_mask
        Read-only mask selecting exactly the target cells.
    height_index, width_index
        Zero-based categorical targets for dimensions 1 through 30.
    """

    colors: IntArray
    valid_mask: BoolArray
    height_index: int
    width_index: int

    def __post_init__(self) -> None:
        colors = np.array(self.colors, dtype=np.int32, copy=True)
        mask = np.array(self.valid_mask, dtype=np.bool_, copy=True)
        colors.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "colors", colors)
        object.__setattr__(self, "valid_mask", mask)


def encode_target_grid(
    grid: ArcGrid, *, max_grid_size: int = MAX_GRID_SIZE
) -> GridTarget:
    """Pad a supervised output without exposing it as model input.

    Parameters
    ----------
    grid
        Target output grid.
    max_grid_size
        Static output side length.

    Returns
    -------
    GridTarget
        Padded colors, exact valid mask, and dimension class indices.
    """

    if not 1 <= max_grid_size <= MAX_GRID_SIZE:
        raise ValueError("max_grid_size must be in 1..30")
    if grid.height > max_grid_size or grid.width > max_grid_size:
        raise ValueError(
            f"target shape {grid.height}x{grid.width} exceeds capacity {max_grid_size}"
        )
    colors = np.zeros((max_grid_size, max_grid_size), dtype=np.int32)
    mask = np.zeros_like(colors, dtype=np.bool_)
    colors[: grid.height, : grid.width] = grid.as_array()
    mask[: grid.height, : grid.width] = True
    return GridTarget(
        colors=colors,
        valid_mask=mask,
        height_index=grid.height - 1,
        width_index=grid.width - 1,
    )


def _encode_side(
    event: FloatArray,
    grid: ArcGrid,
    row_index: int,
    *,
    is_output: bool,
    config: RowEventConfig,
) -> None:
    side_offset = 1 if is_output else 0
    event[config.side_valid_slice.start + side_offset] = 1.0
    normal_offset = 3 if is_output else 1
    event[config.normalized_slice.start + normal_offset] = (
        grid.height / config.max_grid_size
    )
    event[config.normalized_slice.start + normal_offset + 1] = (
        grid.width / config.max_grid_size
    )
    height_slice = (
        config.output_height_slice if is_output else config.input_height_slice
    )
    width_slice = config.output_width_slice if is_output else config.input_width_slice
    mask_slice = config.output_mask_slice if is_output else config.input_mask_slice
    color_slice = config.output_color_slice if is_output else config.input_color_slice
    event[height_slice.start + grid.height - 1] = 1.0
    event[width_slice.start + grid.width - 1] = 1.0
    event[mask_slice.start : mask_slice.start + grid.width] = 1.0
    for column_index, color in enumerate(grid.cells[row_index]):
        event[color_slice.start + column_index * config.color_count + color] = 1.0


def encode_query_episode(
    task: ArcTask,
    query_index: int,
    config: RowEventConfig = RowEventConfig(),
    *,
    task_index: int = 0,
) -> EncodedQueryEpisode:
    """Encode one query and all demonstrations as lossless row events.

    Parameters
    ----------
    task
        Standard ARC task.
    query_index
        Index into ``task.test``.
    config
        Static demonstration, grid, and feature capacities.
    task_index
        Stable parent-task collection index.

    Returns
    -------
    EncodedQueryEpisode
        Fixed demo/query blocks, explicit zero padding, and target out-of-band.
    """

    if isinstance(query_index, bool) or not isinstance(query_index, (int, np.integer)):
        raise ValueError("query_index must be an integer")
    query_index = int(query_index)
    if not 0 <= query_index < len(task.test):
        raise ValueError(
            f"query_index {query_index} outside task.test size {len(task.test)}"
        )
    if len(task.train) > config.max_demonstrations:
        raise ValueError(
            f"task has {len(task.train)} demonstrations, exceeds capacity "
            f"{config.max_demonstrations}"
        )
    if isinstance(task_index, bool) or not isinstance(task_index, (int, np.integer)):
        raise ValueError("task_index must be a non-negative integer")
    task_index = int(task_index)
    if task_index < 0:
        raise ValueError("task_index must be a non-negative integer")
    all_grids = [pair.input for pair in task.train]
    all_grids.extend(pair.output for pair in task.train if pair.output is not None)
    all_grids.append(task.test[query_index].input)
    oversized = [
        f"{grid.height}x{grid.width}"
        for grid in all_grids
        if grid.height > config.max_grid_size or grid.width > config.max_grid_size
    ]
    if oversized:
        raise ValueError(
            f"grid shapes {oversized} exceed row-event capacity {config.max_grid_size}"
        )

    events = np.zeros((config.max_events, config.input_width), dtype=np.float32)
    valid_event_count = 0
    demonstration_spans: list[tuple[int, int]] = []
    for demonstration_index, pair in enumerate(task.train):
        assert pair.output is not None
        demonstration_start = demonstration_index * config.max_grid_size
        demonstration_stop = demonstration_start + config.max_grid_size
        for row_index in range(max(pair.input.height, pair.output.height)):
            event = events[demonstration_start + row_index]
            event[config.valid_slice] = 1.0
            event[config.phase_slice.start] = 1.0
            event[config.demonstration_slice.start + demonstration_index] = 1.0
            event[config.normalized_slice.start] = (
                row_index + 1
            ) / config.max_grid_size
            event[config.row_index_slice.start + row_index] = 1.0
            if row_index < pair.input.height:
                _encode_side(
                    event, pair.input, row_index, is_output=False, config=config
                )
            if row_index < pair.output.height:
                _encode_side(
                    event, pair.output, row_index, is_output=True, config=config
                )
            valid_event_count += 1
        demonstration_spans.append((demonstration_start, demonstration_stop))

    query_start = config.max_demonstrations * config.max_grid_size
    query = task.test[query_index]
    for row_index in range(query.input.height):
        event = events[query_start + row_index]
        event[config.valid_slice] = 1.0
        event[config.phase_slice.start + 1] = 1.0
        event[config.normalized_slice.start] = (row_index + 1) / config.max_grid_size
        event[config.row_index_slice.start + row_index] = 1.0
        _encode_side(event, query.input, row_index, is_output=False, config=config)
        valid_event_count += 1
    query_stop = query_start + query.input.height
    return EncodedQueryEpisode(
        events=events,
        valid_event_count=valid_event_count,
        query_start=query_start,
        query_stop=query_stop,
        demonstration_spans=tuple(demonstration_spans),
        task_index=task_index,
        query_index=query_index,
        task_id=task.task_id,
        task_fingerprint=canonical_task_fingerprint(task),
        target=query.output,
    )


def encode_arc_query_episode(
    episode: ArcQueryEpisode,
    config: RowEventConfig = RowEventConfig(),
) -> EncodedQueryEpisode:
    """Encode a prepared ARC query episode as target-free row events.

    The demonstration context and query input are encoded exactly as supplied.
    The optional target remains out-of-band and cannot affect the returned
    event bytes.  Parent identity metadata is copied from ``episode`` instead
    of being recomputed from its reduced context, which is important for
    leave-one-demonstration-out training episodes.

    Parameters
    ----------
    episode
        Prepared query episode, including ordinary evaluation queries or
        leave-one-demonstration-out training queries.
    config
        Static demonstration, grid, and feature capacities.

    Returns
    -------
    EncodedQueryEpisode
        Fixed demonstration/query blocks with the original parent metadata and
        optional target kept outside model input.
    """

    sanitized_task = ArcTask(
        train=episode.demonstrations,
        test=(ArcPair(episode.query_input, None),),
        task_id=episode.task_id,
    )
    encoded = encode_query_episode(
        sanitized_task,
        0,
        config,
        task_index=episode.task_index,
    )
    return EncodedQueryEpisode(
        events=encoded.events,
        valid_event_count=encoded.valid_event_count,
        query_start=encoded.query_start,
        query_stop=encoded.query_stop,
        demonstration_spans=encoded.demonstration_spans,
        task_index=episode.task_index,
        query_index=episode.query_index,
        task_id=episode.task_id,
        task_fingerprint=episode.task_fingerprint,
        target=episode.target,
    )


def encode_task_queries(
    task: ArcTask,
    config: RowEventConfig = RowEventConfig(),
    *,
    task_index: int = 0,
) -> tuple[EncodedQueryEpisode, ...]:
    """Encode every test query with the same complete demonstrations.

    Parameters
    ----------
    task, config, task_index
        See :func:`encode_query_episode`.

    Returns
    -------
    tuple of EncodedQueryEpisode
        Query episodes preserving parent task and query indices.
    """

    return tuple(
        encode_query_episode(task, index, config, task_index=task_index)
        for index in range(len(task.test))
    )


def _decode_one_hot(row: FloatArray, feature_slice: slice, *, field: str) -> int:
    values = row[feature_slice]
    active = np.flatnonzero(values == 1.0)
    if active.size != 1 or np.any((values != 0.0) & (values != 1.0)):
        raise ValueError(f"{field} must be exactly one-hot")
    return int(active[0])


def _decode_grid_side(
    rows: Sequence[FloatArray], *, is_output: bool, config: RowEventConfig
) -> ArcGrid:
    side_offset = 1 if is_output else 0
    selected = [
        row for row in rows if row[config.side_valid_slice.start + side_offset] == 1.0
    ]
    if not selected:
        raise ValueError("encoded grid side contains no rows")
    height_slice = (
        config.output_height_slice if is_output else config.input_height_slice
    )
    width_slice = config.output_width_slice if is_output else config.input_width_slice
    mask_slice = config.output_mask_slice if is_output else config.input_mask_slice
    color_slice = config.output_color_slice if is_output else config.input_color_slice
    height = _decode_one_hot(selected[0], height_slice, field="height") + 1
    width = _decode_one_hot(selected[0], width_slice, field="width") + 1
    recovered: dict[int, tuple[int, ...]] = {}
    for row in selected:
        if _decode_one_hot(row, height_slice, field="height") + 1 != height:
            raise ValueError("encoded height changes across grid rows")
        if _decode_one_hot(row, width_slice, field="width") + 1 != width:
            raise ValueError("encoded width changes across grid rows")
        row_index = _decode_one_hot(row, config.row_index_slice, field="row index")
        expected_mask = np.zeros(config.max_grid_size, dtype=np.float32)
        expected_mask[:width] = 1.0
        if not np.array_equal(row[mask_slice], expected_mask):
            raise ValueError("encoded cell mask disagrees with width")
        colors = tuple(
            _decode_one_hot(
                row,
                slice(
                    color_slice.start + column * config.color_count,
                    color_slice.start + (column + 1) * config.color_count,
                ),
                field=f"color[{row_index}][{column}]",
            )
            for column in range(width)
        )
        if row_index in recovered:
            raise ValueError(f"duplicate encoded row index {row_index}")
        recovered[row_index] = colors
    if set(recovered) != set(range(height)):
        raise ValueError("encoded row indices do not cover the declared height")
    return ArcGrid(tuple(recovered[index] for index in range(height)))


def decode_row_events(
    events: NDArray[np.floating[Any]], config: RowEventConfig = RowEventConfig()
) -> DecodedQueryContext:
    """Recover demonstrations and query input from encoded model features.

    Parameters
    ----------
    events
        Row-event array produced by :func:`encode_query_episode`.
    config
        Exact layout configuration used during encoding.

    Returns
    -------
    DecodedQueryContext
        Losslessly recovered context.  No test target can be recovered because
        it is never present in the event tensor.
    """

    array = np.asarray(events)
    if array.shape != (config.max_events, config.input_width):
        raise ValueError(
            f"events shape {array.shape} != {(config.max_events, config.input_width)}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("events contain non-finite values")
    valid_values = array[:, config.valid_slice.start]
    if np.any((valid_values != 0.0) & (valid_values != 1.0)):
        raise ValueError("event-valid flags must be binary")
    valid = valid_values == 1.0
    if np.any(array[~valid] != 0.0):
        raise ValueError("padding rows must be exactly zero")
    demonstration_rows: dict[int, list[FloatArray]] = {}
    query_rows: list[FloatArray] = []
    query_started = False
    query_block_start = config.max_demonstrations * config.max_grid_size
    for event_index in np.flatnonzero(valid):
        row = array[event_index]
        phase = _decode_one_hot(row, config.phase_slice, field="phase")
        if phase == 0:
            if query_started:
                raise ValueError("demonstration event appears after query")
            demo = _decode_one_hot(
                row, config.demonstration_slice, field="demonstration index"
            )
            block_start = demo * config.max_grid_size
            if not block_start <= event_index < block_start + config.max_grid_size:
                raise ValueError("demonstration event is outside its fixed block")
            demonstration_rows.setdefault(demo, []).append(row)
        else:
            query_started = True
            if event_index < query_block_start:
                raise ValueError("query event is outside the fixed query block")
            if np.any(row[config.demonstration_slice] != 0.0):
                raise ValueError("query event must not contain a demonstration index")
            if row[config.side_valid_slice.start + 1] != 0.0 or np.any(
                row[config.output_color_slice] != 0.0
            ):
                raise ValueError("query event contains forbidden output features")
            query_rows.append(row)
    if not query_rows:
        raise ValueError("encoded episode contains no query rows")
    expected_demos = list(range(len(demonstration_rows)))
    if sorted(demonstration_rows) != expected_demos:
        raise ValueError("demonstration indices must be contiguous from zero")
    demonstrations = tuple(
        ArcPair(
            input=_decode_grid_side(
                demonstration_rows[index], is_output=False, config=config
            ),
            output=_decode_grid_side(
                demonstration_rows[index], is_output=True, config=config
            ),
        )
        for index in expected_demos
    )
    query = _decode_grid_side(query_rows, is_output=False, config=config)
    return DecodedQueryContext(demonstrations=demonstrations, query_input=query)


def smoke_arc_tasks() -> tuple[ArcTask, ...]:
    """Return tiny hand-authored tasks for tests and ``--smoke`` only.

    Returns
    -------
    tuple of ArcTask
        Multi-demonstration and multi-query plumbing fixtures.  Their scores are
        not ARC benchmark or scientific evidence.
    """

    transpose = ArcTask(
        task_id="fixture-transpose",
        train=(
            ArcPair(ArcGrid(((1, 2, 3), (4, 5, 6))), ArcGrid(((1, 4), (2, 5), (3, 6)))),
            ArcPair(ArcGrid(((7, 8),)), ArcGrid(((7,), (8,)))),
        ),
        test=(
            ArcPair(ArcGrid(((2, 0), (3, 4), (5, 6))), ArcGrid(((2, 3, 5), (0, 4, 6)))),
            ArcPair(ArcGrid(((9, 1, 0),)), ArcGrid(((9,), (1,), (0,)))),
        ),
    )
    recolor = ArcTask(
        task_id="fixture-recolor-nonzero",
        train=(
            ArcPair(ArcGrid(((0, 2), (2, 0))), ArcGrid(((0, 9), (9, 0)))),
            ArcPair(ArcGrid(((2, 2, 0),)), ArcGrid(((9, 9, 0),))),
        ),
        test=(
            ArcPair(ArcGrid(((0, 2, 0), (2, 2, 0))), ArcGrid(((0, 9, 0), (9, 9, 0)))),
        ),
    )
    return (transpose, recolor)


def smoke_loaded_dataset() -> LoadedDataset:
    """Return embedded fixture tasks with an explicit plumbing-only manifest.

    Returns
    -------
    LoadedDataset
        Fixture-role tasks and synthetic hashes of their committed content.
    """

    tasks = smoke_arc_tasks()
    fingerprints = tuple(canonical_task_fingerprint(task) for task in tasks)
    source = DatasetSource(
        name="Example 21 embedded ARC fixture",
        role="fixture",
        version="1",
        path="<embedded>",
        license_reference="repository-authored test fixture",
        format="task_json",
    )
    files = tuple(
        SourceFileHash(
            path=f"{task.task_id}.json",
            sha256=fingerprint,
            size_bytes=len(
                json.dumps(arc_task_to_mapping(task), separators=(",", ":")).encode(
                    "utf-8"
                )
            ),
        )
        for task, fingerprint in zip(tasks, fingerprints, strict=True)
    )
    manifest = SourceManifest(
        source=source,
        resolved_path="<embedded>",
        files=files,
        parsed_task_count=len(tasks),
        valid_task_count=len(tasks),
        rejected=(),
        duplicate_fingerprints=(),
        task_fingerprints=fingerprints,
        plumbing_only=True,
    )
    return LoadedDataset(tasks=tasks, manifest=manifest)
