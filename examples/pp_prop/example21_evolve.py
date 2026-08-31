"""Adapter-driven orchestration for iterative Example 21 ARC evolution.

This module owns lifecycle policy and durable bookkeeping.  It deliberately
does not import or drive the BrainCell model.  The numbered Example 21 entry
point supplies an adapter whose training and scoring blocks use compiled
BrainState transforms.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TextIO, TypedDict

DEFAULT_OPTIMIZER = "muon"
DEFAULT_UPDATES = 128
PROOF_UPDATES = 8
DEFAULT_ROUNDS = 8
DEFAULT_PATIENCE = 2
DEFAULT_MAX_NEURONS = 4_096
DEFAULT_MAX_RECURRENT_EDGES = 65_536
DEFAULT_MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
LOSS_ABSOLUTE_IMPROVEMENT = 1e-6
LOSS_RELATIVE_IMPROVEMENT = 1e-4
EXPECTED_ARC_TASKS = 400
DEFAULT_SCREEN_TASKS = 64
STATE_SCHEMA_VERSION = 2
OPEN_STAGES = frozenset(
    {
        "train",
        "edge",
        "neuron",
        "edge-revisit",
        "dale",
        "compression-edge",
        "compression-neuron",
        "round-screen",
        "round-score",
        "round-end",
        "terminal-evaluation",
    }
)
TERMINAL_REASONS = frozenset({"mastery", "stable", "round-budget"})
MODEL_STAGES = frozenset(
    {
        "train",
        "edge",
        "neuron",
        "edge-revisit",
        "dale",
        "compression-edge",
        "compression-neuron",
    }
)
OPERATION_STAGES = frozenset({"edge", "neuron", "edge-revisit", "dale"})
RESCORE_STAGES: Mapping[str, str] = {"round-screen": "screen", "round-score": "full"}
RESCORE_ARM = "rescore"
STAGE_ARMS: Mapping[str, tuple[str, ...]] = {
    "train": ("training",),
    "edge": ("add", "prune"),
    "neuron": ("add", "prune"),
    "edge-revisit": ("add", "prune"),
    "dale": ("excitatory", "inhibitory"),
    "compression-edge": ("prune",),
    "compression-neuron": ("prune",),
    "round-screen": (RESCORE_ARM,),
    "round-score": (RESCORE_ARM,),
}


@dataclass(frozen=True)
class ProgressEvent:
    """One immutable operator-facing lifecycle event.

    Parameters
    ----------
    event : str
        Stable event name such as ``candidate-start`` or ``selection``.
    fields : mapping
        Event-specific scalar evidence. The mapping is copied and made
        read-only during construction.
    """

    event: str
    fields: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.event, str) or not self.event:
            raise ValueError("Progress event name must be a nonempty string.")
        if not isinstance(self.fields, Mapping):
            raise TypeError("Progress event fields must be a mapping.")
        copied: dict[str, object] = {}
        for name, value in self.fields.items():
            if not isinstance(name, str) or not name or name == "event":
                raise ValueError(
                    "Progress event field names must be nonempty and cannot be 'event'."
                )
            copied[name] = value
        object.__setattr__(self, "fields", MappingProxyType(copied))

    def to_dict(self) -> dict[str, object]:
        """Return one mutable serialization for tests or other reporters.

        Returns
        -------
        dict
            Event name followed by its event-specific fields.
        """

        return {"event": self.event, **self.fields}


class ProgressReporter(Protocol):
    """Consume optional evolution progress without affecting model policy.

    Attributes
    ----------
    emit : callable
        Consume one immutable progress event.
    close : callable
        Stop any reporter-owned background activity and flush output.
    """

    def emit(self, event: ProgressEvent) -> None:
        """Consume one lifecycle event."""

    def close(self) -> None:
        """Release reporter resources and flush pending output."""


class _NullProgressReporter:
    def emit(self, event: ProgressEvent) -> None:
        del event

    def close(self) -> None:
        return None


class ConsoleProgressReporter:
    """Render GEPA-style Example 21 progress to one terminal stream.

    Parameters
    ----------
    stream : text stream, optional
        Destination for progress. ``None`` resolves stderr at construction.
    clock : callable, optional
        Monotonic clock used for elapsed display time.
    refresh_interval : float, optional
        Seconds between TTY live-line refreshes.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        refresh_interval: float = 1.0,
    ) -> None:
        if refresh_interval <= 0.0 or not math.isfinite(refresh_interval):
            raise ValueError("Progress refresh interval must be positive and finite.")
        self._stream = sys.stderr if stream is None else stream
        self._clock = clock
        self._refresh_interval = float(refresh_interval)
        self._started_at = float(clock())
        try:
            self._is_tty = bool(self._stream.isatty())
        except (AttributeError, OSError):
            self._is_tty = False
        self._write_lock = threading.Lock()
        self._live_stop: threading.Event | None = None
        self._live_thread: threading.Thread | None = None
        self._live_width = 0
        self._closed = False

    def emit(self, event: ProgressEvent) -> None:
        """Render and immediately flush one progress event.

        Parameters
        ----------
        event : ProgressEvent
            Immutable lifecycle event from ``run_evolution``.
        """

        if self._closed:
            return
        if not isinstance(event, ProgressEvent):
            raise TypeError("Console progress requires a ProgressEvent instance.")
        if event.event in {"candidate-start", "terminal-start"}:
            self._stop_live(clear=True)
            if self._is_tty:
                self._start_live(event)
            else:
                self._write_line(self._permanent_line(event))
            return
        self._stop_live(clear=True)
        self._write_line(self._permanent_line(event))

    def close(self) -> None:
        """Stop live animation and flush the configured stream."""

        if self._closed:
            return
        self._stop_live(clear=True)
        with self._write_lock:
            self._stream.flush()
        self._closed = True

    def _elapsed_text(self) -> str:
        seconds = max(0, int(float(self._clock()) - self._started_at))
        minutes, second = divmod(seconds, 60)
        hours, minute = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minute:02d}:{second:02d}"
        return f"{minutes:02d}:{second:02d}"

    def _write_line(self, line: str) -> None:
        with self._write_lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def _render_live(self, event: ProgressEvent) -> None:
        fields = event.fields
        if event.event == "candidate-start":
            activity = f"{fields['stage']}:{fields['arm']}"
        else:
            activity = "terminal evaluation"
        line = (
            f"Example 21 ARC | Round {fields['round']}/{fields['rounds']} | "
            f"{activity} | running {self._elapsed_text()}"
        )
        with self._write_lock:
            self._stream.write("\r" + line)
            self._stream.flush()
            self._live_width = max(self._live_width, len(line))

    def _start_live(self, event: ProgressEvent) -> None:
        stop = threading.Event()
        self._live_stop = stop
        self._render_live(event)

        def refresh() -> None:
            while not stop.wait(self._refresh_interval):
                self._render_live(event)

        thread = threading.Thread(
            target=refresh,
            name="example21-progress",
            daemon=True,
        )
        self._live_thread = thread
        thread.start()

    def _stop_live(self, *, clear: bool) -> None:
        stop = self._live_stop
        thread = self._live_thread
        self._live_stop = None
        self._live_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        if clear and self._live_width:
            with self._write_lock:
                self._stream.write("\r" + (" " * self._live_width) + "\r")
                self._stream.flush()
            self._live_width = 0

    @staticmethod
    def _one_line(value: object) -> str:
        return " ".join(str(value).split())

    def _permanent_line(self, event: ProgressEvent) -> str:
        fields = event.fields
        elapsed = f"[{self._elapsed_text()}]"
        if event.event == "candidate-start":
            return (
                f"{elapsed} Round {fields['round']}/{fields['rounds']} "
                f"{fields['stage']}:{fields['arm']} started"
            )
        if event.event == "candidate-result":
            prefix = (
                f"{elapsed} Round {fields['round']}/{fields['rounds']} "
                f"{fields['stage']}:{fields['arm']} {fields['status']}"
            )
            if fields["status"] == "completed":
                return (
                    f"{prefix} | score {fields['score_exact']}/{fields['score_total']}"
                    f" | loss {float(fields['loss']):.4f}"
                    f" | neurons {fields['neurons']}"
                    f" | recurrent edges {fields['recurrent_edges']}"
                    f" | updates {fields['executed_updates']}"
                )
            return (
                f"{prefix} | reason {self._one_line(fields['reason'])}"
                f" | updates {fields['executed_updates']}"
            )
        if event.event == "selection":
            return (
                f"{elapsed} Round {fields['round']}/{fields['rounds']} "
                f"{fields['stage']} selected {fields['selected_arm']}"
                f" | best {fields['best_exact']}/{fields['best_total']}"
                f" | neurons {fields['neurons']}"
                f" | recurrent edges {fields['recurrent_edges']}"
            )
        if event.event == "round-end":
            line = (
                f"{elapsed} Round {fields['round']}/{fields['rounds']} completed"
                f" | best {fields['best_exact']}/{fields['best_total']}"
                f" | neurons {fields['neurons']}"
                f" | recurrent edges {fields['recurrent_edges']}"
                f" | next {fields['next_stage']}"
            )
            if fields.get("terminal_reason") is not None:
                line += f" | reason {fields['terminal_reason']}"
            return line
        if event.event == "resume":
            return (
                f"{elapsed} Restored Round {fields['round']}/{fields['rounds']}"
                f" | next {fields['next_stage']}"
                f" | accepted {fields['score_exact']}/{fields['score_total']}"
                f" | neurons {fields['neurons']}"
                f" | recurrent edges {fields['recurrent_edges']}"
                f" | checkpoint {self._one_line(fields['checkpoint_path'])}"
                f" | sha256 {fields['checkpoint_sha256']}"
            )
        if event.event == "terminal-start":
            return (
                f"{elapsed} Round {fields['round']}/{fields['rounds']} "
                "terminal evaluation started"
            )
        if event.event == "terminal-result":
            return (
                f"{elapsed} Terminal evaluation {fields['status']}"
                f" | score {fields['score_exact']}/{fields['score_total']}"
                f" | loss {float(fields['loss']):.4f}"
                f" | checkpoint {fields['checkpoint_sha256']}"
                f" | reason {fields['terminal_reason']}"
            )
        raise ValueError(f"Unknown Example 21 progress event {event.event!r}.")


class _PendingTransitionParts(TypedDict):
    """Validated fields reconstructed from one pending transition."""

    stage_id: str
    stage: str
    parent: CandidateSnapshot
    selected: CandidateSnapshot
    attempts: tuple[CandidateAttempt, ...]
    dispositions: dict[str, str]
    selected_attempt: str | None
    elapsed_seconds: float
    successor: RunState


class PipelineError(RuntimeError):
    """Base error for a rejected evolution lifecycle.

    Attributes
    ----------
    args : tuple
        Inherited exception arguments.
    """


class ResumeMismatchError(PipelineError):
    """Raised when persisted provenance differs from the requested run.

    Attributes
    ----------
    args : tuple
        Inherited exception arguments.
    """


class ClosedRunError(PipelineError):
    """Raised when a caller tries to resume an already closed run.

    Attributes
    ----------
    args : tuple
        Inherited exception arguments.
    """


class ProgressConflictError(PipelineError):
    """Raised when durable progress has conflicting stage identities.

    Attributes
    ----------
    args : tuple
        Inherited exception arguments.
    """


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration that determines one evolution lineage.

    Parameters
    ----------
    optimizer : str, optional
        Optimizer policy.  Example 21 currently permits Muon only.
    updates : int, optional
        PP-Prop episode updates in every non-proof training block.
    rounds, patience : int, optional
        Maximum rounds and consecutive stable rounds before termination.
    max_neurons, max_recurrent_edges : int, optional
        Configured topology caps checked before promotion.
    max_checkpoint_bytes : int, optional
        Maximum serialized checkpoint size.
    operations_per_round : int or None, optional
        Structural operations executed per round.  ``None`` performs exactly
        one pass of the operation cycle, the historical lifecycle.
    screen_tasks : int, optional
        Training tasks in the intra-round screen subset.  ``0`` scores every
        operation on the complete training corpus.
    """

    optimizer: str = DEFAULT_OPTIMIZER
    updates: int = DEFAULT_UPDATES
    rounds: int = DEFAULT_ROUNDS
    patience: int = DEFAULT_PATIENCE
    max_neurons: int = DEFAULT_MAX_NEURONS
    max_recurrent_edges: int = DEFAULT_MAX_RECURRENT_EDGES
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES
    operations_per_round: int | None = None
    screen_tasks: int = DEFAULT_SCREEN_TASKS

    def __post_init__(self) -> None:
        numeric = {
            "updates": self.updates,
            "rounds": self.rounds,
            "patience": self.patience,
            "max_neurons": self.max_neurons,
            "max_recurrent_edges": self.max_recurrent_edges,
            "max_checkpoint_bytes": self.max_checkpoint_bytes,
        }
        invalid_types = [
            name for name, value in numeric.items() if type(value) is not int
        ]
        if type(self.screen_tasks) is not int:
            invalid_types.append("screen_tasks")
        if self.operations_per_round is not None and (
            type(self.operations_per_round) is not int
        ):
            invalid_types.append("operations_per_round")
        if invalid_types:
            raise TypeError(
                "Evolution numeric configuration must use JSON integers; correct "
                + ", ".join(invalid_types)
                + "."
            )
        if self.operations_per_round is not None and self.operations_per_round < 1:
            raise ValueError(
                "Evolution operations per round must be positive; pass at least one "
                "operation or omit the budget."
            )
        if not 0 <= self.screen_tasks <= EXPECTED_ARC_TASKS:
            raise ValueError(
                "Evolution screen tasks must fall between zero and "
                f"{EXPECTED_ARC_TASKS}; correct screen_tasks."
            )
        if self.optimizer != DEFAULT_OPTIMIZER:
            raise ValueError(
                "Example 21 evolution requires Muon; pass optimizer='muon'."
            )
        if self.updates != DEFAULT_UPDATES:
            raise ValueError(
                "Example 21 evolution requires exactly 128 updates per block; "
                "pass updates=128."
            )
        invalid = [name for name, value in numeric.items() if value < 1]
        if invalid:
            raise ValueError(
                "Evolution limits must be positive; correct " + ", ".join(invalid) + "."
            )
        hard_maxima = {
            "max_neurons": (self.max_neurons, DEFAULT_MAX_NEURONS),
            "max_recurrent_edges": (
                self.max_recurrent_edges,
                DEFAULT_MAX_RECURRENT_EDGES,
            ),
            "max_checkpoint_bytes": (
                self.max_checkpoint_bytes,
                DEFAULT_MAX_CHECKPOINT_BYTES,
            ),
        }
        loosened = [
            name
            for name, (value, hard_maximum) in hard_maxima.items()
            if value > hard_maximum
        ]
        if loosened:
            raise ValueError(
                "Evolution limits cannot exceed the shipped maximum; correct "
                + ", ".join(loosened)
                + "."
            )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible configuration.

        Returns
        -------
        dict
            Stable configuration fields used for resume verification.
        """

        record = {
            "optimizer": self.optimizer,
            "updates": self.updates,
            "rounds": self.rounds,
            "patience": self.patience,
            "max_neurons": self.max_neurons,
            "max_recurrent_edges": self.max_recurrent_edges,
            "max_checkpoint_bytes": self.max_checkpoint_bytes,
            "operations_per_round": self.operations_per_round,
            "screen_tasks": self.screen_tasks,
        }
        return record

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> PipelineConfig:
        """Build configuration from persisted scalar fields.

        Parameters
        ----------
        values : mapping
            JSON-decoded configuration fields.

        Returns
        -------
        PipelineConfig
            Validated configuration.
        """

        document = _json_object(values, "configuration")
        return cls(
            optimizer=_json_string(document["optimizer"], "optimizer"),
            updates=_json_integer(document["updates"], "updates"),
            rounds=_json_integer(document["rounds"], "rounds"),
            patience=_json_integer(document["patience"], "patience"),
            max_neurons=_json_integer(document["max_neurons"], "max_neurons"),
            max_recurrent_edges=_json_integer(
                document["max_recurrent_edges"], "max_recurrent_edges"
            ),
            max_checkpoint_bytes=_json_integer(
                document["max_checkpoint_bytes"], "max_checkpoint_bytes"
            ),
            operations_per_round=_json_optional_integer(
                document.get("operations_per_round"), "operations_per_round"
            ),
            screen_tasks=_json_integer(
                document.get("screen_tasks", DEFAULT_SCREEN_TASKS), "screen_tasks"
            ),
        )

    @property
    def digest(self) -> str:
        """Return the SHA-256 identity of the canonical configuration.

        Returns
        -------
        str
            Lowercase hexadecimal configuration digest.
        """

        return _json_digest(self.to_dict())

    @property
    def screens(self) -> bool:
        """Return whether intra-round operations use the screen subset.

        Returns
        -------
        bool
            True when a nonzero screen subset gates structural operations.
        """

        return self.screen_tasks > 0


def screen_task_ids(
    manifest: CorpusManifest, config: PipelineConfig
) -> tuple[str, ...]:
    """Return the deterministic screen subset for one lineage.

    Parameters
    ----------
    manifest : CorpusManifest
        Sorted, digest-bound training corpus.
    config : PipelineConfig
        Lineage configuration supplying the screen size.

    Returns
    -------
    tuple of str
        Leading manifest identifiers.  Empty when screening is disabled or the
        requested size would not be a proper subset of the corpus, so a screen
        that cannot save any work never costs a scope transition.

    Examples
    --------
    .. code-block:: python

        >>> ids = screen_task_ids(manifest, PipelineConfig(screen_tasks=2))
        >>> len(ids)
        2
    """

    if not config.screens or config.screen_tasks >= len(manifest.task_ids):
        return ()
    return tuple(manifest.task_ids[: config.screen_tasks])


@dataclass(frozen=True)
class CorpusManifest:
    """Deterministic ARC corpus membership and query order.

    Parameters
    ----------
    role : {"training", "evaluation"}
        Corpus role.  The coordinator reads evaluation only at termination.
    task_ids, source_digests : tuple of str
        Sorted task identifiers and aligned source-file SHA-256 values.
    query_order : tuple of tuple
        Ordered ``(task_id, query_index)`` entries used by the cursor.
    """

    role: str
    task_ids: tuple[str, ...]
    source_digests: tuple[str, ...]
    query_order: tuple[tuple[str, int], ...]

    def validate(self, *, expected_tasks: int = EXPECTED_ARC_TASKS) -> None:
        """Validate complete sorted membership and all-query ordering.

        Parameters
        ----------
        expected_tasks : int, optional
            Required number of task files.

        Raises
        ------
        ValueError
            If membership, digests, or query order are malformed.
        """

        if self.role not in {"training", "evaluation"}:
            raise ValueError(
                "Corpus role must be training or evaluation; pass a supported role."
            )
        if len(self.task_ids) != expected_tasks:
            raise ValueError(
                f"ARC {self.role} manifest must contain exactly {expected_tasks} tasks; "
                f"found {len(self.task_ids)}."
            )
        if self.task_ids != tuple(sorted(self.task_ids)):
            raise ValueError(
                "ARC task identifiers must be sorted; build the manifest in path order."
            )
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError(
                "ARC task identifiers must be unique; remove duplicate task files."
            )
        if len(self.source_digests) != len(self.task_ids) or any(
            not _is_sha256(value) for value in self.source_digests
        ):
            raise ValueError(
                "ARC source digests must align with tasks; pass one SHA-256 per task."
            )
        if not self.query_order:
            raise ValueError(
                "ARC query order must not be empty; include every supervised query."
            )
        position = {task_id: index for index, task_id in enumerate(self.task_ids)}
        if any(
            task_id not in position or query_index < 0
            for task_id, query_index in self.query_order
        ):
            raise ValueError(
                "ARC query order references an invalid task or query; rebuild the manifest."
            )
        expected_order = tuple(
            sorted(
                self.query_order,
                key=lambda item: (position[item[0]], item[1]),
            )
        )
        if self.query_order != expected_order or len(set(self.query_order)) != len(
            self.query_order
        ):
            raise ValueError(
                "ARC query order must be sorted and unique; include each query once."
            )
        represented = {task_id for task_id, _ in self.query_order}
        if represented != set(self.task_ids):
            raise ValueError(
                "ARC query order must represent every task; add each task's queries."
            )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible manifest.

        Returns
        -------
        dict
            Role, membership, digests, and ordered queries.
        """

        return {
            "role": self.role,
            "task_ids": list(self.task_ids),
            "source_digests": list(self.source_digests),
            "query_order": [list(item) for item in self.query_order],
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CorpusManifest:
        """Build a manifest from JSON-decoded fields.

        Parameters
        ----------
        values : mapping
            Persisted corpus fields.

        Returns
        -------
        CorpusManifest
            Reconstructed manifest.
        """

        document = _json_object(values, "corpus manifest")
        task_values = _json_list(document["task_ids"], "task_ids")
        digest_values = _json_list(document["source_digests"], "source_digests")
        query_values = _json_list(document["query_order"], "query_order")
        queries = []
        for index, value in enumerate(query_values):
            item = _json_list(value, f"query_order[{index}]")
            if len(item) != 2:
                raise ValueError(
                    "Persisted query entries must contain task ID and query index."
                )
            queries.append(
                (
                    _json_string(item[0], f"query_order[{index}][0]"),
                    _json_integer(item[1], f"query_order[{index}][1]"),
                )
            )
        return cls(
            role=_json_string(document["role"], "corpus role"),
            task_ids=tuple(
                _json_string(value, f"task_ids[{index}]")
                for index, value in enumerate(task_values)
            ),
            source_digests=tuple(
                _json_string(value, f"source_digests[{index}]")
                for index, value in enumerate(digest_values)
            ),
            query_order=tuple(queries),
        )

    @property
    def digest(self) -> str:
        """Return the manifest SHA-256 identity.

        Returns
        -------
        str
            Lowercase hexadecimal digest of the complete manifest.
        """

        return _json_digest(self.to_dict())


@dataclass(frozen=True)
class ScheduleEntry:
    """One deterministic task-query update selection.

    Parameters
    ----------
    ordinal : int
        Absolute update cursor, retained across wraparound.
    task_id : str
        Training-task identifier.
    query_index : int
        Query within the selected task.
    seed : int
        Stable random seed supplied to the model adapter.
    random_input_sha256 : str
        Identity for any adapter-derived random input.
    """

    ordinal: int
    task_id: str
    query_index: int
    seed: int
    random_input_sha256: str


@dataclass(frozen=True)
class UpdateSchedule:
    """One immutable schedule shared by sibling candidates.

    Parameters
    ----------
    cursor_start, cursor_end : int
        Absolute cursor before and after this training block.
    entries : tuple of ScheduleEntry
        Exact ordered task, query, seed, and random-input identities.
    """

    cursor_start: int
    cursor_end: int
    entries: tuple[ScheduleEntry, ...]


def build_update_schedule(
    manifest: CorpusManifest, cursor: int, updates: int = DEFAULT_UPDATES
) -> UpdateSchedule:
    """Build a reproducible wrapped all-query training schedule.

    Parameters
    ----------
    manifest : CorpusManifest
        Validated training corpus and query order.
    cursor : int
        Nonnegative absolute update cursor.
    updates : int, optional
        Positive number of selected query episodes.

    Returns
    -------
    UpdateSchedule
        Immutable schedule suitable for every sibling in one comparison.
    """

    manifest.validate()
    if manifest.role != "training":
        raise ValueError(
            "Update schedules require the training manifest; do not schedule evaluation."
        )
    if cursor < 0 or updates < 1:
        raise ValueError(
            "Schedule cursor must be nonnegative and updates positive; correct both values."
        )
    query_count = len(manifest.query_order)
    entries = []
    for ordinal in range(cursor, cursor + updates):
        task_id, query_index = manifest.query_order[ordinal % query_count]
        random_digest = hashlib.sha256(
            f"{manifest.digest}:{ordinal}:{task_id}:{query_index}".encode()
        ).hexdigest()
        entries.append(
            ScheduleEntry(
                ordinal=ordinal,
                task_id=task_id,
                query_index=query_index,
                seed=int(random_digest[:8], 16),
                random_input_sha256=random_digest,
            )
        )
    return UpdateSchedule(cursor, cursor + updates, tuple(entries))


@dataclass(frozen=True)
class ScoreSnapshot:
    """Direct task exactness and supervised loss for one state.

    Parameters
    ----------
    task_ids : tuple of str
        Ordered training-task identifiers.
    task_exact : tuple of bool
        Direct strict pass@1 result for each task.
    task_loss : tuple of float
        Height, width, and valid-cell loss aggregated per task.
    finite : bool, optional
        Adapter-level finiteness evidence.  Non-finite losses force false.
    """

    task_ids: tuple[str, ...]
    task_exact: tuple[bool, ...]
    task_loss: tuple[float, ...]
    finite: bool = True

    def __post_init__(self) -> None:
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError(
                "Score task identifiers must be nonempty and unique; pass corpus order."
            )
        if not (len(self.task_ids) == len(self.task_exact) == len(self.task_loss)):
            raise ValueError(
                "Score arrays must align by task; pass one exact flag and loss per task."
            )
        losses = tuple(float(value) for value in self.task_loss)
        if any(value < 0.0 for value in losses if math.isfinite(value)):
            raise ValueError(
                "Supervised task losses must be nonnegative; correct the scorer output."
            )
        object.__setattr__(self, "task_exact", tuple(bool(v) for v in self.task_exact))
        object.__setattr__(self, "task_loss", losses)
        object.__setattr__(
            self, "finite", bool(self.finite and all(math.isfinite(v) for v in losses))
        )

    @property
    def exact_count(self) -> int:
        """Return the number of exact training tasks.

        Returns
        -------
        int
            Count of true strict-task results.
        """

        return sum(self.task_exact)

    @property
    def all_exact(self) -> bool:
        """Return whether every training task is exact.

        Returns
        -------
        bool
            True only for complete direct training mastery.
        """

        return self.exact_count == len(self.task_exact)

    @property
    def solved_task_ids(self) -> tuple[str, ...]:
        """Return task identifiers whose direct result is exact.

        Returns
        -------
        tuple of str
            Solved identifiers in manifest order.
        """

        return tuple(
            task_id for task_id, exact in zip(self.task_ids, self.task_exact) if exact
        )

    @property
    def unresolved_loss(self) -> float:
        """Return mean per-task loss over unresolved tasks.

        Returns
        -------
        float
            Zero at mastery, otherwise the unresolved-task arithmetic mean.
        """

        values = [
            loss for exact, loss in zip(self.task_exact, self.task_loss) if not exact
        ]
        return 0.0 if not values else float(sum(values) / len(values))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible direct score evidence.

        Returns
        -------
        dict
            Ordered exact flags, losses, and finiteness.
        """

        return {
            "task_ids": list(self.task_ids),
            "task_exact": list(self.task_exact),
            "task_loss": list(self.task_loss),
            "finite": self.finite,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ScoreSnapshot:
        """Build a score snapshot from persisted fields.

        Parameters
        ----------
        values : mapping
            JSON-decoded score evidence.

        Returns
        -------
        ScoreSnapshot
            Reconstructed direct score.
        """

        document = _json_object(values, "score")
        task_ids = _json_list(document["task_ids"], "score.task_ids")
        task_exact = _json_list(document["task_exact"], "score.task_exact")
        task_loss = _json_list(document["task_loss"], "score.task_loss")
        return cls(
            task_ids=tuple(
                _json_string(value, f"score.task_ids[{index}]")
                for index, value in enumerate(task_ids)
            ),
            task_exact=tuple(
                _json_boolean(value, f"score.task_exact[{index}]")
                for index, value in enumerate(task_exact)
            ),
            task_loss=tuple(
                _json_float(value, f"score.task_loss[{index}]")
                for index, value in enumerate(task_loss)
            ),
            finite=_json_boolean(document["finite"], "score.finite"),
        )


@dataclass(frozen=True)
class ResourceUsage:
    """Persistent and observed resources for one candidate.

    Parameters
    ----------
    persistent_bytes, checkpoint_bytes : int
        Continuation state bytes and serialized checkpoint bytes.
    neurons, recurrent_edges : int
        Accepted topology size.
    peak_host_ram_bytes, device_memory_bytes : int or None, optional
        Engineering measurements that do not affect candidate ordering.
    """

    persistent_bytes: int
    checkpoint_bytes: int
    neurons: int
    recurrent_edges: int
    peak_host_ram_bytes: int | None = None
    device_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        required = (
            self.persistent_bytes,
            self.checkpoint_bytes,
            self.neurons,
            self.recurrent_edges,
        )
        optional = (self.peak_host_ram_bytes, self.device_memory_bytes)
        if any(value < 0 for value in required) or any(
            value is not None and value < 0 for value in optional
        ):
            raise ValueError(
                "Resource measurements must be nonnegative; correct candidate evidence."
            )
        if self.neurons < 1:
            raise ValueError(
                "Candidate resources require at least one neuron; reject an empty topology."
            )

    def to_dict(self) -> dict[str, int | None]:
        """Return JSON-compatible resource evidence.

        Returns
        -------
        dict
            Persistent, checkpoint, topology, host, and device measurements.
        """

        return {
            "persistent_bytes": self.persistent_bytes,
            "checkpoint_bytes": self.checkpoint_bytes,
            "neurons": self.neurons,
            "recurrent_edges": self.recurrent_edges,
            "peak_host_ram_bytes": self.peak_host_ram_bytes,
            "device_memory_bytes": self.device_memory_bytes,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ResourceUsage:
        """Build resource evidence from persisted fields.

        Parameters
        ----------
        values : mapping
            JSON-decoded resource measurements.

        Returns
        -------
        ResourceUsage
            Reconstructed resource evidence.
        """

        document = _json_object(values, "resource usage")
        return cls(
            persistent_bytes=_json_integer(
                document["persistent_bytes"], "persistent_bytes"
            ),
            checkpoint_bytes=_json_integer(
                document["checkpoint_bytes"], "checkpoint_bytes"
            ),
            neurons=_json_integer(document["neurons"], "neurons"),
            recurrent_edges=_json_integer(
                document["recurrent_edges"], "recurrent_edges"
            ),
            peak_host_ram_bytes=_json_optional_integer(
                document.get("peak_host_ram_bytes"), "peak_host_ram_bytes"
            ),
            device_memory_bytes=_json_optional_integer(
                document.get("device_memory_bytes"), "device_memory_bytes"
            ),
        )


@dataclass(frozen=True)
class CandidateSnapshot:
    """Immutable handoff identity for one fully scored continuation state.

    Parameters
    ----------
    candidate_id : str
        Stable candidate label within the lineage.
    checkpoint_path, checkpoint_sha256 : str
        Serialized continuation location and content identity.
    topology_sha256, parameters_sha256, optimizer_sha256 : str
        Explicit topology, model-parameter, and active-Muon identities.
    score : ScoreSnapshot
        Direct complete-training-corpus score.
    resources : ResourceUsage
        Selection and engineering resource evidence.
    topology_changed : bool
        Whether this candidate changed topology relative to its stage parent.
    """

    candidate_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    topology_sha256: str
    parameters_sha256: str
    optimizer_sha256: str
    score: ScoreSnapshot
    resources: ResourceUsage
    topology_changed: bool

    def __post_init__(self) -> None:
        labels = (
            self.candidate_id,
            self.checkpoint_path,
        )
        digests = (
            self.checkpoint_sha256,
            self.topology_sha256,
            self.parameters_sha256,
            self.optimizer_sha256,
        )
        if any(not value for value in labels) or any(
            not _is_sha256(value) for value in digests
        ):
            raise ValueError(
                "Candidate handoff requires labels and lowercase SHA-256 identities."
            )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible checkpoint handoff evidence.

        Returns
        -------
        dict
            Candidate identities, score, resources, and topology-change flag.
        """

        return {
            "candidate_id": self.candidate_id,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "topology_sha256": self.topology_sha256,
            "parameters_sha256": self.parameters_sha256,
            "optimizer_sha256": self.optimizer_sha256,
            "score": self.score.to_dict(),
            "resources": self.resources.to_dict(),
            "topology_changed": self.topology_changed,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CandidateSnapshot:
        """Build a candidate handoff from persisted fields.

        Parameters
        ----------
        values : mapping
            JSON-decoded candidate evidence.

        Returns
        -------
        CandidateSnapshot
            Reconstructed immutable candidate.
        """

        document = _json_object(values, "candidate")
        return cls(
            candidate_id=_json_string(document["candidate_id"], "candidate_id"),
            checkpoint_path=_json_string(
                document["checkpoint_path"], "checkpoint_path"
            ),
            checkpoint_sha256=_json_string(
                document["checkpoint_sha256"], "checkpoint_sha256"
            ),
            topology_sha256=_json_string(
                document["topology_sha256"], "topology_sha256"
            ),
            parameters_sha256=_json_string(
                document["parameters_sha256"], "parameters_sha256"
            ),
            optimizer_sha256=_json_string(
                document["optimizer_sha256"], "optimizer_sha256"
            ),
            score=ScoreSnapshot.from_dict(document["score"]),
            resources=ResourceUsage.from_dict(document["resources"]),
            topology_changed=_json_boolean(
                document["topology_changed"], "topology_changed"
            ),
        )


@dataclass(frozen=True)
class CandidateAttempt:
    """Completed, blocked, or failed mutation-arm outcome.

    Parameters
    ----------
    name : str
        Sibling arm name, such as ``add`` or ``excitatory``.
    status : {"completed", "blocked", "failed"}
        Adapter disposition before coordinator selection.
    candidate : CandidateSnapshot or None
        Scored candidate for a completed arm.
    reason : str or None
        Corrective or failure detail for non-completed arms.
    executed_updates : int, optional
        Literal update-block work: 128 for a completed training arm, zero for a
        completed ``rescore`` arm, zero for blocked, and zero through 128 for a
        partially failed arm.
    """

    name: str
    status: str
    candidate: CandidateSnapshot | None = None
    reason: str | None = None
    executed_updates: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Candidate arm name must be a nonempty string.")
        if self.status not in {"completed", "blocked", "failed"}:
            raise ValueError(
                "Candidate status must be completed, blocked, or failed; correct the adapter."
            )
        if (self.status == "completed") != (self.candidate is not None):
            raise ValueError(
                "Completed candidates require a snapshot and other statuses forbid one."
            )
        if self.status == "completed" and self.reason is not None:
            raise ValueError("Completed candidates cannot carry a failure reason.")
        if self.status != "completed" and (
            not isinstance(self.reason, str) or not self.reason
        ):
            raise ValueError("Blocked and failed candidates require a concrete reason.")
        if (
            type(self.executed_updates) is not int
            or self.executed_updates < 0
            or self.executed_updates > DEFAULT_UPDATES
        ):
            raise ValueError(
                "Candidate executed updates must be an integer from 0 to 128."
            )
        if self.status == "completed" and self.executed_updates != (
            0 if self.name == RESCORE_ARM else DEFAULT_UPDATES
        ):
            raise ValueError(
                "Completed candidates must execute exactly 128 updates, and a "
                "rescore arm must execute none."
            )
        if self.status == "blocked" and self.executed_updates != 0:
            raise ValueError("Blocked candidates must execute zero updates.")

    @classmethod
    def completed(
        cls,
        name: str,
        candidate: CandidateSnapshot,
        *,
        executed_updates: int = DEFAULT_UPDATES,
    ) -> CandidateAttempt:
        """Return a completed candidate attempt.

        Parameters
        ----------
        name : str
            Sibling arm name.
        candidate : CandidateSnapshot
            Fully trained and scored state.
        executed_updates : int, optional
            Literal completed work; must remain exactly 128.

        Returns
        -------
        CandidateAttempt
            Completed attempt ready for selection.
        """

        return cls(
            name=name,
            status="completed",
            candidate=candidate,
            executed_updates=executed_updates,
        )

    @classmethod
    def blocked(
        cls,
        name: str,
        reason: str,
        *,
        executed_updates: int = 0,
    ) -> CandidateAttempt:
        """Return a safely blocked candidate attempt.

        Parameters
        ----------
        name, reason : str
            Sibling arm and concrete blocking reason.
        executed_updates : int, optional
            Literal work; blocked attempts require zero.

        Returns
        -------
        CandidateAttempt
            Blocked attempt without a candidate state.
        """

        return cls(
            name=name,
            status="blocked",
            reason=reason,
            executed_updates=executed_updates,
        )

    @classmethod
    def failed(
        cls,
        name: str,
        reason: str,
        *,
        executed_updates: int = 0,
    ) -> CandidateAttempt:
        """Return a failed candidate attempt.

        Parameters
        ----------
        name, reason : str
            Sibling arm and concrete failure reason.
        executed_updates : int, optional
            Literal partial work from zero through 128.

        Returns
        -------
        CandidateAttempt
            Failed attempt without a promotable state.
        """

        return cls(
            name=name,
            status="failed",
            reason=reason,
            executed_updates=executed_updates,
        )


@dataclass(frozen=True)
class SelectionResult:
    """Deterministic candidate selection and per-arm disposition.

    Parameters
    ----------
    selected : CandidateSnapshot
        Winning child or retained parent.
    selected_attempt : str or None
        Arm name for an accepted child; ``None`` means parent retained.
    dispositions : mapping
        Final disposition keyed by sibling arm.
    """

    selected: CandidateSnapshot
    selected_attempt: str | None
    dispositions: Mapping[str, str]

    @property
    def parent_retained(self) -> bool:
        """Return whether no sibling was accepted.

        Returns
        -------
        bool
            True when the stage keeps its immutable parent.
        """

        return self.selected_attempt is None


def select_candidate(
    parent: CandidateSnapshot,
    attempts: Sequence[CandidateAttempt],
    *,
    config: PipelineConfig | None = None,
    compression: bool = False,
) -> SelectionResult:
    """Select the best protected candidate under the approved ordering.

    Parameters
    ----------
    parent : CandidateSnapshot
        Immutable stage parent.
    attempts : sequence of CandidateAttempt
        Trained sibling outcomes from the same update schedule.
    config : PipelineConfig, optional
        Resource caps; defaults to the shipped configuration.
    compression : bool, optional
        Require mastery-preserving persistent-byte reduction.

    Returns
    -------
    SelectionResult
        Winning state and deterministic final dispositions.
    """

    config = config or PipelineConfig()
    _reject_candidate_aliases(attempts)
    dispositions: dict[str, str] = {}
    eligible: list[tuple[tuple[object, ...], int, CandidateAttempt]] = []
    for order, attempt in enumerate(attempts):
        if attempt.name in dispositions:
            raise ValueError(
                "Candidate arm names must be unique; provide each sibling once."
            )
        if attempt.status != "completed":
            dispositions[attempt.name] = attempt.status
            continue
        assert attempt.candidate is not None
        allowed, reason = _candidate_improves(
            parent, attempt.candidate, config, compression=compression
        )
        dispositions[attempt.name] = "eligible" if allowed else reason
        if allowed:
            candidate = attempt.candidate
            key = (
                -candidate.score.exact_count,
                candidate.score.unresolved_loss,
                candidate.resources.persistent_bytes,
                candidate.resources.neurons,
                candidate.resources.recurrent_edges,
                order,
            )
            eligible.append((key, order, attempt))
    if not eligible:
        return SelectionResult(parent, None, dispositions)
    _, _, winner = min(eligible, key=lambda item: item[0])
    assert winner.candidate is not None
    for attempt in attempts:
        if attempt.status == "completed" and dispositions[attempt.name] == "eligible":
            dispositions[attempt.name] = (
                "accepted" if attempt.name == winner.name else "rejected"
            )
    return SelectionResult(winner.candidate, winner.name, dispositions)


def _reject_candidate_aliases(attempts: Sequence[CandidateAttempt]) -> None:
    completed = [
        attempt.candidate
        for attempt in attempts
        if attempt.status == "completed" and attempt.candidate is not None
    ]
    identities = (
        ("candidate ID", [candidate.candidate_id for candidate in completed]),
        (
            "checkpoint path",
            [str(Path(candidate.checkpoint_path).resolve()) for candidate in completed],
        ),
        (
            "checkpoint digest",
            [candidate.checkpoint_sha256 for candidate in completed],
        ),
    )
    for label, values in identities:
        if len(values) != len(set(values)):
            raise ValueError(f"Completed candidate attempts alias the same {label}.")


@dataclass(frozen=True)
class StageContext:
    """Bounded adapter context for one compiled training stage.

    Parameters
    ----------
    round_index : int
        Zero-based round.
    stage, stage_id : str
        Logical phase and stable durable identity.
    output_dir : pathlib.Path
        Run artifact directory.
    config : PipelineConfig
        Immutable lineage configuration.
    operation_index : int, optional
        Zero-based structural operation within the round.
    score_task_ids : tuple of str, optional
        Screen subset for this stage.  Empty means the complete corpus.
    """

    round_index: int
    stage: str
    stage_id: str
    output_dir: Path
    config: PipelineConfig
    operation_index: int = 0
    score_task_ids: tuple[str, ...] = ()

    @property
    def screens(self) -> bool:
        """Return whether this stage scores the screen subset.

        Returns
        -------
        bool
            True when a nonempty screen subset gates this stage.
        """

        return bool(self.score_task_ids)


class EvolutionAdapter(Protocol):
    """Model-facing operations required by the pure coordinator.

    Attributes
    ----------
    training_manifest : callable
        Training-only corpus provider.  Implementations also supply the model
        operations declared below.
    """

    def training_manifest(self) -> CorpusManifest:
        """Return the complete target-bearing training manifest.

        Returns
        -------
        CorpusManifest
            Sorted training membership and all-query order.
        """

    def evaluation_manifest(self) -> CorpusManifest:
        """Return the held-out manifest only during terminal evaluation.

        Returns
        -------
        CorpusManifest
            Sorted held-out evaluation membership.
        """

    def initialize(self, config: PipelineConfig, output_dir: Path) -> CandidateSnapshot:
        """Build and score the initial continuation state.

        Parameters
        ----------
        config : PipelineConfig
            Validated lineage configuration.
        output_dir : pathlib.Path
            Artifact directory for temporary state.

        Returns
        -------
        CandidateSnapshot
            Initial scored continuation state.
        """

    def restore(self, candidate: CandidateSnapshot) -> CandidateSnapshot:
        """Restore and verify a persisted continuation state.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Expected durable handoff identity.

        Returns
        -------
        CandidateSnapshot
            Restored state with unchanged identities.
        """

    def rescore(
        self, parent: CandidateSnapshot, context: StageContext
    ) -> CandidateAttempt:
        """Re-score one accepted state under a new score scope.

        The parameters, optimizer state, and graph size must not change, and
        the result is never a topology change.  Only the recorded score and
        the task ownership it derives may differ, so the topology digest can
        move: ownership is serialized alongside the graph.

        Parameters
        ----------
        parent : CandidateSnapshot
            Accepted state whose score scope is being changed.
        context : StageContext
            Stage context whose ``score_task_ids`` names the target scope;
            empty means the complete training corpus.

        Returns
        -------
        CandidateAttempt
            Completed ``rescore`` arm carrying the newly scored state.
        """

    def train_parent(
        self,
        parent: CandidateSnapshot,
        schedule: UpdateSchedule,
        context: StageContext,
    ) -> CandidateSnapshot:
        """Run one compiled ordinary PP-Prop block.

        Parameters
        ----------
        parent : CandidateSnapshot
            Immutable durable parent.
        schedule : UpdateSchedule
            Exact 128-query schedule.
        context : StageContext
            Round, stage, artifact, and limit context.

        Returns
        -------
        CandidateSnapshot
            Trained and fully rescored continuation candidate.
        """

    def run_candidate(
        self,
        parent: CandidateSnapshot,
        arm: str,
        schedule: UpdateSchedule,
        context: StageContext,
    ) -> CandidateAttempt:
        """Build, train, and score one sibling from the immutable parent.

        Parameters
        ----------
        parent : CandidateSnapshot
            Immutable durable parent.
        arm : str
            Requested mutation sibling.
        schedule : UpdateSchedule
            Schedule shared exactly with its sibling.
        context : StageContext
            Round, stage, artifact, and limit context.

        Returns
        -------
        CandidateAttempt
            Completed, blocked, or failed sibling evidence.
        """

    def persist(
        self,
        candidate: CandidateSnapshot,
        destination: Path,
        *,
        parent_checkpoint_sha256: str | None,
        stage_id: str,
    ) -> CandidateSnapshot:
        """Atomically write a selected continuation checkpoint.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Selected in-memory or temporary continuation state.
        destination : pathlib.Path
            Versioned accepted checkpoint path.
        parent_checkpoint_sha256 : str or None
            Immediate checkpoint ancestor.
        stage_id : str
            Stable durable stage identity.

        Returns
        -------
        CandidateSnapshot
            Persisted candidate with the exact destination and digest.
        """

    def attest_pending(
        self,
        candidate: CandidateSnapshot,
        *,
        parent_checkpoint_sha256: str,
        stage_id: str,
    ) -> CandidateSnapshot:
        """Re-register durable ancestry for a recovered temporary child.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Journal-validated child whose temporary checkpoint still exists.
        parent_checkpoint_sha256 : str
            Durable stage-parent digest proven by the coordinator journal.
        stage_id : str
            Stable pending stage identity.

        Returns
        -------
        CandidateSnapshot
            Byte- and component-verified unchanged temporary child.
        """

    def discard(self, attempt: CandidateAttempt) -> None:
        """Remove one candidate's temporary staging state after commit.

        Parameters
        ----------
        attempt : CandidateAttempt
            Durably recorded sibling outcome.  A selected attempt's accepted
            checkpoint is already stored at a separate versioned path.
        """

    def render_topology(self, candidate: CandidateSnapshot, output_path: Path) -> None:
        """Render the selected graph to the requested temporary PNG.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Latest accepted checkpoint identity.
        output_path : pathlib.Path
            Temporary PNG path replaced by the coordinator.
        """

    def evaluate_terminal(
        self, candidate: CandidateSnapshot, manifest: CorpusManifest
    ) -> Mapping[str, Any]:
        """Score the unchanged terminal checkpoint on held-out evaluation.

        Parameters
        ----------
        candidate : CandidateSnapshot
            Unchanged terminal checkpoint.
        manifest : CorpusManifest
            Complete held-out evaluation membership.

        Returns
        -------
        mapping
            Direct exact held-out evaluation result.
        """


@dataclass(frozen=True)
class RunState:
    """Durable cursor and accepted lineage state.

    Parameters
    ----------
    config, training_manifest : object
        Immutable lineage provenance.
    sequence, cursor, round_index : int
        Durable transition, update-query, and round positions.
    operation_index : int
        Structural operations completed in the current round.
    next_stage : str
        Phase that has not yet executed.
    accepted, round_entry : CandidateSnapshot
        Current continuation state and round-entry comparison state.
    stable_rounds : int
        Consecutive rounds without protected progress.
    closed : bool
        Whether terminal evaluation is durably complete.
    terminal_reason : str or None
        Mastery, stability, or round-budget reason.
    evaluation_completed, evaluation_digest : object
        Exactly-once terminal evaluation evidence.
    """

    config: PipelineConfig
    training_manifest: CorpusManifest
    sequence: int
    cursor: int
    round_index: int
    next_stage: str
    accepted: CandidateSnapshot
    round_entry: CandidateSnapshot
    operation_index: int = 0
    stable_rounds: int = 0
    closed: bool = False
    terminal_reason: str | None = None
    evaluation_completed: bool = False
    evaluation_digest: str | None = None

    def __post_init__(self) -> None:
        self.training_manifest.validate()
        if self.training_manifest.role != "training":
            raise ValueError(
                "Run state requires a training manifest; evaluation is terminal-only."
            )
        budget = self.config.operations_per_round
        if (
            self.sequence < 0
            or self.cursor < 0
            or self.cursor % self.config.updates
            or self.round_index < 0
            or self.round_index >= self.config.rounds
            or self.operation_index < 0
            or (budget is not None and self.operation_index > budget)
            or self.stable_rounds < 0
            or self.stable_rounds > self.config.patience
        ):
            raise ValueError(
                "Run lifecycle counters are inconsistent; recover run-state.json."
            )
        task_ids = self.training_manifest.task_ids
        screen_ids = screen_task_ids(self.training_manifest, self.config)
        if (
            self.accepted.score.task_ids not in (task_ids, screen_ids)
            or self.round_entry.score.task_ids != task_ids
            or not self.accepted.score.finite
            or not self.round_entry.score.finite
            or _limit_violation(self.accepted, self.config) is not None
            or _limit_violation(self.round_entry, self.config) is not None
        ):
            raise ValueError(
                "Run candidate evidence is inconsistent with its training lineage."
            )
        if self.closed:
            lifecycle_valid = (
                self.next_stage == "closed"
                and self.evaluation_completed
                and self.terminal_reason in TERMINAL_REASONS
                and self.evaluation_digest is not None
                and _is_sha256(self.evaluation_digest)
            )
        else:
            lifecycle_valid = (
                self.next_stage in OPEN_STAGES
                and not self.evaluation_completed
                and self.evaluation_digest is None
                and (
                    (
                        self.next_stage == "terminal-evaluation"
                        and self.terminal_reason in TERMINAL_REASONS
                    )
                    or (
                        self.next_stage != "terminal-evaluation"
                        and self.terminal_reason is None
                    )
                )
            )
        if not lifecycle_valid:
            raise ValueError(
                "Run lifecycle fields are inconsistent; recover run-state.json."
            )
        if self.sequence == 0 and (
            self.cursor != 0
            or self.round_index != 0
            or self.operation_index != 0
            or self.next_stage != "train"
            or self.accepted != self.round_entry
            or self.stable_rounds != 0
            or self.terminal_reason is not None
        ):
            raise ValueError(
                "Initial run state is inconsistent; recover run-state.json."
            )

    @classmethod
    def initial(
        cls,
        config: PipelineConfig,
        manifest: CorpusManifest,
        accepted: CandidateSnapshot,
    ) -> RunState:
        """Return the initial open lifecycle position.

        Parameters
        ----------
        config, manifest, accepted : object
            Validated lineage configuration, training corpus, and checkpoint.

        Returns
        -------
        RunState
            Open state positioned before round-zero training.
        """

        return cls(
            config=config,
            training_manifest=manifest,
            sequence=0,
            cursor=0,
            round_index=0,
            next_stage="train",
            accepted=accepted,
            round_entry=accepted,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible run state.

        Returns
        -------
        dict
            Provenance, position, lineage, and closure fields.
        """

        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "config_sha256": self.config.digest,
            "training_manifest": self.training_manifest.to_dict(),
            "training_manifest_sha256": self.training_manifest.digest,
            **self.position_dict(),
        }

    def position_dict(self) -> dict[str, object]:
        """Return fields permitted to advance in a progress transition.

        Returns
        -------
        dict
            Sequence, cursor, stage, accepted state, and terminal fields.
        """

        return {
            "sequence": self.sequence,
            "cursor": self.cursor,
            "round_index": self.round_index,
            "operation_index": self.operation_index,
            "next_stage": self.next_stage,
            "accepted": self.accepted.to_dict(),
            "round_entry": self.round_entry.to_dict(),
            "stable_rounds": self.stable_rounds,
            "closed": self.closed,
            "terminal_reason": self.terminal_reason,
            "evaluation_completed": self.evaluation_completed,
            "evaluation_digest": self.evaluation_digest,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> RunState:
        """Build and validate run state from JSON fields.

        Parameters
        ----------
        values : mapping
            Persisted run-state document.

        Returns
        -------
        RunState
            Reconstructed lifecycle position.
        """

        document = _json_object(values, "run state")
        if (
            _json_integer(document.get("schema_version"), "schema_version")
            != STATE_SCHEMA_VERSION
        ):
            raise ResumeMismatchError(
                "Run-state schema is incompatible; use a matching implementation."
            )
        state = cls(
            config=PipelineConfig.from_dict(document["config"]),
            training_manifest=CorpusManifest.from_dict(document["training_manifest"]),
            sequence=_json_integer(document["sequence"], "sequence"),
            cursor=_json_integer(document["cursor"], "cursor"),
            round_index=_json_integer(document["round_index"], "round_index"),
            operation_index=_json_integer(
                document["operation_index"], "operation_index"
            ),
            next_stage=_json_string(document["next_stage"], "next_stage"),
            accepted=CandidateSnapshot.from_dict(document["accepted"]),
            round_entry=CandidateSnapshot.from_dict(document["round_entry"]),
            stable_rounds=_json_integer(document["stable_rounds"], "stable_rounds"),
            closed=_json_boolean(document["closed"], "closed"),
            terminal_reason=_json_optional_string(
                document.get("terminal_reason"), "terminal_reason"
            ),
            evaluation_completed=_json_boolean(
                document["evaluation_completed"], "evaluation_completed"
            ),
            evaluation_digest=_json_optional_string(
                document.get("evaluation_digest"), "evaluation_digest"
            ),
        )
        if (
            _json_string(document.get("config_sha256"), "config_sha256")
            != state.config.digest
        ):
            raise ResumeMismatchError(
                "Persisted configuration digest is invalid; repair run-state.json."
            )
        if (
            _json_string(
                document.get("training_manifest_sha256"),
                "training_manifest_sha256",
            )
            != state.training_manifest.digest
        ):
            raise ResumeMismatchError(
                "Persisted training manifest digest is invalid; repair run-state.json."
            )
        return state

    def advance_from(self, values: Mapping[str, Any]) -> RunState:
        """Apply one durable progress position to unchanged provenance.

        Parameters
        ----------
        values : mapping
            JSON-compatible ``state_after`` position.

        Returns
        -------
        RunState
            Reconciled successor state.
        """

        document = _json_object(values, "state transition")
        return replace(
            self,
            sequence=_json_integer(document["sequence"], "sequence"),
            cursor=_json_integer(document["cursor"], "cursor"),
            round_index=_json_integer(document["round_index"], "round_index"),
            operation_index=_json_integer(
                document["operation_index"], "operation_index"
            ),
            next_stage=_json_string(document["next_stage"], "next_stage"),
            accepted=CandidateSnapshot.from_dict(document["accepted"]),
            round_entry=CandidateSnapshot.from_dict(document["round_entry"]),
            stable_rounds=_json_integer(document["stable_rounds"], "stable_rounds"),
            closed=_json_boolean(document["closed"], "closed"),
            terminal_reason=_json_optional_string(
                document.get("terminal_reason"), "terminal_reason"
            ),
            evaluation_completed=_json_boolean(
                document["evaluation_completed"], "evaluation_completed"
            ),
            evaluation_digest=_json_optional_string(
                document.get("evaluation_digest"), "evaluation_digest"
            ),
        )


class PipelineStore:
    """Atomic run-state and append-only progress storage.

    Parameters
    ----------
    output_dir : path-like
        Directory containing state, checkpoints, progress, and plots.
    """

    def __init__(self, output_dir: str | os.PathLike[str]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "run-state.json"
        self.progress_path = self.output_dir / "progress.jsonl"
        self.evaluation_path = self.output_dir / "evaluation.json"
        self.evaluation_intent_path = self.output_dir / "evaluation-intent.json"
        self.pending_path = self.output_dir / "pending-transition.json"
        self.checkpoint_dir = self.output_dir / "checkpoints"

    def write_state(self, state: RunState) -> None:
        """Atomically replace the authoritative run state.

        Parameters
        ----------
        state : RunState
            Validated lifecycle position.
        """

        _atomic_json(self.state_path, state.to_dict())

    def write_pending(self, document: Mapping[str, Any]) -> None:
        """Atomically journal a selected transition before persistence.

        Parameters
        ----------
        document : mapping
            Complete pending transition with immutable provenance.
        """

        if self.pending_path.exists():
            existing = self.read_pending()
            if existing is not None and _json_bytes(existing) == _json_bytes(document):
                return
            raise ProgressConflictError(
                "A different pending transition already exists; recover it first."
            )
        _atomic_json(self.pending_path, document)

    def read_pending(self) -> dict[str, Any] | None:
        """Read the pending transition journal when present.

        Returns
        -------
        dict or None
            JSON-decoded transition, or ``None`` when no journal exists.
        """

        if not self.pending_path.exists():
            return None
        try:
            value = json.loads(self.pending_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProgressConflictError(
                "Pending transition is unreadable; recover it before resume."
            ) from error
        if not isinstance(value, dict):
            raise ProgressConflictError(
                "Pending transition must be a JSON object; recover the journal."
            )
        return value

    def clear_pending(self) -> None:
        """Durably remove the completed pending transition journal."""

        _unlink_and_sync(self.pending_path)

    def read_progress(self) -> list[dict[str, Any]]:
        """Read and validate every complete progress record.

        Returns
        -------
        list of dict
            Append order records, or an empty list when absent.
        """

        if not self.progress_path.exists():
            return []
        records = []
        try:
            with self.progress_path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.endswith("\n"):
                        raise ProgressConflictError(
                            f"Progress line {line_number} is incomplete; recover the durable record."
                        )
                    records.append(
                        _json_object(json.loads(line), f"progress line {line_number}")
                    )
        except json.JSONDecodeError as error:
            raise ProgressConflictError(
                "Progress JSON is malformed; recover the append-only record."
            ) from error
        return records

    def append_progress(self, record: Mapping[str, Any]) -> None:
        """Durably append one idempotent stable-stage record.

        Parameters
        ----------
        record : mapping
            Complete progress document containing ``stage_id`` and sequence.
        """

        document = _json_object(record, "progress record")
        if (
            "stage_id" not in document
            or not {"state_before", "state_after"} <= document.keys()
        ):
            raise ValueError(
                "Progress requires stage_id, state_before, and state_after."
            )
        stage_id = _json_string(document["stage_id"], "progress.stage_id")
        if not stage_id:
            raise ValueError("Progress stage_id must not be empty.")
        encoded = _json_bytes(document) + b"\n"
        for existing in self.read_progress():
            if existing.get("stage_id") == stage_id:
                if _json_bytes(existing) == _json_bytes(document):
                    return
                raise ProgressConflictError(
                    f"Progress stage {stage_id} conflicts with durable evidence."
                )
            if existing.get("sequence_after") == document.get("sequence_after"):
                raise ProgressConflictError(
                    "Progress sequence is already used; preserve the existing transition."
                )
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_path.open("ab", buffering=0) as stream:
            stream.write(encoded)
            os.fsync(stream.fileno())

    def commit(self, record: Mapping[str, Any], state: RunState) -> None:
        """Append progress before atomically advancing run state.

        Parameters
        ----------
        record : mapping
            Durable transition evidence.
        state : RunState
            Exact ``state_after`` successor.
        """

        self.append_progress(record)
        self.write_state(state)

    def load_state(self, config: PipelineConfig, manifest: CorpusManifest) -> RunState:
        """Load, verify, and reconcile an interrupted run.

        Parameters
        ----------
        config, manifest : object
            Requested configuration and freshly rebuilt training manifest.

        Returns
        -------
        RunState
            Latest state after replaying durable progress transitions.
        """

        try:
            values = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResumeMismatchError(
                "Run state cannot be read; recover or choose a new output directory."
            ) from error
        try:
            state = RunState.from_dict(values)
        except (KeyError, TypeError, ValueError) as error:
            raise ResumeMismatchError(
                f"Persisted run state is inconsistent: {error}"
            ) from error
        if state.config.digest != config.digest:
            raise ResumeMismatchError(
                "Resume configuration differs from run-state.json; use the original configuration."
            )
        manifest.validate()
        if state.training_manifest.digest != manifest.digest:
            raise ResumeMismatchError(
                "Resume training manifest differs from run-state.json; restore the original corpus."
            )
        records = self.read_progress()
        transitions: list[tuple[RunState, RunState]] = []
        previous_after: RunState | None = None
        for record in records:
            try:
                before_state = state.advance_from(record["state_before"])
                after_state = state.advance_from(record["state_after"])
                _validate_progress_transition(
                    before_state,
                    after_state,
                    record,
                    store=self,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ProgressConflictError(
                    "Progress lifecycle evidence is inconsistent."
                ) from error
            if previous_after is None:
                if before_state.sequence != 0:
                    raise ProgressConflictError(
                        "Progress sequence does not begin at the initial state."
                    )
            elif before_state != previous_after:
                raise ProgressConflictError(
                    "Progress sequence has a gap; recover the missing transition."
                )
            transitions.append((before_state, after_state))
            previous_after = after_state
        if state.sequence > 0:
            current_records = [
                after_state
                for _, after_state in transitions
                if after_state.sequence == state.sequence
            ]
            if len(current_records) != 1 or current_records[0] != state:
                raise ProgressConflictError(
                    "Durable progress conflicts with the current run state."
                )
        reconciled = state
        for before_state, after_state in transitions:
            if after_state.sequence <= reconciled.sequence:
                continue
            if before_state != reconciled:
                raise ProgressConflictError(
                    "Progress sequence has a gap; recover the missing durable transition."
                )
            reconciled = after_state
        if reconciled != state:
            self.write_state(reconciled)
        if reconciled.closed and (
            not self.evaluation_path.is_file()
            or _file_sha256(self.evaluation_path) != reconciled.evaluation_digest
        ):
            raise ProgressConflictError(
                "Closed evaluation digest differs from evaluation.json."
            )
        return reconciled

    def progress_record(
        self,
        before: RunState,
        after: RunState,
        *,
        stage_id: str,
        stage: str,
        parent: CandidateSnapshot,
        selected: CandidateSnapshot,
        attempts: Sequence[CandidateAttempt],
        elapsed_seconds: float,
        dispositions: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Build one complete durable progress document.

        Parameters
        ----------
        before, after : RunState
            Adjacent lifecycle positions.
        stage_id, stage : str
            Stable identity and logical phase.
        parent, selected : CandidateSnapshot
            Immutable stage parent and carried state.
        attempts : sequence of CandidateAttempt
            Sibling outcomes before final disposition.
        elapsed_seconds : float
            Complete coordinator stage wall time.
        dispositions : mapping, optional
            Final selection result by arm.

        Returns
        -------
        dict
            JSON-compatible progress record with reconciliation state.
        """

        if after.sequence != before.sequence + 1:
            raise ValueError(
                "Progress states must be adjacent; advance sequence by exactly one."
            )
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
            raise ValueError(
                "Progress elapsed time must be finite and nonnegative; correct the timer."
            )
        final_dispositions = dict(dispositions or {})
        attempt_records = []
        total_executed_updates = 0
        for attempt in attempts:
            candidate = attempt.candidate
            disposition = final_dispositions.get(attempt.name, attempt.status)
            executed_updates = attempt.executed_updates
            total_executed_updates += executed_updates
            attempt_records.append(
                {
                    "name": attempt.name,
                    "status": attempt.status,
                    "disposition": disposition,
                    "reason": attempt.reason,
                    "executed_updates": executed_updates,
                    "candidate": None if candidate is None else candidate.to_dict(),
                    "candidate_id": (
                        None if candidate is None else candidate.candidate_id
                    ),
                    "checkpoint_sha256": (
                        None if candidate is None else candidate.checkpoint_sha256
                    ),
                    "topology_sha256": (
                        None if candidate is None else candidate.topology_sha256
                    ),
                    "parameters_sha256": (
                        None if candidate is None else candidate.parameters_sha256
                    ),
                    "optimizer_sha256": (
                        None if candidate is None else candidate.optimizer_sha256
                    ),
                    "persistent_bytes": (
                        None
                        if candidate is None
                        else candidate.resources.persistent_bytes
                    ),
                    "checkpoint_bytes": (
                        None
                        if candidate is None
                        else candidate.resources.checkpoint_bytes
                    ),
                    "neurons": (
                        None if candidate is None else candidate.resources.neurons
                    ),
                    "recurrent_edges": (
                        None
                        if candidate is None
                        else candidate.resources.recurrent_edges
                    ),
                    "peak_host_ram_bytes": (
                        None
                        if candidate is None
                        else candidate.resources.peak_host_ram_bytes
                    ),
                    "device_memory_bytes": (
                        None
                        if candidate is None
                        else candidate.resources.device_memory_bytes
                    ),
                    "topology_changed": (
                        None if candidate is None else candidate.topology_changed
                    ),
                }
            )
        cursor_advance = after.cursor - before.cursor
        record = {
            "schema_version": STATE_SCHEMA_VERSION,
            "stage_id": stage_id,
            "stage": stage,
            "round": before.round_index,
            "operation_index": before.operation_index,
            "score_scope": (
                "full"
                if selected.score.task_ids == before.training_manifest.task_ids
                else "screen"
            ),
            "sequence_before": before.sequence,
            "sequence_after": after.sequence,
            "parent_checkpoint_sha256": parent.checkpoint_sha256,
            "child_checkpoint_sha256": selected.checkpoint_sha256,
            "selected_candidate_id": selected.candidate_id,
            "disposition": (
                "terminal"
                if after.closed
                else "retained-parent"
                if selected.checkpoint_sha256 == parent.checkpoint_sha256
                else "accepted"
            ),
            "siblings": attempt_records,
            "exact_task_count": selected.score.exact_count,
            "solved_task_ids": list(selected.score.solved_task_ids),
            "unresolved_task_loss": selected.score.unresolved_loss,
            "updates": total_executed_updates,
            "total_executed_updates": total_executed_updates,
            "cursor_advance": cursor_advance,
            "neurons": selected.resources.neurons,
            "recurrent_edges": selected.resources.recurrent_edges,
            "persistent_bytes": selected.resources.persistent_bytes,
            "checkpoint_bytes": selected.resources.checkpoint_bytes,
            "elapsed_seconds": float(elapsed_seconds),
            "peak_host_ram_bytes": selected.resources.peak_host_ram_bytes,
            "device_memory_bytes": selected.resources.device_memory_bytes,
            "state_before": before.position_dict(),
            "state_after": after.position_dict(),
        }
        _validate_progress_transition(before, after, record)
        return record


def plot_score_history(
    records: Sequence[Mapping[str, Any]], output_path: str | os.PathLike[str]
) -> dict[str, object]:
    """Atomically render accepted score and resource history.

    Parameters
    ----------
    records : sequence of mapping
        Durable progress records in append order.
    output_path : path-like
        Destination PNG.

    Returns
    -------
    dict
        Output path and number of represented accepted stages.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    accepted: list[Mapping[str, Any]] = []
    seen_checkpoint_digests: set[str] = set()
    if records:
        first_state = _json_object(
            records[0].get("state_before"), "history initial state"
        )
        baseline = CandidateSnapshot.from_dict(first_state["accepted"])
        accepted.append(
            {
                "stage_id": "initial",
                "exact_task_count": baseline.score.exact_count,
                "unresolved_task_loss": baseline.score.unresolved_loss,
                "neurons": baseline.resources.neurons,
                "recurrent_edges": baseline.resources.recurrent_edges,
                "persistent_bytes": baseline.resources.persistent_bytes,
            }
        )
        seen_checkpoint_digests.add(baseline.checkpoint_sha256)
    for record in records:
        if record.get("disposition") != "accepted":
            continue
        digest = _json_string(
            record.get("child_checkpoint_sha256"),
            "history child_checkpoint_sha256",
        )
        if digest in seen_checkpoint_digests:
            continue
        accepted.append(record)
        seen_checkpoint_digests.add(digest)
    labels = [
        str(record.get("stage_id", index)) for index, record in enumerate(accepted)
    ]
    x = list(range(len(accepted)))
    exact_values = [int(record["exact_task_count"]) for record in accepted]
    loss_values = [float(record["unresolved_task_loss"]) for record in accepted]
    neuron_values = [int(record["neurons"]) for record in accepted]
    edge_values = [int(record["recurrent_edges"]) for record in accepted]
    persistent_values = [int(record["persistent_bytes"]) for record in accepted]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    exact_axis, loss_axis, topology_axis, bytes_axis = axes.flat
    exact_axis.plot(x, exact_values, marker="o")
    exact_axis.set_title("Exact training tasks")
    exact_axis.set_ylim(bottom=0, top=EXPECTED_ARC_TASKS)
    loss_axis.plot(x, loss_values, marker="o", color="#e45756")
    loss_axis.set_title("Unresolved-task loss")
    topology_axis.plot(x, neuron_values, label="neurons")
    topology_axis.plot(x, edge_values, label="recurrent edges")
    topology_axis.set_title("Neurons / recurrent edges")
    topology_axis.legend(frameon=False)
    bytes_axis.plot(x, persistent_values, marker="o", color="#72b7b2")
    bytes_axis.set_title("Persistent bytes")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        if labels and len(labels) <= 12:
            axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(output, suffix=".png")
    try:
        fig.savefig(temporary, dpi=150, format="png")
        plt.close(fig)
        _replace_and_sync(temporary, output)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
    return {"accepted_stage_count": len(accepted), "output": str(output)}


def _round_progress_fields(state: RunState) -> dict[str, object]:
    return {
        "round": state.round_index + 1,
        "rounds": state.config.rounds,
    }


def _accepted_progress_fields(candidate: CandidateSnapshot) -> dict[str, object]:
    return {
        "score_exact": candidate.score.exact_count,
        "score_total": len(candidate.score.task_ids),
        "loss": candidate.score.unresolved_loss,
        "neurons": candidate.resources.neurons,
        "recurrent_edges": candidate.resources.recurrent_edges,
    }


def _emit_resume(reporter: ProgressReporter, state: RunState) -> None:
    reporter.emit(
        ProgressEvent(
            "resume",
            {
                **_round_progress_fields(state),
                "next_stage": state.next_stage,
                **_accepted_progress_fields(state.accepted),
                "checkpoint_path": state.accepted.checkpoint_path,
                "checkpoint_sha256": state.accepted.checkpoint_sha256,
            },
        )
    )


def _emit_candidate_start(
    reporter: ProgressReporter,
    state: RunState,
    context: StageContext,
    arm: str,
) -> None:
    reporter.emit(
        ProgressEvent(
            "candidate-start",
            {
                **_round_progress_fields(state),
                "stage": context.stage,
                "stage_id": context.stage_id,
                "arm": arm,
                "updates": state.config.updates,
            },
        )
    )


def _emit_candidate_result(
    reporter: ProgressReporter,
    state: RunState,
    context: StageContext,
    attempt: CandidateAttempt,
) -> None:
    fields: dict[str, object] = {
        **_round_progress_fields(state),
        "stage": context.stage,
        "stage_id": context.stage_id,
        "arm": attempt.name,
        "status": attempt.status,
    }
    if attempt.candidate is None:
        fields.update(
            {
                "reason": attempt.reason,
                "executed_updates": attempt.executed_updates,
            }
        )
    else:
        fields.update(_accepted_progress_fields(attempt.candidate))
        fields["executed_updates"] = attempt.executed_updates
    reporter.emit(ProgressEvent("candidate-result", fields))


def _emit_selection(
    reporter: ProgressReporter,
    before: RunState,
    context: StageContext,
    selection: SelectionResult,
    successor: RunState,
) -> None:
    selected = successor.accepted
    reporter.emit(
        ProgressEvent(
            "selection",
            {
                **_round_progress_fields(before),
                "stage": context.stage,
                "stage_id": context.stage_id,
                "selected_arm": selection.selected_attempt or "parent",
                "parent_retained": selection.parent_retained,
                "best_exact": selected.score.exact_count,
                "best_total": len(selected.score.task_ids),
                "loss": selected.score.unresolved_loss,
                "neurons": selected.resources.neurons,
                "recurrent_edges": selected.resources.recurrent_edges,
                "checkpoint_sha256": selected.checkpoint_sha256,
                "next_stage": successor.next_stage,
            },
        )
    )


def _emit_round_end(
    reporter: ProgressReporter,
    before: RunState,
    successor: RunState,
    stage_id: str,
) -> None:
    selected = successor.accepted
    reporter.emit(
        ProgressEvent(
            "round-end",
            {
                **_round_progress_fields(before),
                "stage": "round-end",
                "stage_id": stage_id,
                "best_exact": selected.score.exact_count,
                "best_total": len(selected.score.task_ids),
                "loss": selected.score.unresolved_loss,
                "neurons": selected.resources.neurons,
                "recurrent_edges": selected.resources.recurrent_edges,
                "next_stage": successor.next_stage,
                "stable_rounds": successor.stable_rounds,
                "terminal_reason": successor.terminal_reason,
            },
        )
    )


def _emit_terminal_start(reporter: ProgressReporter, state: RunState) -> None:
    reporter.emit(
        ProgressEvent(
            "terminal-start",
            {
                **_round_progress_fields(state),
                "stage": "terminal",
                "stage_id": "terminal-evaluation",
                **_accepted_progress_fields(state.accepted),
                "checkpoint_sha256": state.accepted.checkpoint_sha256,
                "terminal_reason": state.terminal_reason,
            },
        )
    )


def _emit_terminal_result(
    reporter: ProgressReporter,
    state: RunState,
    result: Mapping[str, Any],
) -> None:
    reporter.emit(
        ProgressEvent(
            "terminal-result",
            {
                **_round_progress_fields(state),
                "stage": "terminal",
                "stage_id": "terminal-evaluation",
                "status": "completed",
                "score_exact": int(result["strict_task_pass_at_1_count"]),
                "score_total": int(result["task_count"]),
                "loss": float(result["mean_unresolved_task_loss"]),
                "checkpoint_sha256": state.accepted.checkpoint_sha256,
                "terminal_reason": state.terminal_reason,
            },
        )
    )


def run_evolution(
    adapter: EvolutionAdapter,
    output_dir: str | os.PathLike[str],
    *,
    config: PipelineConfig | None = None,
    history_plotter: Callable[
        [Sequence[Mapping[str, Any]], Path], object
    ] = plot_score_history,
    progress_reporter: ProgressReporter | None = None,
) -> RunState:
    """Run or resume the complete adapter-driven evolution lifecycle.

    Parameters
    ----------
    adapter : EvolutionAdapter
        Real model, compiled training, scoring, mutation, and checkpoint bridge.
    output_dir : path-like
        Durable run directory.  Compatible unfinished state resumes implicitly.
    config : PipelineConfig, optional
        Lineage configuration; defaults to Muon, 128 updates, and eight rounds.
    history_plotter : callable, optional
        Atomic history renderer.  Injection supports isolated coordinator tests.
    progress_reporter : ProgressReporter, optional
        Operator-facing event consumer. Omission keeps library execution silent.

    Returns
    -------
    RunState
        Closed terminal state after one held-out evaluation.
    """

    resolved_config = config or PipelineConfig()
    reporter: ProgressReporter = progress_reporter or _NullProgressReporter()
    try:
        return _run_evolution(
            adapter,
            output_dir,
            config=resolved_config,
            history_plotter=history_plotter,
            progress_reporter=reporter,
        )
    finally:
        reporter.close()


def _run_evolution(
    adapter: EvolutionAdapter,
    output_dir: str | os.PathLike[str],
    *,
    config: PipelineConfig,
    history_plotter: Callable[[Sequence[Mapping[str, Any]], Path], object],
    progress_reporter: ProgressReporter,
) -> RunState:
    """Run the coordinator with one resolved configuration and reporter."""

    store = PipelineStore(output_dir)
    manifest = adapter.training_manifest()
    manifest.validate()
    if manifest.role != "training":
        raise ValueError(
            "Evolution requires a training manifest; keep evaluation isolated."
        )
    if store.state_path.exists():
        state = store.load_state(config, manifest)
        if state.closed:
            restored = adapter.restore(state.accepted)
            _verify_restored(state.accepted, restored)
            state = replace(state, accepted=restored)
            _emit_resume(progress_reporter, state)
            _reconcile_closed_evaluation_intent(store, state)
            _refresh_artifacts(adapter, store, state.accepted, history_plotter)
            return state
        restored = adapter.restore(state.accepted)
        _verify_restored(state.accepted, restored)
        state = replace(state, accepted=restored)
        state = _recover_pending_transition(adapter, store, state)
        _emit_resume(progress_reporter, state)
    else:
        initial = adapter.initialize(config, store.output_dir)
        _require_candidate(initial, config, manifest.task_ids)
        initial = _persist_or_restore_selected(
            adapter,
            store,
            initial,
            stage_id="initial",
            parent_checkpoint_sha256=None,
            config=config,
        )
        state = RunState.initial(config, manifest, initial)
        store.write_state(state)
    _refresh_artifacts(adapter, store, state.accepted, history_plotter)

    while not state.closed:
        if state.next_stage == "train":
            state = _run_parent_training(
                adapter,
                store,
                state,
                history_plotter,
                progress_reporter,
            )
        elif state.next_stage in RESCORE_STAGES:
            state = _run_rescore_stage(
                adapter,
                store,
                state,
                history_plotter=history_plotter,
                progress_reporter=progress_reporter,
            )
        elif state.next_stage in MODEL_STAGES:
            state = _run_sibling_stage(
                adapter,
                store,
                state,
                state.next_stage,
                history_plotter=history_plotter,
                progress_reporter=progress_reporter,
            )
        elif state.next_stage == "round-end":
            state = _finish_round(
                adapter,
                store,
                state,
                history_plotter,
                progress_reporter,
            )
        elif state.next_stage == "terminal-evaluation":
            state = _terminal_evaluation(
                adapter,
                store,
                state,
                history_plotter,
                progress_reporter,
            )
        else:
            raise ResumeMismatchError(
                f"Unknown next stage {state.next_stage!r}; repair run-state.json."
            )
    return state


def _stage_identity(state: RunState, stage: str) -> str:
    """Return the canonical durable identity of one stage.

    Parameters
    ----------
    state : RunState
        Lifecycle position entering the stage.
    stage : str
        Logical phase name.

    Returns
    -------
    str
        Round-scoped identity, carrying the operation index for structural
        operations so a repeated operation kind stays uniquely addressable.
    """

    if stage in OPERATION_STAGES:
        return f"r{state.round_index:03d}-op{state.operation_index:02d}-{stage}"
    return f"r{state.round_index:03d}-{stage}"


def _stage_context(
    state: RunState, stage: str, stage_id: str, output_dir: Path
) -> StageContext:
    screened = state.config.screens and stage in OPERATION_STAGES
    return StageContext(
        state.round_index,
        stage,
        stage_id,
        output_dir,
        state.config,
        operation_index=state.operation_index,
        score_task_ids=(
            screen_task_ids(state.training_manifest, state.config) if screened else ()
        ),
    )


def _run_parent_training(
    adapter: EvolutionAdapter,
    store: PipelineStore,
    state: RunState,
    history_plotter: Callable[[Sequence[Mapping[str, Any]], Path], object],
    progress_reporter: ProgressReporter,
) -> RunState:
    started = time.perf_counter()
    stage_id = _stage_identity(state, "train")
    schedule = build_update_schedule(
        state.training_manifest, state.cursor, state.config.updates
    )
    context = _stage_context(state, "train", stage_id, store.output_dir)
    _emit_candidate_start(progress_reporter, state, context, "training")
    trained = adapter.train_parent(state.accepted, schedule, context)
    attempt = CandidateAttempt.completed("training", trained)
    _emit_candidate_result(progress_reporter, state, context, attempt)
    selection = select_candidate(state.accepted, (attempt,), config=state.config)
    selected = selection.selected
    successor = _expected_model_successor(
        state,
        selected,
        stage="train",
        accepted_child=not selection.parent_retained,
    )
    pending = _pending_transition_document(
        state,
        successor,
        stage_id=stage_id,
        stage="train",
        parent=state.accepted,
        selected=selected,
        attempts=(attempt,),
        dispositions=selection.dispositions,
        elapsed_seconds=time.perf_counter() - started,
    )
    store.write_pending(pending)
    successor = _complete_pending_transition(adapter, store, state, pending)
    _refresh_artifacts(adapter, store, successor.accepted, history_plotter)
    _emit_selection(progress_reporter, state, context, selection, successor)
    return successor


def _run_sibling_stage(
    adapter: EvolutionAdapter,
    store: PipelineStore,
    state: RunState,
    stage: str,
    *,
    history_plotter: Callable[[Sequence[Mapping[str, Any]], Path], object],
    progress_reporter: ProgressReporter,
) -> RunState:
    started = time.perf_counter()
    arms = STAGE_ARMS[stage]
    compression = stage.startswith("compression-")
    stage_id = _stage_identity(state, stage)
    schedule = build_update_schedule(
        state.training_manifest, state.cursor, state.config.updates
    )
    context = _stage_context(state, stage, stage_id, store.output_dir)
    parent = state.accepted
    attempt_values: list[CandidateAttempt] = []
    for arm in arms:
        _emit_candidate_start(progress_reporter, state, context, arm)
        attempt = adapter.run_candidate(parent, arm, schedule, context)
        _emit_candidate_result(progress_reporter, state, context, attempt)
        attempt_values.append(attempt)
    attempts = tuple(attempt_values)
    if tuple(attempt.name for attempt in attempts) != tuple(arms):
        raise ValueError(
            "Adapter candidate names must match requested arms; preserve stage ordering."
        )
    selection = select_candidate(
        parent, attempts, config=state.config, compression=compression
    )
    selected = selection.selected
    successor = _expected_model_successor(
        state,
        selected,
        stage=stage,
        accepted_child=not selection.parent_retained,
    )
    pending = _pending_transition_document(
        state,
        successor,
        stage_id=stage_id,
        stage=stage,
        parent=parent,
        selected=selected,
        attempts=attempts,
        dispositions=selection.dispositions,
        elapsed_seconds=time.perf_counter() - started,
    )
    store.write_pending(pending)
    successor = _complete_pending_transition(adapter, store, state, pending)
    _refresh_artifacts(adapter, store, successor.accepted, history_plotter)
    _emit_selection(progress_reporter, state, context, selection, successor)
    return successor


def _run_rescore_stage(
    adapter: EvolutionAdapter,
    store: PipelineStore,
    state: RunState,
    *,
    history_plotter: Callable[[Sequence[Mapping[str, Any]], Path], object],
    progress_reporter: ProgressReporter,
) -> RunState:
    started = time.perf_counter()
    stage = state.next_stage
    stage_id = _stage_identity(state, stage)
    context = StageContext(
        state.round_index,
        stage,
        stage_id,
        store.output_dir,
        state.config,
        operation_index=state.operation_index,
        score_task_ids=_rescore_scope_ids(state, stage),
    )
    parent = state.accepted
    _emit_candidate_start(progress_reporter, state, context, RESCORE_ARM)
    attempt = adapter.rescore(parent, context)
    _emit_candidate_result(progress_reporter, state, context, attempt)
    if attempt.status != "completed" or attempt.candidate is None:
        raise PipelineError(
            f"Rescoring the accepted state failed at {stage_id}; "
            "retain the last durable checkpoint."
        )
    rescored = attempt.candidate
    dispositions = {RESCORE_ARM: "accepted"}
    _verify_rescore_evidence(
        state,
        stage=stage,
        parent=parent,
        selected=rescored,
        attempts=(attempt,),
        dispositions=dispositions,
        selected_attempt=RESCORE_ARM,
    )
    successor = _expected_rescore_successor(state, rescored)
    pending = _pending_transition_document(
        state,
        successor,
        stage_id=stage_id,
        stage=stage,
        parent=parent,
        selected=rescored,
        attempts=(attempt,),
        dispositions=dispositions,
        elapsed_seconds=time.perf_counter() - started,
    )
    store.write_pending(pending)
    successor = _complete_pending_transition(adapter, store, state, pending)
    _refresh_artifacts(adapter, store, successor.accepted, history_plotter)
    _emit_selection(
        progress_reporter,
        state,
        context,
        SelectionResult(rescored, RESCORE_ARM, dispositions),
        successor,
    )
    return successor


def _pending_transition_document(
    before: RunState,
    after: RunState,
    *,
    stage_id: str,
    stage: str,
    parent: CandidateSnapshot,
    selected: CandidateSnapshot,
    attempts: Sequence[CandidateAttempt],
    dispositions: Mapping[str, str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    accepted_arms = [
        name for name, disposition in dispositions.items() if disposition == "accepted"
    ]
    if len(accepted_arms) > 1:
        raise ValueError(
            "A transition can accept at most one sibling; correct selection evidence."
        )
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "config_sha256": before.config.digest,
        "training_manifest_sha256": before.training_manifest.digest,
        "stage_id": stage_id,
        "stage": stage,
        "sequence_before": before.sequence,
        "state_before_sha256": _json_digest(before.position_dict()),
        "state_before": before.position_dict(),
        "parent": parent.to_dict(),
        "selected": selected.to_dict(),
        "selected_attempt": accepted_arms[0] if accepted_arms else None,
        "attempts": [_attempt_to_dict(attempt) for attempt in attempts],
        "dispositions": dict(dispositions),
        "elapsed_seconds": float(elapsed_seconds),
        "state_after": after.position_dict(),
    }


def _lineage_screens(state: RunState) -> bool:
    """Return whether this lineage screens its structural operations.

    Parameters
    ----------
    state : RunState
        Lifecycle position supplying manifest and configuration.

    Returns
    -------
    bool
        True when a proper screen subset exists for the training corpus.
    """

    return bool(screen_task_ids(state.training_manifest, state.config))


def _is_full_score(state: RunState, candidate: CandidateSnapshot) -> bool:
    """Return whether one candidate carries a complete-corpus score.

    Parameters
    ----------
    state : RunState
        Lifecycle position supplying the training manifest.
    candidate : CandidateSnapshot
        Snapshot whose score scope is in question.

    Returns
    -------
    bool
        True when the score covers every training task in manifest order.
    """

    return candidate.score.task_ids == state.training_manifest.task_ids


def _reached_mastery(state: RunState, candidate: CandidateSnapshot) -> bool:
    """Return whether one candidate proves complete training mastery.

    A screen subset can be entirely exact without the corpus being mastered, so
    mastery requires the complete-corpus scope as well as complete exactness.

    Parameters
    ----------
    state : RunState
        Lifecycle position supplying the training manifest.
    candidate : CandidateSnapshot
        Snapshot whose exactness is in question.

    Returns
    -------
    bool
        True only for complete exactness over the complete corpus.
    """

    return _is_full_score(state, candidate) and candidate.score.all_exact


def _next_operation_stage(stage: str, *, revisit: bool) -> str:
    """Return the operation kind that follows one operation in the cycle.

    Parameters
    ----------
    stage : str
        Structural operation kind that has just completed.
    revisit : bool
        Whether a neuron operation accepted a topology change.

    Returns
    -------
    str
        Next operation kind, wrapping from Dale back to edge.
    """

    if stage == "edge":
        return "neuron"
    if stage == "neuron":
        return "edge-revisit" if revisit else "dale"
    if stage == "edge-revisit":
        return "dale"
    return "edge"


def _operations_exhausted(before: RunState, stage: str) -> bool:
    """Return whether one round has spent its structural operation budget.

    Parameters
    ----------
    before : RunState
        Lifecycle position entering the completed operation.
    stage : str
        Structural operation kind that has just completed.

    Returns
    -------
    bool
        True when no further operation may run in this round.
    """

    budget = before.config.operations_per_round
    if budget is None:
        return stage == "dale"
    return before.operation_index + 1 >= budget


def _round_closing_stage(before: RunState, selected: CandidateSnapshot) -> str:
    """Return the stage that closes a round's structural phase.

    Parameters
    ----------
    before : RunState
        Lifecycle position entering the completed operation.
    selected : CandidateSnapshot
        State carried out of the last operation.

    Returns
    -------
    str
        ``round-score`` when the carried score must be restored to the
        complete corpus, otherwise ``round-end``.
    """

    return "round-end" if _is_full_score(before, selected) else "round-score"


def _expected_model_successor(
    before: RunState,
    selected: CandidateSnapshot,
    *,
    stage: str,
    accepted_child: bool,
) -> RunState:
    if stage not in MODEL_STAGES or before.next_stage != stage:
        raise ValueError("Model transition stage differs from the current lifecycle.")
    operation_index = before.operation_index
    if stage == "train":
        following = (
            "compression-edge"
            if _reached_mastery(before, selected)
            else ("round-screen" if _lineage_screens(before) else "edge")
        )
        operation_index = 0
    elif stage in OPERATION_STAGES:
        following, operation_index = _operation_successor_stage(
            before, selected, stage=stage, accepted_child=accepted_child
        )
    elif stage == "compression-edge":
        following = "compression-neuron"
    else:
        following = "round-end"
    return replace(
        before,
        sequence=before.sequence + 1,
        cursor=before.cursor + before.config.updates,
        operation_index=operation_index,
        next_stage=following,
        accepted=selected,
    )


def _operation_successor_stage(
    before: RunState,
    selected: CandidateSnapshot,
    *,
    stage: str,
    accepted_child: bool,
) -> tuple[str, int]:
    if _reached_mastery(before, selected):
        return "compression-edge", 0
    if _operations_exhausted(before, stage):
        return _round_closing_stage(before, selected), before.operation_index + 1
    revisit = stage == "neuron" and accepted_child and selected.topology_changed
    return _next_operation_stage(stage, revisit=revisit), before.operation_index + 1


def _expected_rescore_successor(
    before: RunState,
    rescored: CandidateSnapshot,
) -> RunState:
    """Return the successor of one score-scope transition.

    Parameters
    ----------
    before : RunState
        Lifecycle position entering the rescore stage.
    rescored : CandidateSnapshot
        Accepted state carrying its new score scope.

    Returns
    -------
    RunState
        Successor with the rescored state and no cursor advance.
    """

    stage = before.next_stage
    if stage not in RESCORE_STAGES:
        raise ValueError("Rescore transition stage differs from the current lifecycle.")
    if stage == "round-screen":
        following = "edge"
        operation_index = 0
    else:
        following = "round-end"
        operation_index = before.operation_index
    return replace(
        before,
        sequence=before.sequence + 1,
        operation_index=operation_index,
        next_stage=following,
        accepted=rescored,
    )


def _expected_round_successor(state: RunState) -> RunState:
    comparison = select_candidate(
        state.round_entry,
        (CandidateAttempt.completed("round-result", state.accepted),),
        config=state.config,
        compression=state.round_entry.score.all_exact,
    )
    stable_rounds = state.stable_rounds + int(comparison.parent_retained)
    if not comparison.parent_retained:
        stable_rounds = 0
    mastery = _reached_mastery(state, state.accepted)
    round_budget_reached = state.round_index + 1 >= state.config.rounds
    reason = None
    if mastery and (stable_rounds >= state.config.patience or round_budget_reached):
        reason = "mastery"
    elif not mastery and stable_rounds >= state.config.patience:
        reason = "stable"
    elif not mastery and round_budget_reached:
        reason = "round-budget"
    following = (
        "terminal-evaluation"
        if reason
        else ("compression-edge" if mastery else "train")
    )
    return replace(
        state,
        sequence=state.sequence + 1,
        round_index=state.round_index + (reason is None),
        operation_index=0,
        next_stage=following,
        round_entry=state.accepted,
        stable_rounds=stable_rounds,
        terminal_reason=reason,
    )


def _validate_progress_transition(
    before: RunState,
    after: RunState,
    record: Mapping[str, Any],
    *,
    store: PipelineStore | None = None,
) -> None:
    document = _json_object(record, "progress record")
    if (
        _json_integer(document["schema_version"], "progress.schema_version")
        != STATE_SCHEMA_VERSION
    ):
        raise ProgressConflictError("Progress schema is incompatible.")
    stage = _json_string(document["stage"], "progress.stage")
    stage_id = _json_string(document["stage_id"], "progress.stage_id")
    disposition = _json_string(document["disposition"], "progress.disposition")
    if before.next_stage in MODEL_STAGES:
        if stage != before.next_stage:
            raise ProgressConflictError(
                "Progress stage differs from the current lifecycle stage."
            )
        if stage_id != _stage_identity(before, stage):
            raise ProgressConflictError("Progress stage identity is not canonical.")
        expected = _expected_model_successor(
            before,
            after.accepted,
            stage=stage,
            accepted_child=disposition == "accepted",
        )
        expected_parent = before.accepted
    elif before.next_stage in RESCORE_STAGES:
        if stage != before.next_stage or stage_id != _stage_identity(before, stage):
            raise ProgressConflictError("Rescore progress identity is invalid.")
        expected = _expected_rescore_successor(before, after.accepted)
        expected_parent = before.accepted
    elif before.next_stage == "round-end":
        if stage != "round-end" or stage_id != (f"r{before.round_index:03d}-round-end"):
            raise ProgressConflictError("Round-end progress identity is invalid.")
        expected = _expected_round_successor(before)
        expected_parent = before.round_entry
    elif before.next_stage == "terminal-evaluation":
        if stage != "terminal" or stage_id != "terminal-evaluation":
            raise ProgressConflictError("Terminal progress identity is invalid.")
        expected = replace(
            before,
            sequence=before.sequence + 1,
            next_stage="closed",
            closed=True,
            evaluation_completed=True,
            evaluation_digest=after.evaluation_digest,
        )
        expected_parent = before.accepted
    else:
        raise ProgressConflictError("Closed state cannot have another transition.")
    if after != expected:
        raise ProgressConflictError(
            "Progress successor is not allowed from the current lifecycle stage."
        )
    expected_disposition = (
        "terminal"
        if after.closed
        else "retained-parent"
        if after.accepted.checkpoint_sha256 == expected_parent.checkpoint_sha256
        else "accepted"
    )
    if disposition != expected_disposition:
        raise ProgressConflictError("Progress disposition is inconsistent.")
    if (
        _json_integer(document["sequence_before"], "progress.sequence_before")
        != before.sequence
        or _json_integer(document["sequence_after"], "progress.sequence_after")
        != after.sequence
    ):
        raise ProgressConflictError("Progress sequence evidence is inconsistent.")
    if _json_integer(document["round"], "progress.round") != before.round_index:
        raise ProgressConflictError("Progress round evidence is inconsistent.")
    if (
        _json_integer(document["operation_index"], "progress.operation_index")
        != before.operation_index
    ):
        raise ProgressConflictError("Progress operation evidence is inconsistent.")
    expected_scope = (
        "full"
        if after.accepted.score.task_ids == before.training_manifest.task_ids
        else "screen"
    )
    if _json_string(document["score_scope"], "progress.score_scope") != expected_scope:
        raise ProgressConflictError("Progress score scope is inconsistent.")
    if (
        CandidateSnapshot.from_dict(
            _json_object(document["state_before"], "progress.state_before")["accepted"]
        )
        != before.accepted
    ):
        raise ProgressConflictError("Progress parent state is inconsistent.")
    if _json_bytes(document["state_before"]) != _json_bytes(
        before.position_dict()
    ) or _json_bytes(document["state_after"]) != _json_bytes(after.position_dict()):
        raise ProgressConflictError("Progress lifecycle snapshots are inconsistent.")
    if (
        _json_string(
            document["parent_checkpoint_sha256"],
            "progress.parent_checkpoint_sha256",
        )
        != expected_parent.checkpoint_sha256
        or _json_string(
            document["child_checkpoint_sha256"],
            "progress.child_checkpoint_sha256",
        )
        != after.accepted.checkpoint_sha256
    ):
        raise ProgressConflictError("Progress checkpoint lineage is inconsistent.")
    if (
        _json_string(
            document["selected_candidate_id"], "progress.selected_candidate_id"
        )
        != after.accepted.candidate_id
    ):
        raise ProgressConflictError("Progress selected candidate is inconsistent.")
    cursor_advance = after.cursor - before.cursor
    if (
        _json_integer(document["cursor_advance"], "progress.cursor_advance")
        != cursor_advance
    ):
        raise ProgressConflictError("Progress cursor advance is inconsistent.")
    solved_ids = _json_list(document["solved_task_ids"], "progress.solved_task_ids")
    if (
        tuple(
            _json_string(value, f"progress.solved_task_ids[{index}]")
            for index, value in enumerate(solved_ids)
        )
        != after.accepted.score.solved_task_ids
    ):
        raise ProgressConflictError("Progress solved-task evidence is inconsistent.")
    scalar_evidence = (
        (
            "exact_task_count",
            after.accepted.score.exact_count,
        ),
        ("neurons", after.accepted.resources.neurons),
        ("recurrent_edges", after.accepted.resources.recurrent_edges),
        ("persistent_bytes", after.accepted.resources.persistent_bytes),
        ("checkpoint_bytes", after.accepted.resources.checkpoint_bytes),
    )
    for field, expected_value in scalar_evidence:
        if _json_integer(document[field], f"progress.{field}") != expected_value:
            raise ProgressConflictError(f"Progress {field} evidence is inconsistent.")
    unresolved_loss = _json_float(
        document["unresolved_task_loss"], "progress.unresolved_task_loss"
    )
    if not math.isclose(
        unresolved_loss,
        after.accepted.score.unresolved_loss,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ProgressConflictError("Progress unresolved loss is inconsistent.")
    if (
        _json_optional_integer(
            document.get("peak_host_ram_bytes"), "progress.peak_host_ram_bytes"
        )
        != after.accepted.resources.peak_host_ram_bytes
        or _json_optional_integer(
            document.get("device_memory_bytes"), "progress.device_memory_bytes"
        )
        != after.accepted.resources.device_memory_bytes
    ):
        raise ProgressConflictError(
            "Progress selected resource evidence is inconsistent."
        )
    try:
        attempts, dispositions, total_executed_updates = _progress_attempts(
            document, stage=stage
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProgressConflictError(
            "Progress sibling evidence is inconsistent."
        ) from error
    if (
        _json_integer(document["updates"], "progress.updates") != total_executed_updates
        or _json_integer(
            document["total_executed_updates"],
            "progress.total_executed_updates",
        )
        != total_executed_updates
    ):
        raise ProgressConflictError(
            "Progress executed-update evidence is inconsistent."
        )
    if stage in RESCORE_STAGES:
        try:
            _verify_rescore_evidence(
                before,
                stage=stage,
                parent=expected_parent,
                selected=after.accepted,
                attempts=attempts,
                dispositions=dispositions,
                selected_attempt=RESCORE_ARM,
                carried=True,
            )
        except ValueError as error:
            raise ProgressConflictError(
                "Progress rescore evidence is inconsistent."
            ) from error
        if disposition != "accepted":
            raise ProgressConflictError("Progress rescore disposition is invalid.")
        source = attempts[0].candidate
        if store is not None and source is not None:
            _verify_progress_checkpoint_lineage(
                store,
                stage_id=stage_id,
                parent=expected_parent,
                source=source,
                accepted=after.accepted,
            )
    elif stage in MODEL_STAGES:
        try:
            selection = select_candidate(
                expected_parent,
                attempts,
                config=before.config,
                compression=stage.startswith("compression-"),
            )
        except ValueError as error:
            raise ProgressConflictError(
                "Progress sibling selection evidence is inconsistent."
            ) from error
        if dict(selection.dispositions) != dispositions:
            raise ProgressConflictError(
                "Progress sibling dispositions are inconsistent."
            )
        if selection.parent_retained:
            selection_valid = (
                disposition == "retained-parent" and after.accepted == expected_parent
            )
        else:
            selection_valid = disposition == "accepted" and _carried_state_identity(
                selection.selected
            ) == _carried_state_identity(after.accepted)
        if not selection_valid:
            raise ProgressConflictError("Progress sibling selection is inconsistent.")
        if store is not None and not selection.parent_retained:
            _verify_progress_checkpoint_lineage(
                store,
                stage_id=stage_id,
                parent=expected_parent,
                source=selection.selected,
                accepted=after.accepted,
            )
    elif attempts or dispositions:
        raise ProgressConflictError(
            "Boundary progress cannot contain sibling attempts."
        )
    elapsed = _json_float(document["elapsed_seconds"], "progress.elapsed_seconds")
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ProgressConflictError("Progress elapsed time is invalid.")


def _progress_attempts(
    document: Mapping[str, Any],
    *,
    stage: str,
) -> tuple[tuple[CandidateAttempt, ...], dict[str, str], int]:
    siblings = _json_list(document["siblings"], "progress.siblings")
    attempts: list[CandidateAttempt] = []
    dispositions: dict[str, str] = {}
    for index, sibling_value in enumerate(siblings):
        field_prefix = f"progress.siblings[{index}]"
        sibling = _json_object(sibling_value, field_prefix)
        name = _json_string(sibling["name"], f"{field_prefix}.name")
        status = _json_string(sibling["status"], f"{field_prefix}.status")
        disposition = _json_string(
            sibling["disposition"], f"{field_prefix}.disposition"
        )
        reason = _json_optional_string(sibling.get("reason"), f"{field_prefix}.reason")
        executed_updates = _json_integer(
            sibling["executed_updates"], f"{field_prefix}.executed_updates"
        )
        candidate_value = sibling.get("candidate")
        candidate = (
            None
            if candidate_value is None
            else CandidateSnapshot.from_dict(candidate_value)
        )
        attempt = CandidateAttempt(
            name=name,
            status=status,
            candidate=candidate,
            reason=reason,
            executed_updates=executed_updates,
        )
        if name in dispositions:
            raise ProgressConflictError("Progress sibling names must be unique.")
        direct_evidence: dict[str, object] = {
            "candidate_id": None if candidate is None else candidate.candidate_id,
            "checkpoint_sha256": (
                None if candidate is None else candidate.checkpoint_sha256
            ),
            "topology_sha256": None if candidate is None else candidate.topology_sha256,
            "parameters_sha256": (
                None if candidate is None else candidate.parameters_sha256
            ),
            "optimizer_sha256": (
                None if candidate is None else candidate.optimizer_sha256
            ),
            "persistent_bytes": (
                None if candidate is None else candidate.resources.persistent_bytes
            ),
            "checkpoint_bytes": (
                None if candidate is None else candidate.resources.checkpoint_bytes
            ),
            "neurons": None if candidate is None else candidate.resources.neurons,
            "recurrent_edges": (
                None if candidate is None else candidate.resources.recurrent_edges
            ),
            "peak_host_ram_bytes": (
                None if candidate is None else candidate.resources.peak_host_ram_bytes
            ),
            "device_memory_bytes": (
                None if candidate is None else candidate.resources.device_memory_bytes
            ),
            "topology_changed": None
            if candidate is None
            else candidate.topology_changed,
        }
        string_fields = {
            "candidate_id",
            "checkpoint_sha256",
            "topology_sha256",
            "parameters_sha256",
            "optimizer_sha256",
        }
        integer_fields = {
            "persistent_bytes",
            "checkpoint_bytes",
            "neurons",
            "recurrent_edges",
            "peak_host_ram_bytes",
            "device_memory_bytes",
        }
        parsed_evidence: dict[str, object] = {
            field: _json_optional_string(sibling.get(field), f"{field_prefix}.{field}")
            for field in string_fields
        }
        parsed_evidence.update(
            {
                field: _json_optional_integer(
                    sibling.get(field), f"{field_prefix}.{field}"
                )
                for field in integer_fields
            }
        )
        topology_changed = sibling.get("topology_changed")
        parsed_evidence["topology_changed"] = (
            None
            if topology_changed is None
            else _json_boolean(topology_changed, f"{field_prefix}.topology_changed")
        )
        if parsed_evidence != direct_evidence:
            raise ProgressConflictError(
                "Progress sibling candidate evidence is inconsistent."
            )
        attempts.append(attempt)
        dispositions[name] = disposition
    expected_names = STAGE_ARMS.get(stage, ())
    if tuple(attempt.name for attempt in attempts) != expected_names:
        raise ProgressConflictError(
            "Progress sibling arms or ordering are inconsistent."
        )
    return (
        tuple(attempts),
        dispositions,
        sum(attempt.executed_updates for attempt in attempts),
    )


def _complete_pending_transition(
    adapter: EvolutionAdapter,
    store: PipelineStore,
    state: RunState,
    document: Mapping[str, Any],
    *,
    recovering: bool = False,
) -> RunState:
    parts = _pending_transition_parts(state, document)
    successor = parts["successor"]
    parent = parts["parent"]
    selected = parts["selected"]
    pending_selected = selected
    attempts = parts["attempts"]
    selected_attempt = parts["selected_attempt"]
    if selected_attempt is not None:
        destination = store.checkpoint_dir / f"{parts['stage_id']}.npz"
        if recovering and not destination.exists():
            source = _recovered_candidate_path(store, selected)
            if (
                not source.is_file()
                or _file_sha256(source) != selected.checkpoint_sha256
            ):
                raise ProgressConflictError(
                    "Recovered selected candidate bytes differ from the pending journal."
                )
            attested = adapter.attest_pending(
                selected,
                parent_checkpoint_sha256=parent.checkpoint_sha256,
                stage_id=parts["stage_id"],
            )
            _verify_restored(selected, attested)
            selected = attested
        selected = _persist_or_restore_selected(
            adapter,
            store,
            selected,
            stage_id=parts["stage_id"],
            parent_checkpoint_sha256=parent.checkpoint_sha256,
            config=state.config,
        )
    successor = replace(successor, accepted=selected)
    record = store.progress_record(
        state,
        successor,
        stage_id=parts["stage_id"],
        stage=parts["stage"],
        parent=parent,
        selected=selected,
        attempts=attempts,
        dispositions=parts["dispositions"],
        elapsed_seconds=parts["elapsed_seconds"],
    )
    store.commit(record, successor)
    if recovering and selected_attempt is not None:
        _cleanup_recovered_selected_source(store, pending_selected, selected)
    discardable = tuple(
        attempt for attempt in attempts if attempt.name != selected_attempt
    )
    if recovering:
        _discard_recovered_attempts(adapter, store, discardable)
    else:
        for attempt in discardable:
            adapter.discard(attempt)
    store.clear_pending()
    return successor


def _recover_pending_transition(
    adapter: EvolutionAdapter,
    store: PipelineStore,
    state: RunState,
) -> RunState:
    document = store.read_pending()
    if document is None:
        return state
    try:
        sequence_before = _json_integer(
            document["sequence_before"], "pending.sequence_before"
        )
        stage_id = _json_string(document["stage_id"], "pending.stage_id")
    except (KeyError, TypeError, ValueError) as error:
        raise ProgressConflictError(
            "Pending transition identity is inconsistent; recover the journal."
        ) from error
    if sequence_before == state.sequence:
        return _complete_pending_transition(
            adapter, store, state, document, recovering=True
        )
    if sequence_before + 1 != state.sequence:
        raise ProgressConflictError(
            "Pending transition sequence differs from run state; recover the journal."
        )
    matching = [
        record
        for record in store.read_progress()
        if record.get("stage_id") == stage_id
        and record.get("sequence_before") == sequence_before
        and record.get("sequence_after") == state.sequence
    ]
    if len(matching) != 1:
        raise ProgressConflictError(
            "Committed pending transition lacks matching progress evidence."
        )
    parts = _pending_transition_parts(
        state,
        document,
        committed_record=matching[0],
    )
    if parts["selected_attempt"] is not None:
        _cleanup_recovered_selected_source(store, parts["selected"], state.accepted)
    discardable = tuple(
        attempt
        for attempt in parts["attempts"]
        if attempt.name != parts["selected_attempt"]
    )
    _discard_recovered_attempts(adapter, store, discardable)
    store.clear_pending()
    return state


def _discard_recovered_attempts(
    adapter: EvolutionAdapter,
    store: PipelineStore,
    attempts: Sequence[CandidateAttempt],
) -> None:
    validated: list[tuple[CandidateAttempt, Path | None]] = []
    for attempt in attempts:
        path = None
        if attempt.candidate is not None:
            path = _recovered_candidate_path(store, attempt.candidate)
        validated.append((attempt, path))
    for attempt, path in validated:
        adapter.discard(attempt)
        if path is not None:
            _unlink_and_sync(path)


def _cleanup_recovered_selected_source(
    store: PipelineStore,
    selected: CandidateSnapshot,
    accepted: CandidateSnapshot,
) -> None:
    source = _recovered_candidate_path(store, selected)
    destination = Path(accepted.checkpoint_path).resolve()
    if source == destination:
        raise ProgressConflictError(
            "Recovered selected source aliases the accepted checkpoint destination."
        )
    if not source.exists():
        return
    if not source.is_file() or _file_sha256(source) != selected.checkpoint_sha256:
        raise ProgressConflictError(
            "Recovered selected source differs from pending checkpoint evidence."
        )
    _unlink_and_sync(source)


def _recovered_candidate_path(
    store: PipelineStore, candidate: CandidateSnapshot
) -> Path:
    candidate_root = (store.output_dir / ".candidates").resolve()
    path = Path(candidate.checkpoint_path).resolve()
    if path.parent != candidate_root:
        raise ProgressConflictError(
            "Recovered candidate path escapes the run .candidates directory."
        )
    return path


def _allowed_score_scopes(state: RunState) -> tuple[tuple[str, ...], ...]:
    """Return the score task orders a lineage may carry.

    Parameters
    ----------
    state : RunState
        Lifecycle position supplying manifest and configuration.

    Returns
    -------
    tuple of tuple of str
        Complete-corpus order, plus the screen subset when screening is on.
    """

    full = state.training_manifest.task_ids
    screen = screen_task_ids(state.training_manifest, state.config)
    return (full,) if not screen else (full, screen)


def _rescore_scope_ids(state: RunState, stage: str) -> tuple[str, ...]:
    """Return the score task order one rescore stage must produce.

    Parameters
    ----------
    state : RunState
        Lifecycle position supplying manifest and configuration.
    stage : str
        Rescore stage name.

    Returns
    -------
    tuple of str
        Screen subset for ``round-screen``, complete corpus for ``round-score``.
    """

    if RESCORE_STAGES[stage] == "screen":
        return screen_task_ids(state.training_manifest, state.config)
    return state.training_manifest.task_ids


def _verify_rescore_evidence(
    state: RunState,
    *,
    stage: str,
    parent: CandidateSnapshot,
    selected: CandidateSnapshot,
    attempts: Sequence[CandidateAttempt],
    dispositions: Mapping[str, str],
    selected_attempt: str | None,
    carried: bool = False,
) -> None:
    if (
        len(attempts) != 1
        or attempts[0].status != "completed"
        or attempts[0].candidate is None
        or selected_attempt != RESCORE_ARM
        or dict(dispositions) != {RESCORE_ARM: "accepted"}
    ):
        raise ValueError("rescore evidence")
    source = attempts[0].candidate
    matches = (
        _carried_state_identity(source) == _carried_state_identity(selected)
        if carried
        else source == selected
    )
    if not matches:
        raise ValueError("rescore evidence")
    if selected.score.task_ids != _rescore_scope_ids(state, stage):
        raise ValueError("rescore scope")
    if (
        selected.parameters_sha256 != parent.parameters_sha256
        or selected.optimizer_sha256 != parent.optimizer_sha256
        or selected.resources.neurons != parent.resources.neurons
        or selected.resources.recurrent_edges != parent.resources.recurrent_edges
        or selected.topology_changed
    ):
        raise ValueError("rescore mutated the model")
    if not selected.score.finite:
        raise ValueError("rescore is non-finite")


def _pending_transition_parts(
    state: RunState,
    document: Mapping[str, Any],
    *,
    committed_record: Mapping[str, Any] | None = None,
) -> _PendingTransitionParts:
    try:
        pending = _json_object(document, "pending transition")
        if (
            _json_integer(pending["schema_version"], "pending.schema_version")
            != STATE_SCHEMA_VERSION
        ):
            raise ValueError("schema")
        if (
            _json_string(pending["config_sha256"], "pending.config_sha256")
            != state.config.digest
            or _json_string(
                pending["training_manifest_sha256"],
                "pending.training_manifest_sha256",
            )
            != state.training_manifest.digest
        ):
            raise ValueError("provenance")
        stage_id = _json_string(pending["stage_id"], "pending.stage_id")
        stage = _json_string(pending["stage"], "pending.stage")
        sequence_before = _json_integer(
            pending["sequence_before"], "pending.sequence_before"
        )
        parent = CandidateSnapshot.from_dict(pending["parent"])
        selected = CandidateSnapshot.from_dict(pending["selected"])
        attempt_values = _json_list(pending["attempts"], "pending.attempts")
        attempts = tuple(_attempt_from_dict(value) for value in attempt_values)
        disposition_values = _json_object(
            pending["dispositions"], "pending.dispositions"
        )
        dispositions = {
            _json_string(name, "pending disposition name"): _json_string(
                value, f"pending disposition {name}"
            )
            for name, value in disposition_values.items()
        }
        selected_attempt = _json_optional_string(
            pending.get("selected_attempt"), "pending.selected_attempt"
        )
        elapsed_seconds = _json_float(
            pending["elapsed_seconds"], "pending.elapsed_seconds"
        )
        if not stage_id or not stage or not math.isfinite(elapsed_seconds):
            raise ValueError("stage")
        if elapsed_seconds < 0.0 or len({item.name for item in attempts}) != len(
            attempts
        ):
            raise ValueError("attempts")
        rescoring = stage in RESCORE_STAGES
        if (
            (stage not in MODEL_STAGES and not rescoring)
            or tuple(attempt.name for attempt in attempts) != STAGE_ARMS[stage]
        ):
            raise ValueError("attempt arms")
        if rescoring:
            _verify_rescore_evidence(
                state,
                stage=stage,
                parent=parent,
                selected=selected,
                attempts=attempts,
                dispositions=dispositions,
                selected_attempt=selected_attempt,
            )
        else:
            selection = select_candidate(
                parent,
                attempts,
                config=state.config,
                compression=stage.startswith("compression-"),
            )
            if (
                dict(selection.dispositions) != dispositions
                or selection.selected_attempt != selected_attempt
                or selection.selected != selected
            ):
                raise ValueError("selection")
        allowed_scopes = _allowed_score_scopes(state)
        if parent.score.task_ids not in allowed_scopes:
            raise ValueError("score lineage")
        if selected.score.task_ids not in allowed_scopes:
            raise ValueError("score lineage")
        journal_before = state.advance_from(pending["state_before"])
        if journal_before.sequence != sequence_before:
            raise ValueError("state before sequence")
        if _json_string(
            pending["state_before_sha256"], "pending.state_before_sha256"
        ) != _json_digest(journal_before.position_dict()):
            raise ValueError("state before")
        if parent != journal_before.accepted:
            raise ValueError("parent")
        journal_successor = journal_before.advance_from(pending["state_after"])
        expected_successor = (
            _expected_rescore_successor(journal_before, selected)
            if rescoring
            else _expected_model_successor(
                journal_before,
                selected,
                stage=stage,
                accepted_child=selected_attempt is not None,
            )
        )
        if stage_id != _stage_identity(journal_before, stage):
            raise ValueError("stage identity")
        if journal_successor != expected_successor:
            raise ValueError("successor transition")
        if committed_record is None:
            if sequence_before != state.sequence or journal_before != state:
                raise ValueError("sequence")
            successor = journal_successor
        else:
            if sequence_before + 1 != state.sequence:
                raise ValueError("committed sequence")
            expected_id = selected.candidate_id
            record = _json_object(committed_record, "committed progress")
            record_attempts, record_dispositions, _ = _progress_attempts(
                record,
                stage=stage,
            )
            record_elapsed_seconds = _json_float(
                record["elapsed_seconds"],
                "progress.elapsed_seconds",
            )
            if (
                _json_string(record.get("stage"), "progress.stage") != stage
                or _json_string(record.get("stage_id"), "progress.stage_id") != stage_id
                or _json_string(
                    record.get("selected_candidate_id"),
                    "progress.selected_candidate_id",
                )
                != expected_id
                or record_attempts != attempts
                or record_dispositions != dispositions
                or record_elapsed_seconds != elapsed_seconds
            ):
                raise ValueError("committed selection")
            if _carried_state_identity(selected) != _carried_state_identity(
                state.accepted
            ) or state != replace(journal_successor, accepted=state.accepted):
                raise ValueError("committed state")
            successor = state
    except (KeyError, TypeError, ValueError) as error:
        raise ProgressConflictError(
            "Pending transition evidence is inconsistent; recover the journal."
        ) from error
    return {
        "stage_id": stage_id,
        "stage": stage,
        "parent": parent,
        "selected": selected,
        "attempts": attempts,
        "dispositions": dispositions,
        "selected_attempt": selected_attempt,
        "elapsed_seconds": elapsed_seconds,
        "successor": successor,
    }


def _finish_round(
    adapter: EvolutionAdapter,
    store: PipelineStore,
    state: RunState,
    history_plotter: Callable[[Sequence[Mapping[str, Any]], Path], object],
    progress_reporter: ProgressReporter,
) -> RunState:
    started = time.perf_counter()
    stage_id = f"r{state.round_index:03d}-round-end"
    successor = _expected_round_successor(state)
    record = store.progress_record(
        state,
        successor,
        stage_id=stage_id,
        stage="round-end",
        parent=state.round_entry,
        selected=state.accepted,
        attempts=(),
        elapsed_seconds=time.perf_counter() - started,
    )
    store.commit(record, successor)
    _refresh_artifacts(adapter, store, state.accepted, history_plotter)
    _emit_round_end(progress_reporter, state, successor, stage_id)
    return successor


def _terminal_evaluation(
    adapter: EvolutionAdapter,
    store: PipelineStore,
    state: RunState,
    history_plotter: Callable[[Sequence[Mapping[str, Any]], Path], object],
    progress_reporter: ProgressReporter,
) -> RunState:
    """Durably score and close one terminal checkpoint.

    The coordinator writes an intent before invoking the evaluator.  A crash
    before ``evaluation.json`` is durable may repeat that pure forward score;
    the current adapter protocol cannot close that call-to-write gap.  Once
    ``evaluation.json`` exists, every retry reuses it and finalization is
    idempotent without another evaluation call.
    """

    started = time.perf_counter()
    _emit_terminal_start(progress_reporter, state)
    manifest = adapter.evaluation_manifest()
    manifest.validate()
    if manifest.role != "evaluation":
        raise ValueError(
            "Terminal scoring requires the evaluation manifest; keep roles isolated."
        )
    intent = _evaluation_intent_document(state, manifest)
    if store.evaluation_intent_path.exists():
        durable_intent = _read_evaluation_intent(store.evaluation_intent_path)
        if durable_intent != intent:
            raise ProgressConflictError(
                "Terminal evaluation intent differs from the accepted lineage."
            )
    else:
        _atomic_json(store.evaluation_intent_path, intent)
    new_evaluation = False
    if store.evaluation_path.exists():
        try:
            document = json.loads(store.evaluation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProgressConflictError(
                "Terminal evaluation artifact is unreadable; recover it before resume."
            ) from error
    else:
        result = dict(adapter.evaluate_terminal(state.accepted, manifest))
        document = {
            "schema_version": STATE_SCHEMA_VERSION,
            "checkpoint_sha256": state.accepted.checkpoint_sha256,
            "evaluation_manifest_sha256": manifest.digest,
            "result": result,
        }
        new_evaluation = True
    _validate_terminal_evaluation(document, state, manifest)
    if new_evaluation:
        _atomic_json(store.evaluation_path, document)
    evaluation_digest = hashlib.sha256(store.evaluation_path.read_bytes()).hexdigest()
    successor = replace(
        state,
        sequence=state.sequence + 1,
        next_stage="closed",
        closed=True,
        evaluation_completed=True,
        evaluation_digest=evaluation_digest,
    )
    stage_id = "terminal-evaluation"
    record = store.progress_record(
        state,
        successor,
        stage_id=stage_id,
        stage="terminal",
        parent=state.accepted,
        selected=state.accepted,
        attempts=(),
        elapsed_seconds=time.perf_counter() - started,
    )
    store.commit(record, successor)
    _unlink_and_sync(store.evaluation_intent_path)
    _refresh_artifacts(adapter, store, state.accepted, history_plotter)
    result_document = _json_object(document["result"], "evaluation result")
    _emit_terminal_result(progress_reporter, successor, result_document)
    return successor


def _evaluation_intent_document(
    state: RunState, manifest: CorpusManifest
) -> dict[str, object]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "checkpoint_sha256": state.accepted.checkpoint_sha256,
        "evaluation_manifest_sha256": manifest.digest,
    }


def _read_evaluation_intent(path: Path) -> dict[str, object]:
    try:
        document = _json_object(
            json.loads(path.read_text(encoding="utf-8")), "evaluation intent"
        )
        intent: dict[str, object] = {
            "schema_version": _json_integer(
                document["schema_version"], "evaluation_intent.schema_version"
            ),
            "checkpoint_sha256": _json_string(
                document["checkpoint_sha256"],
                "evaluation_intent.checkpoint_sha256",
            ),
            "evaluation_manifest_sha256": _json_string(
                document["evaluation_manifest_sha256"],
                "evaluation_intent.evaluation_manifest_sha256",
            ),
        }
        if intent["schema_version"] != STATE_SCHEMA_VERSION or any(
            not _is_sha256(str(intent[field]))
            for field in ("checkpoint_sha256", "evaluation_manifest_sha256")
        ):
            raise ValueError("evaluation intent")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ProgressConflictError(
            "Terminal evaluation intent is incomplete or inconsistent."
        ) from error
    return intent


def _reconcile_closed_evaluation_intent(store: PipelineStore, state: RunState) -> None:
    if not store.evaluation_intent_path.exists():
        return
    intent = _read_evaluation_intent(store.evaluation_intent_path)
    try:
        evaluation = _json_object(
            json.loads(store.evaluation_path.read_text(encoding="utf-8")),
            "terminal evaluation",
        )
        expected = {
            "schema_version": _json_integer(
                evaluation["schema_version"], "evaluation.schema_version"
            ),
            "checkpoint_sha256": _json_string(
                evaluation["checkpoint_sha256"], "evaluation.checkpoint_sha256"
            ),
            "evaluation_manifest_sha256": _json_string(
                evaluation["evaluation_manifest_sha256"],
                "evaluation.evaluation_manifest_sha256",
            ),
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ProgressConflictError(
            "Closed terminal evaluation cannot reconcile its intent."
        ) from error
    if (
        intent != expected
        or intent["checkpoint_sha256"] != state.accepted.checkpoint_sha256
    ):
        raise ProgressConflictError(
            "Closed terminal evaluation intent differs from durable result lineage."
        )
    _unlink_and_sync(store.evaluation_intent_path)


def _validate_terminal_evaluation(
    document: Mapping[str, Any],
    state: RunState,
    manifest: CorpusManifest,
) -> None:
    try:
        evaluation = _json_object(document, "terminal evaluation")
        if (
            _json_integer(evaluation["schema_version"], "evaluation.schema_version")
            != STATE_SCHEMA_VERSION
        ):
            raise ValueError("schema")
        if (
            _json_string(
                evaluation["checkpoint_sha256"],
                "evaluation.checkpoint_sha256",
            )
            != state.accepted.checkpoint_sha256
            or _json_string(
                evaluation["evaluation_manifest_sha256"],
                "evaluation.evaluation_manifest_sha256",
            )
            != manifest.digest
        ):
            raise ValueError("lineage")
        result = _json_object(evaluation["result"], "evaluation.result")
        task_count = _json_integer(result["task_count"], "evaluation.result.task_count")
        exact_count = _json_integer(
            result["strict_task_pass_at_1_count"],
            "evaluation.result.strict_task_pass_at_1_count",
        )
        task_ids = _json_list(result["task_ids"], "evaluation.result.task_ids")
        task_exact = _json_list(result["task_exact"], "evaluation.result.task_exact")
        task_loss = _json_list(result["task_loss"], "evaluation.result.task_loss")
        score = ScoreSnapshot(
            task_ids=tuple(
                _json_string(value, f"evaluation.result.task_ids[{index}]")
                for index, value in enumerate(task_ids)
            ),
            task_exact=tuple(
                _json_boolean(value, f"evaluation.result.task_exact[{index}]")
                for index, value in enumerate(task_exact)
            ),
            task_loss=tuple(
                _json_float(value, f"evaluation.result.task_loss[{index}]")
                for index, value in enumerate(task_loss)
            ),
            finite=_json_boolean(result["finite"], "evaluation.result.finite"),
        )
        unresolved_loss = _json_float(
            result["mean_unresolved_task_loss"],
            "evaluation.result.mean_unresolved_task_loss",
        )
        if (
            task_count != EXPECTED_ARC_TASKS
            or task_count != len(manifest.task_ids)
            or score.task_ids != manifest.task_ids
            or not score.finite
            or exact_count != score.exact_count
            or exact_count < 0
            or exact_count > task_count
            or not math.isfinite(unresolved_loss)
            or not math.isclose(
                unresolved_loss,
                score.unresolved_loss,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("result")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ProgressConflictError(
            "Terminal evaluation result is incomplete or inconsistent; recover it."
        ) from error


def _candidate_improves(
    parent: CandidateSnapshot,
    candidate: CandidateSnapshot,
    config: PipelineConfig,
    *,
    compression: bool,
) -> tuple[bool, str]:
    if not candidate.score.finite:
        return False, "rejected-nonfinite"
    if _limit_violation(candidate, config):
        return False, "rejected-limit"
    if candidate.score.task_ids != parent.score.task_ids:
        return False, "rejected-score-mismatch"
    if any(
        old and not new
        for old, new in zip(parent.score.task_exact, candidate.score.task_exact)
    ):
        return False, "rejected-regression"
    if compression:
        if not parent.score.all_exact or not candidate.score.all_exact:
            return False, "rejected-regression"
        return (
            (True, "eligible")
            if candidate.resources.persistent_bytes < parent.resources.persistent_bytes
            else (False, "rejected-no-improvement")
        )
    if candidate.score.exact_count > parent.score.exact_count:
        return True, "eligible"
    if candidate.score.exact_count < parent.score.exact_count:
        return False, "rejected-regression"
    parent_loss = parent.score.unresolved_loss
    candidate_loss = candidate.score.unresolved_loss
    threshold = max(
        LOSS_ABSOLUTE_IMPROVEMENT,
        LOSS_RELATIVE_IMPROVEMENT * parent_loss,
    )
    if parent_loss - candidate_loss >= threshold:
        return True, "eligible"
    if candidate_loss == parent_loss and _resource_key(candidate) < _resource_key(
        parent
    ):
        return True, "eligible"
    return False, "rejected-no-improvement"


def _limit_violation(
    candidate: CandidateSnapshot, config: PipelineConfig
) -> str | None:
    resources = candidate.resources
    if resources.neurons > config.max_neurons:
        return "neuron-cap"
    if resources.recurrent_edges > config.max_recurrent_edges:
        return "recurrent-edge-cap"
    if resources.checkpoint_bytes > config.max_checkpoint_bytes:
        return "checkpoint-cap"
    return None


def _resource_key(candidate: CandidateSnapshot) -> tuple[int, int, int]:
    return (
        candidate.resources.persistent_bytes,
        candidate.resources.neurons,
        candidate.resources.recurrent_edges,
    )


def _require_candidate(
    candidate: CandidateSnapshot,
    config: PipelineConfig,
    expected_task_ids: tuple[str, ...] | None = None,
) -> None:
    if not candidate.score.finite:
        raise PipelineError(
            "Selected continuation is non-finite; retain the last durable checkpoint."
        )
    violation = _limit_violation(candidate, config)
    if violation is not None:
        raise PipelineError(
            f"Selected continuation exceeds {violation}; reduce the candidate."
        )
    if expected_task_ids is not None and candidate.score.task_ids != expected_task_ids:
        raise PipelineError(
            "Selected score does not match the training manifest; score every task in order."
        )


def _checkpoint_lineage_path(store: PipelineStore, stage_id: str) -> Path:
    return store.checkpoint_dir / f"{stage_id}.lineage.json"


def _checkpoint_lineage_core(
    candidate: CandidateSnapshot,
    *,
    stage_id: str,
    parent_checkpoint_sha256: str | None,
    destination: Path,
) -> dict[str, object]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "stage_id": stage_id,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "candidate_id": candidate.candidate_id,
        "source_checkpoint_path": str(Path(candidate.checkpoint_path).resolve()),
        "source_checkpoint_sha256": candidate.checkpoint_sha256,
        "topology_sha256": candidate.topology_sha256,
        "parameters_sha256": candidate.parameters_sha256,
        "optimizer_sha256": candidate.optimizer_sha256,
        "destination_path": str(destination.resolve()),
    }


def _read_checkpoint_lineage(path: Path) -> dict[str, Any]:
    try:
        document = _json_object(
            json.loads(path.read_text(encoding="utf-8")), "checkpoint lineage"
        )
        _json_integer(document["schema_version"], "lineage.schema_version")
        _json_string(document["stage_id"], "lineage.stage_id")
        _json_optional_string(
            document.get("parent_checkpoint_sha256"),
            "lineage.parent_checkpoint_sha256",
        )
        for field in (
            "candidate_id",
            "source_checkpoint_path",
            "source_checkpoint_sha256",
            "topology_sha256",
            "parameters_sha256",
            "optimizer_sha256",
            "destination_path",
        ):
            _json_string(document[field], f"lineage.{field}")
        destination_digest = _json_optional_string(
            document.get("destination_checkpoint_sha256"),
            "lineage.destination_checkpoint_sha256",
        )
        destination_bytes = _json_optional_integer(
            document.get("destination_checkpoint_bytes"),
            "lineage.destination_checkpoint_bytes",
        )
        if (destination_digest is None) != (destination_bytes is None):
            raise ValueError("lineage completion")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ProgressConflictError(
            "Checkpoint lineage evidence is inconsistent; recover the sidecar."
        ) from error
    return document


def _verify_checkpoint_lineage(
    store: PipelineStore,
    candidate: CandidateSnapshot,
    *,
    stage_id: str,
    parent_checkpoint_sha256: str | None,
    destination: Path,
) -> dict[str, Any]:
    path = _checkpoint_lineage_path(store, stage_id)
    if not path.is_file():
        raise ProgressConflictError(
            "Checkpoint lineage sidecar is missing; do not infer accepted ancestry."
        )
    document = _read_checkpoint_lineage(path)
    core = _checkpoint_lineage_core(
        candidate,
        stage_id=stage_id,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        destination=destination,
    )
    if any(document.get(field) != value for field, value in core.items()):
        raise ProgressConflictError(
            "Checkpoint lineage does not match the immediate accepted parent."
        )
    destination_digest = document.get("destination_checkpoint_sha256")
    destination_bytes = document.get("destination_checkpoint_bytes")
    if destination_digest is not None and (
        destination_digest != _file_sha256(destination)
        or destination_bytes != destination.stat().st_size
    ):
        raise ProgressConflictError(
            "Checkpoint lineage completion differs from destination bytes."
        )
    return document


def _verify_progress_checkpoint_lineage(
    store: PipelineStore,
    *,
    stage_id: str,
    parent: CandidateSnapshot,
    source: CandidateSnapshot,
    accepted: CandidateSnapshot,
) -> None:
    destination = (store.checkpoint_dir / f"{stage_id}.npz").resolve()
    if (
        not destination.is_file()
        or Path(accepted.checkpoint_path).resolve() != destination
    ):
        raise ProgressConflictError(
            "Progress checkpoint destination differs from its accepted lineage."
        )
    document = _verify_checkpoint_lineage(
        store,
        source,
        stage_id=stage_id,
        parent_checkpoint_sha256=parent.checkpoint_sha256,
        destination=destination,
    )
    if (
        document.get("destination_checkpoint_sha256") != accepted.checkpoint_sha256
        or document.get("destination_checkpoint_bytes")
        != accepted.resources.checkpoint_bytes
    ):
        raise ProgressConflictError(
            "Progress accepted checkpoint differs from completed lineage evidence."
        )


def _write_checkpoint_lineage_intent(
    store: PipelineStore,
    candidate: CandidateSnapshot,
    *,
    stage_id: str,
    parent_checkpoint_sha256: str | None,
    destination: Path,
) -> None:
    path = _checkpoint_lineage_path(store, stage_id)
    core = _checkpoint_lineage_core(
        candidate,
        stage_id=stage_id,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        destination=destination,
    )
    if path.exists():
        document = _read_checkpoint_lineage(path)
        if any(document.get(field) != value for field, value in core.items()):
            raise ProgressConflictError(
                "Checkpoint lineage intent conflicts with the selected transition."
            )
        return
    _atomic_json(path, core)


def _finalize_checkpoint_lineage(
    store: PipelineStore,
    candidate: CandidateSnapshot,
    *,
    stage_id: str,
    parent_checkpoint_sha256: str | None,
    destination: Path,
) -> None:
    document = _verify_checkpoint_lineage(
        store,
        candidate,
        stage_id=stage_id,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        destination=destination,
    )
    completed = {
        **document,
        "destination_checkpoint_sha256": _file_sha256(destination),
        "destination_checkpoint_bytes": destination.stat().st_size,
    }
    _atomic_json(_checkpoint_lineage_path(store, stage_id), completed)


def _persist_or_restore_selected(
    adapter: EvolutionAdapter,
    store: PipelineStore,
    candidate: CandidateSnapshot,
    *,
    stage_id: str,
    parent_checkpoint_sha256: str | None,
    config: PipelineConfig,
) -> CandidateSnapshot:
    destination = store.checkpoint_dir / f"{stage_id}.npz"
    if not destination.exists():
        return _persist_selected(
            adapter,
            store,
            candidate,
            stage_id=stage_id,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            config=config,
        )
    if not destination.is_file():
        raise PipelineError(
            "Pending checkpoint destination is not a file; recover the lineage path."
        )
    _verify_checkpoint_lineage(
        store,
        candidate,
        stage_id=stage_id,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        destination=destination,
    )
    expected = replace(
        candidate,
        checkpoint_path=str(destination),
        checkpoint_sha256=_file_sha256(destination),
        resources=replace(
            candidate.resources, checkpoint_bytes=destination.stat().st_size
        ),
    )
    _require_candidate(expected, config)
    restored = adapter.restore(expected)
    _verify_restored(expected, restored)
    _require_candidate(restored, config)
    _finalize_checkpoint_lineage(
        store,
        candidate,
        stage_id=stage_id,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        destination=destination,
    )
    return restored


def _persist_selected(
    adapter: EvolutionAdapter,
    store: PipelineStore,
    candidate: CandidateSnapshot,
    *,
    stage_id: str,
    parent_checkpoint_sha256: str | None,
    config: PipelineConfig,
) -> CandidateSnapshot:
    destination = store.checkpoint_dir / f"{stage_id}.npz"
    if destination.exists():
        raise PipelineError(
            "Checkpoint destination already exists; reconcile its lineage before persistence."
        )
    _write_checkpoint_lineage_intent(
        store,
        candidate,
        stage_id=stage_id,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        destination=destination,
    )
    persisted = adapter.persist(
        candidate,
        destination,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        stage_id=stage_id,
    )
    if persisted.candidate_id != candidate.candidate_id or (
        persisted.topology_sha256,
        persisted.parameters_sha256,
        persisted.optimizer_sha256,
    ) != (
        candidate.topology_sha256,
        candidate.parameters_sha256,
        candidate.optimizer_sha256,
    ):
        raise PipelineError(
            "Persisted checkpoint changed carried topology, parameters, or Muon state."
        )
    if Path(persisted.checkpoint_path).resolve() != destination.resolve():
        raise PipelineError(
            "Persisted checkpoint path differs from lineage destination; use the stage path."
        )
    if not destination.is_file():
        raise PipelineError(
            "Persisted checkpoint is missing; atomically write the requested stage path."
        )
    actual_digest = _file_sha256(destination)
    if persisted.checkpoint_sha256 != actual_digest:
        raise PipelineError(
            "Persisted checkpoint digest differs from its actual bytes; reject the handoff."
        )
    persisted = replace(
        persisted,
        resources=replace(
            persisted.resources, checkpoint_bytes=destination.stat().st_size
        ),
    )
    _require_candidate(persisted, config)
    _finalize_checkpoint_lineage(
        store,
        candidate,
        stage_id=stage_id,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        destination=destination,
    )
    return persisted


def _verify_restored(expected: CandidateSnapshot, restored: CandidateSnapshot) -> None:
    if _continuation_identity(expected) != _continuation_identity(restored):
        raise ResumeMismatchError(
            "Restored checkpoint changed continuation evidence; reject resume."
        )


def _refresh_artifacts(
    adapter: EvolutionAdapter,
    store: PipelineStore,
    candidate: CandidateSnapshot,
    history_plotter: Callable[[Sequence[Mapping[str, Any]], Path], object],
) -> None:
    topology_output = store.output_dir / "topology.png"
    topology_temporary = _temporary_sibling(topology_output, suffix=".png")
    try:
        adapter.render_topology(candidate, topology_temporary)
        if not topology_temporary.is_file() or topology_temporary.stat().st_size == 0:
            raise PipelineError(
                "Topology renderer did not write a PNG; produce the requested output."
            )
        _replace_and_sync(topology_temporary, topology_output)
    finally:
        topology_temporary.unlink(missing_ok=True)
    history_output = store.output_dir / "score-history.png"
    history_temporary = _temporary_sibling(history_output, suffix=".png")
    try:
        history_records: Sequence[Mapping[str, Any]] = store.read_progress()
        if not history_records:
            history_records = (
                {
                    "stage_id": "initial",
                    "state_before": {"accepted": candidate.to_dict()},
                },
            )
        history_plotter(history_records, history_temporary)
        if not history_temporary.is_file() or history_temporary.stat().st_size == 0:
            raise PipelineError(
                "History renderer did not write a PNG; produce the requested output."
            )
        _replace_and_sync(history_temporary, history_output)
    finally:
        history_temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path, suffix=".json")
    try:
        with temporary.open("wb") as stream:
            stream.write(_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_and_sync(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_and_sync(source: Path, target: Path) -> None:
    os.replace(source, target)
    try:
        descriptor = os.open(target.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _unlink_and_sync(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _temporary_sibling(path: Path, *, suffix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _json_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attempt_to_dict(attempt: CandidateAttempt) -> dict[str, object]:
    return {
        "name": attempt.name,
        "status": attempt.status,
        "candidate": (
            None if attempt.candidate is None else attempt.candidate.to_dict()
        ),
        "reason": attempt.reason,
        "executed_updates": attempt.executed_updates,
    }


def _attempt_from_dict(value: Mapping[str, Any]) -> CandidateAttempt:
    document = _json_object(value, "candidate attempt")
    candidate_value = document.get("candidate")
    candidate = (
        None
        if candidate_value is None
        else CandidateSnapshot.from_dict(candidate_value)
    )
    return CandidateAttempt(
        name=_json_string(document["name"], "attempt.name"),
        status=_json_string(document["status"], "attempt.status"),
        candidate=candidate,
        reason=_json_optional_string(document.get("reason"), "attempt.reason"),
        executed_updates=_json_integer(
            document["executed_updates"], "attempt.executed_updates"
        ),
    )


def _continuation_identity(candidate: CandidateSnapshot) -> tuple[object, ...]:
    resources = candidate.resources
    return (
        candidate.candidate_id,
        str(Path(candidate.checkpoint_path).resolve()),
        candidate.checkpoint_sha256,
        candidate.topology_sha256,
        candidate.parameters_sha256,
        candidate.optimizer_sha256,
        candidate.score,
        resources.persistent_bytes,
        resources.checkpoint_bytes,
        resources.neurons,
        resources.recurrent_edges,
        candidate.topology_changed,
    )


def _carried_state_identity(candidate: CandidateSnapshot) -> tuple[object, ...]:
    resources = candidate.resources
    return (
        candidate.candidate_id,
        candidate.topology_sha256,
        candidate.parameters_sha256,
        candidate.optimizer_sha256,
        candidate.score,
        resources.persistent_bytes,
        resources.neurons,
        resources.recurrent_edges,
        candidate.topology_changed,
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _json_object(value: Any, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"Persisted {field} must be a JSON object.")
    return value


def _json_list(value: Any, field: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"Persisted {field} must be a JSON list.")
    return value


def _json_string(value: Any, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"Persisted {field} must be a JSON string.")
    return value


def _json_integer(value: Any, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"Persisted {field} must be a JSON integer.")
    return value


def _json_float(value: Any, field: str) -> float:
    if type(value) is not float:
        raise TypeError(f"Persisted {field} must be a JSON float.")
    return value


def _json_boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"Persisted {field} must be a JSON boolean.")
    return value


def _json_optional_integer(value: Any, field: str) -> int | None:
    return None if value is None else _json_integer(value, field)


def _json_optional_string(value: Any, field: str) -> str | None:
    return None if value is None else _json_string(value, field)
