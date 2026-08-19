"""Exact ARC scoring and latent-trajectory evidence for Example 21."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
GridArray = NDArray[np.int8]

_AXIS_SIZE = 30
_COLOR_COUNT = 10
_MODEL_ONLY_REQUIRED_TASK_COUNT = 400
_MODEL_ONLY_REQUIRED_EXACT_TASK_COUNT = 160


def _finite_array(
    value: ArrayLike,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _grid_array(value: ArrayLike, name: str) -> GridArray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular integer grid") from error
    if array.ndim != 2 or any(size < 1 or size > _AXIS_SIZE for size in array.shape):
        raise ValueError(
            f"{name} shape must be a nonempty 1..{_AXIS_SIZE} by "
            f"1..{_AXIS_SIZE} grid; got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f"{name} must contain non-boolean integer colors")
    integer = array.astype(np.int64, copy=False)
    if np.any(integer < 0) or np.any(integer >= _COLOR_COUNT):
        raise ValueError(f"{name} colors must be in [0, {_COLOR_COUNT})")
    result = np.ascontiguousarray(integer, dtype=np.int8)
    result.setflags(write=False)
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


@dataclass(frozen=True)
class OutputLogits:
    """Hold one query's independently predicted ARC output factors.

    Parameters
    ----------
    height : array-like
        Thirty logits for output heights 1 through 30.
    width : array-like
        Thirty logits for output widths 1 through 30.
    colors : array-like
        Color logits shaped ``(30, 30, 10)``. Target shape never masks these
        logits during decoding.

    Raises
    ------
    ValueError
        If a head has the wrong shape or contains a non-finite value.
    """

    height: ArrayLike
    width: ArrayLike
    colors: ArrayLike

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "height", _finite_array(self.height, "height logits", (_AXIS_SIZE,))
        )
        object.__setattr__(
            self, "width", _finite_array(self.width, "width logits", (_AXIS_SIZE,))
        )
        object.__setattr__(
            self,
            "colors",
            _finite_array(
                self.colors,
                "color logits",
                (_AXIS_SIZE, _AXIS_SIZE, _COLOR_COUNT),
            ),
        )


@dataclass(frozen=True)
class DecodedCandidate:
    """Describe one deterministic candidate grid.

    Parameters
    ----------
    grid : array-like
        Rectangular ARC grid with colors 0 through 9.
    changed_decision : str or None, default=None
        Decision changed from joint argmax. Candidate one uses ``None``;
        candidate two names ``height``, ``width``, or ``cell:r,c``.
    log_probability : float, default=0.0
        Sum of log probabilities for the chosen shape and valid cells.
    """

    grid: ArrayLike
    changed_decision: str | None = None
    log_probability: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", _grid_array(self.grid, "candidate grid"))
        if self.changed_decision is not None and not isinstance(
            self.changed_decision, str
        ):
            raise ValueError("changed_decision must be a string or None")
        if isinstance(self.log_probability, (bool, np.bool_)) or not isinstance(
            self.log_probability, Real
        ):
            raise ValueError("log_probability must be finite")
        value = float(self.log_probability)
        if not math.isfinite(value):
            raise ValueError("log_probability must be finite")
        object.__setattr__(self, "log_probability", value)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation of the candidate.

        Returns
        -------
        dict
            Candidate dimensions, cells, changed decision, and score.
        """
        grid = np.asarray(self.grid)
        return {
            "height": int(grid.shape[0]),
            "width": int(grid.shape[1]),
            "grid": grid.tolist(),
            "changed_decision": self.changed_decision,
            "log_probability": self.log_probability,
        }


SelectionRole = Literal[
    "latest_sweep_joint_argmax",
    "earlier_sweep_joint_argmax",
    "latest_sweep_logit_runner_up",
]


@dataclass(frozen=True)
class SelectedModelCandidate:
    """Attach target-free neural-checkpoint provenance to one ARC candidate.

    Parameters
    ----------
    candidate : DecodedCandidate
        Candidate decoded exclusively from model logits.
    source_checkpoint : int
        Positive completed-sweep checkpoint that supplied the candidate.
    selection_role : {"latest_sweep_joint_argmax", \
            "earlier_sweep_joint_argmax", "latest_sweep_logit_runner_up"}
        Deterministic role occupied by the candidate in the two-slot policy.

    Raises
    ------
    TypeError
        If ``candidate`` is not a :class:`DecodedCandidate`.
    ValueError
        If the checkpoint or selection role is invalid, or the candidate's
        changed-decision metadata conflicts with the role.
    """

    candidate: DecodedCandidate
    source_checkpoint: int
    selection_role: SelectionRole

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, DecodedCandidate):
            raise TypeError("candidate must be a DecodedCandidate")
        checkpoint = _nonnegative_integer(self.source_checkpoint, "source_checkpoint")
        if checkpoint == 0:
            raise ValueError("source_checkpoint must be positive")
        object.__setattr__(self, "source_checkpoint", checkpoint)
        valid_roles = {
            "latest_sweep_joint_argmax",
            "earlier_sweep_joint_argmax",
            "latest_sweep_logit_runner_up",
        }
        if self.selection_role not in valid_roles:
            raise ValueError("selection_role is invalid")
        is_runner_up = self.selection_role == "latest_sweep_logit_runner_up"
        if is_runner_up and self.candidate.changed_decision is None:
            raise ValueError("runner-up candidate must name its changed decision")
        if not is_runner_up and self.candidate.changed_decision is not None:
            raise ValueError("joint-argmax candidate cannot name a changed decision")

    def to_dict(self) -> dict[str, object]:
        """Return candidate cells and model-only selection provenance.

        Returns
        -------
        dict
            JSON-safe candidate data, source checkpoint, and selection role.
        """

        return {
            **self.candidate.to_dict(),
            "provenance": "model",
            "source_checkpoint": self.source_checkpoint,
            "selection_role": self.selection_role,
        }


@dataclass(frozen=True)
class QueryScore:
    """Hold exact and diagnostic scores for one ARC test query.

    Parameters
    ----------
    task_id : str
        Stable task identity used for conjunctive task scoring.
    query_index : int
        Zero-based query index within the task.
    pass_at_1, pass_at_2 : bool
        Exact shape-and-cell success under one or two candidates.
    shape_accuracy : bool
        Candidate-one shape diagnostic. This never satisfies exact success.
    valid_cell_pixel_accuracy : float
        Candidate-one overlap matches divided by target valid-cell count.
        Missing predicted cells receive no credit.
    candidate_count : int
        Number of distinct candidates retained, either one or two.
    """

    task_id: str
    query_index: int
    pass_at_1: bool
    pass_at_2: bool
    shape_accuracy: bool
    valid_cell_pixel_accuracy: float
    candidate_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a nonempty string")
        object.__setattr__(
            self, "query_index", _nonnegative_integer(self.query_index, "query_index")
        )
        for name in ("pass_at_1", "pass_at_2", "shape_accuracy"):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
            object.__setattr__(self, name, bool(value))
        if self.pass_at_1 and not self.pass_at_2:
            raise ValueError("pass_at_2 cannot be false when pass_at_1 is true")
        pixel = float(self.valid_cell_pixel_accuracy)
        if not math.isfinite(pixel) or not 0.0 <= pixel <= 1.0:
            raise ValueError("valid_cell_pixel_accuracy must be finite and in [0, 1]")
        object.__setattr__(self, "valid_cell_pixel_accuracy", pixel)
        count = _nonnegative_integer(self.candidate_count, "candidate_count")
        if count not in (1, 2):
            raise ValueError("candidate_count must be one or two")
        object.__setattr__(self, "candidate_count", count)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe query-score mapping.

        Returns
        -------
        dict
            Exact pass indicators and explicitly labelled diagnostics.
        """
        return {
            "task_id": self.task_id,
            "query_index": self.query_index,
            "pass_at_1": self.pass_at_1,
            "pass_at_2": self.pass_at_2,
            "shape_accuracy_diagnostic": self.shape_accuracy,
            "valid_cell_pixel_accuracy_diagnostic": self.valid_cell_pixel_accuracy,
            "candidate_count": self.candidate_count,
        }


def _top_two(values: FloatArray) -> tuple[int, int, float]:
    order = np.argsort(-values, kind="stable")
    first, second = int(order[0]), int(order[1])
    return first, second, float(values[first] - values[second])


def _log_softmax_choice(values: FloatArray, index: int) -> float:
    maximum = float(np.max(values))
    shifted = values - maximum
    return float(shifted[index] - np.log(np.sum(np.exp(shifted))))


def _candidate_log_probability(
    logits: OutputLogits, height_index: int, width_index: int, grid: GridArray
) -> float:
    score = _log_softmax_choice(np.asarray(logits.height), height_index)
    score += _log_softmax_choice(np.asarray(logits.width), width_index)
    colors = np.asarray(logits.colors)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            score += _log_softmax_choice(colors[row, column], int(grid[row, column]))
    return score


def decode_candidates(
    logits: OutputLogits,
    max_candidates: int = 2,
) -> tuple[DecodedCandidate, ...]:
    """Decode joint argmax and one deterministic runner-up ARC grid.

    Candidate two flips the globally smallest top-two logit margin among the
    height, width, and candidate-one valid cells. Ties prefer height, then
    width, then row-major cells. The changed grid is regenerated from the same
    color logits, so a shape alternative is a genuine complete ARC candidate.

    Parameters
    ----------
    logits : OutputLogits
        Validated output heads for one query and one checkpoint.
    max_candidates : int, default=2
        One or two candidates to retain.

    Returns
    -------
    tuple of DecodedCandidate
        One or two distinct candidates in rank order.

    Raises
    ------
    ValueError
        If ``max_candidates`` is not one or two.
    TypeError
        If ``logits`` is not :class:`OutputLogits`.
    """
    if not isinstance(logits, OutputLogits):
        raise TypeError("logits must be an OutputLogits instance")
    if (
        isinstance(max_candidates, (bool, np.bool_))
        or not isinstance(max_candidates, Integral)
        or int(max_candidates) not in (1, 2)
    ):
        raise ValueError("max_candidates must be one or two")
    height_first, height_second, height_margin = _top_two(np.asarray(logits.height))
    width_first, width_second, width_margin = _top_two(np.asarray(logits.width))
    height, width = height_first + 1, width_first + 1
    color_logits = np.asarray(logits.colors)
    first_grid = np.argmax(color_logits[:height, :width], axis=-1).astype(np.int8)
    first = DecodedCandidate(
        first_grid,
        log_probability=_candidate_log_probability(
            logits, height_first, width_first, first_grid
        ),
    )
    if int(max_candidates) == 1:
        return (first,)

    alternatives: list[tuple[float, int, str, int, int, int]] = [
        (height_margin, 0, "height", -1, -1, height_second),
        (width_margin, 1, "width", -1, -1, width_second),
    ]
    order = 2
    for row in range(height):
        for column in range(width):
            _, second_color, margin = _top_two(color_logits[row, column])
            alternatives.append((margin, order, "cell", row, column, second_color))
            order += 1
    _, _, kind, row, column, replacement = min(
        alternatives, key=lambda item: (item[0], item[1])
    )
    second_height_index = height_first
    second_width_index = width_first
    changed_decision: str
    if kind == "height":
        second_height_index = replacement
        changed_decision = "height"
    elif kind == "width":
        second_width_index = replacement
        changed_decision = "width"
    else:
        changed_decision = f"cell:{row},{column}"
    second_height, second_width = second_height_index + 1, second_width_index + 1
    second_grid = np.argmax(
        color_logits[:second_height, :second_width], axis=-1
    ).astype(np.int8)
    if kind == "cell":
        second_grid[row, column] = replacement
    second = DecodedCandidate(
        second_grid,
        changed_decision=changed_decision,
        log_probability=_candidate_log_probability(
            logits, second_height_index, second_width_index, second_grid
        ),
    )
    if np.array_equal(first.grid, second.grid):
        return (first,)
    return first, second


def select_checkpoint_candidates(
    logits_by_checkpoint: Mapping[int, OutputLogits],
    *,
    latest_checkpoint: int,
    sweep_size: int = _AXIS_SIZE,
) -> tuple[SelectedModelCandidate, SelectedModelCandidate]:
    """Select two target-free neural candidates across refinement checkpoints.

    Candidate one is the joint argmax at ``latest_checkpoint``. Candidate two
    is the joint argmax from the newest earlier completed sweep whose grid is
    distinct from candidate one. If no earlier sweep supplies a distinct grid,
    candidate two is the deterministic logit runner-up at the latest sweep.

    Checkpoint zero may be supplied as a pre-refinement diagnostic but is never
    eligible for submission. Selection depends only on validated
    :class:`OutputLogits` and checkpoint order; targets and rule outputs are not
    accepted by this API.

    Parameters
    ----------
    logits_by_checkpoint : mapping of int to OutputLogits
        Model logits keyed by checkpoint. Positive checkpoints must be complete
        sweeps no later than ``latest_checkpoint``.
    latest_checkpoint : int
        Positive completed sweep used for candidate one and fallback decoding.
    sweep_size : int, default=30
        Positive number of row ticks in one complete refinement sweep.

    Returns
    -------
    tuple of SelectedModelCandidate
        Exactly two distinct model candidates in submission order.

    Raises
    ------
    TypeError
        If the checkpoint collection is not a mapping or a value is not
        :class:`OutputLogits`.
    ValueError
        If checkpoint metadata is malformed, the latest checkpoint is absent,
        or the latest logits cannot supply a distinct fallback runner-up.
    """

    if not isinstance(logits_by_checkpoint, Mapping):
        raise TypeError("logits_by_checkpoint must be a mapping")
    latest = _nonnegative_integer(latest_checkpoint, "latest_checkpoint")
    sweep = _nonnegative_integer(sweep_size, "sweep_size")
    if sweep == 0:
        raise ValueError("sweep_size must be positive")
    if latest == 0 or latest % sweep:
        raise ValueError("latest_checkpoint must identify a completed sweep")

    validated: dict[int, OutputLogits] = {}
    for raw_checkpoint, logits in logits_by_checkpoint.items():
        checkpoint = _nonnegative_integer(raw_checkpoint, "checkpoint")
        if checkpoint > latest:
            raise ValueError("checkpoint history contains a checkpoint after latest")
        if checkpoint and checkpoint % sweep:
            raise ValueError("positive checkpoints must identify completed sweeps")
        if not isinstance(logits, OutputLogits):
            raise TypeError("checkpoint values must be OutputLogits")
        validated[checkpoint] = logits
    if latest not in validated:
        raise ValueError("latest checkpoint is absent from checkpoint history")

    latest_logits = validated[latest]
    latest_argmax = decode_candidates(latest_logits, max_candidates=1)[0]
    first = SelectedModelCandidate(
        latest_argmax,
        source_checkpoint=latest,
        selection_role="latest_sweep_joint_argmax",
    )
    for checkpoint in sorted(
        (value for value in validated if 0 < value < latest), reverse=True
    ):
        earlier = decode_candidates(validated[checkpoint], max_candidates=1)[0]
        if not np.array_equal(earlier.grid, latest_argmax.grid):
            return first, SelectedModelCandidate(
                earlier,
                source_checkpoint=checkpoint,
                selection_role="earlier_sweep_joint_argmax",
            )

    latest_candidates = decode_candidates(latest_logits, max_candidates=2)
    if len(latest_candidates) != 2 or np.array_equal(
        latest_argmax.grid, latest_candidates[1].grid
    ):
        raise ValueError("latest logits did not produce a distinct runner-up")
    return first, SelectedModelCandidate(
        latest_candidates[1],
        source_checkpoint=latest,
        selection_role="latest_sweep_logit_runner_up",
    )


def _distinct_candidate_grids(
    candidates: Sequence[DecodedCandidate | ArrayLike],
) -> tuple[GridArray, ...]:
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("candidates must be a sequence of one or two ARC grids")
    if not 1 <= len(candidates) <= 2:
        raise ValueError("candidates must contain one or two ARC grids")
    grids: list[GridArray] = []
    for index, candidate in enumerate(candidates):
        value = candidate.grid if isinstance(candidate, DecodedCandidate) else candidate
        grid = _grid_array(value, f"candidate {index + 1}")
        if not any(np.array_equal(grid, prior) for prior in grids):
            grids.append(grid)
    return tuple(grids)


def score_query_candidates(
    candidates: Sequence[DecodedCandidate | ArrayLike],
    target: ArrayLike,
    *,
    task_id: str,
    query_index: int,
) -> QueryScore:
    """Score at most two candidates with exact ARC semantics.

    Parameters
    ----------
    candidates : sequence
        One or two candidates, usually from :func:`decode_candidates`.
        Duplicate candidates are removed before pass@2 is computed.
    target : array-like
        Held-out output grid, used only by this scorer.
    task_id : str
        Task identity for later strict task aggregation.
    query_index : int
        Zero-based query index within ``task_id``.

    Returns
    -------
    QueryScore
        Exact pass@1/pass@2 and candidate-one diagnostics.

    Raises
    ------
    ValueError
        If the target, candidates, identity, or index is invalid.
    """
    grids = _distinct_candidate_grids(candidates)
    truth = _grid_array(target, "target grid")

    exact = [
        grid.shape == truth.shape and np.array_equal(grid, truth) for grid in grids
    ]
    first = grids[0]
    overlap_height = min(first.shape[0], truth.shape[0])
    overlap_width = min(first.shape[1], truth.shape[1])
    matching = np.sum(
        first[:overlap_height, :overlap_width] == truth[:overlap_height, :overlap_width]
    )
    return QueryScore(
        task_id=task_id,
        query_index=query_index,
        pass_at_1=bool(exact[0]),
        pass_at_2=bool(any(exact)),
        shape_accuracy=first.shape == truth.shape,
        valid_cell_pixel_accuracy=float(matching / truth.size),
        candidate_count=len(grids),
    )


def aggregate_arc_metrics(scores: Sequence[QueryScore]) -> dict[str, object]:
    """Aggregate query scores and conjunctive strict task pass rates.

    Parameters
    ----------
    scores : sequence of QueryScore
        Exactly one score for every evaluated task/query pair.

    Returns
    -------
    dict
        Query pass rates, strict task pass rates, labelled diagnostics, and
        per-task conjunctive outcomes.

    Raises
    ------
    ValueError
        If no scores are provided, an item is not a ``QueryScore``, or a task
        and query index occurs more than once.
    """
    if not scores:
        raise ValueError("scores must contain at least one query")
    grouped: dict[str, list[QueryScore]] = defaultdict(list)
    identities: set[tuple[str, int]] = set()
    for score in scores:
        if not isinstance(score, QueryScore):
            raise ValueError("scores must contain only QueryScore instances")
        identity = (score.task_id, score.query_index)
        if identity in identities:
            raise ValueError(f"duplicate task/query score {identity!r}")
        identities.add(identity)
        grouped[score.task_id].append(score)

    task_results = {
        task_id: {
            "query_count": len(task_scores),
            "pass_at_1": all(score.pass_at_1 for score in task_scores),
            "pass_at_2": all(score.pass_at_2 for score in task_scores),
        }
        for task_id, task_scores in sorted(grouped.items())
    }
    query_count = len(scores)
    task_count = len(task_results)
    return {
        "query_count": query_count,
        "task_count": task_count,
        "query_pass_at_1": float(
            sum(score.pass_at_1 for score in scores) / query_count
        ),
        "query_pass_at_2": float(
            sum(score.pass_at_2 for score in scores) / query_count
        ),
        "strict_task_pass_at_1": float(
            sum(result["pass_at_1"] for result in task_results.values()) / task_count
        ),
        "strict_task_pass_at_2": float(
            sum(result["pass_at_2"] for result in task_results.values()) / task_count
        ),
        "shape_accuracy_diagnostic": float(
            sum(score.shape_accuracy for score in scores) / query_count
        ),
        "valid_cell_pixel_accuracy_diagnostic": float(
            sum(score.valid_cell_pixel_accuracy for score in scores) / query_count
        ),
        "tasks": task_results,
    }


def _selected_candidate_from_record(value: object, name: str) -> SelectedModelCandidate:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a candidate provenance mapping")
    required = {
        "height",
        "width",
        "grid",
        "changed_decision",
        "log_probability",
        "provenance",
        "source_checkpoint",
        "selection_role",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"{name} is missing provenance fields {missing}")
    if value["provenance"] != "model":
        raise ValueError(f"{name} must carry model provenance")
    grid = _grid_array(value["grid"], f"{name} grid")
    height = _nonnegative_integer(value["height"], f"{name} height")
    width = _nonnegative_integer(value["width"], f"{name} width")
    if (height, width) != grid.shape:
        raise ValueError(f"{name} declared shape does not match its grid")
    decoded = DecodedCandidate(
        grid,
        changed_decision=value["changed_decision"],  # type: ignore[arg-type]
        log_probability=value["log_probability"],  # type: ignore[arg-type]
    )
    role = value["selection_role"]
    if not isinstance(role, str):
        raise ValueError(f"{name} selection_role is invalid")
    return SelectedModelCandidate(
        decoded,
        source_checkpoint=value["source_checkpoint"],  # type: ignore[arg-type]
        selection_role=cast(SelectionRole, role),
    )


def _query_score_from_record(value: object, name: str) -> QueryScore:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} score must be a mapping")
    required = {
        "task_id",
        "query_index",
        "pass_at_1",
        "pass_at_2",
        "shape_accuracy_diagnostic",
        "valid_cell_pixel_accuracy_diagnostic",
        "candidate_count",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"{name} score is missing fields {missing}")
    return QueryScore(
        task_id=value["task_id"],  # type: ignore[arg-type]
        query_index=value["query_index"],  # type: ignore[arg-type]
        pass_at_1=value["pass_at_1"],  # type: ignore[arg-type]
        pass_at_2=value["pass_at_2"],  # type: ignore[arg-type]
        shape_accuracy=value["shape_accuracy_diagnostic"],  # type: ignore[arg-type]
        valid_cell_pixel_accuracy=value[  # type: ignore[arg-type]
            "valid_cell_pixel_accuracy_diagnostic"
        ],
        candidate_count=value["candidate_count"],  # type: ignore[arg-type]
    )


def assess_model_only_completion(
    query_records: Sequence[Mapping[str, object]],
    expected_queries_by_task: Mapping[str, int],
) -> dict[str, object]:
    """Assess the strict 160-of-400 model-only ARC completion gate.

    A task is exact only when every official query for that task has exact
    ``pass_at_2`` scorer evidence. The expected-query manifest fixes the full
    evaluation population and makes missing, duplicate, or unexpected queries
    fail closed. Candidate records must contain exactly two distinct neural
    candidates produced by the checkpoint-selection policy.

    Parameters
    ----------
    query_records : sequence of mappings
        Retained per-query records containing model-only candidate provenance
        and a JSON-safe :meth:`QueryScore.to_dict` score.
    expected_queries_by_task : mapping of str to int
        Official query count for each of exactly 400 evaluated ARC tasks.

    Returns
    -------
    dict
        JSON-safe task outcomes, exact task count, strict pass@2 rate, fixed
        target thresholds, and completion status.

    Raises
    ------
    ValueError
        If the task/query manifest or any score/candidate record is incomplete,
        duplicated, inconsistent, non-model, or invalid.
    """

    if not isinstance(expected_queries_by_task, Mapping):
        raise ValueError("expected_queries_by_task must be a mapping")
    if len(expected_queries_by_task) != _MODEL_ONLY_REQUIRED_TASK_COUNT:
        raise ValueError("completion requires exactly 400 expected ARC tasks")
    expected: dict[str, int] = {}
    for task_id, raw_count in expected_queries_by_task.items():
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("expected task ids must be nonempty strings")
        count = _nonnegative_integer(raw_count, f"query count for {task_id}")
        if count == 0:
            raise ValueError("every expected task must have an official query")
        expected[task_id] = count
    if not isinstance(query_records, Sequence) or isinstance(
        query_records, (str, bytes)
    ):
        raise ValueError("query_records must be a sequence")

    expected_identities = {
        (task_id, query_index)
        for task_id, count in expected.items()
        for query_index in range(count)
    }
    identities: set[tuple[str, int]] = set()
    scores_by_task: dict[str, list[QueryScore]] = defaultdict(list)
    for record_index, record in enumerate(query_records):
        name = f"query_records[{record_index}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{name} must be a mapping")
        if record.get("primary_candidate_mode") != "model_only":
            raise ValueError(f"{name} primary_candidate_mode must be model_only")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or task_id not in expected:
            raise ValueError(f"{name} has an unexpected task_id")
        query_index = _nonnegative_integer(record.get("query_index"), "query_index")
        identity = (task_id, query_index)
        if identity not in expected_identities:
            raise ValueError(f"{name} has an unexpected official query")
        if identity in identities:
            raise ValueError(f"duplicate task/query record {identity!r}")
        identities.add(identity)

        candidate_values = record.get("candidates")
        if not isinstance(candidate_values, Sequence) or isinstance(
            candidate_values, (str, bytes)
        ):
            raise ValueError(f"{name} candidates must be a sequence")
        if len(candidate_values) != 2:
            raise ValueError(f"{name} must contain exactly two candidates")
        first = _selected_candidate_from_record(candidate_values[0], "candidate 1")
        second = _selected_candidate_from_record(candidate_values[1], "candidate 2")
        latest = first.source_checkpoint
        if first.selection_role != "latest_sweep_joint_argmax":
            raise ValueError("candidate 1 selection_role must identify latest argmax")
        if latest % _AXIS_SIZE:
            raise ValueError("candidate 1 source checkpoint must be a completed sweep")
        if second.selection_role == "earlier_sweep_joint_argmax":
            if not 0 < second.source_checkpoint < latest:
                raise ValueError("earlier candidate checkpoint must precede latest")
            if second.source_checkpoint % _AXIS_SIZE:
                raise ValueError(
                    "earlier candidate checkpoint must be a completed sweep"
                )
        elif second.selection_role == "latest_sweep_logit_runner_up":
            if second.source_checkpoint != latest:
                raise ValueError("runner-up checkpoint must equal latest checkpoint")
        else:
            raise ValueError("candidate 2 selection_role is invalid")
        if np.array_equal(first.candidate.grid, second.candidate.grid):
            raise ValueError("model-only candidates must be distinct")

        score = _query_score_from_record(record.get("score"), name)
        if (score.task_id, score.query_index) != identity:
            raise ValueError(f"{name} score identity does not match its record")
        if score.candidate_count != 2:
            raise ValueError(f"{name} score candidate_count must be two")
        scores_by_task[task_id].append(score)

    missing = expected_identities - identities
    if missing:
        raise ValueError(f"missing official query records: {len(missing)}")
    task_results = {
        task_id: {
            "query_count": expected[task_id],
            "pass_at_2": all(score.pass_at_2 for score in scores_by_task[task_id]),
        }
        for task_id in sorted(expected)
    }
    exact_task_count = sum(result["pass_at_2"] for result in task_results.values())
    strict_pass_at_2 = exact_task_count / _MODEL_ONLY_REQUIRED_TASK_COUNT
    return {
        "primary_candidate_mode": "model_only",
        "required_task_count": _MODEL_ONLY_REQUIRED_TASK_COUNT,
        "evaluated_task_count": len(task_results),
        "evaluated_query_count": len(identities),
        "required_exact_task_count": _MODEL_ONLY_REQUIRED_EXACT_TASK_COUNT,
        "exact_task_count": exact_task_count,
        "strict_task_pass_at_2": strict_pass_at_2,
        "passed": exact_task_count >= _MODEL_ONLY_REQUIRED_EXACT_TASK_COUNT,
        "tasks": task_results,
    }


def _logit_sequences(
    height: ArrayLike, width: ArrayLike, colors: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    height_array = _finite_array(height, "height logit sequence")
    width_array = _finite_array(width, "width logit sequence")
    color_array = _finite_array(colors, "color logit sequence")
    if height_array.ndim != 2 or height_array.shape[1:] != (_AXIS_SIZE,):
        raise ValueError(
            f"height logit sequence shape must be (steps, 30); got {height_array.shape}"
        )
    steps = height_array.shape[0]
    if steps < 1:
        raise ValueError("logit sequences must contain at least one step")
    if width_array.shape != (steps, _AXIS_SIZE):
        raise ValueError(
            f"width logit sequence shape must be ({steps}, 30); got {width_array.shape}"
        )
    expected_colors = (steps, _AXIS_SIZE, _AXIS_SIZE, _COLOR_COUNT)
    if color_array.shape != expected_colors:
        raise ValueError(
            f"color logit sequence shape must be {expected_colors}; "
            f"got {color_array.shape}"
        )
    return height_array, width_array, color_array


def _state_sequences(
    spikes: ArrayLike, voltages: ArrayLike, expected_steps: int | None = None
) -> tuple[NDArray[np.bool_], FloatArray]:
    try:
        spike_array = np.asarray(spikes)
    except (TypeError, ValueError) as error:
        raise ValueError("spikes must be a rectangular binary array") from error
    if spike_array.ndim != 2 or any(size < 1 for size in spike_array.shape):
        raise ValueError(
            f"spikes shape must be (steps, neurons) with nonempty axes; got {spike_array.shape}"
        )
    if expected_steps is not None and spike_array.shape[0] != expected_steps:
        raise ValueError(
            f"spikes have {spike_array.shape[0]} steps; expected {expected_steps}"
        )
    if not (
        np.issubdtype(spike_array.dtype, np.bool_)
        or np.issubdtype(spike_array.dtype, np.number)
    ):
        raise ValueError("spikes must contain binary numeric values")
    if not np.all(np.isfinite(spike_array)) or not np.all(
        (spike_array == 0) | (spike_array == 1)
    ):
        raise ValueError("spikes must contain only finite binary values")
    spike_result = np.ascontiguousarray(spike_array, dtype=np.bool_)
    spike_result.setflags(write=False)
    voltage_result = _finite_array(voltages, "voltages")
    if voltage_result.shape != spike_result.shape:
        raise ValueError(
            "voltages shape must match spikes; got "
            f"voltages {voltage_result.shape}, spikes {spike_result.shape}"
        )
    return spike_result, voltage_result


def _step_indices(values: Sequence[int] | ArrayLike | None, count: int) -> list[int]:
    if values is None:
        return list(range(count))
    array = np.asarray(values)
    if (
        array.shape != (count,)
        or not np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"step_indices must contain {count} integers")
    result = [int(value) for value in array]
    if any(value < 0 for value in result) or any(
        right <= left for left, right in zip(result, result[1:])
    ):
        raise ValueError("step_indices must be nonnegative and strictly increasing")
    return result


def _changed_grid_cells(previous: GridArray, current: GridArray) -> tuple[int, float]:
    height = max(previous.shape[0], current.shape[0])
    width = max(previous.shape[1], current.shape[1])
    previous_canvas = np.full((height, width), -1, dtype=np.int16)
    current_canvas = np.full((height, width), -1, dtype=np.int16)
    previous_canvas[: previous.shape[0], : previous.shape[1]] = previous
    current_canvas[: current.shape[0], : current.shape[1]] = current
    union = (previous_canvas >= 0) | (current_canvas >= 0)
    changed = int(np.sum((previous_canvas != current_canvas) & union))
    return changed, float(changed / np.sum(union))


def _decision_uncertainty(
    logits: OutputLogits, candidate: DecodedCandidate
) -> tuple[float, float]:
    decisions = [np.asarray(logits.height), np.asarray(logits.width)]
    grid = np.asarray(candidate.grid)
    colors = np.asarray(logits.colors)
    decisions.extend(
        colors[row, column]
        for row in range(grid.shape[0])
        for column in range(grid.shape[1])
    )
    entropies: list[float] = []
    margins: list[float] = []
    for decision in decisions:
        maximum = float(np.max(decision))
        probabilities = np.exp(decision - maximum)
        probabilities /= np.sum(probabilities)
        positive = probabilities > 0.0
        entropies.append(
            float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        )
        margins.append(_top_two(decision)[2])
    return float(np.mean(entropies)), float(np.mean(margins))


def _state_hash(
    spikes: NDArray[np.bool_], voltages: FloatArray, *additional: FloatArray
) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(spikes).tobytes())
    digest.update(np.ascontiguousarray(voltages).tobytes())
    for state in additional:
        digest.update(np.ascontiguousarray(state).tobytes())
    return digest.hexdigest()


def _bounded_fraction(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def analyze_latent_trajectory(
    height_logits: ArrayLike,
    width_logits: ArrayLike,
    color_logits: ArrayLike,
    spikes: ArrayLike,
    voltages: ArrayLike,
    *,
    feedforward_current: ArrayLike | None = None,
    recurrent_current: ArrayLike | None = None,
    target: ArrayLike | None = None,
    task_id: str = "trajectory",
    query_index: int = 0,
    step_indices: Sequence[int] | ArrayLike | None = None,
    convergence_atol: float = 0.0,
    silence_rate: float = 0.001,
    saturation_rate: float = 0.95,
    raster_neurons: int = 64,
) -> dict[str, object]:
    """Decode and measure every state in one query's latent rollout.

    Parameters
    ----------
    height_logits, width_logits, color_logits : array-like
        Unbatched output-head stacks shaped ``(steps, 30)``, ``(steps, 30)``,
        and ``(steps, 30, 30, 10)``.
    spikes, voltages : array-like
        Matched state stacks shaped ``(steps, neurons)``. Step zero may be the
        query-terminal state and later rows zero-input recurrent states.
    feedforward_current, recurrent_current : array-like or None, default=None
        Optional separate Expon-current stacks shaped ``(steps, neurons)``.
        Supply both or neither. They participate in state hashes, displacement,
        and convergence.
    target : array-like or None, default=None
        Optional held-out grid for exact provisional scoring.
    task_id : str, default="trajectory"
        Task identity attached to optional scores.
    query_index : int, default=0
        Query index attached to optional scores.
    step_indices : sequence of int or None, default=None
        Nonnegative, strictly increasing external checkpoint labels.
    convergence_atol : float, default=0.0
        Absolute voltage tolerance. Convergence also requires identical spikes
        and an unchanged candidate-one grid.
    silence_rate : float, default=0.001
        Inclusive firing-rate threshold for near-silence.
    saturation_rate : float, default=0.95
        Inclusive firing-rate threshold for near-saturation.
    raster_neurons : int, default=64
        Prefix width retained as active neuron indices for a bounded raster.

    Returns
    -------
    dict
        JSON-safe per-step provisional outputs, exact scores when available,
        uncertainty, activity, voltage, displacement, convergence, and flags.

    Raises
    ------
    ValueError
        If logits, state stacks, thresholds, indices, or target are invalid.
    """
    height, width, colors = _logit_sequences(height_logits, width_logits, color_logits)
    spike_array, voltage_array = _state_sequences(
        spikes, voltages, expected_steps=height.shape[0]
    )
    if (feedforward_current is None) != (recurrent_current is None):
        raise ValueError(
            "feedforward_current and recurrent_current must be provided together"
        )
    feedforward_array: FloatArray | None = None
    recurrent_array: FloatArray | None = None
    if feedforward_current is not None and recurrent_current is not None:
        feedforward_array = _finite_array(
            feedforward_current, "feedforward synaptic current"
        )
        recurrent_array = _finite_array(recurrent_current, "recurrent synaptic current")
        if feedforward_array.shape != spike_array.shape:
            raise ValueError(
                "feedforward synaptic current shape must match spikes; got "
                f"{feedforward_array.shape} and {spike_array.shape}"
            )
        if recurrent_array.shape != spike_array.shape:
            raise ValueError(
                "recurrent synaptic current shape must match spikes; got "
                f"{recurrent_array.shape} and {spike_array.shape}"
            )
    labels = _step_indices(step_indices, height.shape[0])
    if (
        isinstance(convergence_atol, (bool, np.bool_))
        or not isinstance(convergence_atol, Real)
        or not math.isfinite(float(convergence_atol))
        or float(convergence_atol) < 0.0
    ):
        raise ValueError("convergence_atol must be finite and nonnegative")
    silence = _bounded_fraction(silence_rate, "silence_rate")
    saturation = _bounded_fraction(saturation_rate, "saturation_rate")
    if silence >= saturation:
        raise ValueError("silence_rate must be smaller than saturation_rate")
    raster_width = _nonnegative_integer(raster_neurons, "raster_neurons")
    if raster_width < 1:
        raise ValueError("raster_neurons must be positive")
    truth = None if target is None else _grid_array(target, "target grid")

    reports: list[dict[str, object]] = []
    previous_grid: GridArray | None = None
    previous_spikes: NDArray[np.bool_] | None = None
    previous_voltage: FloatArray | None = None
    previous_feedforward: FloatArray | None = None
    previous_recurrent: FloatArray | None = None
    for offset, step in enumerate(labels):
        output = OutputLogits(height[offset], width[offset], colors[offset])
        candidates = decode_candidates(output)
        first_grid = np.asarray(candidates[0].grid)
        entropy, margin = _decision_uncertainty(output, candidates[0])
        spike_row = spike_array[offset]
        voltage_row = voltage_array[offset]
        feedforward_row = (
            None if feedforward_array is None else feedforward_array[offset]
        )
        recurrent_row = None if recurrent_array is None else recurrent_array[offset]
        rate = float(np.mean(spike_row))
        if previous_grid is None:
            changed_count: int | None = None
            changed_fraction: float | None = None
            spike_hamming: int | None = None
            spike_hamming_fraction: float | None = None
            voltage_displacement: float | None = None
            feedforward_displacement: float | None = None
            recurrent_displacement: float | None = None
            converged = False
        else:
            changed_count, changed_fraction = _changed_grid_cells(
                previous_grid, first_grid
            )
            spike_hamming = int(np.count_nonzero(spike_row != previous_spikes))
            spike_hamming_fraction = float(spike_hamming / spike_row.size)
            voltage_displacement = float(np.linalg.norm(voltage_row - previous_voltage))
            feedforward_displacement = (
                None
                if feedforward_row is None or previous_feedforward is None
                else float(np.linalg.norm(feedforward_row - previous_feedforward))
            )
            recurrent_displacement = (
                None
                if recurrent_row is None or previous_recurrent is None
                else float(np.linalg.norm(recurrent_row - previous_recurrent))
            )
            converged = bool(
                changed_count == 0
                and spike_hamming == 0
                and np.allclose(
                    voltage_row,
                    previous_voltage,
                    rtol=0.0,
                    atol=float(convergence_atol),
                )
                and (
                    feedforward_row is None
                    or (
                        previous_feedforward is not None
                        and np.allclose(
                            feedforward_row,
                            previous_feedforward,
                            rtol=0.0,
                            atol=float(convergence_atol),
                        )
                    )
                )
                and (
                    recurrent_row is None
                    or (
                        previous_recurrent is not None
                        and np.allclose(
                            recurrent_row,
                            previous_recurrent,
                            rtol=0.0,
                            atol=float(convergence_atol),
                        )
                    )
                )
            )
        report: dict[str, object] = {
            "step": step,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "changed_cell_count": changed_count,
            "changed_cell_fraction": changed_fraction,
            "predictive_entropy": entropy,
            "top_two_logit_margin": margin,
            "spike_count": int(np.sum(spike_row)),
            "firing_rate": rate,
            "raster_active_indices": np.flatnonzero(
                spike_row[: min(raster_width, spike_row.size)]
            )
            .astype(int)
            .tolist(),
            "voltage_mean": float(np.mean(voltage_row)),
            "voltage_std": float(np.std(voltage_row)),
            "voltage_mean_absolute": float(np.mean(np.abs(voltage_row))),
            "voltage_l2": float(np.linalg.norm(voltage_row)),
            "spike_hamming_displacement": spike_hamming,
            "spike_hamming_fraction": spike_hamming_fraction,
            "voltage_l2_displacement": voltage_displacement,
            "feedforward_current_mean_absolute": (
                None
                if feedforward_row is None
                else float(np.mean(np.abs(feedforward_row)))
            ),
            "feedforward_current_l2": (
                None
                if feedforward_row is None
                else float(np.linalg.norm(feedforward_row))
            ),
            "feedforward_current_l2_displacement": feedforward_displacement,
            "recurrent_current_mean_absolute": (
                None if recurrent_row is None else float(np.mean(np.abs(recurrent_row)))
            ),
            "recurrent_current_l2": (
                None if recurrent_row is None else float(np.linalg.norm(recurrent_row))
            ),
            "recurrent_current_l2_displacement": recurrent_displacement,
            "converged": converged,
            "near_silence": rate <= silence,
            "near_saturation": rate >= saturation,
            "state_sha256": _state_hash(
                spike_row,
                voltage_row,
                *(
                    ()
                    if feedforward_row is None or recurrent_row is None
                    else (feedforward_row, recurrent_row)
                ),
            ),
        }
        if truth is not None:
            report["score"] = score_query_candidates(
                candidates,
                truth,
                task_id=task_id,
                query_index=query_index,
            ).to_dict()
        reports.append(report)
        previous_grid = first_grid
        previous_spikes = spike_row
        previous_voltage = voltage_row
        previous_feedforward = feedforward_row
        previous_recurrent = recurrent_row

    rates = [float(report["firing_rate"]) for report in reports]
    entropy_values = [float(report["predictive_entropy"]) for report in reports]
    voltage_norms = [float(report["voltage_l2"]) for report in reports]
    return {
        "step_count": len(reports),
        "neuron_count": int(spike_array.shape[1]),
        "steps": reports,
        "converged_steps": [
            int(report["step"]) for report in reports if report["converged"]
        ],
        "near_silence_steps": [
            int(report["step"]) for report in reports if report["near_silence"]
        ],
        "near_saturation_steps": [
            int(report["step"]) for report in reports if report["near_saturation"]
        ],
        "firing_rate_distribution": {
            "minimum": min(rates),
            "mean": float(np.mean(rates)),
            "maximum": max(rates),
        },
        "predictive_entropy_distribution": {
            "minimum": min(entropy_values),
            "mean": float(np.mean(entropy_values)),
            "maximum": max(entropy_values),
        },
        "voltage_l2_distribution": {
            "minimum": min(voltage_norms),
            "mean": float(np.mean(voltage_norms)),
            "maximum": max(voltage_norms),
        },
    }


def _byte_identical(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype.str == right.dtype.str
        and np.ascontiguousarray(left).tobytes()
        == np.ascontiguousarray(right).tobytes()
    )


def _flatten_numeric(value: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten_numeric(item, path))
        elif isinstance(item, (bool, np.bool_)):
            result[path] = float(bool(item))
        elif isinstance(item, Real) and math.isfinite(float(item)):
            result[path] = float(item)
    return result


def compare_control_trajectories(
    intact_spikes: ArrayLike,
    intact_voltages: ArrayLike,
    control_spikes: ArrayLike,
    control_voltages: ArrayLike,
    *,
    control_name: str,
    intact_scores: Mapping[str, Any] | None = None,
    control_scores: Mapping[str, Any] | None = None,
    intact_synaptic_currents: Mapping[str, ArrayLike] | None = None,
    control_synaptic_currents: Mapping[str, ArrayLike] | None = None,
) -> dict[str, object]:
    """Compare one frozen control with its matched intact trajectory.

    Parameters
    ----------
    intact_spikes, intact_voltages : array-like
        Intact state stacks shaped ``(steps, neurons)``.
    control_spikes, control_voltages : array-like
        Matched control stacks with the same shapes.
    control_name : str
        Human-readable intervention identifier.
    intact_scores, control_scores : mapping or None, default=None
        Optional matched metric mappings. Numeric leaves are reported as
        ``control - intact`` deltas. Both mappings must be present together.
    intact_synaptic_currents, control_synaptic_currents : mapping or None, default=None
        Optional matched named current stacks shaped ``(steps, features)``.
        Both mappings must be present together and carry identical names.

    Returns
    -------
    dict
        Per-step state differences, exact-byte null evidence, score deltas,
        and a plain-language causal interpretation.

    Raises
    ------
    ValueError
        If names, state shapes, binary spikes, or score pairing are invalid.
    """
    if not isinstance(control_name, str) or not control_name:
        raise ValueError("control_name must be a nonempty string")
    raw_intact_spikes = np.asarray(intact_spikes)
    raw_intact_voltages = np.asarray(intact_voltages)
    raw_control_spikes = np.asarray(control_spikes)
    raw_control_voltages = np.asarray(control_voltages)
    intact_spike_array, intact_voltage_array = _state_sequences(
        raw_intact_spikes, raw_intact_voltages
    )
    control_spike_array, control_voltage_array = _state_sequences(
        raw_control_spikes, raw_control_voltages
    )
    if control_spike_array.shape != intact_spike_array.shape:
        raise ValueError(
            "control state shape must match intact state shape; got "
            f"control {control_spike_array.shape}, intact {intact_spike_array.shape}"
        )

    byte_identical_steps = [
        _byte_identical(raw_intact_spikes[index], raw_control_spikes[index])
        and _byte_identical(raw_intact_voltages[index], raw_control_voltages[index])
        for index in range(intact_spike_array.shape[0])
    ]
    hamming = np.count_nonzero(
        intact_spike_array != control_spike_array, axis=1
    ).astype(int)
    voltage_distance = np.linalg.norm(
        intact_voltage_array - control_voltage_array, axis=1
    )
    if (intact_synaptic_currents is None) != (control_synaptic_currents is None):
        raise ValueError(
            "intact_synaptic_currents and control_synaptic_currents must be provided together"
        )
    current_distances: dict[str, list[float]] = {}
    current_byte_identity: dict[str, list[bool]] = {}
    if intact_synaptic_currents is not None and control_synaptic_currents is not None:
        if not intact_synaptic_currents or (
            intact_synaptic_currents.keys() != control_synaptic_currents.keys()
        ):
            raise ValueError(
                "matched synaptic current mappings must have identical nonempty names"
            )
        for name in sorted(intact_synaptic_currents):
            raw_intact_current = np.asarray(intact_synaptic_currents[name])
            raw_control_current = np.asarray(control_synaptic_currents[name])
            intact_current = _finite_array(
                raw_intact_current, f"intact {name} synaptic current"
            )
            control_current = _finite_array(
                raw_control_current, f"control {name} synaptic current"
            )
            if (
                intact_current.ndim != 2
                or intact_current.shape[0] != intact_spike_array.shape[0]
                or intact_current.shape[1] < 1
                or control_current.shape != intact_current.shape
            ):
                raise ValueError(
                    f"matched {name} synaptic current shapes must be "
                    f"(steps, features); got intact {intact_current.shape}, "
                    f"control {control_current.shape}"
                )
            identical = [
                _byte_identical(raw_intact_current[index], raw_control_current[index])
                for index in range(intact_spike_array.shape[0])
            ]
            current_byte_identity[name] = identical
            current_distances[name] = np.linalg.norm(
                intact_current - control_current, axis=1
            ).tolist()
            byte_identical_steps = [
                state_identical and current_identical
                for state_identical, current_identical in zip(
                    byte_identical_steps, identical, strict=True
                )
            ]
    causal_null = bool(all(byte_identical_steps))

    if (intact_scores is None) != (control_scores is None):
        raise ValueError("intact_scores and control_scores must be provided together")
    score_deltas: dict[str, float] = {}
    if intact_scores is not None and control_scores is not None:
        intact_flat = _flatten_numeric(intact_scores)
        control_flat = _flatten_numeric(control_scores)
        for key in sorted(intact_flat.keys() & control_flat.keys()):
            score_deltas[key] = control_flat[key] - intact_flat[key]

    interpretation = (
        f"{control_name} is causally null at measured precision: its spike, "
        "voltage, and measured synaptic-current states are byte-identical to "
        "the matched intact trajectory."
        if causal_null
        else f"{control_name} changed measured latent states relative to the matched intact trajectory."
    )
    return {
        "control_name": control_name,
        "causally_null_at_measured_precision": causal_null,
        "state_byte_identical_by_step": byte_identical_steps,
        "spike_hamming_by_step": hamming.tolist(),
        "spike_hamming_fraction_by_step": (
            hamming / intact_spike_array.shape[1]
        ).tolist(),
        "voltage_l2_by_step": voltage_distance.tolist(),
        "synaptic_current_l2_by_step": current_distances,
        "synaptic_current_byte_identical_by_step": current_byte_identity,
        "score_deltas_control_minus_intact": score_deltas,
        "interpretation": interpretation,
    }
