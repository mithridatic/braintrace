"""Direct checkpoint-owned ARC decoding and strict pass-at-one scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

MAX_GRID_SIZE = 30
COLOR_COUNT = 10
STRICT_TASK_TARGET = 16


@dataclass(frozen=True)
class DirectPredictionLogits:
    """Hold direct shape and cell logits emitted by one model execution.

    Parameters
    ----------
    height : numpy.ndarray
        Logits for output heights 1 through 30.
    width : numpy.ndarray
        Logits for output widths 1 through 30.
    colors : numpy.ndarray
        Per-cell colour logits shaped ``(30, 30, 10)``.
    parameter_dependencies : tuple of str
        Ordered checkpoint leaf paths executed by the answer path.
    """

    height: np.ndarray
    width: np.ndarray
    colors: np.ndarray
    parameter_dependencies: tuple[str, ...]


def _logit_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} logits must have shape {shape}.")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} logits must have a floating dtype.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} logits must contain only finite values.")
    return array


def _dependencies(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("parameter_dependencies must be an ordered sequence.")
    dependencies = tuple(value)
    if (
        not dependencies
        or any(not isinstance(name, str) or not name for name in dependencies)
        or len(set(dependencies)) != len(dependencies)
    ):
        raise ValueError(
            "parameter_dependencies must contain unique nonempty leaf paths."
        )
    return dependencies


def _grid(value: object, *, height: object = None, width: object = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not 1 <= array.shape[0] <= MAX_GRID_SIZE or not 1 <= array.shape[1] <= MAX_GRID_SIZE:
        raise ValueError("A predicted grid must have two dimensions from 1 through 30.")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("A predicted grid must contain integer ARC colours.")
    if np.any(array < 0) or np.any(array >= COLOR_COUNT):
        raise ValueError("A predicted grid must contain ARC colours from 0 through 9.")
    if height is not None and height != array.shape[0]:
        raise ValueError("Candidate height must match its grid.")
    if width is not None and width != array.shape[1]:
        raise ValueError("Candidate width must match its grid.")
    return np.asarray(array, dtype=np.int8)


def decode_first_candidate(logits: DirectPredictionLogits) -> dict[str, object]:
    """Decode one greedy ARC grid directly from trained-model logits.

    Parameters
    ----------
    logits : DirectPredictionLogits
        Shape and cell logits returned by an executed BrainTrace network.

    Returns
    -------
    dict
        Canonical direct-model candidate with explicit checkpoint dependencies.

    Raises
    ------
    TypeError
        If ``logits`` is not :class:`DirectPredictionLogits`.
    ValueError
        If logits or dependency paths are invalid.
    """

    if not isinstance(logits, DirectPredictionLogits):
        raise TypeError("logits must be a DirectPredictionLogits instance.")
    height_logits = _logit_array(logits.height, (MAX_GRID_SIZE,), "height")
    width_logits = _logit_array(logits.width, (MAX_GRID_SIZE,), "width")
    color_logits = _logit_array(
        logits.colors, (MAX_GRID_SIZE, MAX_GRID_SIZE, COLOR_COUNT), "color"
    )
    dependencies = _dependencies(logits.parameter_dependencies)
    height = int(np.argmax(height_logits)) + 1
    width = int(np.argmax(width_logits)) + 1
    grid = np.argmax(color_logits[:height, :width], axis=-1).astype(np.int8)
    return {
        "rank": 1,
        "height": height,
        "width": width,
        "grid": grid.tolist(),
        "provenance": "model",
        "dependency_class": "model_checkpoint",
        "proposal_source": "direct_model_logits",
        "ranking_source": "none_single_greedy_candidate",
        "answer_head_version": "direct_model_generation_v1",
        "selection_role": "greedy_argmax",
        "parameter_dependencies": list(dependencies),
    }


def validate_direct_candidate(value: object) -> None:
    """Validate that a candidate is a direct neural argmax serialization.

    Parameters
    ----------
    value : object
        Candidate provenance mapping to validate.

    Raises
    ------
    ValueError
        If the candidate can depend on proposals, rules, forests, or fitters,
        or if its direct-model evidence is malformed.
    """

    if not isinstance(value, Mapping):
        raise TypeError("A direct candidate must be a provenance mapping.")
    required = {
        "rank": 1,
        "provenance": "model",
        "dependency_class": "model_checkpoint",
        "proposal_source": "direct_model_logits",
        "ranking_source": "none_single_greedy_candidate",
        "answer_head_version": "direct_model_generation_v1",
        "selection_role": "greedy_argmax",
    }
    for name, expected in required.items():
        if value.get(name) != expected:
            raise ValueError(
                f"Candidate {name} must prove generation from direct model logits; "
                "forest or external proposals are ineligible."
            )
    forbidden = {
        "forest_log_probability",
        "network_log_probability",
        "combined_log_probability",
        "forest_rank",
        "rule_name",
        "retrieval_source",
    }
    if forbidden.intersection(value):
        raise ValueError("Forest, rule, retrieval, and proposal scores are ineligible.")
    _dependencies(value.get("parameter_dependencies"))
    _grid(value.get("grid"), height=value.get("height"), width=value.get("width"))


def first_prediction_bytes(candidates: Sequence[Mapping[str, object]]) -> bytes:
    """Serialize ordered first-prediction grids without metadata or scores.

    Parameters
    ----------
    candidates : sequence of mappings
        One direct first candidate per query in manifest order.

    Returns
    -------
    bytes
        Count-prefixed shape and row-major colour bytes.
    """

    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be an ordered sequence.")
    payload = bytearray(len(candidates).to_bytes(4, "little", signed=False))
    for candidate in candidates:
        validate_direct_candidate(candidate)
        grid = _grid(
            candidate["grid"],
            height=candidate["height"],
            width=candidate["width"],
        )
        payload.extend(grid.shape[0].to_bytes(1, "little"))
        payload.extend(grid.shape[1].to_bytes(1, "little"))
        payload.extend(np.ascontiguousarray(grid).tobytes())
    return bytes(payload)


def qualifies_strict_task_pass_at_1(count: int) -> bool:
    """Return whether the sole strict ARC acceptance threshold is met.

    Parameters
    ----------
    count : int
        Strictly solved task count on the 400-task evaluation manifest.

    Returns
    -------
    bool
        ``True`` exactly when ``count`` is at least 16.
    """

    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
        raise TypeError("strict_task_pass_at_1_count must be an integer.")
    integer = int(count)
    if not 0 <= integer <= 400:
        raise ValueError("strict_task_pass_at_1_count must be from 0 through 400.")
    return integer >= STRICT_TASK_TARGET


def strict_task_pass_at_1(
    predictions: Sequence[object],
    targets: Sequence[object],
    task_ids: Sequence[str],
) -> dict[str, object]:
    """Score strict task pass-at-one from scorer-only target grids.

    Parameters
    ----------
    predictions : sequence of array-like
        First predicted grids in query-manifest order.
    targets : sequence of array-like
        Held-out scorer-only targets in the same order.
    task_ids : sequence of str
        Task identifier for every query.

    Returns
    -------
    dict
        Strict count, solved task identifiers, and per-task membership.

    Raises
    ------
    ValueError
        If the sequences disagree or contain invalid task identifiers.
    """

    if not len(predictions) == len(targets) == len(task_ids):
        raise ValueError("Predictions, targets, and task_ids must have equal lengths.")
    membership: dict[str, bool] = {}
    for prediction, target, task_id in zip(
        predictions, targets, task_ids, strict=True
    ):
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Every task_id must be a nonempty string.")
        predicted_grid = _grid(prediction)
        target_grid = _grid(target)
        exact = predicted_grid.shape == target_grid.shape and np.array_equal(
            predicted_grid, target_grid
        )
        membership[task_id] = membership.get(task_id, True) and exact
    strict_ids = [task_id for task_id, passed in membership.items() if passed]
    return {
        "strict_task_pass_at_1_count": len(strict_ids),
        "strict_task_ids": strict_ids,
        "task_membership": membership,
    }
