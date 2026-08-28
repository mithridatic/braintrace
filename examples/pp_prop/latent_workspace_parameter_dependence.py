"""Checkpoint-conditioned ARC candidates and causal qualification gates."""

# Ruff's TRY004 conflicts with this module's public fail-closed contract: all
# malformed external evidence is intentionally reported as ValueError.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import brainstate
import msgspec
import numpy as np

try:
    from examples.pp_prop.latent_workspace_analysis import (
        DecodedCandidate,
        OutputLogits,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script import fallback.
    from latent_workspace_analysis import (
        DecodedCandidate,
        OutputLogits,
    )


_EXACT_COUNT_NAMES = (
    "query_pass_at_1_count",
    "query_pass_at_2_count",
    "strict_task_pass_at_1_count",
    "strict_task_pass_at_2_count",
)
_PERTURBATION_NAMES = ("scale", "swap", "reseed")
_ACCEPTED_FULL_PROFILE = {
    "schema_version": 2,
    "protocol_version": 2,
    "neuron_count": 4096,
    "recurrent_edges": 4_194_304,
    "latent_steps": 60,
    "checkpoints": [0, 30, 60],
    "submission_checkpoint": 60,
    "seed": 31337,
    "answer_head": "checkpoint_conditioned",
    "primary_candidate_mode": "model_only",
    "checkpoint_conditioning_coupling": 1.0,
    "requested_device": "gpu",
    "actual_platform": "gpu",
    "smoke": False,
    "structural_only": False,
    "evaluation_task_limit": None,
}


def _positive_finite_real(value: object, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite real number.")
    return float(value)


def _finite_real(value: object, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(msgspec.json.encode(value, order="sorted")).hexdigest()


def _log_softmax_choice(values: np.ndarray, index: int) -> float:
    if not np.all(np.isfinite(values)):
        raise ValueError("Model logits must contain only finite values.")
    maximum = float(np.max(values))
    shifted = np.asarray(values, dtype=np.float64) - maximum
    return float(shifted[index] - np.log(np.sum(np.exp(shifted))))


def _network_candidate_log_probability(
    logits: OutputLogits, candidate: DecodedCandidate
) -> float:
    grid = np.asarray(candidate.grid)
    height, width = grid.shape
    score = _log_softmax_choice(np.asarray(logits.height), height - 1)
    score += _log_softmax_choice(np.asarray(logits.width), width - 1)
    colors = np.asarray(logits.colors, dtype=np.float64)[:height, :width]
    if not np.all(np.isfinite(colors)):
        raise ValueError("Model logits must contain only finite values.")
    maxima = np.max(colors, axis=-1, keepdims=True)
    shifted = colors - maxima
    log_probabilities = shifted - np.log(
        np.sum(np.exp(shifted), axis=-1, keepdims=True)
    )
    selected = np.take_along_axis(log_probabilities, grid[..., None], axis=-1)
    return score + float(np.sum(selected, dtype=np.float64))


def checkpoint_conditioned_candidates(
    demonstration_candidates: Sequence[DecodedCandidate],
    model_logits: OutputLogits,
    coupling: float = 1.0,
) -> tuple[DecodedCandidate, DecodedCandidate]:
    """Rank two target-free proposals with trained-network likelihood.

    The proposal score and the factorized model log likelihood are combined as
    an unweighted product of experts when ``coupling`` is one.  Stable sorting
    preserves the demonstration order only when the combined scores tie.

    Parameters
    ----------
    demonstration_candidates : sequence of DecodedCandidate
        Exactly two distinct target-free proposals.  Their ``log_probability``
        values are the demonstration-model terms.
    model_logits : OutputLogits
        Height, width, and cell-colour logits produced by the executed trained
        network for the same query.
    coupling : float, default=1.0
        Positive finite multiplier on the network log-likelihood term.

    Returns
    -------
    tuple of DecodedCandidate
        The same two grids in descending combined-score order.  Returned
        ``log_probability`` values contain the combined scores.

    Raises
    ------
    TypeError
        If a proposal or ``model_logits`` has the wrong type.
    ValueError
        If there are not exactly two distinct proposals or ``coupling`` is not
        positive and finite.
    """

    weight = _positive_finite_real(coupling, "coupling")
    if not isinstance(model_logits, OutputLogits):
        raise TypeError("model_logits must be an OutputLogits instance.")
    if isinstance(demonstration_candidates, (str, bytes)):
        raise TypeError(
            "demonstration_candidates must contain DecodedCandidate values."
        )
    proposals = tuple(demonstration_candidates)
    if len(proposals) != 2:
        raise ValueError("demonstration_candidates must contain exactly two proposals.")
    if not all(isinstance(candidate, DecodedCandidate) for candidate in proposals):
        raise TypeError("Every demonstration proposal must be a DecodedCandidate.")
    if np.array_equal(proposals[0].grid, proposals[1].grid):
        raise ValueError("The two demonstration proposals must be distinct grids.")

    scored: list[tuple[int, DecodedCandidate]] = []
    for index, proposal in enumerate(proposals):
        forest_score = _finite_real(
            proposal.log_probability, "proposal log_probability"
        )
        network_score = _network_candidate_log_probability(model_logits, proposal)
        combined = forest_score + weight * network_score
        _finite_real(combined, "combined candidate score")
        scored.append(
            (
                index,
                DecodedCandidate(
                    np.asarray(proposal.grid).copy(),
                    changed_decision=proposal.changed_decision,
                    log_probability=combined,
                ),
            )
        )
    scored.sort(key=lambda item: (-item[1].log_probability, item[0]))
    return scored[0][1], scored[1][1]


def _parameter_digest(names: Sequence[str], values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in names:
        array = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _checkpoint_values(
    path: pathlib.Path,
) -> tuple[list[str], dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as stored:
            names = list(stored.files)
            values = {name: np.asarray(stored[name]) for name in names}
    except (OSError, ValueError) as error:
        raise ValueError(f"Checkpoint cannot be loaded safely: {path}") from error
    if "__architecture__" not in values:
        raise ValueError("Checkpoint schema requires __architecture__ metadata.")
    parameter_names = [name for name in names if name != "__architecture__"]
    expected = [f"arr_{index}" for index in range(len(parameter_names))]
    if parameter_names != expected:
        raise ValueError("Checkpoint parameter schema must use contiguous arr_N names.")
    if not parameter_names or not all(
        np.issubdtype(values[name].dtype, np.floating) for name in parameter_names
    ):
        raise ValueError("Checkpoint parameters must be nonempty floating arrays.")
    if not all(np.all(np.isfinite(values[name])) for name in parameter_names):
        raise ValueError("Checkpoint parameters must contain only finite values.")
    architecture = values["__architecture__"]
    if architecture.dtype != np.dtype(np.uint8) or architecture.ndim != 1:
        raise ValueError("Checkpoint architecture must be one-dimensional uint8 JSON.")
    try:
        decoded = msgspec.json.decode(architecture.tobytes())
    except msgspec.DecodeError as error:
        raise ValueError(
            "Checkpoint architecture metadata must be valid JSON."
        ) from error
    if not isinstance(decoded, Mapping):
        raise ValueError("Checkpoint architecture metadata must be a JSON mapping.")
    return parameter_names, values


def _leaf_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _checkpoint_audit(path: str | pathlib.Path) -> dict[str, object]:
    checkpoint_path = pathlib.Path(path)
    names, values = _checkpoint_values(checkpoint_path)
    architecture = values["__architecture__"]
    leaf_schema = [
        {
            "archive_name": name,
            "shape": list(values[name].shape),
            "dtype": values[name].dtype.str,
            "sha256": _leaf_sha256(values[name]),
        }
        for name in names
    ]
    architecture_sha256 = hashlib.sha256(architecture.tobytes()).hexdigest()
    return {
        "path": str(checkpoint_path),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "size_bytes": checkpoint_path.stat().st_size,
        "parameter_sha256": _parameter_digest(names, values),
        "architecture_sha256": architecture_sha256,
        "architecture": msgspec.json.decode(architecture.tobytes()),
        "schema_sha256": _canonical_sha256(
            {
                "architecture_sha256": architecture_sha256,
                "leaves": [
                    {
                        "archive_name": leaf["archive_name"],
                        "shape": leaf["shape"],
                        "dtype": leaf["dtype"],
                    }
                    for leaf in leaf_schema
                ],
            }
        ),
        "ordered_leaves": leaf_schema,
        "finite_all_leaves": True,
    }


def _reseed_parameter_values(
    parameter_names: Sequence[str],
    source_values: Mapping[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    generator = brainstate.random.RandomState(seed)
    output: dict[str, np.ndarray] = {}
    for name in parameter_names:
        source_value = source_values[name]
        draw = np.asarray(
            generator.normal(size=source_value.shape, dtype=np.float32),
            dtype=source_value.dtype,
        )
        source_rms = float(np.sqrt(np.mean(np.square(source_value.astype(np.float64)))))
        draw_rms = float(np.sqrt(np.mean(np.square(draw.astype(np.float64)))))
        target_rms = max(source_rms, float(np.finfo(source_value.dtype).eps))
        if not math.isfinite(draw_rms) or draw_rms == 0.0:
            raise ValueError("BrainState reseed produced an invalid random leaf.")
        output[name] = np.asarray(
            draw * np.asarray(target_rms / draw_rms, dtype=draw.dtype),
            dtype=source_value.dtype,
        )
    return output


def write_checkpoint_perturbation(
    source: str | pathlib.Path,
    destination: str | pathlib.Path,
    *,
    kind: str,
    factor: float | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Write an exact-schema scale or deterministic-reseed checkpoint.

    Parameters
    ----------
    source, destination : path-like
        Existing source checkpoint and new output path.  The source is never
        overwritten.
    kind : {"scale", "reseed"}
        Parameter intervention to apply to every floating leaf.
    factor : float or None, default=None
        Positive finite non-unit scale, required only for ``kind="scale"``.
    seed : int or None, default=None
        Nonnegative BrainState seed, required only for ``kind="reseed"``.

    Returns
    -------
    dict
        Source/output file and ordered-parameter digests plus schema metadata.

    Raises
    ------
    ValueError
        If the intervention or checkpoint schema is invalid.
    FileExistsError
        If ``destination`` already exists.
    """

    if kind not in ("scale", "reseed"):
        raise ValueError("kind must be 'scale' or 'reseed'.")
    source_path = pathlib.Path(source)
    destination_path = pathlib.Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("destination must differ from the source checkpoint.")
    if destination_path.exists():
        raise FileExistsError(f"Destination already exists: {destination_path}")
    parameter_names, source_values = _checkpoint_values(source_path)
    output_values: dict[str, np.ndarray] = {}
    report_detail: dict[str, object]
    if kind == "scale":
        scale = _positive_finite_real(factor, "factor")
        if scale == 1.0:
            raise ValueError("factor must be non-unit.")
        for name in parameter_names:
            value = source_values[name]
            output_values[name] = np.asarray(
                value * np.asarray(scale, dtype=value.dtype), dtype=value.dtype
            )
        report_detail = {"factor": scale}
    else:
        reseed = _integer(seed, "seed")
        output_values = _reseed_parameter_values(parameter_names, source_values, reseed)
        report_detail = {"seed": reseed}
    serialized_values = {
        name: (
            source_values[name].copy()
            if name == "__architecture__"
            else output_values[name]
        )
        for name in source_values
    }
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staged = destination_path.with_name(destination_path.name + ".partial")
    try:
        with staged.open("wb") as handle:
            np.savez(
                handle,
                **serialized_values,
            )
        os.replace(staged, destination_path)
    finally:
        if staged.exists():
            staged.unlink()
    output_names, restored = _checkpoint_values(destination_path)
    schema_identical = bool(
        output_names == parameter_names
        and list(restored) == list(source_values)
        and all(
            restored[name].shape == source_values[name].shape
            and restored[name].dtype == source_values[name].dtype
            for name in source_values
        )
        and np.array_equal(
            restored["__architecture__"], source_values["__architecture__"]
        )
    )
    if not schema_identical:
        raise ValueError("Perturbed checkpoint schema differs from its source.")
    return {
        "kind": kind,
        **report_detail,
        "schema_identical": True,
        "parameter_names": parameter_names,
        "source_checkpoint_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(destination_path.read_bytes()).hexdigest(),
        "source_parameter_sha256": _parameter_digest(parameter_names, source_values),
        "parameter_sha256": _parameter_digest(parameter_names, restored),
    }


def _candidate_bytes(candidate: Mapping[str, object], rank: int | None) -> bytes:
    if rank is not None and candidate.get("rank") != rank:
        raise ValueError("Candidate ranks must be the ordered values one and two.")
    raw_grid = candidate.get("grid")
    try:
        raw_array = np.asarray(raw_grid)
    except (TypeError, ValueError) as error:
        raise ValueError("Candidate grid must be a rectangular ARC grid.") from error
    if (
        raw_array.ndim != 2
        or not 1 <= raw_array.shape[0] <= 30
        or not 1 <= raw_array.shape[1] <= 30
        or not np.issubdtype(raw_array.dtype, np.number)
        or not np.all(np.isfinite(raw_array))
        or np.any(raw_array != np.floor(raw_array))
        or np.any(raw_array < 0)
        or np.any(raw_array > 9)
        or candidate.get("height") != raw_array.shape[0]
        or candidate.get("width") != raw_array.shape[1]
    ):
        raise ValueError("Candidate dimensions or ARC colours are invalid.")
    grid = np.asarray(raw_array, dtype=np.int8)
    payload = bytearray()
    if rank is not None:
        payload.extend(int(rank).to_bytes(1, "little"))
    payload.extend(int(grid.shape[0]).to_bytes(1, "little"))
    payload.extend(int(grid.shape[1]).to_bytes(1, "little"))
    payload.extend(np.ascontiguousarray(grid).tobytes())
    return bytes(payload)


def _candidate_sha256(candidate: Mapping[str, object], rank: int) -> str:
    return hashlib.sha256(_candidate_bytes(candidate, rank)).hexdigest()


def _validate_checkpoint_candidate(
    value: object,
    rank: int,
    submission_checkpoint: int,
    coupling: float,
    parameter_dependencies: Sequence[str],
) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("Every candidate must be a provenance mapping.")
    required = {
        "provenance": "model",
        "dependency_class": "model_checkpoint",
        "proposal_source": "demonstration_fitted_forest",
        "ranking_source": "trained_network_factorized_candidate_log_probability",
        "answer_head_version": "checkpoint_conditioned_v1",
        "source_checkpoint": submission_checkpoint,
        "selection_role": "checkpoint_conditioned_rank",
    }
    for name, expected in required.items():
        if value.get(name) != expected:
            raise ValueError(
                f"Candidate {name} does not prove checkpoint model provenance."
            )
    dependencies = value.get("parameter_dependencies")
    if not isinstance(dependencies, list) or dependencies != list(
        parameter_dependencies
    ):
        raise ValueError(
            "Candidate parameter_dependencies must exactly match all ordered "
            "participating parameter leaves."
        )
    forest = _finite_real(value.get("forest_log_probability"), "candidate forest score")
    network = _finite_real(
        value.get("network_log_probability"), "candidate network score"
    )
    combined = _finite_real(
        value.get("combined_log_probability"), "candidate combined score"
    )
    reported = _finite_real(value.get("log_probability"), "candidate score")
    expected = forest + coupling * network
    if not math.isclose(combined, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(
            "Candidate combined score must equal forest plus coupling times "
            "network score."
        )
    if not math.isclose(reported, combined, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("Candidate score must equal its combined score.")
    forest_rank = value.get("forest_rank")
    if forest_rank not in (1, 2):
        raise ValueError("Candidate forest_rank must be one or two.")
    return _candidate_sha256(value, rank)


def _manifest_sha256(manifest: Sequence[Sequence[object]]) -> str:
    digest = hashlib.sha256()
    for task_id, query_index in manifest:
        encoded = str(task_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        query_number = _integer(query_index, "query_index")
        digest.update(query_number.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _source_revision(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("implementation.source_revision must be a 40/64 hex digest.")
    return value


def _ordered_parameter_leaves(
    value: object,
) -> tuple[list[dict[str, object]], list[str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("parameter_binding.ordered_leaves must be a nonempty list.")
    leaves: list[dict[str, object]] = []
    names: list[str] = []
    for index, raw_leaf in enumerate(value):
        if not isinstance(raw_leaf, Mapping):
            raise ValueError("Every ordered parameter leaf must be a mapping.")
        name = raw_leaf.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(
                "Ordered parameter leaf names must be unique and nonempty."
            )
        raw_shape = raw_leaf.get("shape")
        if not isinstance(raw_shape, list) or not raw_shape:
            raise ValueError("Ordered parameter leaf shape must be a nonempty list.")
        shape = [
            _integer(dimension, f"ordered_leaves[{index}].shape")
            for dimension in raw_shape
        ]
        if any(dimension == 0 for dimension in shape):
            raise ValueError(
                "Ordered parameter leaves must contain at least one value."
            )
        raw_dtype = raw_leaf.get("dtype")
        if not isinstance(raw_dtype, str):
            raise ValueError("Ordered parameter leaf dtype must be valid.")
        try:
            dtype = np.dtype(raw_dtype)
        except (TypeError, ValueError) as error:
            raise ValueError("Ordered parameter leaf dtype must be valid.") from error
        if raw_dtype != dtype.str or not np.issubdtype(dtype, np.floating):
            raise ValueError(
                "Ordered parameter leaf dtype must be a canonical floating dtype.str."
            )
        leaf: dict[str, object] = {
            "name": name,
            "shape": shape,
            "dtype": dtype.str,
            "sha256": _sha256(
                raw_leaf.get("sha256"), f"ordered_leaves[{index}].sha256"
            ),
        }
        names.append(name)
        leaves.append(leaf)
    return leaves, names


def _dale_dependency(model: Mapping[str, object]) -> dict[str, object]:
    neuron_typing = model.get("neuron_typing")
    if not isinstance(neuron_typing, Mapping):
        raise ValueError("model.neuron_typing must explicitly report Dale status.")
    mode = neuron_typing.get("mode")
    if mode == "none":
        return {"claimed": False, "status": "not_claimed"}
    if mode != "ei_dale":
        raise ValueError("model.neuron_typing.mode must be none or ei_dale.")
    violations = _integer(
        neuron_typing.get("recurrent_sign_violation_count"),
        "recurrent_sign_violation_count",
    )
    if violations != 0:
        raise ValueError("Dale compliance cannot be claimed with sign violations.")
    excitatory = _integer(neuron_typing.get("excitatory_count"), "excitatory_count")
    inhibitory = _integer(neuron_typing.get("inhibitory_count"), "inhibitory_count")
    if excitatory + inhibitory == 0:
        raise ValueError("Dale evidence must cover a nonempty neuron population.")
    return {
        "claimed": True,
        "status": "verified_zero_recurrent_sign_violations",
        "excitatory_count": excitatory,
        "inhibitory_count": inhibitory,
        "recurrent_sign_violation_count": 0,
    }


def _cross_check_reported_metrics(
    evaluation: Mapping[str, object],
    checkpoint: int,
    exact_counts: Mapping[str, int],
    query_count: int,
    task_count: int,
    strict_task_ids: Sequence[str],
    strict_task_membership: Sequence[Sequence[bool]],
) -> None:
    metrics_by_effort = evaluation.get("metrics_by_effort")
    if metrics_by_effort is not None:
        if not isinstance(metrics_by_effort, Mapping):
            raise ValueError("evaluation.metrics_by_effort must be a mapping.")
        metrics = metrics_by_effort.get(str(checkpoint))
        if not isinstance(metrics, Mapping):
            raise ValueError("Submission metrics are missing from metrics_by_effort.")
        expected_values: dict[str, int | float] = {
            "query_count": query_count,
            "task_count": task_count,
            "query_pass_at_1": exact_counts["query_pass_at_1_count"] / query_count,
            "query_pass_at_2": exact_counts["query_pass_at_2_count"] / query_count,
            "strict_task_pass_at_1": exact_counts["strict_task_pass_at_1_count"]
            / task_count,
            "strict_task_pass_at_2": exact_counts["strict_task_pass_at_2_count"]
            / task_count,
        }
        for name, expected in expected_values.items():
            observed = metrics.get(name)
            if isinstance(expected, int):
                if observed != expected:
                    raise ValueError(
                        f"Reported metric {name} disagrees with query rows."
                    )
            elif not isinstance(observed, Real) or not math.isclose(
                float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"Reported metric {name} disagrees with query rows.")
        tasks = metrics.get("tasks")
        if tasks is not None:
            if not isinstance(tasks, Mapping) or set(tasks) != set(strict_task_ids):
                raise ValueError(
                    "Reported strict-task metrics have a different manifest."
                )
            for task_id, membership in zip(
                strict_task_ids, strict_task_membership, strict=True
            ):
                task = tasks[task_id]
                if (
                    not isinstance(task, Mapping)
                    or task.get("pass_at_1") is not membership[0]
                    or task.get("pass_at_2") is not membership[1]
                ):
                    raise ValueError(
                        "Reported strict-task membership disagrees with rows."
                    )
    completion = evaluation.get("submitted_completion")
    if completion is not None:
        if not isinstance(completion, Mapping):
            raise ValueError("evaluation.submitted_completion must be a mapping.")
        expected_completion = {
            "evaluated_task_count": task_count,
            "evaluated_query_count": query_count,
            "exact_query_count_at_1": exact_counts["query_pass_at_1_count"],
            "exact_query_count_at_2": exact_counts["query_pass_at_2_count"],
            "exact_task_count_at_1": exact_counts["strict_task_pass_at_1_count"],
            "exact_task_count_at_2": exact_counts["strict_task_pass_at_2_count"],
        }
        for name, expected in expected_completion.items():
            if completion.get(name) != expected:
                raise ValueError(
                    f"Reported submitted completion {name} disagrees with rows."
                )


def parameter_dependence_run_from_result(result: object) -> dict[str, object]:
    """Extract canonical causal-score evidence from one Example 21 result.

    Parameters
    ----------
    result : object
        Parsed full ``result.json`` mapping for a checkpoint-conditioned run.

    Returns
    -------
    dict
        Canonical manifest, ordered candidate hashes, query/task exact
        memberships, exact counts, and bound checkpoint/configuration digests.

    Raises
    ------
    ValueError
        If the result is incomplete, noncausal, inconsistent, or unfrozen.
    """

    if not isinstance(result, Mapping):
        raise ValueError("Result must be a mapping.")
    configuration = result.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("Result configuration is missing.")
    if configuration.get("answer_head") != "checkpoint_conditioned":
        raise ValueError("answer_head must be checkpoint_conditioned.")
    if configuration.get("primary_candidate_mode") != "model_only":
        raise ValueError("checkpoint candidates require model_only mode.")
    submission_checkpoint = _integer(
        configuration.get("submission_checkpoint"), "submission_checkpoint"
    )
    coupling = _positive_finite_real(
        configuration.get("checkpoint_conditioning_coupling"),
        "checkpoint_conditioning_coupling",
    )
    if coupling != 1.0:
        raise ValueError("checkpoint_conditioning_coupling must be exactly 1.0.")
    configuration_sha256 = _sha256(
        result.get("configuration_sha256"), "configuration_sha256"
    )
    if configuration_sha256 != _canonical_sha256(configuration):
        raise ValueError("configuration_sha256 does not bind the configuration.")
    training = result.get("training")
    evaluation = result.get("evaluation")
    if not isinstance(training, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError("Training and evaluation evidence are required.")
    checkpoint_sha256 = _sha256(
        training.get("parameter_checkpoint_sha256"),
        "training.parameter_checkpoint_sha256",
    )
    before = _sha256(
        evaluation.get("parameter_sha256_before"),
        "evaluation.parameter_sha256_before",
    )
    after = _sha256(
        evaluation.get("parameter_sha256_after"),
        "evaluation.parameter_sha256_after",
    )
    if before != after:
        raise ValueError("Evaluation did not preserve frozen parameter bytes.")
    training_before = _sha256(
        training.get("parameter_sha256_before"),
        "training.parameter_sha256_before",
    )
    training_after = _sha256(
        training.get("parameter_sha256_after"),
        "training.parameter_sha256_after",
    )
    if training_before != training_after or training_after != before:
        raise ValueError("Training/evaluation parameter digests are not bound.")
    if evaluation.get("same_frozen_parameter_bytes") is False:
        raise ValueError("Evaluation reports mutable parameter bytes.")

    implementation = result.get("implementation")
    data_summary = result.get("data_summary")
    model = result.get("model")
    if not all(
        isinstance(value, Mapping) for value in (implementation, data_summary, model)
    ):
        raise ValueError("Implementation, data, and model provenance are required.")
    assert isinstance(implementation, Mapping)
    assert isinstance(data_summary, Mapping)
    assert isinstance(model, Mapping)
    revision = _source_revision(implementation.get("source_revision"))
    source_dirty = implementation.get("source_dirty")
    if not isinstance(source_dirty, bool):
        raise ValueError("implementation.source_dirty must be boolean.")
    manifest_sha256 = _sha256(
        data_summary.get("manifest_sha256"), "data_summary.manifest_sha256"
    )
    topology_sha256 = _sha256(model.get("topology_sha256"), "model.topology_sha256")
    dale_dependency = _dale_dependency(model)

    parameter_binding = evaluation.get("parameter_binding")
    if not isinstance(parameter_binding, Mapping):
        raise ValueError("evaluation.parameter_binding is required.")
    binding_checkpoint = _sha256(
        parameter_binding.get("checkpoint_sha256"),
        "parameter_binding.checkpoint_sha256",
    )
    binding_parameter = _sha256(
        parameter_binding.get("parameter_sha256"),
        "parameter_binding.parameter_sha256",
    )
    binding_topology = _sha256(
        parameter_binding.get("topology_sha256"),
        "parameter_binding.topology_sha256",
    )
    if binding_checkpoint != checkpoint_sha256:
        raise ValueError("Evaluation checkpoint binding disagrees with training.")
    if binding_parameter != before:
        raise ValueError("Evaluation parameter binding disagrees with frozen bytes.")
    if binding_topology != topology_sha256:
        raise ValueError("Evaluation topology binding disagrees with the model.")
    ordered_leaves, parameter_dependencies = _ordered_parameter_leaves(
        parameter_binding.get("ordered_leaves")
    )
    checkpoint_queries = evaluation.get("checkpoint_queries")
    rows = (
        checkpoint_queries.get(str(submission_checkpoint))
        if isinstance(checkpoint_queries, Mapping)
        else None
    )
    if not isinstance(rows, list) or not rows:
        raise ValueError("Submission-checkpoint query evidence is missing.")

    query_manifest: list[list[object]] = []
    candidate_sha256: list[list[str]] = []
    proposal_set_sha256: list[list[str]] = []
    exact_membership: list[list[bool]] = []
    observed_queries: set[tuple[str, int]] = set()
    task_membership: dict[str, list[list[bool]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every checkpoint query row must be a mapping.")
        task_id = row.get("task_id")
        query_index = row.get("query_index")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Every query row needs a nonempty task_id.")
        query_number = _integer(query_index, "query_index")
        identity = (task_id, query_number)
        if identity in observed_queries:
            raise ValueError("Duplicate task/query identity in checkpoint evidence.")
        observed_queries.add(identity)
        if (
            row.get("primary_candidate_mode") != "model_only"
            or row.get("submission_role") != "primary_submission"
        ):
            raise ValueError("Submission rows must be primary model-only candidates.")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 2:
            raise ValueError("Every query must contain exactly two candidate ranks.")
        candidate_sha256.append(
            [
                _validate_checkpoint_candidate(
                    candidate,
                    rank,
                    submission_checkpoint,
                    coupling,
                    parameter_dependencies,
                )
                for rank, candidate in enumerate(candidates, start=1)
            ]
        )
        proposal_set_sha256.append(
            sorted(
                hashlib.sha256(_candidate_bytes(candidate, None)).hexdigest()
                for candidate in candidates
            )
        )
        score = row.get("score")
        if (
            not isinstance(score, Mapping)
            or score.get("task_id") != task_id
            or score.get("query_index") != query_number
        ):
            raise ValueError("Query score identity must match its checkpoint row.")
        pass_at_1 = score.get("pass_at_1")
        pass_at_2 = score.get("pass_at_2")
        if not isinstance(pass_at_1, bool) or not isinstance(pass_at_2, bool):
            raise ValueError("Exact query memberships must be boolean.")
        if pass_at_1 and not pass_at_2:
            raise ValueError("pass_at_1 membership implies pass_at_2 membership.")
        membership = [pass_at_1, pass_at_2]
        query_manifest.append([task_id, query_number])
        exact_membership.append(membership)
        task_membership.setdefault(task_id, []).append(membership)

    strict_task_ids = list(task_membership)
    strict_task_membership = [
        [
            all(membership[0] for membership in task_membership[task_id]),
            all(membership[1] for membership in task_membership[task_id]),
        ]
        for task_id in strict_task_ids
    ]
    exact_counts = {
        "query_pass_at_1_count": sum(item[0] for item in exact_membership),
        "query_pass_at_2_count": sum(item[1] for item in exact_membership),
        "strict_task_pass_at_1_count": sum(item[0] for item in strict_task_membership),
        "strict_task_pass_at_2_count": sum(item[1] for item in strict_task_membership),
    }
    _cross_check_reported_metrics(
        evaluation,
        submission_checkpoint,
        exact_counts,
        len(query_manifest),
        len(strict_task_ids),
        strict_task_ids,
        strict_task_membership,
    )
    candidate_output_sha256 = _canonical_sha256(candidate_sha256)
    proposal_set_output_sha256 = _canonical_sha256(proposal_set_sha256)
    exact_membership_sha256 = _canonical_sha256(exact_membership)
    strict_task_membership_sha256 = _canonical_sha256(
        {
            "task_ids": strict_task_ids,
            "membership": strict_task_membership,
        }
    )
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "parameter_sha256": before,
        "configuration_sha256": configuration_sha256,
        "source_revision": revision,
        "source_dirty": source_dirty,
        "manifest_sha256": manifest_sha256,
        "topology_sha256": topology_sha256,
        "ordered_parameter_leaves": ordered_leaves,
        "ordered_parameter_leaves_sha256": _canonical_sha256(ordered_leaves),
        "parameter_schema_sha256": _canonical_sha256(
            [
                {
                    "name": leaf["name"],
                    "shape": leaf["shape"],
                    "dtype": leaf["dtype"],
                }
                for leaf in ordered_leaves
            ]
        ),
        "parameter_dependencies": parameter_dependencies,
        "dale_dependency": dale_dependency,
        "query_manifest": query_manifest,
        "query_manifest_sha256": _manifest_sha256(query_manifest),
        "candidate_sha256": candidate_sha256,
        "candidate_output_sha256": candidate_output_sha256,
        "proposal_set_sha256": proposal_set_sha256,
        "proposal_set_output_sha256": proposal_set_output_sha256,
        "exact_membership": exact_membership,
        "exact_membership_sha256": exact_membership_sha256,
        "strict_task_ids": strict_task_ids,
        "strict_task_membership": strict_task_membership,
        "strict_task_membership_sha256": strict_task_membership_sha256,
        "query_count": len(query_manifest),
        "task_count": len(strict_task_ids),
        "exact_counts": exact_counts,
        "cumulative_score": sum(exact_counts.values()),
    }


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return result


def _candidate_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("candidate_sha256 must be a nonempty sequence of rank pairs.")
    pairs: list[tuple[str, str]] = []
    for query_index, pair in enumerate(value):
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise ValueError("Every candidate_sha256 entry must contain two ranks.")
        pairs.append(
            (
                _sha256(pair[0], f"candidate_sha256[{query_index}][0]"),
                _sha256(pair[1], f"candidate_sha256[{query_index}][1]"),
            )
        )
    return tuple(pairs)


def _membership_pairs(value: object) -> tuple[tuple[bool, bool], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("exact_membership must be a nonempty sequence of rank pairs.")
    pairs: list[tuple[bool, bool]] = []
    for pair in value:
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise ValueError("Every exact_membership entry must contain two ranks.")
        if not all(isinstance(item, (bool, np.bool_)) for item in pair):
            raise ValueError("Every exact_membership value must be boolean.")
        pairs.append((bool(pair[0]), bool(pair[1])))
    return tuple(pairs)


def _validated_run(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    checkpoint = _sha256(value.get("checkpoint_sha256"), f"{name}.checkpoint_sha256")
    parameters = _sha256(value.get("parameter_sha256"), f"{name}.parameter_sha256")
    manifest = _sha256(
        value.get("query_manifest_sha256"), f"{name}.query_manifest_sha256"
    )
    source_revision = _source_revision(value.get("source_revision"))
    source_dirty = value.get("source_dirty")
    if not isinstance(source_dirty, bool):
        raise ValueError(f"{name}.source_dirty must be boolean.")
    data_manifest = _sha256(value.get("manifest_sha256"), f"{name}.manifest_sha256")
    topology = _sha256(value.get("topology_sha256"), f"{name}.topology_sha256")
    parameter_schema = _sha256(
        value.get("parameter_schema_sha256"),
        f"{name}.parameter_schema_sha256",
    )
    ordered_leaves, dependencies = _ordered_parameter_leaves(
        value.get("ordered_parameter_leaves")
    )
    computed_schema = _canonical_sha256(
        [
            {
                "name": leaf["name"],
                "shape": leaf["shape"],
                "dtype": leaf["dtype"],
            }
            for leaf in ordered_leaves
        ]
    )
    if parameter_schema != computed_schema:
        raise ValueError(f"{name} parameter schema digest is inconsistent.")
    ordered_leaves_digest = _sha256(
        value.get("ordered_parameter_leaves_sha256"),
        f"{name}.ordered_parameter_leaves_sha256",
    )
    if ordered_leaves_digest != _canonical_sha256(ordered_leaves):
        raise ValueError(f"{name} ordered parameter leaf digest is inconsistent.")
    if value.get("parameter_dependencies") != dependencies:
        raise ValueError(f"{name} parameter dependency set is inconsistent.")
    dale = value.get("dale_dependency")
    if not isinstance(dale, Mapping) or not isinstance(dale.get("claimed"), bool):
        raise ValueError(f"{name}.dale_dependency must be explicit.")
    raw_counts = value.get("exact_counts")
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(
        _EXACT_COUNT_NAMES
    ):
        raise ValueError(
            f"{name}.exact_counts must contain the four exact count names."
        )
    counts = {
        count_name: _integer(raw_counts[count_name], f"{name}.{count_name}")
        for count_name in _EXACT_COUNT_NAMES
    }
    cumulative = _integer(value.get("cumulative_score"), f"{name}.cumulative_score")
    if cumulative != sum(counts.values()):
        raise ValueError(f"{name}.cumulative_score must equal the four exact counts.")
    candidates = _candidate_pairs(value.get("candidate_sha256"))
    membership = _membership_pairs(value.get("exact_membership"))
    if len(candidates) != len(membership):
        raise ValueError(f"{name} candidate and membership query counts must match.")
    candidate_output = _sha256(
        value.get("candidate_output_sha256"),
        f"{name}.candidate_output_sha256",
    )
    exact_membership_output = _sha256(
        value.get("exact_membership_sha256"),
        f"{name}.exact_membership_sha256",
    )
    if candidate_output != _canonical_sha256(value.get("candidate_sha256")):
        raise ValueError(f"{name} aggregate candidate digest is inconsistent.")
    if exact_membership_output != _canonical_sha256(value.get("exact_membership")):
        raise ValueError(f"{name} aggregate exact-membership digest is inconsistent.")
    return {
        "checkpoint_sha256": checkpoint,
        "parameter_sha256": parameters,
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "manifest_sha256": data_manifest,
        "topology_sha256": topology,
        "parameter_schema_sha256": parameter_schema,
        "ordered_parameter_leaves": ordered_leaves,
        "ordered_parameter_leaves_sha256": ordered_leaves_digest,
        "parameter_dependencies": dependencies,
        "dale_dependency": dict(dale),
        "query_manifest_sha256": manifest,
        "exact_counts": counts,
        "cumulative_score": cumulative,
        "candidate_sha256": candidates,
        "candidate_output_sha256": candidate_output,
        "exact_membership": membership,
        "exact_membership_sha256": exact_membership_output,
    }


def _validated_mutation(value: object, expected: str) -> None:
    if not isinstance(value, Mapping) or value.get("kind") != expected:
        raise ValueError(f"{expected} mutation metadata must name kind {expected!r}.")
    if expected == "scale":
        factor = _positive_finite_real(value.get("factor"), "scale factor")
        if factor == 1.0:
            raise ValueError("scale factor must be non-unit.")
    elif expected == "swap":
        if not isinstance(value.get("run_id"), str) or not value["run_id"]:
            raise ValueError("swap mutation metadata must name a nonempty run_id.")
    else:
        _integer(value.get("seed"), "reseed seed")


def _failed_report(
    minimum: int, checks: Mapping[str, bool], reasons: Sequence[str]
) -> dict[str, object]:
    return {
        "passed": False,
        "minimum_cumulative_score": minimum,
        "baseline_cumulative_score": None,
        "checks": dict(checks),
        "reasons": list(reasons),
    }


def assess_parameter_dependence(
    evidence: object, *, minimum_cumulative_score: int = 16
) -> dict[str, object]:
    """Validate a matched checkpoint-perturbation qualification matrix.

    Parameters
    ----------
    evidence : object
        Raw baseline, repeat, and named scale/swap/reseed evidence.  Malformed
        values fail closed instead of raising.
    minimum_cumulative_score : int, default=16
        Smallest accepted sum of the four integer exact-count fields.

    Returns
    -------
    dict
        A JSON-safe report with ``passed``, individual ``checks``, and explicit
        failure ``reasons``.
    """

    try:
        minimum = _integer(minimum_cumulative_score, "minimum_cumulative_score")
    except (TypeError, ValueError) as error:
        return _failed_report(16, {"minimum_valid": False}, [str(error)])
    checks: dict[str, bool] = {"evidence_complete": False}
    reasons: list[str] = []
    try:
        if not isinstance(evidence, Mapping):
            raise ValueError("parameter-dependence evidence must be a mapping.")
        baseline = _validated_run(evidence.get("baseline"), "baseline")
        repeat = _validated_run(evidence.get("repeat_intact"), "repeat_intact")
        raw_perturbations = evidence.get("perturbations")
        if not isinstance(raw_perturbations, Mapping) or set(raw_perturbations) != set(
            _PERTURBATION_NAMES
        ):
            raise ValueError(
                "perturbations must contain exactly scale, swap, and reseed."
            )
        perturbations: dict[str, dict[str, Any]] = {}
        for name in _PERTURBATION_NAMES:
            raw_run = raw_perturbations[name]
            perturbations[name] = _validated_run(raw_run, name)
            assert isinstance(raw_run, Mapping)
            _validated_mutation(raw_run.get("mutation"), name)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        reasons.append(str(error))
        return _failed_report(minimum, checks, reasons)

    checks["evidence_complete"] = True
    checks["baseline_score"] = baseline["cumulative_score"] >= minimum
    if not checks["baseline_score"]:
        reasons.append(
            f"baseline cumulative score must be at least {minimum}; "
            f"observed {baseline['cumulative_score']}."
        )
    repeat_fields = (
        "checkpoint_sha256",
        "parameter_sha256",
        "source_revision",
        "source_dirty",
        "manifest_sha256",
        "topology_sha256",
        "parameter_schema_sha256",
        "ordered_parameter_leaves",
        "ordered_parameter_leaves_sha256",
        "parameter_dependencies",
        "dale_dependency",
        "query_manifest_sha256",
        "exact_counts",
        "cumulative_score",
        "candidate_sha256",
        "candidate_output_sha256",
        "exact_membership",
        "exact_membership_sha256",
    )
    checks["repeat_intact"] = all(
        repeat[field] == baseline[field] for field in repeat_fields
    )
    if not checks["repeat_intact"]:
        reasons.append(
            "repeat_intact must exactly match every baseline causal output field."
        )

    for name, run in perturbations.items():
        same_manifest = bool(
            run["query_manifest_sha256"] == baseline["query_manifest_sha256"]
            and run["manifest_sha256"] == baseline["manifest_sha256"]
        )
        same_execution_contract = bool(
            run["source_revision"] == baseline["source_revision"]
            and run["source_dirty"] == baseline["source_dirty"]
            and run["topology_sha256"] == baseline["topology_sha256"]
            and run["parameter_schema_sha256"] == baseline["parameter_schema_sha256"]
            and run["parameter_dependencies"] == baseline["parameter_dependencies"]
            and run["dale_dependency"] == baseline["dale_dependency"]
        )
        same_length = len(run["candidate_sha256"]) == len(baseline["candidate_sha256"])
        parameter_moved = bool(
            run["checkpoint_sha256"] != baseline["checkpoint_sha256"]
            and run["parameter_sha256"] != baseline["parameter_sha256"]
            and run["ordered_parameter_leaves_sha256"]
            != baseline["ordered_parameter_leaves_sha256"]
        )
        candidates_moved = bool(
            same_length and run["candidate_sha256"] != baseline["candidate_sha256"]
        )
        membership_moved = bool(
            same_length and run["exact_membership"] != baseline["exact_membership"]
        )
        score_moved = bool(
            run["exact_counts"] != baseline["exact_counts"]
            and run["cumulative_score"] != baseline["cumulative_score"]
        )
        arm_checks = {
            f"{name}_manifest_matched": same_manifest,
            f"{name}_execution_contract_matched": same_execution_contract,
            f"{name}_parameter_moved": parameter_moved,
            f"{name}_candidate_moved": candidates_moved,
            f"{name}_membership_moved": membership_moved,
            f"{name}_score_moved": score_moved,
        }
        checks.update(arm_checks)
        for check_name, passed in arm_checks.items():
            if not passed:
                reasons.append(check_name.replace("_", " ") + " check failed.")

    return {
        "passed": all(checks.values()),
        "minimum_cumulative_score": minimum,
        "baseline_cumulative_score": baseline["cumulative_score"],
        "checks": checks,
        "reasons": reasons,
    }


def _normalized_result_configuration(result: Mapping[str, object]) -> object:
    configuration = result.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("Every matrix result requires a configuration mapping.")
    normalized = dict(configuration)
    for name in ("output_dir", "parameter_checkpoint", "initial_checkpoint"):
        normalized.pop(name, None)
    return normalized


def _accepted_full_profile(result: Mapping[str, object]) -> dict[str, object]:
    configuration = result.get("configuration")
    model = result.get("model")
    device = result.get("device")
    config = configuration if isinstance(configuration, Mapping) else {}
    model_report = model if isinstance(model, Mapping) else {}
    device_report = device if isinstance(device, Mapping) else {}
    platform = device_report.get("platform")
    checks = {
        "result_schema_v2": result.get("schema_version") == 2,
        "protocol_v2": result.get("protocol_version") == 2,
        "configuration_model_scale": bool(
            config.get("neuron_count") == 4096
            and config.get("recurrent_edges") == 4_194_304
        ),
        "configuration_effort_profile": bool(
            config.get("latent_steps") == 60
            and config.get("checkpoints") == [0, 30, 60]
            and config.get("submission_checkpoint") == 60
        ),
        "configuration_seed": config.get("seed") == 31337,
        "configuration_answer_path": bool(
            config.get("answer_head") == "checkpoint_conditioned"
            and config.get("primary_candidate_mode") == "model_only"
            and config.get("checkpoint_conditioning_coupling") == 1.0
        ),
        "configuration_gpu_requested": config.get("device") == "gpu",
        "configuration_not_smoke": config.get("smoke") is False,
        "configuration_not_structural_only": (config.get("structural_only") is False),
        "configuration_complete_split": (
            "evaluation_task_limit" in config
            and config.get("evaluation_task_limit") is None
        ),
        "model_scale": bool(
            model_report.get("neuron_count") == 4096
            and model_report.get("recurrent_edge_count") == 4_194_304
        ),
        "actual_gpu_backend": bool(
            isinstance(platform, str) and platform.casefold() == "gpu"
        ),
    }
    observed = {
        "schema_version": result.get("schema_version"),
        "protocol_version": result.get("protocol_version"),
        "neuron_count": config.get("neuron_count"),
        "recurrent_edges": config.get("recurrent_edges"),
        "latent_steps": config.get("latent_steps"),
        "checkpoints": config.get("checkpoints"),
        "submission_checkpoint": config.get("submission_checkpoint"),
        "seed": config.get("seed"),
        "answer_head": config.get("answer_head"),
        "primary_candidate_mode": config.get("primary_candidate_mode"),
        "checkpoint_conditioning_coupling": config.get(
            "checkpoint_conditioning_coupling"
        ),
        "requested_device": config.get("device"),
        "actual_platform": platform,
        "smoke": config.get("smoke"),
        "structural_only": config.get("structural_only"),
        "evaluation_task_limit": config.get("evaluation_task_limit"),
        "model_neuron_count": model_report.get("neuron_count"),
        "model_recurrent_edge_count": model_report.get("recurrent_edge_count"),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": observed,
    }


def _training_origin(
    value: object,
    name: str,
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} training origin must be a result mapping.")
    configuration = value.get("configuration")
    training = value.get("training")
    if not isinstance(configuration, Mapping) or not isinstance(training, Mapping):
        raise ValueError(f"{name} training origin lacks configuration/training.")
    configuration_sha256 = _sha256(
        value.get("configuration_sha256"),
        f"{name} origin configuration_sha256",
    )
    if configuration_sha256 != _canonical_sha256(configuration):
        raise ValueError(f"{name} origin configuration hash is invalid.")
    if training.get("performed") is not True:
        raise ValueError(f"{name} origin training.performed must be True.")
    checkpoint_sha256 = _sha256(
        training.get("parameter_checkpoint_sha256"),
        f"{name} origin parameter_checkpoint_sha256",
    )
    if checkpoint_sha256 != checkpoint["sha256"]:
        raise ValueError(f"{name} origin does not bind the audited checkpoint.")
    seed = _integer(configuration.get("seed"), f"{name} origin seed")
    final_parameter = _sha256(
        training.get("parameter_sha256_after"),
        f"{name} origin parameter_sha256_after",
    )
    return {
        "result_sha256": _canonical_sha256(value),
        "configuration_sha256": configuration_sha256,
        "seed": seed,
        "training_performed": True,
        "checkpoint_sha256": checkpoint_sha256,
        "final_parameter_sha256": final_parameter,
    }


def _bind_run_checkpoint(
    run: Mapping[str, object],
    checkpoint: Mapping[str, object],
    name: str,
) -> None:
    if run.get("checkpoint_sha256") != checkpoint.get("sha256"):
        raise ValueError(f"{name} result does not bind its checkpoint file SHA-256.")
    result_leaves = run.get("ordered_parameter_leaves")
    archive_leaves = checkpoint.get("ordered_leaves")
    if not isinstance(result_leaves, list) or not isinstance(archive_leaves, list):
        raise ValueError(f"{name} checkpoint leaves are missing.")
    if len(result_leaves) != len(archive_leaves):
        raise ValueError(f"{name} result/checkpoint leaf counts differ.")
    for result_leaf, archive_leaf in zip(result_leaves, archive_leaves, strict=True):
        if not isinstance(result_leaf, Mapping) or not isinstance(
            archive_leaf, Mapping
        ):
            raise ValueError(f"{name} checkpoint leaf evidence is malformed.")
        for field in ("shape", "dtype", "sha256"):
            if result_leaf.get(field) != archive_leaf.get(field):
                raise ValueError(
                    f"{name} executed leaf {field} differs from checkpoint bytes."
                )


def _same_checkpoint_schema(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> bool:
    return bool(
        baseline.get("schema_sha256") == candidate.get("schema_sha256")
        and baseline.get("architecture_sha256") == candidate.get("architecture_sha256")
    )


def _verify_exact_scale(
    baseline_path: pathlib.Path,
    scale_path: pathlib.Path,
    factor: float,
) -> None:
    baseline_names, baseline_values = _checkpoint_values(baseline_path)
    scale_names, scale_values = _checkpoint_values(scale_path)
    if baseline_names != scale_names:
        raise ValueError("Scale checkpoint schema differs from baseline.")
    for name in baseline_names:
        expected = np.asarray(
            baseline_values[name]
            * np.asarray(factor, dtype=baseline_values[name].dtype),
            dtype=baseline_values[name].dtype,
        )
        if not np.array_equal(scale_values[name], expected):
            raise ValueError("Scale checkpoint is not the exact declared factor.")


def _verify_exact_reseed(
    baseline_path: pathlib.Path,
    reseed_path: pathlib.Path,
    seed: int,
) -> None:
    baseline_names, baseline_values = _checkpoint_values(baseline_path)
    reseed_names, reseed_values = _checkpoint_values(reseed_path)
    if baseline_names != reseed_names:
        raise ValueError("Reseed checkpoint schema differs from baseline.")
    expected = _reseed_parameter_values(baseline_names, baseline_values, seed)
    for name in baseline_names:
        if not np.array_equal(reseed_values[name], expected[name]):
            raise ValueError(
                "Reseed checkpoint does not reproduce the deterministic "
                "BrainState algorithm."
            )


def build_parameter_dependence_matrix(
    results: Mapping[str, object],
    *,
    training_origins: Mapping[str, object],
    checkpoints: Mapping[str, str | pathlib.Path],
    scale_factor: float,
    reseed_seed: int,
    minimum_cumulative_score: int = 16,
    require_complete_arc: bool = True,
) -> dict[str, object]:
    """Build and assess one matched five-run causal qualification matrix.

    Parameters
    ----------
    results : mapping
        Parsed results named ``baseline``, ``repeat_intact``, ``scale``,
        ``swap``, and ``reseed``.
    training_origins : mapping
        Original completed training results named ``baseline`` and ``swap``.
    checkpoints : mapping
        Explicit checkpoint files named ``baseline``, ``scale``, ``swap``, and
        ``reseed``.
    scale_factor : float
        Applied non-unit checkpoint scale.
    reseed_seed : int
        BrainState seed used by the deterministic untrained control.
    minimum_cumulative_score : int, default=16
        Required baseline cumulative score.
    require_complete_arc : bool, default=True
        Require 419 queries and 400 tasks in every arm.

    Returns
    -------
    dict
        Raw evidence and a recomputed fail-closed qualification report.

    Raises
    ------
    ValueError
        If a named arm is missing or a result cannot be canonically extracted.
    """

    expected = ("baseline", "repeat_intact", "scale", "swap", "reseed")
    if set(results) != set(expected):
        raise ValueError("results must contain the five named matrix arms.")
    if set(training_origins) != {"baseline", "swap"}:
        raise ValueError("training_origins must contain baseline and swap.")
    checkpoint_names = ("baseline", "scale", "swap", "reseed")
    if set(checkpoints) != set(checkpoint_names):
        raise ValueError("checkpoints must contain baseline, scale, swap, and reseed.")
    factor = _positive_finite_real(scale_factor, "scale_factor")
    if factor == 1.0:
        raise ValueError("scale_factor must be non-unit.")
    seed = _integer(reseed_seed, "reseed_seed")
    checkpoint_paths = {
        name: pathlib.Path(checkpoints[name]) for name in checkpoint_names
    }
    resolved_paths = [path.resolve() for path in checkpoint_paths.values()]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Every named checkpoint file must be distinct.")
    checkpoint_audits = {
        name: _checkpoint_audit(path) for name, path in checkpoint_paths.items()
    }
    baseline_checkpoint = checkpoint_audits["baseline"]
    for name in ("scale", "swap", "reseed"):
        if not _same_checkpoint_schema(baseline_checkpoint, checkpoint_audits[name]):
            raise ValueError(f"{name} checkpoint schema/architecture differs.")
        if checkpoint_audits[name]["sha256"] == baseline_checkpoint["sha256"]:
            raise ValueError(f"{name} checkpoint must differ from baseline.")
        if (
            checkpoint_audits[name]["parameter_sha256"]
            == baseline_checkpoint["parameter_sha256"]
        ):
            raise ValueError(f"{name} parameter bytes must differ from baseline.")
    _verify_exact_scale(checkpoint_paths["baseline"], checkpoint_paths["scale"], factor)
    _verify_exact_reseed(checkpoint_paths["baseline"], checkpoint_paths["reseed"], seed)
    origins = {
        name: _training_origin(training_origins[name], name, checkpoint_audits[name])
        for name in ("baseline", "swap")
    }
    if origins["baseline"]["seed"] == origins["swap"]["seed"]:
        raise ValueError("Baseline and swap origins need distinct training seeds.")

    parsed_results: dict[str, Mapping[str, object]] = {}
    runs: dict[str, dict[str, object]] = {}
    accepted_profile_arms: dict[str, dict[str, object]] = {}
    for name in expected:
        raw_result = results[name]
        if not isinstance(raw_result, Mapping):
            raise ValueError(f"{name} result must be a mapping.")
        parsed_results[name] = raw_result
        runs[name] = parameter_dependence_run_from_result(raw_result)
        accepted_profile_arms[name] = _accepted_full_profile(raw_result)
    run_checkpoint_names = {
        "baseline": "baseline",
        "repeat_intact": "baseline",
        "scale": "scale",
        "swap": "swap",
        "reseed": "reseed",
    }
    for name, checkpoint_name in run_checkpoint_names.items():
        _bind_run_checkpoint(runs[name], checkpoint_audits[checkpoint_name], name)
    runs["scale"]["mutation"] = {"kind": "scale", "factor": factor}
    runs["swap"]["mutation"] = {
        "kind": "swap",
        "run_id": origins["swap"]["result_sha256"],
    }
    runs["reseed"]["mutation"] = {"kind": "reseed", "seed": seed}
    evidence = {
        "baseline": runs["baseline"],
        "repeat_intact": runs["repeat_intact"],
        "perturbations": {name: runs[name] for name in ("scale", "swap", "reseed")},
    }
    qualification = assess_parameter_dependence(
        evidence, minimum_cumulative_score=minimum_cumulative_score
    )
    raw_checks = qualification.get("checks")
    raw_reasons = qualification.get("reasons")
    if not isinstance(raw_checks, Mapping) or not isinstance(raw_reasons, list):
        raise RuntimeError("Parameter-dependence assessor returned malformed output.")
    checks: dict[str, bool] = {
        str(name): bool(passed) for name, passed in raw_checks.items()
    }
    reasons: list[str] = [str(reason) for reason in raw_reasons]
    normalized = [
        _normalized_result_configuration(parsed_results[name]) for name in expected
    ]
    checks["matched_configuration"] = all(
        configuration == normalized[0] for configuration in normalized[1:]
    )
    if not checks["matched_configuration"]:
        reasons.append("Matrix arms do not share one matched evaluation configuration.")
    checks["accepted_full_profile"] = all(
        report["passed"] is True for report in accepted_profile_arms.values()
    )
    if not checks["accepted_full_profile"]:
        failures = []
        for name in expected:
            arm_checks = accepted_profile_arms[name]["checks"]
            assert isinstance(arm_checks, Mapping)
            failed = [
                str(check_name)
                for check_name, passed in arm_checks.items()
                if passed is not True
            ]
            if failed:
                failures.append(f"{name} ({', '.join(failed)})")
        reasons.append(
            "Accepted full-matrix profile failed for " + "; ".join(failures) + "."
        )
    checks["complete_arc_manifest"] = bool(
        not require_complete_arc
        or all(
            run["query_count"] == 419 and run["task_count"] == 400
            for run in runs.values()
        )
    )
    if not checks["complete_arc_manifest"]:
        reasons.append("Every matrix arm must contain all 419 queries and 400 tasks.")
    checks["strict_task_manifest_matched"] = all(
        run["strict_task_ids"] == runs["baseline"]["strict_task_ids"]
        for run in runs.values()
    )
    if not checks["strict_task_manifest_matched"]:
        reasons.append("Strict-task order differs across matrix arms.")
    checks["proposal_sets_matched"] = all(
        run["proposal_set_sha256"] == runs["baseline"]["proposal_set_sha256"]
        for run in runs.values()
    )
    if not checks["proposal_sets_matched"]:
        reasons.append("Unordered forest proposal sets differ across matrix arms.")
    checks["origin_training_verified"] = True
    checks["checkpoint_bytes_verified"] = True
    checks["result_checkpoint_bindings_verified"] = True
    qualification["checks"] = checks
    qualification["reasons"] = reasons
    qualification["passed"] = all(checks.values())
    arms = {
        "baseline": {
            "run": runs["baseline"],
            "checkpoint": checkpoint_audits["baseline"],
            "verified_intervention": {
                "kind": "trained_baseline",
                "origin_result_sha256": origins["baseline"]["result_sha256"],
            },
        },
        "repeat_intact": {
            "run": runs["repeat_intact"],
            "checkpoint": checkpoint_audits["baseline"],
            "verified_intervention": {
                "kind": "intact_repeat",
                "same_checkpoint_bytes": True,
            },
        },
        "scale": {
            "run": runs["scale"],
            "checkpoint": checkpoint_audits["scale"],
            "verified_intervention": {
                "kind": "scale",
                "factor": factor,
                "exact_all_leaves": True,
            },
        },
        "swap": {
            "run": runs["swap"],
            "checkpoint": checkpoint_audits["swap"],
            "verified_intervention": {
                "kind": "swap",
                "independently_trained": True,
                "origin_result_sha256": origins["swap"]["result_sha256"],
                "training_seed": origins["swap"]["seed"],
            },
        },
        "reseed": {
            "run": runs["reseed"],
            "checkpoint": checkpoint_audits["reseed"],
            "verified_intervention": {
                "kind": "reseed",
                "seed": seed,
                "exact_all_leaves": True,
                "generator": "brainstate.random.RandomState",
            },
        },
    }
    baseline_result_sha256 = _canonical_sha256(parsed_results["baseline"])
    return {
        "schema_version": 2,
        "authority": "parameter_dependence_matrix",
        "baseline_result_sha256": baseline_result_sha256,
        "approved_completion_target_passed": qualification["passed"],
        "accepted_full_profile": {
            "required": {
                **_ACCEPTED_FULL_PROFILE,
                "checkpoints": list(_ACCEPTED_FULL_PROFILE["checkpoints"]),
            },
            "arms": accepted_profile_arms,
        },
        "origins": origins,
        "arms": arms,
        "evidence": evidence,
        "qualification": qualification,
    }


def _load_json(path: pathlib.Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Example 21 checkpoint perturbations and causal evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    perturb = commands.add_parser("perturb-checkpoint")
    perturb.add_argument("--source", type=pathlib.Path, required=True)
    perturb.add_argument("--destination", type=pathlib.Path, required=True)
    perturb.add_argument("--kind", choices=("scale", "reseed"), required=True)
    perturb.add_argument("--factor", type=float)
    perturb.add_argument("--seed", type=int)
    matrix = commands.add_parser("assess-matrix")
    for name in ("baseline", "repeat_intact", "scale", "swap", "reseed"):
        matrix.add_argument(
            f"--{name.replace('_', '-')}", type=pathlib.Path, required=True
        )
    matrix.add_argument("--baseline-origin", type=pathlib.Path, required=True)
    matrix.add_argument("--swap-origin", type=pathlib.Path, required=True)
    for name in ("baseline", "scale", "swap", "reseed"):
        matrix.add_argument(f"--{name}-checkpoint", type=pathlib.Path, required=True)
    matrix.add_argument("--scale-factor", type=float, required=True)
    matrix.add_argument("--reseed-seed", type=int, required=True)
    matrix.add_argument("--minimum-cumulative-score", type=int, default=16)
    matrix.add_argument("--allow-incomplete-arc", action="store_true")
    matrix.add_argument("--output", type=pathlib.Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checkpoint-perturbation or matrix-assessment command.

    Parameters
    ----------
    argv : sequence of str or None, default=None
        Arguments excluding the executable name. ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Zero for a created perturbation or passing matrix, one for a completed
        but nonqualifying matrix.
    """

    args = _parser().parse_args(argv)
    if args.command == "perturb-checkpoint":
        report = write_checkpoint_perturbation(
            args.source,
            args.destination,
            kind=args.kind,
            factor=args.factor,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    results = {
        name: _load_json(getattr(args, name))
        for name in ("baseline", "repeat_intact", "scale", "swap", "reseed")
    }
    artifact = build_parameter_dependence_matrix(
        results,
        training_origins={
            "baseline": _load_json(args.baseline_origin),
            "swap": _load_json(args.swap_origin),
        },
        checkpoints={
            name: getattr(args, f"{name}_checkpoint")
            for name in ("baseline", "scale", "swap", "reseed")
        },
        scale_factor=args.scale_factor,
        reseed_seed=args.reseed_seed,
        minimum_cumulative_score=args.minimum_cumulative_score,
        require_complete_arc=not args.allow_incomplete_arc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged = args.output.with_name(args.output.name + ".partial")
    try:
        staged.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staged, args.output)
    finally:
        if staged.exists():
            staged.unlink()
    artifact_qualification = artifact.get("qualification")
    if not isinstance(artifact_qualification, Mapping):
        raise RuntimeError("Matrix artifact qualification is malformed.")
    print(json.dumps(artifact_qualification, indent=2, sort_keys=True))
    return 0 if artifact_qualification.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
