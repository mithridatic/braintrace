"""Checkpoint-dependence qualification for the online direct-model line.

Implements the causal matrix of the direct-model-generation contract on the
online exact-schema checkpoint format: baseline, intact repeat, predeclared
0.5x scale, deterministic BrainState reseed (seed 73), and an independently
trained same-schema swap. Each arm is evaluated target-free on the complete
fixed ARC evaluation manifest, and qualification requires the repeat to be
byte-identical while every perturbation independently moves the parameter
digest, the ordered first-prediction grid bytes, and the strict pass-at-one
count. See ``docs/specs/2026-08-23-example21-direct-model-generation.md``.
"""

from __future__ import annotations

import hashlib
import math
import os
import pathlib

import msgspec
import numpy as np

import brainstate

from examples.pp_prop.latent_workspace_direct_experiment import load_corpora
from examples.pp_prop.latent_workspace_expert_model import TaskGatedOnlineRNN
from examples.pp_prop.latent_workspace_expert_training import (
    evaluate_task_gated_model,
)
from examples.pp_prop.latent_workspace_online_oracle import _source_revision
from examples.pp_prop.latent_workspace_online_training import (
    evaluation_online_episodes,
    load_online_checkpoint,
)
from examples.pp_prop.latent_workspace_task import (
    RowEventConfig,
    canonical_task_fingerprint,
)

SCALE_FACTOR = 0.5
RESEED_SEED = 73
EXPECTED_EVALUATION_TASK_COUNT = 400
EXPECTED_EVALUATION_QUERY_COUNT = 419


def _floating_leaves(
    archive: np.lib.npyio.NpzFile, leaves_metadata: list[dict[str, object]]
) -> list[tuple[dict[str, object], np.ndarray]]:
    leaves = []
    for item in leaves_metadata:
        array = np.asarray(archive[item["key"]])
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError("online checkpoints must contain only floating leaves.")
        leaves.append((item, array))
    return leaves


def _leaves_digest(leaves: list[tuple[dict[str, object], np.ndarray]]) -> str:
    digest = hashlib.sha256()
    previous_path = None
    for item, array in leaves:
        state_path = str(item["state_path"])
        if state_path != previous_path:
            digest.update(state_path.encode("utf-8"))
            previous_path = state_path
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _reseed_online_leaves(
    leaves: list[tuple[dict[str, object], np.ndarray]], seed: int
) -> dict[str, np.ndarray]:
    generator = brainstate.random.RandomState(seed)
    output: dict[str, np.ndarray] = {}
    for item, source_value in leaves:
        draw = np.asarray(
            generator.normal(size=source_value.shape, dtype=np.float32),
            dtype=source_value.dtype,
        )
        source_rms = float(
            np.sqrt(np.mean(np.square(source_value.astype(np.float64))))
        )
        draw_rms = float(np.sqrt(np.mean(np.square(draw.astype(np.float64)))))
        target_rms = max(source_rms, float(np.finfo(source_value.dtype).eps))
        if not math.isfinite(draw_rms) or draw_rms == 0.0:
            raise ValueError("BrainState reseed produced an invalid random leaf.")
        output[item["key"]] = np.asarray(
            draw * np.asarray(target_rms / draw_rms, dtype=draw.dtype),
            dtype=source_value.dtype,
        )
    return output


def write_online_checkpoint_perturbation(
    source: str | pathlib.Path,
    destination: str | pathlib.Path,
    *,
    kind: str,
    factor: float | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Write an exact-schema scale or deterministic-reseed online checkpoint.

    Parameters
    ----------
    source, destination : path-like
        Existing online checkpoint and new output path. The source is never
        overwritten.
    kind : {"scale", "reseed"}
        Intervention applied to every floating parameter leaf.
    factor : float or None, default=None
        Positive finite non-unit scale, required for ``kind="scale"``.
    seed : int or None, default=None
        Nonnegative BrainState seed, required for ``kind="reseed"``.

    Returns
    -------
    dict
        Intervention detail plus source/output file and parameter digests.

    Raises
    ------
    ValueError
        If the intervention or checkpoint schema is invalid, or the written
        checkpoint fails exact-schema round-trip validation.
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
    with np.load(source_path, allow_pickle=False) as archive:
        names = set(archive.files)
        if "__metadata__" not in names:
            raise ValueError("checkpoint metadata is missing.")
        metadata = msgspec.json.decode(bytes(archive["__metadata__"]))
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ValueError("checkpoint schema_version is unsupported.")
        leaves_metadata = metadata.get("leaves")
        if not isinstance(leaves_metadata, list):
            raise ValueError("checkpoint metadata schema is invalid.")
        if names != {item["key"] for item in leaves_metadata} | {"__metadata__"}:
            raise ValueError("checkpoint leaf set does not match metadata.")
        leaves = _floating_leaves(archive, leaves_metadata)
    source_digest = _leaves_digest(leaves)
    if source_digest != metadata.get("parameter_sha256"):
        raise ValueError("checkpoint parameter digest does not match its contents.")
    if kind == "scale":
        if factor is None or not math.isfinite(float(factor)) or float(factor) <= 0.0:
            raise ValueError("factor must be a positive finite real.")
        scale = float(factor)
        if scale == 1.0:
            raise ValueError("factor must be non-unit.")
        perturbed = {
            item["key"]: np.asarray(
                array * np.asarray(scale, dtype=array.dtype), dtype=array.dtype
            )
            for item, array in leaves
        }
        detail: dict[str, object] = {"factor": scale}
    else:
        if seed is None or isinstance(seed, bool) or int(seed) < 0:
            raise ValueError("seed must be a nonnegative integer.")
        perturbed = _reseed_online_leaves(leaves, int(seed))
        detail = {"seed": int(seed)}
    perturbed_leaves = [
        (item, perturbed[item["key"]]) for item, _ in leaves
    ]
    output_metadata = dict(metadata)
    output_metadata["parameter_sha256"] = _leaves_digest(perturbed_leaves)
    arrays = {
        item["key"]: np.ascontiguousarray(perturbed[item["key"]])
        for item, _ in leaves
    }
    arrays["__metadata__"] = np.frombuffer(
        msgspec.json.encode(output_metadata, order="sorted"), dtype=np.uint8
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staged = destination_path.with_name(destination_path.name + ".partial")
    try:
        with staged.open("wb") as handle:
            np.savez(handle, **arrays)
        os.replace(staged, destination_path)
    finally:
        if staged.exists():
            staged.unlink()
    _, restored_metadata = load_online_checkpoint(destination_path)
    return {
        "kind": kind,
        **detail,
        "leaf_count": len(leaves),
        "source_checkpoint_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(
            destination_path.read_bytes()
        ).hexdigest(),
        "source_parameter_sha256": source_digest,
        "parameter_sha256": restored_metadata["parameter_sha256"],
    }


def evaluate_checkpoint_on_manifest(
    checkpoint_path: str | pathlib.Path,
    corpora,
    row_config: RowEventConfig,
    *,
    batch_size: int = 10,
) -> dict[str, object]:
    """Evaluate one online checkpoint on the complete evaluation manifest.

    Parameters
    ----------
    checkpoint_path : path-like
        Exact-schema online checkpoint.
    corpora : DirectCorpora
        Integrity-checked corpora; only the evaluation role is scored.
    row_config : RowEventConfig
        Fixed lossless row-event configuration.
    batch_size : int, default=10
        Positive static evaluation batch size.

    Returns
    -------
    dict
        Strict count and membership, candidate digest, and bound digests.
    """

    model, metadata = load_online_checkpoint(checkpoint_path)
    if not isinstance(model, TaskGatedOnlineRNN):
        raise TypeError("matrix evaluation requires a task-gated model class.")
    episodes = evaluation_online_episodes(corpora.evaluation, row_config)
    result = evaluate_task_gated_model(
        model, episodes, trace_decay=2.0 ** (-1.0 / 40.0), batch_size=batch_size
    )
    return {
        "checkpoint_file_sha256": hashlib.sha256(
            pathlib.Path(checkpoint_path).read_bytes()
        ).hexdigest(),
        "parameter_sha256": metadata["parameter_sha256"],
        "architecture_version": metadata["architecture"]["architecture_version"],
        "strict_task_pass_at_1_count": result["strict_task_pass_at_1_count"],
        "strict_task_ids": list(result["strict_task_ids"]),
        "task_membership": dict(result["task_membership"]),
        "candidate_sha256": result["candidate_sha256"],
        "candidate_bytes_size": result["candidate_bytes_size"],
        "task_count": result["task_count"],
        "query_count": result["query_count"],
    }


def assess_online_dependence(arms: dict[str, dict[str, object]]) -> dict[str, object]:
    """Apply the causal qualification contract to five evaluated arms.

    Parameters
    ----------
    arms : dict
        Mapping with keys baseline, repeat_intact, scale, reseed, and swap,
        each holding an :func:`evaluate_checkpoint_on_manifest` result.

    Returns
    -------
    dict
        Fail-closed qualification report with per-check booleans and reasons.
    """

    required = ("baseline", "repeat_intact", "scale", "reseed", "swap")
    missing = [name for name in required if name not in arms]
    if missing:
        raise ValueError(f"arms are missing: {missing}")
    baseline = arms["baseline"]
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    repeat = arms["repeat_intact"]
    checks["repeat_checkpoint_identical"] = (
        repeat["checkpoint_file_sha256"] == baseline["checkpoint_file_sha256"]
    )
    checks["repeat_parameter_identical"] = (
        repeat["parameter_sha256"] == baseline["parameter_sha256"]
    )
    checks["repeat_candidate_identical"] = (
        repeat["candidate_sha256"] == baseline["candidate_sha256"]
    )
    checks["repeat_score_identical"] = (
        repeat["strict_task_pass_at_1_count"]
        == baseline["strict_task_pass_at_1_count"]
    )
    checks["repeat_membership_identical"] = (
        repeat["strict_task_ids"] == baseline["strict_task_ids"]
        and repeat["task_membership"] == baseline["task_membership"]
    )
    for name in ("scale", "reseed", "swap"):
        arm = arms[name]
        checks[f"{name}_parameter_moved"] = (
            arm["parameter_sha256"] != baseline["parameter_sha256"]
        )
        checks[f"{name}_candidate_moved"] = (
            arm["candidate_sha256"] != baseline["candidate_sha256"]
        )
        checks[f"{name}_score_moved"] = (
            arm["strict_task_pass_at_1_count"]
            != baseline["strict_task_pass_at_1_count"]
        )
    for name, passed in checks.items():
        if not passed:
            reasons.append(f"{name} failed")
    return {
        "passed": bool(all(checks.values()) and not reasons),
        "checks": checks,
        "reasons": reasons,
    }


def run_online_dependence_matrix(
    *,
    baseline_checkpoint: str | pathlib.Path,
    swap_checkpoint: str | pathlib.Path,
    source_manifest: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    scale_factor: float = SCALE_FACTOR,
    reseed_seed: int = RESEED_SEED,
    batch_size: int = 10,
) -> dict[str, object]:
    """Evaluate and assess the five-arm online dependence matrix.

    Parameters
    ----------
    baseline_checkpoint, swap_checkpoint : path-like
        Independently trained same-schema checkpoints with distinct training
        seeds; the baseline is the nominated qualifying checkpoint.
    source_manifest : path-like
        Integrity-indexed ARC source manifest.
    output_dir : path-like
        New artifact directory receiving perturbed checkpoints and the
        matrix result.
    scale_factor : float, default=0.5
        Predeclared non-unit scale applied to every floating leaf.
    reseed_seed : int, default=73
        Predeclared BrainState reseed seed.
    batch_size : int, default=10
        Static evaluation batch size.

    Returns
    -------
    dict
        Bound per-arm evidence and the fail-closed qualification report.
    """

    baseline_path = pathlib.Path(baseline_checkpoint)
    swap_path = pathlib.Path(swap_checkpoint)
    if baseline_path.resolve() == swap_path.resolve():
        raise ValueError("baseline and swap checkpoints must differ.")
    if not float(scale_factor) > 0.0 and math.isfinite(float(scale_factor)):
        raise ValueError("scale_factor must be a positive finite real.")
    if float(scale_factor) == 1.0:
        raise ValueError("scale_factor must be non-unit.")
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    corpora = load_corpora(pathlib.Path(source_manifest))
    if len(corpora.evaluation) != EXPECTED_EVALUATION_TASK_COUNT:
        raise ValueError("complete evaluation manifest must contain 400 tasks.")
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    manifest_ids = [task.task_id for task in corpora.evaluation]
    manifest_fingerprints = [
        canonical_task_fingerprint(task) for task in corpora.evaluation
    ]
    manifest_digest = hashlib.sha256(
        "\n".join(
            f"{task_id}:{fingerprint}"
            for task_id, fingerprint in zip(manifest_ids, manifest_fingerprints)
        ).encode("ascii")
    ).hexdigest()

    scale_checkpoint = output_path / "checkpoint-scale.npz"
    reseed_checkpoint = output_path / "checkpoint-reseed.npz"
    scale_report = write_online_checkpoint_perturbation(
        baseline_path, scale_checkpoint, kind="scale", factor=float(scale_factor)
    )
    reseed_report = write_online_checkpoint_perturbation(
        baseline_path, reseed_checkpoint, kind="reseed", seed=int(reseed_seed)
    )
    arms = {
        "baseline": evaluate_checkpoint_on_manifest(
            baseline_path, corpora, row_config, batch_size=batch_size
        ),
        "repeat_intact": evaluate_checkpoint_on_manifest(
            baseline_path, corpora, row_config, batch_size=batch_size
        ),
        "scale": evaluate_checkpoint_on_manifest(
            scale_checkpoint, corpora, row_config, batch_size=batch_size
        ),
        "reseed": evaluate_checkpoint_on_manifest(
            reseed_checkpoint, corpora, row_config, batch_size=batch_size
        ),
        "swap": evaluate_checkpoint_on_manifest(
            swap_path, corpora, row_config, batch_size=batch_size
        ),
    }
    for name, arm in arms.items():
        if arm["task_count"] != EXPECTED_EVALUATION_TASK_COUNT:
            raise ValueError(f"arm {name} did not score 400 evaluation tasks.")
        if arm["query_count"] != EXPECTED_EVALUATION_QUERY_COUNT:
            raise ValueError(f"arm {name} did not score 419 evaluation queries.")
    qualification = assess_online_dependence(arms)
    revision, dirty = _source_revision()
    result = {
        "schema_version": 1,
        "authority": "online_parameter_dependence_matrix",
        "source_revision": revision,
        "source_dirty": dirty,
        "sources": [item.manifest.to_dict() for item in corpora.loaded],
        "manifest": {
            "task_count": len(corpora.evaluation),
            "query_count": EXPECTED_EVALUATION_QUERY_COUNT,
            "ordered_task_ids": manifest_ids,
            "manifest_sha256": manifest_digest,
        },
        "interventions": {
            "scale": scale_report,
            "reseed": reseed_report,
        },
        "arms": arms,
        "qualification": qualification,
    }
    (output_path / "matrix-result.json").write_bytes(
        msgspec.json.encode(result, order="sorted")
    )
    return result
