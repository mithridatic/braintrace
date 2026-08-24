"""Sealed synthetic capability screen for V46 paired spatial recurrence."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral, Real
import pathlib
import time

import brainstate
import jax
import msgspec
import numpy as np

from examples.pp_prop.latent_workspace_direct_curriculum import (
    SyntheticCurriculumConfig,
    generate_synthetic_curriculum,
)
from examples.pp_prop.latent_workspace_direct_experiment import (
    training_episode_catalog,
)
from examples.pp_prop.latent_workspace_online_oracle import (
    _source_revision,
    exact_family_summary,
)
from examples.pp_prop.latent_workspace_paired_spatial_model import (
    PairedSpatialARC,
    PairedSpatialConfig,
)
from examples.pp_prop.latent_workspace_paired_spatial_training import (
    PairedSpatialPPPropTrainer,
    encode_paired_spatial_episode,
    evaluate_paired_spatial_model,
    paired_spatial_parameter_arrays,
    paired_spatial_parameter_digest,
    paired_spatial_parameter_leaf_arrays,
    sample_paired_spatial_training_chunk,
    save_paired_spatial_checkpoint,
)

STRUCTURAL_FAMILIES = frozenset(
    (
        "dihedral",
        "crop",
        "upscale",
        "project_marker",
        "complete_corner",
        "mirror_concat",
    )
)
RECURRENT_ROOTS = frozenset(
    (
        "demo_input_conv",
        "demo_recurrent_conv",
        "query_input_conv",
        "query_recurrent_conv",
    )
)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return integer


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real.")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite real.")
    return number


@dataclass(frozen=True)
class PairedSpatialOracleConfig:
    """Configure one sealed V46 synthetic capability screen.

    Parameters
    ----------
    output_dir : pathlib.Path
        New artifact directory.
    device : {"cpu", "gpu"}, default="gpu"
        Required JAX backend.
    seed : int, default=2108
        BrainState model-initialization seed.
    training_updates, training_chunk_size, training_batch_size : int
        Positive compiled training schedule.
    synthetic_task_count : int, default=800
        Number of training-only generated tasks.
    synthetic_demonstrations : int, default=4
        Demonstrations per generated task.
    synthetic_max_grid_size : int, default=12
        Maximum generated input or output side.
    synthetic_seed : int, default=27108
        Independent training-curriculum seed.
    oracle_task_count : int, default=60
        Number of untouched synthetic screen tasks.
    oracle_seed : int, default=132108
        Independent screen seed.
    learning_rate : float, default=0.001
        Positive Adam learning rate.
    spatial_channels : int, default=32
        Channels in each recurrent canvas.
    refinement_steps : int, default=8
        Target-free query message-passing steps.
    retention : float, default=0.8
        Fixed recurrent retention in ``[0, 1)``.
    minimum_strict_task_count : int, default=5
        Inclusive strict screen threshold.
    trace_decay : float, default=2 ** (-1 / 40)
        Single-step PP-prop trace decay.
    """

    output_dir: pathlib.Path
    device: str = "gpu"
    seed: int = 2108
    training_updates: int = 400
    training_chunk_size: int = 20
    training_batch_size: int = 8
    synthetic_task_count: int = 800
    synthetic_demonstrations: int = 4
    synthetic_max_grid_size: int = 12
    synthetic_seed: int = 27108
    oracle_task_count: int = 60
    oracle_seed: int = 132108
    learning_rate: float = 0.001
    spatial_channels: int = 32
    refinement_steps: int = 8
    retention: float = 0.8
    minimum_strict_task_count: int = 5
    trace_decay: float = 2.0 ** (-1.0 / 40.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", pathlib.Path(self.output_dir))
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'.")
        for name in ("seed", "synthetic_seed", "oracle_seed"):
            object.__setattr__(
                self, name, _nonnegative_integer(getattr(self, name), name)
            )
        if self.synthetic_seed == self.oracle_seed:
            raise ValueError("synthetic_seed and oracle_seed must be different.")
        for name in (
            "training_updates",
            "training_chunk_size",
            "training_batch_size",
            "synthetic_task_count",
            "oracle_task_count",
            "spatial_channels",
            "refinement_steps",
            "minimum_strict_task_count",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        if self.training_updates % self.training_chunk_size:
            raise ValueError("training_chunk_size must divide training_updates.")
        curriculum = SyntheticCurriculumConfig(
            task_count=self.synthetic_task_count,
            demonstrations=self.synthetic_demonstrations,
            max_grid_size=self.synthetic_max_grid_size,
        )
        object.__setattr__(
            self, "synthetic_demonstrations", curriculum.demonstrations
        )
        object.__setattr__(
            self, "synthetic_max_grid_size", curriculum.max_grid_size
        )
        object.__setattr__(
            self,
            "learning_rate",
            _positive_real(self.learning_rate, "learning_rate"),
        )
        retention = float(self.retention)
        model_config = PairedSpatialConfig(
            spatial_channels=self.spatial_channels,
            refinement_steps=self.refinement_steps,
            retention=retention,
            seed=self.seed,
        )
        object.__setattr__(self, "retention", model_config.retention)
        decay = _positive_real(self.trace_decay, "trace_decay")
        if decay > 1.0:
            raise ValueError("trace_decay must be at most 1.0.")
        object.__setattr__(self, "trace_decay", decay)

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-ready configuration.

        Returns
        -------
        dict
            Dataclass fields with the artifact path serialized as text.
        """

        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def paired_spatial_promotion_gate(
    strict_count: int,
    family_summary: dict[str, object],
    mechanism_passed: bool,
    anti_collapse_passed: bool,
    *,
    minimum: int,
) -> bool:
    """Apply the sealed V46 strict, diversity, and structural boundary.

    Parameters
    ----------
    strict_count : int
        Observed strict task pass-at-one count.
    family_summary : dict
        Exact synthetic family membership summary.
    mechanism_passed, anti_collapse_passed : bool
        Whether both independent diagnostic gates passed.
    minimum : int
        Inclusive predeclared strict-count threshold.

    Returns
    -------
    bool
        True only when every predeclared V46 condition passes.
    """

    observed = _nonnegative_integer(strict_count, "strict_count")
    required = _positive_integer(minimum, "minimum")
    if not isinstance(family_summary, dict):
        raise TypeError("family_summary must be a dictionary.")
    if not isinstance(mechanism_passed, bool):
        raise TypeError("mechanism_passed must be boolean.")
    if not isinstance(anti_collapse_passed, bool):
        raise TypeError("anti_collapse_passed must be boolean.")
    counts = family_summary.get("family_counts", {})
    structural = bool(
        isinstance(counts, dict) and STRUCTURAL_FAMILIES.intersection(counts)
    )
    return bool(
        observed >= required
        and mechanism_passed
        and anti_collapse_passed
        and len(family_summary.get("non_label_families", ())) >= 2
        and structural
    )


def _compiler_summary(
    trainer: PairedSpatialPPPropTrainer,
) -> dict[str, object]:
    report = trainer.learner.report
    return {
        "counts": dict(report.counts),
        "etrace_weight_paths": [
            ".".join(map(str, path)) for path, _ in report.etrace_weights
        ],
        "excluded_paths": [
            ".".join(map(str, path)) for path, _ in report.excluded_weights
        ],
        "recurrent_excluded_paths": [
            ".".join(map(str, path))
            for path, _ in report.excluded_weights
            if path[0] in RECURRENT_ROOTS
        ],
        "diagnostic_kinds": [item.kind.value for item in report.diagnostics],
    }


def _parameter_leaves_finite(model: PairedSpatialARC) -> bool:
    return all(
        bool(np.isfinite(np.asarray(leaf)).all())
        for state in model.states(brainstate.ParamState).values()
        for leaf in jax.tree.leaves(state.value)
    )


def run_paired_spatial_oracle(
    config: PairedSpatialOracleConfig,
) -> dict[str, object]:
    """Train and score one sealed V46 synthetic capability screen.

    Parameters
    ----------
    config : PairedSpatialOracleConfig
        Fully predeclared architecture, data, schedule, and gate.

    Returns
    -------
    dict
        Bound model, compiler, checkpoint, candidate, and gate evidence.
    """

    if not isinstance(config, PairedSpatialOracleConfig):
        raise TypeError("config must be a PairedSpatialOracleConfig instance.")
    devices = jax.devices(config.device)
    if not devices:
        raise RuntimeError(f"Requested JAX {config.device} backend is unavailable.")
    device = devices[0]
    with jax.default_device(device):
        training_curriculum = generate_synthetic_curriculum(
            SyntheticCurriculumConfig(
                task_count=config.synthetic_task_count,
                demonstrations=config.synthetic_demonstrations,
                max_grid_size=config.synthetic_max_grid_size,
            ),
            brainstate.random.RandomState(config.synthetic_seed),
        )
        oracle_curriculum = generate_synthetic_curriculum(
            SyntheticCurriculumConfig(
                task_count=config.oracle_task_count,
                demonstrations=config.synthetic_demonstrations,
                max_grid_size=config.synthetic_max_grid_size,
            ),
            brainstate.random.RandomState(config.oracle_seed),
        )
        catalog = training_episode_catalog(training_curriculum.tasks)
        model_config = PairedSpatialConfig(
            spatial_channels=config.spatial_channels,
            refinement_steps=config.refinement_steps,
            retention=config.retention,
            seed=config.seed,
        )
        oracle_episodes = tuple(
            encode_paired_spatial_episode(task, query_index, model_config)
            for task in oracle_curriculum.tasks
            for query_index in range(len(task.test))
        )
        model = PairedSpatialARC(model_config)
        parameter_before = paired_spatial_parameter_digest(model)
        groups_before = paired_spatial_parameter_arrays(model)
        leaves_before = paired_spatial_parameter_leaf_arrays(model)
        evaluation_before = evaluate_paired_spatial_model(
            model,
            oracle_episodes,
            trace_decay=config.trace_decay,
            batch_size=config.training_batch_size,
        )
        trainer = PairedSpatialPPPropTrainer(
            model,
            batch_size=config.training_batch_size,
            learning_rate=config.learning_rate,
            trace_decay=config.trace_decay,
        )
        compiler = _compiler_summary(trainer)
        sampling_rng = brainstate.random.RandomState(config.synthetic_seed + 1)
        losses: list[float] = []
        gradient_norms = {name: 0.0 for name in trainer.groups}
        chunk_count = config.training_updates // config.training_chunk_size
        started = time.perf_counter()
        for chunk_index in range(chunk_count):
            chunk = sample_paired_spatial_training_chunk(
                catalog,
                model_config,
                sampling_rng,
                updates=config.training_chunk_size,
                batch_size=config.training_batch_size,
                augment=False,
            )
            observed_losses, observed_norms = trainer.train_chunk(chunk)
            observed = np.asarray(observed_losses, dtype=np.float64)
            losses.extend(observed.tolist())
            for name, value in observed_norms.items():
                gradient_norms[name] = max(gradient_norms[name], value)
            print(
                f"[paired-spatial] chunk={chunk_index + 1}/{chunk_count} "
                f"loss={observed[-1]:.6f}",
                flush=True,
            )
        training_seconds = time.perf_counter() - started
        parameter_after = paired_spatial_parameter_digest(model)
        groups_after = paired_spatial_parameter_arrays(model)
        leaves_after = paired_spatial_parameter_leaf_arrays(model)
        evaluation = evaluate_paired_spatial_model(
            model,
            oracle_episodes,
            trace_decay=config.trace_decay,
            batch_size=config.training_batch_size,
        )
        config.output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_path = config.output_dir / "checkpoint.npz"
        checkpoint_digest = save_paired_spatial_checkpoint(model, checkpoint_path)
        checkpoint_file_digest = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
    revision, dirty = _source_revision()
    family_summary = exact_family_summary(evaluation["strict_task_ids"])
    groups_moved = {
        name: groups_before[name].tobytes() != groups_after[name].tobytes()
        for name in groups_before
    }
    leaves_moved = {
        name: leaves_before[name].tobytes() != leaves_after[name].tobytes()
        for name in leaves_before
    }
    mechanism_passed = bool(
        not compiler["recurrent_excluded_paths"]
        and parameter_before != parameter_after
        and all(groups_moved.values())
        and leaves_before.keys() == leaves_after.keys()
        and all(leaves_moved.values())
        and all(
            np.isfinite(value) and value > 0.0 for value in gradient_norms.values()
        )
        and evaluation_before["candidate_sha256"]
        != evaluation["candidate_sha256"]
        and _parameter_leaves_finite(model)
        and bool(np.isfinite(losses).all())
    )
    diagnostics = evaluation["diagnostics"]
    predicted_nonzero = sum(
        count
        for color, count in diagnostics["predicted_color_counts"].items()
        if color != "0"
    )
    anti_collapse_passed = bool(
        predicted_nonzero > 0
        and diagnostics["foreground_total"] > 0
        and diagnostics["foreground_accuracy"] >= 0.05
        and diagnostics["background_total"] > 0
        and diagnostics["background_accuracy"] >= 0.5
    )
    gate_passed = paired_spatial_promotion_gate(
        evaluation["strict_task_pass_at_1_count"],
        family_summary,
        mechanism_passed,
        anti_collapse_passed,
        minimum=config.minimum_strict_task_count,
    )
    result = {
        "schema_version": 1,
        "configuration": config.to_dict(),
        "source_revision": revision,
        "source_dirty": dirty,
        "device": {
            "platform": str(device.platform),
            "kind": str(getattr(device, "device_kind", device)),
        },
        "data": {
            "training_schema_version": training_curriculum.schema_version,
            "training_task_count": len(training_curriculum.tasks),
            "training_family_counts": training_curriculum.family_counts,
            "training_task_sha256": training_curriculum.task_sha256,
            "training_episode_count": len(catalog),
            "oracle_task_count": len(oracle_curriculum.tasks),
            "oracle_family_counts": oracle_curriculum.family_counts,
            "oracle_task_sha256": oracle_curriculum.task_sha256,
        },
        "model": {
            "architecture": asdict(model_config),
            "parameter_sha256_before": parameter_before,
            "parameter_sha256_after": parameter_after,
            "parameters_moved": parameter_before != parameter_after,
            "parameter_groups_moved": groups_moved,
            "parameter_leaves_moved": leaves_moved,
            "parameter_leaves_finite": _parameter_leaves_finite(model),
        },
        "learner": {
            "algorithm": trainer.algorithm,
            "vjp_method": trainer.vjp_method,
            "loss_version": trainer.loss_version,
            "trace_decay": config.trace_decay,
            "gradient_norm_maxima": gradient_norms,
            "compiler": compiler,
        },
        "training": {
            "losses": losses,
            "finite": bool(np.isfinite(losses).all()),
            "wall_seconds": training_seconds,
        },
        "checkpoint": {
            "filename": checkpoint_path.name,
            "parameter_sha256": checkpoint_digest,
            "file_sha256": checkpoint_file_digest,
        },
        "evaluation_before_training": evaluation_before,
        "evaluation": evaluation,
        "exact_family_summary": family_summary,
        "mechanism_gate_passed": mechanism_passed,
        "anti_collapse_gate_passed": anti_collapse_passed,
        "promotion_gate_passed": gate_passed,
    }
    (config.output_dir / "result.json").write_bytes(
        msgspec.json.encode(result, order="sorted")
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=2108)
    parser.add_argument("--training-updates", type=int, default=400)
    parser.add_argument("--training-chunk-size", type=int, default=20)
    parser.add_argument("--training-batch-size", type=int, default=8)
    parser.add_argument("--synthetic-task-count", type=int, default=800)
    parser.add_argument("--synthetic-demonstrations", type=int, default=4)
    parser.add_argument("--synthetic-max-grid-size", type=int, default=12)
    parser.add_argument("--synthetic-seed", type=int, default=27108)
    parser.add_argument("--oracle-task-count", type=int, default=60)
    parser.add_argument("--oracle-seed", type=int, default=132108)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--spatial-channels", type=int, default=32)
    parser.add_argument("--refinement-steps", type=int, default=8)
    parser.add_argument("--retention", type=float, default=0.8)
    parser.add_argument("--minimum-strict-task-count", type=int, default=5)
    parser.add_argument("--trace-decay", type=float, default=2.0 ** (-1.0 / 40.0))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run V46 and print its strict promotion score.

    Parameters
    ----------
    argv : list of str, optional
        Explicit command arguments; defaults to process arguments.

    Returns
    -------
    int
        Zero after a complete artifact write.
    """

    args = _parser().parse_args(argv)
    result = run_paired_spatial_oracle(PairedSpatialOracleConfig(**vars(args)))
    print(
        msgspec.json.encode(
            {
                "strict_task_pass_at_1_count": result["evaluation"][
                    "strict_task_pass_at_1_count"
                ],
                "promotion_gate_passed": result["promotion_gate_passed"],
            },
            order="sorted",
        ).decode("utf-8"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
