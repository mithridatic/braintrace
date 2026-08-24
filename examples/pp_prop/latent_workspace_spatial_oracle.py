"""Run the sealed V22 spatial Conv-LIF PP-prop learning oracle."""

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
import brainunit as u

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
from examples.pp_prop.latent_workspace_online_training import (
    evaluation_online_episodes,
    sample_online_training_chunk,
)
from examples.pp_prop.latent_workspace_spatial_model import (
    MODEL_INPUT_WIDTH,
    SpatialARCConvLIF,
    SpatialModelConfig,
)
from examples.pp_prop.latent_workspace_spatial_training import (
    SpatialPPPropTrainer,
    evaluate_spatial_model,
    save_spatial_checkpoint,
    spatial_parameter_arrays,
    spatial_parameter_digest,
)
from examples.pp_prop.latent_workspace_task import RowEventConfig


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
class SpatialOracleConfig:
    """Configure one sealed V22 spatial PP-prop learning oracle.

    Parameters
    ----------
    output_dir : pathlib.Path
        New artifact directory.
    device : {"cpu", "gpu"}, default="gpu"
        Required JAX platform.
    seed : int, default=2108
        BrainState model-initialization seed.
    training_updates, training_chunk_size, training_batch_size : int
        Positive compiled training schedule; the chunk size divides updates.
    synthetic_task_count : int, default=1400
        Number of training-only synthetic tasks.
    synthetic_demonstrations : int, default=4
        Demonstrations per generated task.
    synthetic_max_grid_size : int, default=12
        Maximum generated side length.
    synthetic_seed : int, default=12108
        Independent training-curriculum seed.
    oracle_task_count : int, default=120
        Number of untouched synthetic oracle tasks.
    oracle_seed : int, default=62108
        Independent oracle seed.
    learning_rate : float, default=0.001
        Adam learning rate.
    spatial_channels : int, default=16
        Number of recurrent Conv-LIF feature maps.
    trace_decay : float, default=2 ** (-1 / 40)
        Single-step PP-prop eligibility decay.
    """

    output_dir: pathlib.Path
    device: str = "gpu"
    seed: int = 2108
    training_updates: int = 1000
    training_chunk_size: int = 20
    training_batch_size: int = 8
    synthetic_task_count: int = 1400
    synthetic_demonstrations: int = 4
    synthetic_max_grid_size: int = 12
    synthetic_seed: int = 12108
    oracle_task_count: int = 120
    oracle_seed: int = 62108
    learning_rate: float = 0.001
    spatial_channels: int = 16
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
            self, "learning_rate", _positive_real(self.learning_rate, "learning_rate")
        )
        decay = _positive_real(self.trace_decay, "trace_decay")
        if decay > 1.0:
            raise ValueError("trace_decay must be at most 1.0.")
        object.__setattr__(self, "trace_decay", decay)
        SpatialModelConfig(
            input_width=MODEL_INPUT_WIDTH,
            spatial_channels=self.spatial_channels,
            seed=self.seed,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-ready oracle configuration.

        Returns
        -------
        dict
            Dataclass fields with the artifact path serialized as text.
        """

        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def _parameter_leaves_finite(model: SpatialARCConvLIF) -> bool:
    for state in model.states(brainstate.ParamState).values():
        leaves = jax.tree.flatten(
            state.value, is_leaf=lambda leaf: isinstance(leaf, u.Quantity)
        )[0]
        if not all(np.isfinite(np.asarray(u.get_mantissa(leaf))).all() for leaf in leaves):
            return False
    return True


def run_spatial_oracle(config: SpatialOracleConfig) -> dict[str, object]:
    """Train and score one sealed synthetic V22 learning oracle.

    Parameters
    ----------
    config : SpatialOracleConfig
        Fully predeclared model, training, and data configuration.

    Returns
    -------
    dict
        Bound checkpoint, mechanism, exact-family, and candidate evidence.
    """

    if not isinstance(config, SpatialOracleConfig):
        raise TypeError("config must be a SpatialOracleConfig instance.")
    devices = jax.devices(config.device)
    if not devices:
        raise RuntimeError(f"Requested JAX {config.device} backend is unavailable.")
    device = devices[0]
    with jax.default_device(device):
        row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
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
        oracle_episodes = evaluation_online_episodes(
            oracle_curriculum.tasks, row_config
        )
        model_config = SpatialModelConfig(
            input_width=MODEL_INPUT_WIDTH,
            spatial_channels=config.spatial_channels,
            seed=config.seed,
        )
        model = SpatialARCConvLIF(model_config)
        parameter_before = spatial_parameter_digest(model)
        groups_before = spatial_parameter_arrays(model)
        evaluation_before = evaluate_spatial_model(
            model,
            oracle_episodes,
            trace_decay=config.trace_decay,
            batch_size=config.training_batch_size,
        )
        trainer = SpatialPPPropTrainer(
            model,
            batch_size=config.training_batch_size,
            learning_rate=config.learning_rate,
            trace_decay=config.trace_decay,
        )
        relation_roots = sorted(
            {str(relation.path[0]) for relation in trainer.learner.graph.hidden_param_op_relations}
        )
        sampling_rng = brainstate.random.RandomState(config.synthetic_seed + 1)
        losses: list[float] = []
        gradient_norms = {name: 0.0 for name in trainer.groups}
        chunk_count = config.training_updates // config.training_chunk_size
        started = time.perf_counter()
        for chunk_index in range(chunk_count):
            chunk = sample_online_training_chunk(
                catalog,
                row_config,
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
                f"[spatial-synthetic] chunk={chunk_index + 1}/{chunk_count} "
                f"loss={observed[-1]:.6f}",
                flush=True,
            )
        training_seconds = time.perf_counter() - started
        parameter_after = spatial_parameter_digest(model)
        groups_after = spatial_parameter_arrays(model)
        evaluation = evaluate_spatial_model(
            model,
            oracle_episodes,
            trace_decay=config.trace_decay,
            batch_size=config.training_batch_size,
        )
        config.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = config.output_dir / "checkpoint.npz"
        checkpoint_digest = save_spatial_checkpoint(model, checkpoint_path)
        checkpoint_file_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    revision, dirty = _source_revision()
    family_summary = exact_family_summary(evaluation["strict_task_ids"])
    groups_moved = {
        name: groups_before[name].tobytes() != groups_after[name].tobytes()
        for name in groups_before
    }
    mechanism_passed = bool(
        set(relation_roots)
        == {"input_conv", "recurrent_conv", "color_head", "height_head", "width_head"}
        and parameter_before != parameter_after
        and all(groups_moved.values())
        and all(np.isfinite(value) and value > 0.0 for value in gradient_norms.values())
        and evaluation_before["candidate_sha256"] != evaluation["candidate_sha256"]
        and _parameter_leaves_finite(model)
        and bool(np.all(np.isfinite(losses)))
    )
    gate_passed = bool(
        mechanism_passed
        and evaluation["strict_task_pass_at_1_count"] > 7
        and len(family_summary["non_label_families"]) >= 2
        and family_summary["non_copy_non_label_ids"]
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
            "parameter_leaves_finite": _parameter_leaves_finite(model),
        },
        "learner": {
            "algorithm": trainer.algorithm,
            "vjp_method": trainer.vjp_method,
            "loss_version": trainer.loss_version,
            "trace_decay": config.trace_decay,
            "compiled_relation_roots": relation_roots,
            "gradient_norm_maxima": gradient_norms,
        },
        "training": {
            "losses": losses,
            "finite": bool(np.all(np.isfinite(losses))),
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
    parser.add_argument("--training-updates", type=int, default=1000)
    parser.add_argument("--training-chunk-size", type=int, default=20)
    parser.add_argument("--training-batch-size", type=int, default=8)
    parser.add_argument("--synthetic-task-count", type=int, default=1400)
    parser.add_argument("--synthetic-demonstrations", type=int, default=4)
    parser.add_argument("--synthetic-max-grid-size", type=int, default=12)
    parser.add_argument("--synthetic-seed", type=int, default=12108)
    parser.add_argument("--oracle-task-count", type=int, default=120)
    parser.add_argument("--oracle-seed", type=int, default=62108)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--spatial-channels", type=int, default=16)
    parser.add_argument("--trace-decay", type=float, default=2.0 ** (-1.0 / 40.0))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the V22 spatial oracle command and print only its gate score.

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
    result = run_spatial_oracle(SpatialOracleConfig(**vars(args)))
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
