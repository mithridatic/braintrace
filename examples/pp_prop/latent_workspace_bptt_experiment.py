"""Run the matched pp-prop-versus-BPTT credit-assignment arm.

Trains a fresh V44 checkpoint on the exact V47 training curriculum (same
schema, seed, and task count) with full reverse-mode gradients, and scores
it on the exact V47 holdout. The comparison decides whether credit
assignment or representation limits spatial-operator learning; see
``docs/specs/2026-08-24-example21-causal-phase-map.md``.
"""

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

from examples.pp_prop.latent_workspace_bptt_training import TaskGatedBPTTTrainer
from examples.pp_prop.latent_workspace_direct_experiment import (
    training_episode_catalog,
)
from examples.pp_prop.latent_workspace_diverse_curriculum import (
    DiverseCurriculumConfig,
    generate_diverse_curriculum,
)
from examples.pp_prop.latent_workspace_diverse_experiment import (
    _v4_family_summary,
)
from examples.pp_prop.latent_workspace_expert_training import (
    evaluate_task_gated_model,
    parameter_leaf_arrays,
)
from examples.pp_prop.latent_workspace_gated_memory_model import (
    MODEL_INPUT_WIDTH,
    GatedMemoryConfig,
    PhaseSeparatedGatedMemoryRNN,
)
from examples.pp_prop.latent_workspace_online_oracle import _source_revision
from examples.pp_prop.latent_workspace_online_training import (
    evaluation_online_episodes,
    parameter_arrays,
    parameter_digest,
    sample_online_training_chunk,
    save_online_checkpoint,
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
class BPTTExperimentConfig:
    """Configure the matched reverse-mode credit-assignment arm.

    Parameters
    ----------
    output_dir : pathlib.Path
        New artifact directory.
    device : {"cpu", "gpu"}, default="gpu"
        Required JAX backend.
    seed : int, default=2108
        BrainState model-initialization seed, matched to the PP-prop arm.
    synthetic_seed : int, default=33108
        V4 training-curriculum seed, matched to the PP-prop arm.
    holdout_seed : int, default=44108
        Untouched v4 holdout seed, matched to the PP-prop arm.
    synthetic_task_count, holdout_task_count : int
        Curriculum sizes, matched to the PP-prop arm (1400 and 120).
    max_grid_size, min_demonstrations, max_demonstrations : int
        V4 curriculum dimensions, matched to the PP-prop arm.
    training_updates, training_chunk_size, training_batch_size : int
        Compiled training schedule, matched to the PP-prop arm (800/20/8).
    learning_rate : float, default=0.001
        Adam learning rate, matched to the PP-prop arm.
    memory_width : int, default=128
        Width of each phase-separated MiniLSTM population.
    expert_count : int, default=12
        Number of checkpoint-owned neural colour experts.
    """

    output_dir: pathlib.Path
    device: str = "gpu"
    seed: int = 2108
    synthetic_seed: int = 33108
    holdout_seed: int = 44108
    synthetic_task_count: int = 1400
    holdout_task_count: int = 120
    max_grid_size: int = 30
    min_demonstrations: int = 2
    max_demonstrations: int = 6
    training_updates: int = 800
    training_chunk_size: int = 20
    training_batch_size: int = 8
    learning_rate: float = 0.001
    memory_width: int = 128
    expert_count: int = 12

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", pathlib.Path(self.output_dir))
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'.")
        for name in ("seed", "synthetic_seed", "holdout_seed"):
            object.__setattr__(
                self, name, _nonnegative_integer(getattr(self, name), name)
            )
        if self.synthetic_seed == self.holdout_seed:
            raise ValueError("synthetic_seed and holdout_seed must be different.")
        for name in (
            "synthetic_task_count",
            "holdout_task_count",
            "training_updates",
            "training_chunk_size",
            "training_batch_size",
            "memory_width",
            "expert_count",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        if self.training_updates % self.training_chunk_size:
            raise ValueError("training_chunk_size must divide training_updates.")
        curriculum = DiverseCurriculumConfig(
            task_count=self.synthetic_task_count,
            max_grid_size=self.max_grid_size,
            min_demonstrations=self.min_demonstrations,
            max_demonstrations=self.max_demonstrations,
        )
        object.__setattr__(self, "max_grid_size", curriculum.max_grid_size)
        object.__setattr__(self, "min_demonstrations", curriculum.min_demonstrations)
        object.__setattr__(self, "max_demonstrations", curriculum.max_demonstrations)
        object.__setattr__(
            self,
            "learning_rate",
            _positive_real(self.learning_rate, "learning_rate"),
        )
        GatedMemoryConfig(
            input_width=MODEL_INPUT_WIDTH,
            memory_width=self.memory_width,
            expert_count=self.expert_count,
            seed=self.seed,
        )

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


def _parameter_leaves_finite(model: PhaseSeparatedGatedMemoryRNN) -> bool:
    return all(
        bool(np.isfinite(np.asarray(leaf)).all())
        for state in model.states(brainstate.ParamState).values()
        for leaf in jax.tree.leaves(state.value)
    )


def run_bptt_experiment(config: BPTTExperimentConfig) -> dict[str, object]:
    """Train and score the matched reverse-mode credit-assignment arm.

    Parameters
    ----------
    config : BPTTExperimentConfig
        Fully predeclared architecture, data, and training configuration.

    Returns
    -------
    dict
        Bound model, checkpoint, candidate, and per-family evidence.
    """

    if not isinstance(config, BPTTExperimentConfig):
        raise TypeError("config must be a BPTTExperimentConfig instance.")
    devices = jax.devices(config.device)
    if not devices:
        raise RuntimeError(f"Requested JAX {config.device} backend is unavailable.")
    device = devices[0]
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    curriculum_config = DiverseCurriculumConfig(
        task_count=config.synthetic_task_count,
        max_grid_size=config.max_grid_size,
        min_demonstrations=config.min_demonstrations,
        max_demonstrations=config.max_demonstrations,
    )
    training_curriculum = generate_diverse_curriculum(
        curriculum_config, brainstate.random.RandomState(config.synthetic_seed)
    )
    holdout_curriculum = generate_diverse_curriculum(
        DiverseCurriculumConfig(
            task_count=config.holdout_task_count,
            max_grid_size=config.max_grid_size,
            min_demonstrations=config.min_demonstrations,
            max_demonstrations=config.max_demonstrations,
        ),
        brainstate.random.RandomState(config.holdout_seed),
    )
    catalog = training_episode_catalog(training_curriculum.tasks)
    holdout_episodes = evaluation_online_episodes(
        holdout_curriculum.tasks, row_config
    )

    with jax.default_device(device):
        model_config = GatedMemoryConfig(
            input_width=MODEL_INPUT_WIDTH,
            memory_width=config.memory_width,
            expert_count=config.expert_count,
            seed=config.seed,
        )
        model = PhaseSeparatedGatedMemoryRNN(model_config)
        parameter_before = parameter_digest(model)
        groups_before = parameter_arrays(model)
        leaves_before = parameter_leaf_arrays(model)
        evaluation_before = evaluate_task_gated_model(
            model, holdout_episodes, trace_decay=2.0 ** (-1.0 / 40.0), batch_size=10
        )
        trainer = TaskGatedBPTTTrainer(
            model,
            batch_size=config.training_batch_size,
            learning_rate=config.learning_rate,
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
                f"[bptt-arm] chunk={chunk_index + 1}/{chunk_count} "
                f"loss={observed[-1]:.6f}",
                flush=True,
            )
        training_seconds = time.perf_counter() - started
        parameter_after = parameter_digest(model)
        groups_after = parameter_arrays(model)
        leaves_after = parameter_leaf_arrays(model)
        evaluation = evaluate_task_gated_model(
            model, holdout_episodes, trace_decay=2.0 ** (-1.0 / 40.0), batch_size=10
        )
        config.output_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_path = config.output_dir / "checkpoint.npz"
        checkpoint_digest = save_online_checkpoint(model, checkpoint_path)
        checkpoint_file_digest = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()

    revision, dirty = _source_revision()
    groups_moved = {
        name: groups_before[name].tobytes() != groups_after[name].tobytes()
        for name in groups_before
    }
    leaves_moved = {
        name: leaves_before[name].tobytes() != leaves_after[name].tobytes()
        for name in leaves_before
    }
    mechanism_passed = bool(
        parameter_before != parameter_after
        and all(groups_moved.values())
        and leaves_before.keys() == leaves_after.keys()
        and all(leaves_moved.values())
        and all(
            np.isfinite(value) and value > 0.0
            for value in gradient_norms.values()
        )
        and evaluation_before["candidate_sha256"] != evaluation["candidate_sha256"]
        and _parameter_leaves_finite(model)
        and bool(np.all(np.isfinite(losses)))
    )
    family_summary = _v4_family_summary(evaluation["strict_task_ids"])
    result = {
        "schema_version": 1,
        "configuration": config.to_dict(),
        "source_revision": revision,
        "source_dirty": dirty,
        "device": {
            "platform": str(device.platform),
            "kind": str(getattr(device, "device_kind", device)),
        },
        "matched_ppprop_arm": {
            "artifact": "var/ex21-online-v47-diverse-curriculum-v1",
            "note": (
                "Same model seed, curricula, schedule, loss, and evaluation; "
                "only the gradient algorithm differs."
            ),
        },
        "data": {
            "training_schema_version": training_curriculum.schema_version,
            "training_task_count": len(training_curriculum.tasks),
            "training_family_counts": training_curriculum.family_counts,
            "training_task_sha256": training_curriculum.task_sha256,
            "training_episode_count": len(catalog),
            "holdout_task_count": len(holdout_curriculum.tasks),
            "holdout_family_counts": holdout_curriculum.family_counts,
            "holdout_task_sha256": holdout_curriculum.task_sha256,
        },
        "model": {
            "architecture": asdict(model_config),
            "relation_width": model_config.hidden_width,
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
        "holdout_family_summary": family_summary,
        "mechanism_gate_passed": mechanism_passed,
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
    parser.add_argument("--synthetic-seed", type=int, default=33108)
    parser.add_argument("--holdout-seed", type=int, default=44108)
    parser.add_argument("--synthetic-task-count", type=int, default=1400)
    parser.add_argument("--holdout-task-count", type=int, default=120)
    parser.add_argument("--max-grid-size", type=int, default=30)
    parser.add_argument("--min-demonstrations", type=int, default=2)
    parser.add_argument("--max-demonstrations", type=int, default=6)
    parser.add_argument("--training-updates", type=int, default=800)
    parser.add_argument("--training-chunk-size", type=int, default=20)
    parser.add_argument("--training-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--memory-width", type=int, default=128)
    parser.add_argument("--expert-count", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the matched BPTT arm and print its strict holdout score.

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
    result = run_bptt_experiment(BPTTExperimentConfig(**vars(args)))
    print(
        msgspec.json.encode(
            {
                "holdout_strict": result["evaluation"][
                    "strict_task_pass_at_1_count"
                ],
                "family_counts": result["holdout_family_summary"]["family_counts"],
                "mechanism_gate_passed": result["mechanism_gate_passed"],
            },
            order="sorted",
        ).decode("utf-8"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
