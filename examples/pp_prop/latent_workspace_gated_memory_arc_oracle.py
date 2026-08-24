"""Fine-tune V44 on public ARC and score one bound development fold."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import pathlib
import time

import brainstate
import jax
import msgspec
import numpy as np

from examples.pp_prop.latent_workspace_direct_experiment import (
    load_corpora,
    training_episode_catalog,
)
from examples.pp_prop.latent_workspace_expert_arc_oracle import (
    _nonnegative_integer,
    _positive_integer,
    _positive_real,
    _sha256,
    select_transfer_scope,
)
from examples.pp_prop.latent_workspace_expert_training import (
    TaskGatedPPPropTrainer,
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
    load_online_checkpoint,
    parameter_arrays,
    parameter_digest,
    sample_online_training_chunk,
    save_online_checkpoint,
)
from examples.pp_prop.latent_workspace_task import (
    RowEventConfig,
    canonical_task_fingerprint,
)


@dataclass(frozen=True)
class GatedMemoryARCOracleConfig:
    """Configure one sealed V45 public-ARC transfer pilot.

    Parameters
    ----------
    source_manifest : pathlib.Path
        Integrity-indexed public ARC role declarations.
    pretrained_checkpoint : pathlib.Path
        Exact V44 synthetic checkpoint to reload before fine-tuning.
    expected_pretrained_parameter_sha256 : str
        Predeclared ordered V44 parameter digest.
    expected_pretrained_file_sha256 : str
        Predeclared V44 checkpoint-file digest.
    output_dir : pathlib.Path
        New evidence-artifact directory.
    device : {"cpu", "gpu"}, default="gpu"
        Required JAX backend.
    seed : int, default=2108
        Seed bound into the pretrained architecture metadata.
    sampling_seed : int, default=26108
        BrainState sampling and augmentation seed.
    validation_start_index : int, default=0
        First canonical-fingerprint position of the development fold.
    validation_task_count : int, default=80
        Number of public-training tasks scored after fine-tuning.
    expected_training_task_count, expected_evaluation_task_count : int
        Fail-closed indexed-corpus task counts.
    training_updates, training_chunk_size, training_batch_size : int
        Positive compiled fine-tuning schedule.
    learning_rate : float, default=0.001
        Adam learning rate.
    memory_width, expert_count : int
        Exact V44 checkpoint topology.
    minimum_strict_task_count : int, default=4
        Inclusive strict development threshold.
    trace_decay : float, default=2 ** (-1 / 40)
        Single-step PP-prop eligibility decay.
    augment : bool, default=True
        Apply training-only task-independent augmentations.
    """

    source_manifest: pathlib.Path
    pretrained_checkpoint: pathlib.Path
    expected_pretrained_parameter_sha256: str
    expected_pretrained_file_sha256: str
    output_dir: pathlib.Path
    device: str = "gpu"
    seed: int = 2108
    sampling_seed: int = 26108
    validation_start_index: int = 0
    validation_task_count: int = 80
    expected_training_task_count: int = 399
    expected_evaluation_task_count: int = 400
    training_updates: int = 500
    training_chunk_size: int = 20
    training_batch_size: int = 8
    learning_rate: float = 0.001
    memory_width: int = 128
    expert_count: int = 12
    minimum_strict_task_count: int = 4
    trace_decay: float = 2.0 ** (-1.0 / 40.0)
    augment: bool = True

    def __post_init__(self) -> None:
        for name in ("source_manifest", "pretrained_checkpoint", "output_dir"):
            object.__setattr__(self, name, pathlib.Path(getattr(self, name)))
        for name in (
            "expected_pretrained_parameter_sha256",
            "expected_pretrained_file_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'.")
        for name in ("seed", "sampling_seed", "validation_start_index"):
            object.__setattr__(
                self, name, _nonnegative_integer(getattr(self, name), name)
            )
        for name in (
            "validation_task_count",
            "expected_training_task_count",
            "expected_evaluation_task_count",
            "training_updates",
            "training_chunk_size",
            "training_batch_size",
            "memory_width",
            "expert_count",
            "minimum_strict_task_count",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        if self.validation_task_count >= self.expected_training_task_count:
            raise ValueError(
                "validation_task_count must be smaller than "
                "expected_training_task_count."
            )
        if (
            self.validation_start_index + self.validation_task_count
            > self.expected_training_task_count
        ):
            raise ValueError("validation scope must fit inside the training corpus.")
        if self.training_updates % self.training_chunk_size:
            raise ValueError("training_chunk_size must divide training_updates.")
        object.__setattr__(
            self,
            "learning_rate",
            _positive_real(self.learning_rate, "learning_rate"),
        )
        decay = _positive_real(self.trace_decay, "trace_decay")
        if decay > 1.0:
            raise ValueError("trace_decay must be at most 1.0.")
        object.__setattr__(self, "trace_decay", decay)
        if not isinstance(self.augment, bool):
            raise TypeError("augment must be boolean.")
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
            Dataclass fields with paths serialized as text.
        """

        payload = asdict(self)
        for name in ("source_manifest", "pretrained_checkpoint", "output_dir"):
            payload[name] = str(payload[name])
        return payload


def _compiler_summary(trainer: TaskGatedPPPropTrainer) -> dict[str, object]:
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
            if path[0] == "recurrent"
        ],
        "diagnostic_kinds": [item.kind.value for item in report.diagnostics],
    }


def _parameter_leaves_finite(model: PhaseSeparatedGatedMemoryRNN) -> bool:
    return all(
        bool(np.isfinite(np.asarray(leaf)).all())
        for state in model.states(brainstate.ParamState).values()
        for leaf in jax.tree.leaves(state.value)
    )


def run_gated_memory_arc_oracle(
    config: GatedMemoryARCOracleConfig,
) -> dict[str, object]:
    """Fine-tune V44 and score the bound public-training fold.

    Parameters
    ----------
    config : GatedMemoryARCOracleConfig
        Exact checkpoint, corpus, topology, schedule, and score scope.

    Returns
    -------
    dict
        Bound transfer, compiler, checkpoint, candidate, and gate evidence.
    """

    if not isinstance(config, GatedMemoryARCOracleConfig):
        raise TypeError("config must be a GatedMemoryARCOracleConfig instance.")
    devices = jax.devices(config.device)
    if not devices:
        raise RuntimeError(f"Requested JAX {config.device} backend is unavailable.")
    device = devices[0]
    corpora = load_corpora(config.source_manifest)
    actual_counts = (len(corpora.training), len(corpora.evaluation))
    expected_counts = (
        config.expected_training_task_count,
        config.expected_evaluation_task_count,
    )
    if actual_counts != expected_counts:
        raise ValueError(
            f"Expected {expected_counts[0]} training and {expected_counts[1]} "
            f"evaluation tasks, found {actual_counts[0]} and {actual_counts[1]}."
        )
    scope = select_transfer_scope(
        corpora.training,
        corpora.evaluation,
        validation_start_index=config.validation_start_index,
        validation_task_count=config.validation_task_count,
    )
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    catalog = training_episode_catalog(scope.fit_tasks)
    score_episodes = evaluation_online_episodes(scope.score_tasks, row_config)
    expected_architecture = GatedMemoryConfig(
        input_width=MODEL_INPUT_WIDTH,
        memory_width=config.memory_width,
        expert_count=config.expert_count,
        seed=config.seed,
    )
    pretrained_file_digest = hashlib.sha256(
        config.pretrained_checkpoint.read_bytes()
    ).hexdigest()
    if pretrained_file_digest != config.expected_pretrained_file_sha256:
        raise ValueError(
            "Pretrained checkpoint file digest does not match the predeclared digest."
        )

    with jax.default_device(device):
        loaded_model, checkpoint_metadata = load_online_checkpoint(
            config.pretrained_checkpoint
        )
        if not isinstance(loaded_model, PhaseSeparatedGatedMemoryRNN):
            raise ValueError("Pretrained checkpoint is not a V44 gated-memory model.")
        model = loaded_model
        if asdict(model.config) != asdict(expected_architecture):
            raise ValueError("Pretrained checkpoint architecture does not match V45.")
        parameter_before = parameter_digest(model)
        if parameter_before != config.expected_pretrained_parameter_sha256:
            raise ValueError(
                "Pretrained checkpoint parameter digest does not match the "
                "predeclared digest."
            )
        groups_before = parameter_arrays(model)
        leaves_before = parameter_leaf_arrays(model)
        evaluation_before = evaluate_task_gated_model(
            model,
            score_episodes,
            trace_decay=config.trace_decay,
            batch_size=config.training_batch_size,
        )
        trainer = TaskGatedPPPropTrainer(
            model,
            batch_size=config.training_batch_size,
            learning_rate=config.learning_rate,
            trace_decay=config.trace_decay,
        )
        compiler = _compiler_summary(trainer)
        sampling_rng = brainstate.random.RandomState(config.sampling_seed)
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
                augment=config.augment,
            )
            observed_losses, observed_norms = trainer.train_chunk(chunk)
            observed = np.asarray(observed_losses, dtype=np.float64)
            losses.extend(observed.tolist())
            for name, value in observed_norms.items():
                gradient_norms[name] = max(gradient_norms[name], value)
            print(
                f"[gated-memory-arc] chunk={chunk_index + 1}/{chunk_count} "
                f"loss={observed[-1]:.6f}",
                flush=True,
            )
        training_seconds = time.perf_counter() - started
        parameter_after = parameter_digest(model)
        groups_after = parameter_arrays(model)
        leaves_after = parameter_leaf_arrays(model)
        evaluation = evaluate_task_gated_model(
            model,
            score_episodes,
            trace_decay=config.trace_decay,
            batch_size=config.training_batch_size,
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
    mechanism_gate = bool(
        parameter_before != parameter_after
        and all(groups_moved.values())
        and leaves_before.keys() == leaves_after.keys()
        and all(leaves_moved.values())
        and all(
            np.isfinite(value) and value > 0.0
            for value in gradient_norms.values()
        )
        and evaluation_before["candidate_sha256"]
        != evaluation["candidate_sha256"]
        and bool(np.isfinite(losses).all())
        and _parameter_leaves_finite(model)
        and not compiler["recurrent_excluded_paths"]
    )
    strict_count = int(evaluation["strict_task_pass_at_1_count"])
    pilot_gate = bool(
        mechanism_gate and strict_count >= config.minimum_strict_task_count
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
        "sources": [item.manifest.to_dict() for item in corpora.loaded],
        "scope": {
            "name": scope.scope,
            "fit_task_count": len(scope.fit_tasks),
            "score_task_count": len(scope.score_tasks),
            "validation_start_index": config.validation_start_index,
            "fit_task_ids": list(scope.fit_task_ids),
            "score_task_ids": list(scope.score_task_ids),
            "fit_task_fingerprints": [
                canonical_task_fingerprint(task) for task in scope.fit_tasks
            ],
            "score_task_fingerprints": [
                canonical_task_fingerprint(task) for task in scope.score_tasks
            ],
            "training_episode_count": len(catalog),
        },
        "pretraining_checkpoint": {
            "path": str(config.pretrained_checkpoint),
            "parameter_sha256": parameter_before,
            "file_sha256": pretrained_file_digest,
            "metadata": checkpoint_metadata,
        },
        "model": {
            "architecture": asdict(model.config),
            "relation_width": model.config.hidden_width,
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
        "mechanism_gate_passed": mechanism_gate,
        "pilot_gate_passed": pilot_gate,
    }
    (config.output_dir / "result.json").write_bytes(
        msgspec.json.encode(result, order="sorted")
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument(
        "--expected-pretrained-parameter-sha256", required=True
    )
    parser.add_argument("--expected-pretrained-file-sha256", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=2108)
    parser.add_argument("--sampling-seed", type=int, default=26108)
    parser.add_argument("--validation-start-index", type=int, default=0)
    parser.add_argument("--validation-task-count", type=int, default=80)
    parser.add_argument("--expected-training-task-count", type=int, default=399)
    parser.add_argument("--expected-evaluation-task-count", type=int, default=400)
    parser.add_argument("--training-updates", type=int, default=500)
    parser.add_argument("--training-chunk-size", type=int, default=20)
    parser.add_argument("--training-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--memory-width", type=int, default=128)
    parser.add_argument("--expert-count", type=int, default=12)
    parser.add_argument("--minimum-strict-task-count", type=int, default=4)
    parser.add_argument("--trace-decay", type=float, default=2.0 ** (-1.0 / 40.0))
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run V45 and print its strict development gate score.

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
    result = run_gated_memory_arc_oracle(
        GatedMemoryARCOracleConfig(**vars(args))
    )
    print(
        msgspec.json.encode(
            {
                "strict_task_pass_at_1_count": result["evaluation"][
                    "strict_task_pass_at_1_count"
                ],
                "pilot_gate_passed": result["pilot_gate_passed"],
            },
            order="sorted",
        ).decode("utf-8"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
