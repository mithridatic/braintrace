"""Train one global online BrainTrace checkpoint on public ARC training tasks."""

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

from examples.pp_prop.latent_workspace_direct_experiment import (
    load_corpora,
    training_episode_catalog,
)
from examples.pp_prop.latent_workspace_online_model import (
    OnlineARCVanillaRNN,
    OnlineModelConfig,
)
from examples.pp_prop.latent_workspace_online_oracle import _source_revision
from examples.pp_prop.latent_workspace_online_training import (
    OnlinePPPropTrainer,
    evaluate_online_model,
    evaluation_online_episodes,
    parameter_arrays,
    parameter_digest,
    sample_online_training_chunk,
    save_online_checkpoint,
)
from examples.pp_prop.latent_workspace_task import (
    ArcTask,
    RowEventConfig,
    canonical_task_fingerprint,
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
class OnlineARCOracleConfig:
    """Configure a target-isolated public-ARC online-learning run.

    Parameters
    ----------
    source_manifest : pathlib.Path
        Integrity-indexed public ARC train/evaluation declarations.
    output_dir : pathlib.Path
        New evidence-artifact directory.
    device : {"cpu", "gpu"}, default="gpu"
        Required JAX platform.
    seed, sampling_seed : int
        Independent model and BrainState sampling/augmentation seeds.
    validation_task_count : int, default=80
        Public training tasks held out during a pilot.
    validation_fold_index : int, default=0
        Zero-based nonoverlapping canonical-fingerprint validation fold.
    expected_training_task_count, expected_evaluation_task_count : int
        Fail-closed indexed-corpus task counts.
    training_updates, training_chunk_size, training_batch_size : int
        Compiled optimizer schedule.
    learning_rate : float, default=0.001
        Adam learning rate.
    encoder_width, hidden_width, recurrent_layers : int
        Direct-state recurrent topology.
    trace_decay : float, default=2 ** (-1 / 40)
        Single-step PP-prop eligibility decay.
    augment : bool, default=True
        Apply training-only task-independent augmentations.
    evaluate_complete_manifest : bool, default=False
        Train on all public training tasks and score the complete evaluation
        manifest. False selects the held-out public-training pilot.
    """

    source_manifest: pathlib.Path
    output_dir: pathlib.Path
    device: str = "gpu"
    seed: int = 2108
    sampling_seed: int = 12108
    validation_task_count: int = 80
    validation_fold_index: int = 0
    expected_training_task_count: int = 399
    expected_evaluation_task_count: int = 400
    training_updates: int = 400
    training_chunk_size: int = 20
    training_batch_size: int = 8
    learning_rate: float = 0.001
    encoder_width: int = 128
    hidden_width: int = 256
    recurrent_layers: int = 2
    trace_decay: float = 2.0 ** (-1.0 / 40.0)
    augment: bool = True
    evaluate_complete_manifest: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_manifest", pathlib.Path(self.source_manifest))
        object.__setattr__(self, "output_dir", pathlib.Path(self.output_dir))
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'.")
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        object.__setattr__(
            self,
            "sampling_seed",
            _nonnegative_integer(self.sampling_seed, "sampling_seed"),
        )
        object.__setattr__(
            self,
            "validation_fold_index",
            _nonnegative_integer(
                self.validation_fold_index, "validation_fold_index"
            ),
        )
        for name in (
            "validation_task_count",
            "expected_training_task_count",
            "expected_evaluation_task_count",
            "training_updates",
            "training_chunk_size",
            "training_batch_size",
            "encoder_width",
            "hidden_width",
            "recurrent_layers",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if self.validation_task_count >= self.expected_training_task_count:
            raise ValueError(
                "validation_task_count must be smaller than "
                "expected_training_task_count."
            )
        if (
            self.validation_fold_index + 1
        ) * self.validation_task_count > self.expected_training_task_count:
            raise ValueError(
                "validation fold must fit inside expected_training_task_count."
            )
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
        if not isinstance(self.augment, bool) or not isinstance(
            self.evaluate_complete_manifest, bool
        ):
            raise TypeError(
                "augment and evaluate_complete_manifest must be boolean."
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready exact configuration mapping.

        Returns
        -------
        dict
            Dataclass fields with paths serialized as text.
        """

        payload = asdict(self)
        payload["source_manifest"] = str(self.source_manifest)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass(frozen=True)
class ARCRunScope:
    """Bind fitting and scoring tasks for one public ARC run.

    Parameters
    ----------
    scope : str
        Stable scope identifier.
    fit_tasks, score_tasks : tuple of ArcTask
        Disjoint pilot tasks, or public-training and evaluation-role tasks for
        the complete run.
    """

    scope: str
    fit_tasks: tuple[ArcTask, ...]
    score_tasks: tuple[ArcTask, ...]

    @property
    def fit_task_ids(self) -> tuple[str, ...]:
        """Return ordered fitting task identifiers."""

        return tuple(task.task_id for task in self.fit_tasks)

    @property
    def score_task_ids(self) -> tuple[str, ...]:
        """Return ordered scoring task identifiers."""

        return tuple(task.task_id for task in self.score_tasks)


def select_arc_scope(
    training: tuple[ArcTask, ...],
    evaluation: tuple[ArcTask, ...],
    *,
    validation_task_count: int,
    validation_fold_index: int,
    complete: bool,
) -> ARCRunScope:
    """Select disjoint pilot tasks or the complete public ARC roles.

    Parameters
    ----------
    training, evaluation : tuple of ArcTask
        Integrity-checked role-separated corpora.
    validation_task_count : int
        Held-out public-training task count for a pilot.
    validation_fold_index : int
        Zero-based nonoverlapping canonical-fingerprint fold.
    complete : bool
        Whether to select the complete evaluation role.

    Returns
    -------
    ARCRunScope
        Ordered fitting and scoring tasks with a stable scope name.
    """

    if not isinstance(training, tuple) or not training:
        raise ValueError("training must be a nonempty tuple.")
    if not isinstance(evaluation, tuple) or not evaluation:
        raise ValueError("evaluation must be a nonempty tuple.")
    if not isinstance(complete, bool):
        raise TypeError("complete must be boolean.")
    if complete:
        return ARCRunScope("complete_arc_evaluation", training, evaluation)
    count = _positive_integer(validation_task_count, "validation_task_count")
    fold = _nonnegative_integer(validation_fold_index, "validation_fold_index")
    start = fold * count
    stop = start + count
    if stop > len(training):
        raise ValueError("validation fold must fit inside the training corpus.")
    ordered = tuple(sorted(training, key=canonical_task_fingerprint))
    validation = ordered[start:stop]
    fitting = ordered[:start] + ordered[stop:]
    fit_fingerprints = {canonical_task_fingerprint(task) for task in fitting}
    score_fingerprints = {canonical_task_fingerprint(task) for task in validation}
    if fit_fingerprints.intersection(score_fingerprints):
        raise ValueError("Pilot fitting and scoring fingerprints must be disjoint.")
    return ARCRunScope("held_out_public_training", fitting, validation)


def _compiler_summary(trainer: OnlinePPPropTrainer) -> dict[str, object]:
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


def run_arc_oracle(config: OnlineARCOracleConfig) -> dict[str, object]:
    """Train and score one bound public-ARC online-learning run.

    Parameters
    ----------
    config : OnlineARCOracleConfig
        Exact corpus, topology, learning, and score-scope configuration.

    Returns
    -------
    dict
        Bound corpus, checkpoint, compiler, training, and strict-score evidence.
    """

    if not isinstance(config, OnlineARCOracleConfig):
        raise TypeError("config must be an OnlineARCOracleConfig instance.")
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
    scope = select_arc_scope(
        corpora.training,
        corpora.evaluation,
        validation_task_count=config.validation_task_count,
        validation_fold_index=config.validation_fold_index,
        complete=config.evaluate_complete_manifest,
    )
    row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
    catalog = training_episode_catalog(scope.fit_tasks)
    score_episodes = evaluation_online_episodes(scope.score_tasks, row_config)

    with jax.default_device(device):
        model_config = OnlineModelConfig(
            input_width=row_config.input_width + 31,
            encoder_width=config.encoder_width,
            hidden_width=config.hidden_width,
            recurrent_layers=config.recurrent_layers,
            seed=config.seed,
        )
        model = OnlineARCVanillaRNN(model_config)
        parameter_before = parameter_digest(model)
        groups_before = parameter_arrays(model)
        evaluation_before = evaluate_online_model(
            model,
            score_episodes,
            trace_decay=config.trace_decay,
            batch_size=config.training_batch_size,
        )
        trainer = OnlinePPPropTrainer(
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
                f"[online-arc] chunk={chunk_index + 1}/{chunk_count} "
                f"loss={observed[-1]:.6f}",
                flush=True,
            )
        training_seconds = time.perf_counter() - started
        parameter_after = parameter_digest(model)
        groups_after = parameter_arrays(model)
        evaluation = evaluate_online_model(
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
    mechanism_gate = bool(
        parameter_before != parameter_after
        and all(groups_moved.values())
        and evaluation_before["candidate_sha256"]
        != evaluation["candidate_sha256"]
        and np.isfinite(losses).all()
        and not compiler["recurrent_excluded_paths"]
    )
    strict_count = int(evaluation["strict_task_pass_at_1_count"])
    pilot_gate = bool(
        not config.evaluate_complete_manifest and mechanism_gate and strict_count >= 2
    )
    acceptance_threshold = bool(
        config.evaluate_complete_manifest and strict_count >= 16
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
            "validation_fold_index": (
                None
                if config.evaluate_complete_manifest
                else config.validation_fold_index
            ),
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
        "model": {
            "architecture": asdict(model_config),
            "parameter_sha256_before": parameter_before,
            "parameter_sha256_after": parameter_after,
            "parameters_moved": parameter_before != parameter_after,
            "parameter_groups_moved": groups_moved,
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
        "acceptance_threshold_passed": acceptance_threshold,
    }
    (config.output_dir / "result.json").write_bytes(
        msgspec.json.encode(result, order="sorted")
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=2108)
    parser.add_argument("--sampling-seed", type=int, default=12108)
    parser.add_argument("--validation-task-count", type=int, default=80)
    parser.add_argument("--validation-fold-index", type=int, default=0)
    parser.add_argument("--expected-training-task-count", type=int, default=399)
    parser.add_argument("--expected-evaluation-task-count", type=int, default=400)
    parser.add_argument("--training-updates", type=int, default=400)
    parser.add_argument("--training-chunk-size", type=int, default=20)
    parser.add_argument("--training-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--recurrent-layers", type=int, default=2)
    parser.add_argument("--trace-decay", type=float, default=2.0 ** (-1.0 / 40.0))
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.add_argument("--evaluate-complete-manifest", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the public-ARC online-learning command.

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
    result = run_arc_oracle(OnlineARCOracleConfig(**vars(args)))
    print(
        msgspec.json.encode(
            {
                "strict_task_pass_at_1_count": result["evaluation"][
                    "strict_task_pass_at_1_count"
                ],
                "pilot_gate_passed": result["pilot_gate_passed"],
                "acceptance_threshold_passed": result[
                    "acceptance_threshold_passed"
                ],
            },
            order="sorted",
        ).decode("utf-8"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
