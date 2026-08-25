"""Run the V48 query-routing experiment.

Trains a fresh V48 query-routing gated-memory checkpoint on the exact V47
surface-diversified curriculum (same schema, seeds, and task counts) with
the unchanged single-step PP-prop trainer and loss, then scores the same
fresh v4 holdout and the V47 real diagnostic scopes. The decisive gate is a
strict same-generator routing-family task. See
``docs/specs/2026-08-24-example21-query-routing-v48.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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
from examples.pp_prop.latent_workspace_diverse_curriculum import (
    DiverseCurriculumConfig,
    generate_diverse_curriculum,
)
from examples.pp_prop.latent_workspace_diverse_experiment import (
    _v4_family_summary,
    select_in_library_tasks,
    IN_LIBRARY_FAMILIES,
)
from examples.pp_prop.latent_workspace_expert_training import (
    TaskGatedPPPropTrainer,
    evaluate_task_gated_model,
    parameter_leaf_arrays,
)
from examples.pp_prop.latent_workspace_online_arc_oracle import (
    select_arc_scope,
)
from examples.pp_prop.latent_workspace_online_oracle import _source_revision
from examples.pp_prop.latent_workspace_online_training import (
    evaluation_online_episodes,
    parameter_arrays,
    parameter_digest,
    sample_online_training_chunk,
    save_online_checkpoint,
)
from examples.pp_prop.latent_workspace_query_routing_model import (
    ARCHITECTURE_VERSION,
    QueryRoutingConfig,
    QueryRoutingGatedMemoryRNN,
    extend_events_query_grid,
)
from examples.pp_prop.latent_workspace_task import (
    RowEventConfig,
    canonical_task_fingerprint,
)

ROUTING_FAMILIES = (
    "crop",
    "dihedral",
    "mirror_concat",
    "project_marker",
    "upscale",
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
class QueryRoutingExperimentConfig:
    """Configure the sealed V48 query-routing experiment.

    Parameters
    ----------
    output_dir : pathlib.Path
        New artifact directory.
    source_manifest : pathlib.Path
        Integrity-indexed ARC source manifest; the evaluation role is loaded
        only for corpus-integrity checks and is never encoded or scored.
    device : {"cpu", "gpu"}, default="gpu"
        Required JAX backend.
    seed : int, default=2108
        BrainState model-initialization seed.
    synthetic_seed : int, default=33108
        V4 training-curriculum seed, matched to V47.
    holdout_seed : int, default=44108
        Untouched v4 holdout seed, matched to V47.
    synthetic_task_count, holdout_task_count : int
        Curriculum sizes, matched to V47 (1400 and 120).
    max_grid_size, min_demonstrations, max_demonstrations : int
        V4 curriculum dimensions, matched to V47.
    training_updates, training_chunk_size, training_batch_size : int
        Compiled training schedule, matched to V47 (800/20/8).
    learning_rate : float, default=0.001
        Adam learning rate, matched to V47.
    memory_width : int, default=128
        Width of each phase-separated MiniLSTM population.
    expert_count : int, default=12
        Number of checkpoint-owned neural colour experts.
    program_count : int, default=16
        Number of learned routing programs.
    validation_task_count : int, default=80
        Canonical fold-zero public-training scoring size.
    expected_training_task_count, expected_evaluation_task_count : int
        Fail-closed corpus size checks, defaults 399 and 400.
    trace_decay : float, default=2 ** (-1 / 40)
        Single-step PP-prop trace decay.
    """

    output_dir: pathlib.Path
    source_manifest: pathlib.Path
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
    program_count: int = 16
    validation_task_count: int = 80
    expected_training_task_count: int = 399
    expected_evaluation_task_count: int = 400
    trace_decay: float = 2.0 ** (-1.0 / 40.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", pathlib.Path(self.output_dir))
        object.__setattr__(
            self, "source_manifest", pathlib.Path(self.source_manifest)
        )
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
            "program_count",
            "validation_task_count",
            "expected_training_task_count",
            "expected_evaluation_task_count",
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
        decay = _positive_real(self.trace_decay, "trace_decay")
        if decay > 1.0:
            raise ValueError("trace_decay must be at most 1.0.")
        object.__setattr__(self, "trace_decay", decay)
        QueryRoutingConfig(
            memory_width=self.memory_width,
            expert_count=self.expert_count,
            program_count=self.program_count,
            seed=self.seed,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-ready configuration.

        Returns
        -------
        dict
            Dataclass fields with artifact paths serialized as text.
        """

        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["source_manifest"] = str(self.source_manifest)
        return payload


def _parameter_leaves_finite(model: QueryRoutingGatedMemoryRNN) -> bool:
    return all(
        bool(np.isfinite(np.asarray(leaf)).all())
        for state in model.states(brainstate.ParamState).values()
        for leaf in jax.tree.leaves(state.value)
    )


def _extend_chunk(chunk):
    events = np.asarray(chunk.events)
    extended = extend_events_query_grid(events.transpose(0, 2, 1, 3))
    return replace(chunk, events=extended.transpose(0, 2, 1, 3))


def _extend_episodes(episodes):
    return tuple(
        replace(episode, events=extend_events_query_grid(episode.events))
        for episode in episodes
    )


def run_query_routing_experiment(
    config: QueryRoutingExperimentConfig,
) -> dict[str, object]:
    """Train and score the sealed V48 query-routing experiment.

    Parameters
    ----------
    config : QueryRoutingExperimentConfig
        Fully predeclared architecture, data, scope, and training setup.

    Returns
    -------
    dict
        Bound model, compiler, checkpoint, scope, candidate, and gate evidence.
    """

    if not isinstance(config, QueryRoutingExperimentConfig):
        raise TypeError("config must be a QueryRoutingExperimentConfig instance.")
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
    in_library = select_in_library_tasks(corpora.training)
    in_library_tasks = tuple(
        task for family in IN_LIBRARY_FAMILIES for task in in_library.get(family, ())
    )
    in_library_families = {
        task.task_id: family
        for family, tasks in in_library.items()
        for task in tasks
    }
    fold_zero = select_arc_scope(
        corpora.training,
        corpora.evaluation,
        validation_task_count=config.validation_task_count,
        validation_fold_index=0,
        complete=False,
    )
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
    holdout_episodes = _extend_episodes(
        evaluation_online_episodes(holdout_curriculum.tasks, row_config)
    )
    in_library_episodes = _extend_episodes(
        evaluation_online_episodes(in_library_tasks, row_config)
    )
    fold_zero_episodes = _extend_episodes(
        evaluation_online_episodes(fold_zero.score_tasks, row_config)
    )

    with jax.default_device(device):
        model_config = QueryRoutingConfig(
            memory_width=config.memory_width,
            expert_count=config.expert_count,
            program_count=config.program_count,
            seed=config.seed,
        )
        model = QueryRoutingGatedMemoryRNN(model_config)
        parameter_before = parameter_digest(model)
        groups_before = parameter_arrays(model)
        leaves_before = parameter_leaf_arrays(model)
        evaluation_before = evaluate_task_gated_model(
            model, holdout_episodes, trace_decay=config.trace_decay, batch_size=10
        )
        trainer = TaskGatedPPPropTrainer(
            model,
            batch_size=config.training_batch_size,
            learning_rate=config.learning_rate,
            trace_decay=config.trace_decay,
        )
        report = trainer.learner.report
        compiler_summary = {
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
            observed_losses, observed_norms = trainer.train_chunk(
                _extend_chunk(chunk)
            )
            observed = np.asarray(observed_losses, dtype=np.float64)
            losses.extend(observed.tolist())
            for name, value in observed_norms.items():
                gradient_norms[name] = max(gradient_norms[name], value)
            print(
                f"[routing-v48] chunk={chunk_index + 1}/{chunk_count} "
                f"loss={observed[-1]:.6f}",
                flush=True,
            )
        training_seconds = time.perf_counter() - started
        parameter_after = parameter_digest(model)
        groups_after = parameter_arrays(model)
        leaves_after = parameter_leaf_arrays(model)
        holdout_evaluation = evaluate_task_gated_model(
            model, holdout_episodes, trace_decay=config.trace_decay, batch_size=10
        )
        in_library_evaluation = evaluate_task_gated_model(
            model, in_library_episodes, trace_decay=config.trace_decay, batch_size=10
        )
        fold_zero_evaluation = evaluate_task_gated_model(
            model, fold_zero_episodes, trace_decay=config.trace_decay, batch_size=10
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
        not compiler_summary["recurrent_excluded_paths"]
        and parameter_before != parameter_after
        and all(groups_moved.values())
        and leaves_before.keys() == leaves_after.keys()
        and all(leaves_moved.values())
        and all(
            np.isfinite(value) and value > 0.0
            for value in gradient_norms.values()
        )
        and evaluation_before["candidate_sha256"]
        != holdout_evaluation["candidate_sha256"]
        and _parameter_leaves_finite(model)
        and bool(np.all(np.isfinite(losses)))
    )
    diagnostics = holdout_evaluation["diagnostics"]
    predicted_nonzero = sum(
        count
        for color, count in diagnostics["predicted_color_counts"].items()
        if color != "0"
    )
    anti_collapse_passed = bool(
        mechanism_passed
        and predicted_nonzero > 0
        and diagnostics["foreground_total"] > 0
        and diagnostics["foreground_accuracy"] >= 0.05
        and diagnostics["background_total"] > 0
        and diagnostics["background_accuracy"] >= 0.5
    )
    holdout_family_summary = _v4_family_summary(
        holdout_evaluation["strict_task_ids"]
    )
    routing_exact = {
        family: count
        for family, count in holdout_family_summary["family_counts"].items()
        if family in ROUTING_FAMILIES
    }
    routing_gate_passed = bool(
        mechanism_passed and sum(routing_exact.values()) >= 1
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
        "scopes": {
            "in_library": {
                "name": "in_library_public_training",
                "families": in_library_families,
                "family_counts": {
                    family: len(tasks) for family, tasks in in_library.items()
                },
                "task_count": len(in_library_tasks),
                "query_count": len(in_library_episodes),
                "task_fingerprints": [
                    canonical_task_fingerprint(task) for task in in_library_tasks
                ],
            },
            "fold_zero": {
                "name": fold_zero.scope,
                "validation_fold_index": 0,
                "task_count": len(fold_zero.score_tasks),
                "query_count": len(fold_zero_episodes),
                "task_ids": list(fold_zero.score_task_ids),
                "task_fingerprints": [
                    canonical_task_fingerprint(task)
                    for task in fold_zero.score_tasks
                ],
            },
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
            "trace_decay": config.trace_decay,
            "gradient_norm_maxima": gradient_norms,
            "compiler": compiler_summary,
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
        "evaluation_holdout": holdout_evaluation,
        "evaluation_in_library": in_library_evaluation,
        "evaluation_fold_zero": fold_zero_evaluation,
        "holdout_family_summary": holdout_family_summary,
        "routing_exact_families": routing_exact,
        "mechanism_gate_passed": mechanism_passed,
        "anti_collapse_gate_passed": anti_collapse_passed,
        "routing_gate_passed": routing_gate_passed,
    }
    (config.output_dir / "result.json").write_bytes(
        msgspec.json.encode(result, order="sorted")
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--source-manifest", type=pathlib.Path, required=True)
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
    parser.add_argument("--program-count", type=int, default=16)
    parser.add_argument("--validation-task-count", type=int, default=80)
    parser.add_argument("--trace-decay", type=float, default=2.0 ** (-1.0 / 40.0))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the V48 experiment and print its routing gate score.

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
    result = run_query_routing_experiment(
        QueryRoutingExperimentConfig(**vars(args))
    )
    print(
        msgspec.json.encode(
            {
                "holdout_strict": result["evaluation_holdout"][
                    "strict_task_pass_at_1_count"
                ],
                "holdout_family_counts": result["holdout_family_summary"][
                    "family_counts"
                ],
                "in_library_strict": result["evaluation_in_library"][
                    "strict_task_pass_at_1_count"
                ],
                "fold_zero_strict": result["evaluation_fold_zero"][
                    "strict_task_pass_at_1_count"
                ],
                "mechanism_gate_passed": result["mechanism_gate_passed"],
                "routing_gate_passed": result["routing_gate_passed"],
            },
            order="sorted",
        ).decode("utf-8"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
