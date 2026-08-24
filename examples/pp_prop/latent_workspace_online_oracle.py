"""Run the staged synthetic V20 row-decoded PP-prop oracle."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral, Real
import os
import pathlib
import subprocess
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
from examples.pp_prop.latent_workspace_online_model import (
    OnlineARCGRU,
    OnlineModelConfig,
)
from examples.pp_prop.latent_workspace_online_training import (
    OnlinePPPropTrainer,
    evaluate_online_model,
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
class OnlineOracleConfig:
    """Configure one synthetic V20 PP-prop learning oracle.

    Parameters
    ----------
    output_dir : pathlib.Path
        New artifact directory.
    device : {"cpu", "gpu"}, default="gpu"
        Required JAX platform.
    seed : int, default=2108
        Model initialization seed.
    training_updates, training_chunk_size, training_batch_size : int
        Positive compiled training schedule.
    synthetic_task_count : int, default=1400
        Number of training-only generated tasks.
    synthetic_demonstrations : int, default=4
        Demonstrations per generated task.
    synthetic_max_grid_size : int, default=12
        Maximum generated side length.
    synthetic_seed : int, default=12108
        Independent BrainState training-curriculum seed.
    oracle_task_count : int, default=120
        Number of untouched synthetic evaluation tasks.
    oracle_seed : int, default=42108
        Independent BrainState oracle seed.
    learning_rate : float, default=0.001
        Adam learning rate.
    encoder_width, hidden_width : int
        First and later recurrent-layer widths.
    recurrent_layers : int, default=2
        Recurrent depth, at least two.
    trace_decay : float, default=2 ** (-1 / 40)
        Single-step PP-prop eligibility decay.
    """

    output_dir: pathlib.Path
    device: str = "gpu"
    seed: int = 2108
    training_updates: int = 1000
    training_chunk_size: int = 50
    training_batch_size: int = 8
    synthetic_task_count: int = 1400
    synthetic_demonstrations: int = 4
    synthetic_max_grid_size: int = 12
    synthetic_seed: int = 12108
    oracle_task_count: int = 120
    oracle_seed: int = 42108
    learning_rate: float = 0.001
    encoder_width: int = 128
    hidden_width: int = 256
    recurrent_layers: int = 2
    trace_decay: float = 2.0 ** (-1.0 / 40.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", pathlib.Path(self.output_dir))
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'.")
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        object.__setattr__(
            self, "synthetic_seed", _nonnegative_integer(self.synthetic_seed, "synthetic_seed")
        )
        object.__setattr__(
            self, "oracle_seed", _nonnegative_integer(self.oracle_seed, "oracle_seed")
        )
        if self.synthetic_seed == self.oracle_seed:
            raise ValueError("synthetic_seed and oracle_seed must be different.")
        for name in (
            "training_updates",
            "training_chunk_size",
            "training_batch_size",
            "synthetic_task_count",
            "oracle_task_count",
            "encoder_width",
            "hidden_width",
            "recurrent_layers",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
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
        OnlineModelConfig(
            input_width=1,
            encoder_width=self.encoder_width,
            hidden_width=self.hidden_width,
            recurrent_layers=self.recurrent_layers,
            seed=self.seed,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready exact configuration mapping.

        Returns
        -------
        dict
            Dataclass fields with the artifact path serialized as text.
        """

        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def exact_family_summary(strict_task_ids: list[str]) -> dict[str, object]:
    """Summarize exact synthetic task IDs by transformation family.

    Parameters
    ----------
    strict_task_ids : list of str
        Exact task IDs in ``synthetic-v3:<family>:<index>`` form.

    Returns
    -------
    dict
        Sorted family counts, non-label families, and non-copy/non-label IDs.
    """

    families = []
    non_copy_non_label = []
    for task_id in strict_task_ids:
        parts = task_id.split(":")
        if len(parts) != 3 or parts[0] != "synthetic-v3" or not parts[1]:
            raise ValueError("strict task IDs must use synthetic-v3 family syntax.")
        family = parts[1]
        families.append(family)
        if family not in {"copy", "pattern_label"}:
            non_copy_non_label.append(task_id)
    counts = Counter(families)
    return {
        "family_counts": dict(sorted(counts.items())),
        "non_label_families": sorted(
            family for family in counts if family != "pattern_label"
        ),
        "non_copy_non_label_ids": non_copy_non_label,
    }


def _source_revision() -> tuple[str, bool]:
    explicit_revision = os.environ.get("BRAINTRACE_SOURCE_REVISION")
    explicit_dirty = os.environ.get("BRAINTRACE_SOURCE_DIRTY")
    if explicit_revision is not None or explicit_dirty is not None:
        if explicit_revision is None or explicit_dirty is None:
            raise ValueError(
                "BRAINTRACE_SOURCE_REVISION and BRAINTRACE_SOURCE_DIRTY must be set together."
            )
        if len(explicit_revision) != 40 or any(
            character not in "0123456789abcdefABCDEF"
            for character in explicit_revision
        ):
            raise ValueError("BRAINTRACE_SOURCE_REVISION must be 40 hex characters.")
        normalized_dirty = explicit_dirty.lower()
        if normalized_dirty not in {"true", "false"}:
            raise ValueError("BRAINTRACE_SOURCE_DIRTY must be true or false.")
        return explicit_revision.lower(), normalized_dirty == "true"
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def run_oracle(config: OnlineOracleConfig) -> dict[str, object]:
    """Train and score one sealed synthetic V20 PP-prop oracle.

    Parameters
    ----------
    config : OnlineOracleConfig
        Fully predeclared model, learning, and data configuration.

    Returns
    -------
    dict
        Bound checkpoint, training, exact-family, and candidate evidence.
    """

    if not isinstance(config, OnlineOracleConfig):
        raise TypeError("config must be an OnlineOracleConfig instance.")
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
        model_config = OnlineModelConfig(
            input_width=row_config.input_width + 31,
            encoder_width=config.encoder_width,
            hidden_width=config.hidden_width,
            recurrent_layers=config.recurrent_layers,
            seed=config.seed,
        )
        model = OnlineARCGRU(model_config)
        parameter_before = parameter_digest(model)
        groups_before = parameter_arrays(model)
        evaluation_before = evaluate_online_model(
            model,
            oracle_episodes,
            trace_decay=config.trace_decay,
            batch_size=config.training_batch_size,
        )
        trainer = OnlinePPPropTrainer(
            model,
            batch_size=config.training_batch_size,
            learning_rate=config.learning_rate,
            trace_decay=config.trace_decay,
        )
        sampling_rng = brainstate.random.RandomState(config.synthetic_seed + 1)
        losses = []
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
                f"[online-synthetic] chunk={chunk_index + 1}/{chunk_count} "
                f"loss={observed[-1]:.6f}",
                flush=True,
            )
        training_seconds = time.perf_counter() - started
        parameter_after = parameter_digest(model)
        groups_after = parameter_arrays(model)
        evaluation = evaluate_online_model(
            model,
            oracle_episodes,
            trace_decay=config.trace_decay,
            batch_size=config.training_batch_size,
        )
        config.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = config.output_dir / "checkpoint.npz"
        checkpoint_digest = save_online_checkpoint(model, checkpoint_path)
        checkpoint_file_digest = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
    revision, dirty = _source_revision()
    family_summary = exact_family_summary(evaluation["strict_task_ids"])
    non_label_families = family_summary["non_label_families"]
    non_copy_ids = family_summary["non_copy_non_label_ids"]
    diagnostics = evaluation["diagnostics"]
    predicted_nonzero = sum(
        count
        for color, count in diagnostics["predicted_color_counts"].items()
        if color != "0"
    )
    anti_collapse_passed = bool(
        parameter_before != parameter_after
        and all(
            groups_before[name].tobytes() != groups_after[name].tobytes()
            for name in groups_before
        )
        and evaluation_before["candidate_sha256"] != evaluation["candidate_sha256"]
        and predicted_nonzero > 0
        and diagnostics["foreground_total"] > 0
        and diagnostics["foreground_accuracy"] >= 0.05
        and diagnostics["background_total"] > 0
        and diagnostics["background_accuracy"] >= 0.5
    )
    gate_passed = bool(
        anti_collapse_passed
        and evaluation["strict_task_pass_at_1_count"] > 7
        and len(non_label_families) >= 2
        and non_copy_ids
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
            "parameter_groups_moved": {
                name: groups_before[name].tobytes() != groups_after[name].tobytes()
                for name in groups_before
            },
            "parameter_leaves_finite": all(
                bool(np.isfinite(np.asarray(leaf)).all())
                for state in model.states(brainstate.ParamState).values()
                for leaf in jax.tree.leaves(state.value)
            ),
        },
        "learner": {
            "algorithm": trainer.algorithm,
            "vjp_method": trainer.vjp_method,
            "loss_version": trainer.loss_version,
            "trace_decay": config.trace_decay,
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
    parser.add_argument("--training-updates", type=int, default=1000)
    parser.add_argument("--training-chunk-size", type=int, default=50)
    parser.add_argument("--training-batch-size", type=int, default=8)
    parser.add_argument("--synthetic-task-count", type=int, default=1400)
    parser.add_argument("--synthetic-demonstrations", type=int, default=4)
    parser.add_argument("--synthetic-max-grid-size", type=int, default=12)
    parser.add_argument("--synthetic-seed", type=int, default=12108)
    parser.add_argument("--oracle-task-count", type=int, default=120)
    parser.add_argument("--oracle-seed", type=int, default=42108)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--recurrent-layers", type=int, default=2)
    parser.add_argument("--trace-decay", type=float, default=2.0 ** (-1.0 / 40.0))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the synthetic online-learning oracle command.

    Parameters
    ----------
    argv : list of str, optional
        Explicit command arguments; defaults to the process arguments.

    Returns
    -------
    int
        Zero after a complete artifact write.
    """

    args = _parser().parse_args(argv)
    result = run_oracle(OnlineOracleConfig(**vars(args)))
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
