"""21 - In-context learning through a recurrent latent workspace.

This example instantiates the published system-level interface of BDH-CQ with
an original BrainTrace mechanism: episode-local factored memory, a recurrent
latent workspace, and terminal-only pp_prop learning. The source system's
internal dimensions, update equations, and training recipe are unpublished;
this is an interface experiment, not a reproduction.

The release default keeps the contextual-memory write projections fixed and
trains the remaining eligible parameters. It reports accuracy and latent
geometry only. It makes no source-benchmark, inference-cost, or learning-rule
gradient claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import warnings
from dataclasses import asdict, dataclass, replace
from numbers import Integral
from typing import Any, Dict, Literal, Optional

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np

import braintrace

try:
    from examples.pp_prop.latent_workspace_analysis import analyze_latent_workspace
    from examples.pp_prop.latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
        occupied_slot_derangement,
        parameter_snapshot,
    )
    from examples.pp_prop.latent_workspace_task import (
        Episode,
        MatchedEpisodes,
        TaskConfig,
        generate_episode,
        generate_matched_episodes,
    )
except ModuleNotFoundError:
    from latent_workspace_analysis import analyze_latent_workspace
    from latent_workspace_model import (
        LatentWorkspaceModel,
        ModelConfig,
        occupied_slot_derangement,
        parameter_snapshot,
    )
    from latent_workspace_task import (
        Episode,
        MatchedEpisodes,
        TaskConfig,
        generate_episode,
        generate_matched_episodes,
    )


DeviceName = Literal["cpu", "gpu"]
DEFAULT_DEPTHS = (0, 1, 2, 4, 8)
DEFAULT_BINDING_COUNTS = tuple(range(2, 9))
DEFAULT_COUPLED_JACOBIAN_BUDGET = 67_108_864
COUPLED_STATE_LEAVES = 3
CLAIM_BOUNDARY = (
    "Claim boundary: this is an instantiation of the published system-level "
    "interface only. The source system's internal update rules, dimensions, "
    "and training recipe are unpublished, so this is not a reproduction. No "
    "source benchmark score or inference-cost claim is made, and no property "
    "of pp_prop's gradient estimate is asserted. The memory-write projections "
    "are fixed_random in this release; pp_prop trains only the remaining "
    "eligible parameters, so this is not evidence of a learned write path."
)


def _validated_nonnegative_seed(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative non-boolean integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative non-boolean integer")
    return result


@dataclass(frozen=True)
class ExperimentConfig:
    """Configure training, interventions, analysis, and output.

    Parameters
    ----------
    seed : int
        Seed used for data and parameter initialization.
    codebook_seed : int
        Seed for the fixed local BrainState symbol-code stream.
    projection_seed : int
        Seed for the fixed random memory-write projections.
    device : {"cpu", "gpu"}
        Requested JAX platform. GPU is the fail-closed default.
    depths : tuple of int
        Latent-iteration depths trained and evaluated independently.
    binding_counts : tuple of int
        Demonstration binding counts in the frozen intervention grid.
    batch_size : int
        Native model batch size and episodes per intervention cell.
    training_updates : int
        Number of terminal-only pp_prop updates per depth.
    latent_width : int
        Width of memory-factor rows and the latent workspace.
    code_width : int
        Width of each distributed spike code bank.
    symbol_ticks : int
        Ticks used for each demonstration and query symbol.
    figure_path : pathlib.Path
        Destination for the three-panel Agg PNG.
    learning_rate : float
        Adam learning rate for eligible non-write parameters.
    symbol_count : int
        Number of symbols in each fresh episode rule.
    slot_capacity : int
        Maximum simultaneous contextual bindings.
    spike_rate : float
        Bernoulli activation probability for each symbol-code channel. The
        realized rate is measured from the fixed sampled codebook.
    """

    seed: int = 2108
    codebook_seed: int = 313320
    projection_seed: int = 210848
    device: DeviceName = "gpu"
    depths: tuple[int, ...] = DEFAULT_DEPTHS
    binding_counts: tuple[int, ...] = DEFAULT_BINDING_COUNTS
    batch_size: int = 4
    training_updates: int = 8
    latent_width: int = 32
    code_width: int = 24
    symbol_ticks: int = 4
    figure_path: pathlib.Path = pathlib.Path("latent-reasoning-in-context.png")
    learning_rate: float = 1e-5
    symbol_count: int = 10
    slot_capacity: int = 8
    spike_rate: float = 0.25
    jacobian_budget_elements: int = DEFAULT_COUPLED_JACOBIAN_BUDGET

    def __post_init__(self) -> None:
        for name in ("codebook_seed", "projection_seed"):
            object.__setattr__(
                self,
                name,
                _validated_nonnegative_seed(getattr(self, name), name),
            )
        if self.device not in ("cpu", "gpu"):
            raise ValueError(f"device must be 'cpu' or 'gpu', got {self.device!r}")
        if not self.depths or any(depth < 0 for depth in self.depths):
            raise ValueError("depths must contain nonnegative integers")
        if len(set(self.depths)) != len(self.depths):
            raise ValueError("depths must not contain duplicates")
        if not self.binding_counts:
            raise ValueError("binding_counts must not be empty")
        for binding_count in self.binding_counts:
            if not 1 <= binding_count <= self.slot_capacity:
                raise ValueError(
                    f"binding count {binding_count} exceeds memory capacity "
                    f"{self.slot_capacity}"
                )
        if len(set(self.binding_counts)) != len(self.binding_counts):
            raise ValueError("binding_counts must not contain duplicates")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for held-out probes")
        if self.training_updates < 1:
            raise ValueError("training_updates must be positive")
        if self.latent_width < 1:
            raise ValueError("latent_width must be positive")
        estimated_elements = self.coupled_jacobian_elements
        if self.jacobian_budget_elements < 1:
            raise ValueError("jacobian_budget_elements must be positive")
        if estimated_elements > self.jacobian_budget_elements:
            raise ValueError(
                f"batch_size {self.batch_size} with slot_capacity "
                f"{self.slot_capacity} and latent_width {self.latent_width} "
                f"estimates {estimated_elements} coupled Jacobian elements, "
                f"exceeding the supported budget {self.jacobian_budget_elements}"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        TaskConfig(
            symbol_count=self.symbol_count,
            binding_count=max(self.binding_counts),
            slot_capacity=self.slot_capacity,
            latent_steps=max(self.depths),
            code_width=self.code_width,
            spike_rate=self.spike_rate,
            symbol_ticks=self.symbol_ticks,
            codebook_seed=self.codebook_seed,
        )

    @classmethod
    def smoke(
        cls,
        *,
        seed: int = 2108,
        codebook_seed: int = 313320,
        projection_seed: int = 210848,
        figure_path: pathlib.Path = pathlib.Path("latent-reasoning-smoke.png"),
        device: DeviceName = "cpu",
    ) -> ExperimentConfig:
        """Return a tiny configuration retaining every depth and control arm.

        Parameters
        ----------
        seed : int
            Reproducibility seed.
        codebook_seed : int
            Seed for the fixed sampled symbol codebook.
        projection_seed : int
            Seed for the fixed memory-write projections.
        figure_path : pathlib.Path
            Destination for the smoke PNG.
        device : {"cpu", "gpu"}
            Explicit platform for the smoke run.

        Returns
        -------
        ExperimentConfig
            Reduced-width, one-update configuration with the full grid.
        """
        return cls(
            seed=seed,
            codebook_seed=codebook_seed,
            projection_seed=projection_seed,
            device=device,
            batch_size=2,
            training_updates=1,
            latent_width=4,
            code_width=12,
            symbol_ticks=4,
            figure_path=figure_path,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly configuration mapping."""
        result = asdict(self)
        result["depths"] = list(self.depths)
        result["binding_counts"] = list(self.binding_counts)
        result["figure_path"] = str(self.figure_path)
        result["coupled_jacobian_elements"] = self.coupled_jacobian_elements
        result["coupled_jacobian_state_leaves"] = COUPLED_STATE_LEAVES
        return result

    @property
    def coupled_jacobian_elements(self) -> int:
        """Return the conservative grouped-state Jacobian element estimate."""
        grouped_width = (
            self.batch_size * (2 * self.slot_capacity + 1) * self.latent_width
        )
        return (grouped_width * COUPLED_STATE_LEAVES) ** 2


@dataclass(frozen=True)
class TrainingCorpus:
    """Hold the shared mixed-binding training episodes.

    Parameters
    ----------
    episodes : tuple of Episode
        Supported-context base episodes without latent ticks.
    binding_counts : numpy.ndarray
        Binding count used by each episode.
    targets : numpy.ndarray
        Terminal answer labels.
    rules : numpy.ndarray
        Fresh per-episode bijections retained for reproducibility audits.
    """

    episodes: tuple[Episode, ...]
    binding_counts: np.ndarray
    targets: np.ndarray
    rules: np.ndarray


@dataclass(frozen=True)
class EvaluationBatch:
    """Hold one frozen intervention cell's outputs.

    Parameters
    ----------
    correct : numpy.ndarray
        Per-episode terminal correctness flags.
    workspace : numpy.ndarray
        States ``H_0`` through ``H_R``.
    memory_read : numpy.ndarray
        Exact query-conditioned read of the contextual memory.
    memory_values, memory_keys : numpy.ndarray
        Raw factored memory rows used only for secondary analysis.
    parameters_unchanged : bool
        Whether the frozen-evaluation parameter audit passed.
    """

    correct: np.ndarray
    workspace: np.ndarray
    memory_read: np.ndarray
    memory_values: np.ndarray
    memory_keys: np.ndarray
    parameters_unchanged: bool


def _devices_for_platform(platform: DeviceName) -> list[jax.Device]:
    return list(jax.devices(platform))


def _resolve_device(requested: DeviceName) -> tuple[jax.Device, dict[str, object]]:
    try:
        devices = _devices_for_platform(requested)
    except RuntimeError as error:
        devices = []
        unavailable = error
    else:
        unavailable = None
    if not devices:
        if requested == "gpu":
            detail = f" ({unavailable})" if unavailable is not None else ""
            raise RuntimeError(
                "GPU execution was requested but no JAX GPU is visible"
                f"{detail}; pass --device cpu explicitly for a CPU run"
            )
        raise RuntimeError("CPU execution was requested but no JAX CPU is visible")
    device = devices[0]
    return device, {
        "requested": requested,
        "platform": str(device.platform),
        "id": int(device.id),
        "kind": str(getattr(device, "device_kind", device)),
    }


def _model_task(config: ExperimentConfig, depth: int) -> TaskConfig:
    if depth not in config.depths:
        raise ValueError(f"depth {depth} is not configured in {config.depths}")
    return TaskConfig(
        symbol_count=config.symbol_count,
        binding_count=config.slot_capacity,
        slot_capacity=config.slot_capacity,
        latent_steps=depth,
        code_width=config.code_width,
        spike_rate=config.spike_rate,
        symbol_ticks=config.symbol_ticks,
        codebook_seed=config.codebook_seed,
    )


def _episode_task(config: ExperimentConfig, binding_count: int) -> TaskConfig:
    return TaskConfig(
        symbol_count=config.symbol_count,
        binding_count=binding_count,
        slot_capacity=config.slot_capacity,
        latent_steps=0,
        code_width=config.code_width,
        spike_rate=config.spike_rate,
        symbol_ticks=config.symbol_ticks,
        codebook_seed=config.codebook_seed,
    )


def _mixed_binding_schedule(
    binding_counts: tuple[int, ...], total: int, rng: brainstate.random.RandomState
) -> np.ndarray:
    if total < 2:
        raise ValueError("mixed binding schedule requires at least two episodes")
    values = np.asarray(binding_counts, dtype=np.int32)
    if values.size == 1:
        raise ValueError("mixed binding schedule requires at least two binding counts")
    if total < values.size:
        indices = np.rint(np.linspace(0, values.size - 1, total)).astype(np.int32)
        schedule = values[indices]
    else:
        schedule = np.resize(values, total)
    permutation = np.asarray(rng.permutation(total), dtype=np.int32)
    return schedule[permutation]


def _build_training_corpus(config: ExperimentConfig) -> TrainingCorpus:
    total = config.batch_size * config.training_updates
    rng = brainstate.random.RandomState(config.seed + 10_000)
    schedule = _mixed_binding_schedule(config.binding_counts, total, rng)
    episodes = tuple(
        generate_episode(
            _episode_task(config, int(binding_count)),
            rng,
            condition="supported",
        )
        for binding_count in schedule
    )
    return TrainingCorpus(
        episodes=episodes,
        binding_counts=schedule,
        targets=np.asarray([episode.target for episode in episodes], dtype=np.int32),
        rules=np.stack([episode.rule for episode in episodes]),
    )


def _build_held_out_corpus(
    config: ExperimentConfig,
) -> dict[int, tuple[MatchedEpisodes, ...]]:
    rng = brainstate.random.RandomState(config.seed + 20_000)
    return {
        binding_count: tuple(
            generate_matched_episodes(_episode_task(config, binding_count), rng)
            for _ in range(config.batch_size)
        )
        for binding_count in config.binding_counts
    }


def _canonical_inputs(episode: Episode, destination: TaskConfig) -> np.ndarray:
    source = episode.config
    if source.input_width != destination.input_width:
        raise ValueError(
            "episode input_width does not match destination input_width: "
            f"{source.input_width} != {destination.input_width}"
        )
    if source.symbol_ticks != destination.symbol_ticks:
        raise ValueError(
            "episode symbol_ticks does not match destination symbol_ticks: "
            f"{source.symbol_ticks} != {destination.symbol_ticks}"
        )
    if source.binding_count > destination.binding_count:
        raise ValueError(
            f"episode binding count {source.binding_count} exceeds destination "
            f"capacity span {destination.binding_count}"
        )
    packed = np.zeros(
        (destination.total_steps, destination.input_width), dtype=np.float32
    )
    packed[: destination.demonstration_steps, destination.phase_slice.start] = 1.0
    packed[: source.demonstration_steps] = episode.model_inputs[
        : source.demonstration_steps
    ]
    packed[destination.query_slice] = episode.query_inputs
    packed[destination.latent_slice, destination.phase_slice.start + 3] = 1.0
    if destination.latent_steps:
        packed[destination.latent_slice.start, destination.phase_slice.start + 3] = 0.0
        packed[destination.latent_slice.start, destination.phase_slice.start + 2] = 1.0
    return packed


def _path_name(path: object) -> str:
    if isinstance(path, tuple):
        return ".".join(str(part) for part in path)
    return str(path)


def _json_counts(counts: object) -> dict[str, int]:
    if not isinstance(counts, dict):
        return {}
    return {str(key): int(value) for key, value in counts.items()}


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _parameter_fingerprints(model: LatentWorkspaceModel) -> dict[str, str]:
    return {
        name: _array_fingerprint(value)
        for name, value in parameter_snapshot(model).items()
    }


def _train_depth(
    config: ExperimentConfig, depth: int, corpus: TrainingCorpus
) -> tuple[LatentWorkspaceModel, dict[str, object]]:
    task = _model_task(config, depth)
    model = LatentWorkspaceModel(
        ModelConfig(
            task=task,
            batch_size=config.batch_size,
            latent_width=config.latent_width,
            write_mode="fixed_random",
            seed=config.seed,
            projection_seed=config.projection_seed,
        )
    )
    flat_inputs = np.stack(
        [_canonical_inputs(episode, task) for episode in corpus.episodes]
    )
    input_batches = jnp.asarray(
        flat_inputs.reshape(
            config.training_updates,
            config.batch_size,
            task.total_steps,
            task.input_width,
        ).transpose(0, 2, 1, 3)
    )
    target_batches = jnp.asarray(
        corpus.targets.reshape(config.training_updates, config.batch_size)
    )
    initial_parameter_fingerprints = _parameter_fingerprints(model)
    canonical_prefix_fingerprint = _array_fingerprint(
        flat_inputs[:, : task.query_slice.stop]
    )
    sample = jnp.zeros((config.batch_size, task.input_width), dtype=jnp.float32)

    with warnings.catch_warnings(record=True) as compile_warnings:
        warnings.simplefilter("always")
        learner = braintrace.compile(
            model,
            model.etrace_config(),
            sample,
            batch_size=config.batch_size,
            vmap=False,
            verbose=0,
            snap_max_jacobian_elements=config.jacobian_budget_elements,
        )
        jax.block_until_ready(learner(sample))

    trainable = model.trainable_parameters()
    optimizer = braintools.optim.Adam(lr=config.learning_rate)
    optimizer.register_trainable_weights(trainable)
    terminal_mask = jnp.zeros((task.total_steps,), dtype=jnp.float32)
    terminal_mask = terminal_mask.at[-1].set(1.0)

    def reset_runtime() -> None:
        model.reset_state()
        learner.reset_state()

    @brainstate.transform.jit
    def train_all(batches: jax.Array, targets: jax.Array) -> jax.Array:
        def train_one(inputs: jax.Array, labels: jax.Array):
            reset_runtime()

            def step_loss(step_input: jax.Array) -> jax.Array:
                logits = learner(step_input)
                return braintools.metric.softmax_cross_entropy_with_integer_labels(
                    logits, labels
                ).mean()

            grads, loss = learner.etrace_grad(
                inputs,
                step_fn=step_loss,
                mask=terminal_mask,
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            selected = {path: grads[path] for path in trainable}
            optimizer.update(brainstate.nn.clip_grad_norm(selected, 1.0))
            model.project_dale()
            return loss

        return brainstate.transform.for_loop(train_one, batches, targets)

    before = parameter_snapshot(model)
    with warnings.catch_warnings(record=True) as training_warnings:
        warnings.simplefilter("always")
        losses = train_all(input_batches, target_batches)
        jax.block_until_ready(losses)
    after = parameter_snapshot(model)
    deltas = {
        name: float(np.linalg.norm(after[name] - value))
        for name, value in before.items()
    }
    write_changed = deltas.get("Wk", 0.0) > 0.0 or deltas.get("Wv", 0.0) > 0.0
    if write_changed:
        raise RuntimeError("fixed_random training changed Wk or Wv")
    report = learner.report
    return model, {
        "depth": depth,
        "losses": np.asarray(losses, dtype=float).tolist(),
        "parameter_l2_deltas": deltas,
        "terminal_only_supervision": True,
        "write_mode": model.config.write_mode,
        "write_projections_updated": write_changed,
        "mixed_binding_counts": sorted(set(corpus.binding_counts.tolist())),
        "trainable_parameters": sorted(_path_name(path) for path in trainable),
        "initial_parameter_fingerprints": initial_parameter_fingerprints,
        "canonical_prefix_fingerprint": canonical_prefix_fingerprint,
        "compiler": {
            "warnings": [str(item.message) for item in compile_warnings],
            "training_warnings": [str(item.message) for item in training_warnings],
            "diagnostic_counts": _json_counts(report.counts),
            "recurrence_scope": model.etrace_config().recurrence_scope,
        },
    }


def _evaluate_batch(
    model: LatentWorkspaceModel,
    task: TaskConfig,
    episodes: tuple[Episode, ...],
    *,
    shuffled: bool,
) -> EvaluationBatch:
    return _evaluate_grid(model, task, (episodes,), (shuffled,))[0]


def _cell_report(batch: EvaluationBatch) -> dict[str, object]:
    return {
        "accuracy": float(np.mean(batch.correct)),
        "correct": int(np.sum(batch.correct)),
        "episodes": int(batch.correct.size),
        "parameters_unchanged": batch.parameters_unchanged,
    }


def _run_demonstration_prefix(
    model: LatentWorkspaceModel, demonstration: jax.Array
) -> jax.Array:
    def step(one_tick: jax.Array) -> jax.Array:
        return model.update(one_tick)

    return brainstate.transform.for_loop(step, demonstration)


def _apply_optional_memory_shuffle(
    model: LatentWorkspaceModel,
    permutation: jax.Array,
    shuffled: jax.Array,
) -> None:
    values, keys = model.memory_factors()
    permuted_values = jnp.take(values, permutation, axis=1)
    selected_values = jnp.where(shuffled, permuted_values, values)
    state = jnp.concatenate(
        (selected_values, keys, model.workspace[:, None, :]), axis=1
    )
    model.grouped_state.value = state.reshape(
        model.batch_size * model.state_rows, model.width
    )


def _run_query_and_latent(
    model: LatentWorkspaceModel,
    task: TaskConfig,
    query_and_latent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    def step(one_tick: jax.Array) -> tuple[jax.Array, jax.Array]:
        return model.update(one_tick), model.workspace

    logits, workspace = brainstate.transform.for_loop(step, query_and_latent)
    values, keys = model.memory_factors()
    memory_read = model.memory_read(model.query_encoding_view)
    latent_workspace = workspace[task.symbol_ticks - 1 :].transpose(1, 0, 2)
    return logits[-1], latent_workspace, memory_read, values, keys


def _evaluate_grid(
    model: LatentWorkspaceModel,
    task: TaskConfig,
    episode_cells: tuple[tuple[Episode, ...], ...],
    shuffled_flags: tuple[bool, ...],
) -> tuple[EvaluationBatch, ...]:
    if not episode_cells:
        raise ValueError("evaluation grid must contain at least one cell")
    if len(episode_cells) != len(shuffled_flags):
        raise ValueError(
            "evaluation cells and shuffled flags must have identical lengths"
        )
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    permutations: list[np.ndarray] = []
    for episodes, shuffled in zip(episode_cells, shuffled_flags, strict=True):
        if len(episodes) != model.batch_size:
            raise ValueError(
                f"evaluation episodes {len(episodes)} do not match model "
                f"batch_size {model.batch_size}"
            )
        occupied_count = episodes[0].config.binding_count
        if any(episode.config.binding_count != occupied_count for episode in episodes):
            raise ValueError("all episodes in one cell must share one binding count")
        packed = np.stack(
            [_canonical_inputs(episode, task) for episode in episodes]
        ).transpose(1, 0, 2)
        sequences.append(packed)
        targets.append(
            np.asarray([episode.target for episode in episodes], dtype=np.int32)
        )
        if shuffled:
            permutation = occupied_slot_derangement(task.slot_capacity, occupied_count)
        else:
            permutation = jnp.arange(task.slot_capacity, dtype=jnp.int32)
        permutations.append(np.asarray(permutation, dtype=np.int32))

    sequence_array = jnp.asarray(np.stack(sequences))
    target_array = jnp.asarray(np.stack(targets))
    permutation_array = jnp.asarray(np.stack(permutations))
    shuffled_array = jnp.asarray(shuffled_flags, dtype=jnp.bool_)[:, None, None]
    before = parameter_snapshot(model)

    @brainstate.transform.jit
    def evaluate_all(
        cell_sequences: jax.Array,
        cell_targets: jax.Array,
        cell_permutations: jax.Array,
        cell_shuffled: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        def evaluate_one(
            sequence: jax.Array,
            labels: jax.Array,
            permutation: jax.Array,
            shuffled: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            model.reset_state()
            _run_demonstration_prefix(model, sequence[: task.demonstration_steps])
            _apply_optional_memory_shuffle(model, permutation, shuffled)
            logits, workspace, memory_read, values, keys = _run_query_and_latent(
                model, task, sequence[task.demonstration_steps :]
            )
            return (
                jnp.argmax(logits, axis=-1) == labels,
                workspace,
                memory_read,
                values,
                keys,
            )

        return brainstate.transform.for_loop(
            evaluate_one,
            cell_sequences,
            cell_targets,
            cell_permutations,
            cell_shuffled,
        )

    correct, workspace, memory_read, values, keys = evaluate_all(
        sequence_array,
        target_array,
        permutation_array,
        shuffled_array,
    )
    jax.block_until_ready(correct)
    after = parameter_snapshot(model)
    unchanged = before.keys() == after.keys() and all(
        np.array_equal(before[name], after[name]) for name in before
    )
    if not unchanged:
        raise RuntimeError("frozen intervention evaluation changed a parameter")
    arrays = tuple(
        np.asarray(value) for value in (correct, workspace, memory_read, values, keys)
    )
    if arrays[1].shape[2] != task.latent_steps + 1:
        raise RuntimeError(
            "workspace collection did not produce H0 through HR: "
            f"shape {arrays[1].shape}, R={task.latent_steps}"
        )
    return tuple(
        EvaluationBatch(
            correct=arrays[0][index],
            workspace=arrays[1][index],
            memory_read=arrays[2][index],
            memory_values=arrays[3][index],
            memory_keys=arrays[4][index],
            parameters_unchanged=unchanged,
        )
        for index in range(len(episode_cells))
    )


def _depth_interventions(
    config: ExperimentConfig,
    depth: int,
    model: LatentWorkspaceModel,
    held_out: dict[int, tuple[MatchedEpisodes, ...]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate one depth with one compiled call over the fixed-shape grid.

    The host loops only construct distinct intervention cells and label outputs
    after collection. The outer cell axis and both temporal spans are driven by
    :mod:`brainstate.transform` loops.
    """
    task = _model_task(config, depth)
    labels: list[tuple[int, str, str]] = []
    episode_cells: list[tuple[Episode, ...]] = []
    shuffled_flags: list[bool] = []
    for binding_count in config.binding_counts:
        pairs = held_out[binding_count]
        conditions = {
            "supported": tuple(pair.supported for pair in pairs),
            "short": tuple(pair.short for pair in pairs),
        }
        for condition, episodes in conditions.items():
            for arm, shuffled in (("intact", False), ("shuffled", True)):
                labels.append((binding_count, condition, arm))
                episode_cells.append(episodes)
                shuffled_flags.append(shuffled)
    evaluated_cells = _evaluate_grid(
        model,
        task,
        tuple(episode_cells),
        tuple(shuffled_flags),
    )

    per_binding: dict[str, object] = {
        str(binding_count): {"supported": {}, "short": {}}
        for binding_count in config.binding_counts
    }
    analysis_batches: list[EvaluationBatch] = []
    analysis_episodes: list[Episode] = []
    totals = {
        name: [0, 0]
        for name in (
            "supported_intact",
            "short_intact",
            "supported_shuffled",
            "short_shuffled",
        )
    }
    for label, episodes, evaluated in zip(
        labels, episode_cells, evaluated_cells, strict=True
    ):
        binding_count, condition, arm = label
        per_binding[str(binding_count)][condition][arm] = _cell_report(evaluated)
        total = totals[f"{condition}_{arm}"]
        total[0] += int(np.sum(evaluated.correct))
        total[1] += int(evaluated.correct.size)
        if condition == "supported" and arm == "intact":
            analysis_batches.append(evaluated)
            analysis_episodes.extend(episodes)

    accuracies = {name: correct / count for name, (correct, count) in totals.items()}
    intervention = {
        "depth": depth,
        "overall_accuracy": accuracies["supported_intact"],
        "per_binding_count": per_binding,
        "supported_vs_short": {
            "supported_intact": accuracies["supported_intact"],
            "short_intact": accuracies["short_intact"],
            "supported_minus_short": (
                accuracies["supported_intact"] - accuracies["short_intact"]
            ),
        },
        "intact_vs_shuffled": {
            "supported_intact": accuracies["supported_intact"],
            "supported_shuffled": accuracies["supported_shuffled"],
            "intact_minus_shuffled": (
                accuracies["supported_intact"] - accuracies["supported_shuffled"]
            ),
        },
        "all_frozen_parameter_audits_passed": all(
            cell["parameters_unchanged"]
            for binding in per_binding.values()
            for condition in binding.values()
            for cell in condition.values()
        ),
    }

    states = np.concatenate([batch.workspace for batch in analysis_batches])
    memory_read = np.concatenate([batch.memory_read for batch in analysis_batches])
    memory_values = np.concatenate([batch.memory_values for batch in analysis_batches])
    memory_keys = np.concatenate([batch.memory_keys for batch in analysis_batches])
    answers = np.asarray(
        [episode.target for episode in analysis_episodes], dtype=np.int32
    )
    rules = np.stack([episode.rule for episode in analysis_episodes])
    fit_indices = np.arange(0, answers.size, 2, dtype=np.int32)
    score_indices = np.arange(1, answers.size, 2, dtype=np.int32)
    geometry = analyze_latent_workspace(
        states,
        memory_read,
        answers,
        rules,
        fit_indices,
        score_indices,
        memory_values=memory_values,
        memory_keys=memory_keys,
    )
    return intervention, geometry


def _figure_series(result: dict[str, object]) -> dict[str, object]:
    config = result["config"]
    depths = config["depths"]
    binding_counts = config["binding_counts"]
    interventions = result["interventions"]["depths"]
    geometry = result["geometry"]["depths"]
    final_depth = str(max(depths))
    final_cells = interventions[final_depth]["per_binding_count"]
    final_geometry = geometry[final_depth]
    workspace_scores = final_geometry["answer_decodability"]["workspace_per_iteration"]
    return {
        "accuracy_vs_depth": {
            "x": list(depths),
            "accuracy": [
                interventions[str(depth)]["overall_accuracy"] for depth in depths
            ],
        },
        "accuracy_vs_binding_count": {
            "x": list(binding_counts),
            "depth": int(final_depth),
            "supported": [
                final_cells[str(count)]["supported"]["intact"]["accuracy"]
                for count in binding_counts
            ],
            "short": [
                final_cells[str(count)]["short"]["intact"]["accuracy"]
                for count in binding_counts
            ],
        },
        "decodability_per_iteration": {
            "x": list(range(len(workspace_scores))),
            "workspace": workspace_scores,
            "memory_read": final_geometry["answer_decodability"]["memory_read"],
        },
    }


def _plot_report(result: dict[str, object], path: pathlib.Path) -> pathlib.Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    series = _figure_series(result)
    depth_series = series["accuracy_vs_depth"]
    binding_series = series["accuracy_vs_binding_count"]
    decodability_series = series["decodability_per_iteration"]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(depth_series["x"], depth_series["accuracy"], marker="o")
    axes[0].set(title="Accuracy vs latent depth", xlabel="R", ylabel="accuracy")
    axes[0].set_ylim(0.0, 1.0)

    for condition in ("supported", "short"):
        axes[1].plot(
            binding_series["x"],
            binding_series[condition],
            marker="o",
            label=condition,
        )
    axes[1].set(
        title=f"Context accuracy at R={binding_series['depth']}",
        xlabel="binding count K",
        ylabel="accuracy",
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    axes[2].plot(
        decodability_series["x"],
        decodability_series["workspace"],
        marker="o",
        label="workspace",
    )
    axes[2].axhline(
        decodability_series["memory_read"],
        color="tab:orange",
        linestyle="--",
        label="memory read",
    )
    axes[2].set(
        title="Answer decodability",
        xlabel="latent iteration",
        ylabel="held-out accuracy",
    )
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend()
    figure.tight_layout()
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _render_report(result: dict[str, object]) -> str:
    config = result["config"]
    lines = [
        "Example 21 - in-context latent reasoning",
        (
            f"seed={config['seed']} codebook_seed={config['codebook_seed']} "
            f"projection_seed={config['projection_seed']} "
            f"device={result['device']['requested']} "
            f"batch={config['batch_size']} updates={config['training_updates']} "
            f"width={config['latent_width']} write_mode=fixed_random"
        ),
        f"canonical_metadata_json={result['canonical_metadata_json']}",
    ]
    interventions = result["interventions"]["depths"]
    geometry = result["geometry"]["depths"]
    training = result["training"]["depths"]
    for depth in config["depths"]:
        entry = interventions[str(depth)]
        training_entry = training[str(depth)]
        lines.append(
            f"R={depth} overall supported/intact accuracy="
            f"{entry['overall_accuracy']:.6f}"
        )
        lines.append(
            f"  terminal loss={training_entry['losses'][-1]:.6f} "
            f"parameter_l2_deltas={training_entry['parameter_l2_deltas']}"
        )
        for binding_count in config["binding_counts"]:
            cell = entry["per_binding_count"][str(binding_count)]
            lines.append(
                f"  K={binding_count} supported intact="
                f"{cell['supported']['intact']['accuracy']:.6f} "
                f"shuffled={cell['supported']['shuffled']['accuracy']:.6f}; "
                f"short intact={cell['short']['intact']['accuracy']:.6f} "
                f"shuffled={cell['short']['shuffled']['accuracy']:.6f}"
            )
        lines.append(
            "  supported-short="
            f"{entry['supported_vs_short']['supported_minus_short']:.6f}; "
            "intact-shuffled="
            f"{entry['intact_vs_shuffled']['intact_minus_shuffled']:.6f}"
        )
        measures = geometry[str(depth)]
        lines.append(
            "  geometry participation_ratio="
            f"{measures['participation_ratio']} trajectory_step_norm="
            f"{measures['trajectory_step_norm']}"
        )
        lines.append(
            "  answer_decodability="
            f"{measures['answer_decodability']} full_rule_decodability="
            f"{measures['rule_decodability']}"
        )
        lines.append(
            "  primary exact_query_memory_read="
            f"{measures['answer_decodability']['memory_read']}; "
            "secondary raw_memory_factors="
            f"{measures['raw_memory_factor_decodability']}"
        )
        split = measures["probe_split"]
        lines.append(
            f"  probe split fit={split['fit_count']} score={split['score_count']}; "
            f"{measures['comparison']}"
        )
    lines.extend(
        (
            f"figure={result['figure_path']}",
            str(result["claim_boundary"]),
        )
    )
    return "\n".join(lines)


def _active_device_report(requested: DeviceName) -> dict[str, object]:
    device = jax.devices()[0]
    return {
        "requested": requested,
        "platform": str(device.platform),
        "id": int(device.id),
        "kind": str(getattr(device, "device_kind", device)),
    }


def _run_experiment(
    config: ExperimentConfig,
    device_report: Optional[dict[str, object]] = None,
) -> Dict[str, Any]:
    training_corpus = _build_training_corpus(config)
    held_out = _build_held_out_corpus(config)
    training: dict[str, object] = {}
    interventions: dict[str, object] = {}
    geometry: dict[str, object] = {}
    for depth in config.depths:
        model, training_report = _train_depth(config, depth, training_corpus)
        intervention_report, geometry_report = _depth_interventions(
            config, depth, model, held_out
        )
        training[str(depth)] = training_report
        interventions[str(depth)] = intervention_report
        geometry[str(depth)] = geometry_report

    initial_parameter_sets = [
        training[str(depth)]["initial_parameter_fingerprints"]
        for depth in config.depths
    ]
    prefix_fingerprints = [
        training[str(depth)]["canonical_prefix_fingerprint"] for depth in config.depths
    ]
    initial_parameters_identical = all(
        value == initial_parameter_sets[0] for value in initial_parameter_sets[1:]
    )
    demo_query_inputs_identical = all(
        value == prefix_fingerprints[0] for value in prefix_fingerprints[1:]
    )
    if not initial_parameters_identical:
        raise RuntimeError("initial parameter fingerprints differ across depths")
    if not demo_query_inputs_identical:
        raise RuntimeError("demonstration/query tensors differ across depths")

    result: Dict[str, Any] = {
        "config": config.to_dict(),
        "device": device_report or _active_device_report(config.device),
        "training": {
            "shared_binding_counts": training_corpus.binding_counts.tolist(),
            "shared_targets": training_corpus.targets.tolist(),
            "depths": training,
        },
        "interventions": {
            "frozen_no_retraining": True,
            "depths": interventions,
        },
        "geometry": {
            "primary_memory_representation": "exact_query_memory_read",
            "raw_factors_are_secondary": True,
            "depths": geometry,
        },
        "reproducibility": {
            "seed": config.seed,
            "codebook_seed": config.codebook_seed,
            "projection_seed": config.projection_seed,
            "numerical_tolerance": 1e-6,
            "same_training_corpus_for_every_depth": True,
            "initial_parameters_identical_across_depths": (
                initial_parameters_identical
            ),
            "demo_query_inputs_identical_across_depths": (demo_query_inputs_identical),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    metadata_config = dict(result["config"])
    metadata_config.pop("figure_path")
    metadata = {
        "config": metadata_config,
        "device": result["device"],
        "reproducibility": {
            "seed": result["reproducibility"]["seed"],
            "codebook_seed": result["reproducibility"]["codebook_seed"],
            "projection_seed": result["reproducibility"]["projection_seed"],
            "numerical_tolerance": result["reproducibility"]["numerical_tolerance"],
        },
    }
    result["canonical_metadata_json"] = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    figure_path = _plot_report(result, config.figure_path)
    result["figure_path"] = str(figure_path)
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=2108)
    parser.add_argument("--codebook-seed", type=int, default=313320)
    parser.add_argument("--projection-seed", type=int, default=210848)
    parser.add_argument("--depths", type=int, nargs="+")
    parser.add_argument("--binding-counts", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--training-updates", type=int)
    parser.add_argument("--latent-width", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--figure",
        type=pathlib.Path,
        default=pathlib.Path("latent-reasoning-in-context.png"),
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.smoke:
        config = ExperimentConfig.smoke(
            seed=args.seed,
            codebook_seed=args.codebook_seed,
            projection_seed=args.projection_seed,
            device=args.device,
            figure_path=args.figure,
        )
    else:
        config = ExperimentConfig(
            seed=args.seed,
            codebook_seed=args.codebook_seed,
            projection_seed=args.projection_seed,
            device=args.device,
            figure_path=args.figure,
        )
    changes: dict[str, object] = {}
    for name in (
        "batch_size",
        "training_updates",
        "latent_width",
        "learning_rate",
    ):
        value = getattr(args, name)
        if value is not None:
            changes[name] = value
    if args.depths is not None:
        changes["depths"] = tuple(args.depths)
    if args.binding_counts is not None:
        changes["binding_counts"] = tuple(args.binding_counts)
    return replace(config, **changes)


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    """Run the complete seeded training and frozen intervention experiment.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments excluding the program name. ``None`` reads
        arguments from :mod:`sys.argv`.

    Returns
    -------
    dict
        JSON-friendly configuration, training, interventions, geometry,
        reproducibility metadata, figure path, and claim boundary.

    Raises
    ------
    RuntimeError
        If the requested device is unavailable. GPU requests fail closed and
        require an explicit ``--device cpu`` fallback.
    """
    args = _parse_args(argv)
    config = _config_from_args(args)
    device, device_report = _resolve_device(config.device)
    with jax.default_device(device):
        result = _run_experiment(config, device_report)
    print(_render_report(result))
    return result


if __name__ == "__main__":
    main()
