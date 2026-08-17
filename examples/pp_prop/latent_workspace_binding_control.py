"""Minimal binding capability control for the legacy Example 21 reservoir.

This module asks one deliberately narrow question before the associative
``S_K/H_r`` architecture is enabled: can the current LIF reservoir learn a
fresh one-cell color bijection when it receives exact BPTT credit, and how does
the same initialization and schedule behave under production pp-prop?

The generated tasks use the real :class:`LatentWorkspaceModel`, the real ARC
row-event encoder, and the existing shared factorized decoder.  Training and
validation mapping identifiers are unique and disjoint by construction.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import brainstate
import braintrace
import braintools
import jax
import jax.numpy as jnp
import numpy as np

from braintrace._testing.oracle import (
    bptt_param_gradients,
    chunked_online_param_gradients,
    flat_gradient_leaves,
    gradient_norm,
    relative_deviation,
)
from examples.pp_prop.latent_workspace_model import (
    LatentWorkspaceModel,
    ModelConfig,
    compile_pp_prop,
    expand_compact_logits,
)
from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    RowEventConfig,
    encode_query_episode,
)


COLOR_COUNT = 10
SYMBOL_COUNT = 4
_INPUT_SETS = tuple(itertools.combinations(range(COLOR_COUNT), SYMBOL_COUNT))
_OUTPUT_ASSIGNMENTS = tuple(itertools.permutations(range(COLOR_COUNT), SYMBOL_COUNT))
_MAPPING_COUNT = len(_INPUT_SETS) * len(_OUTPUT_ASSIGNMENTS)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive non-boolean integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive non-boolean integer")
    return result


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


@dataclass(frozen=True)
class BindingControlConfig:
    """Configure the preregistered legacy-reservoir binding control.

    Parameters
    ----------
    training_updates : int, default=10000
        Number of optimizer updates in each learning-rule arm.
    batch_size : int, default=64
        Fresh, unique binding episodes per update.
    validation_episodes : int, default=512
        Unique mappings held out from the complete training mapping set.
    gap_steps : int, default=1
        Advancing zero-input ticks after query ingestion and before loss.
    neuron_count, recurrent_edges, readout_width, color_rank : int
        Legacy :class:`LatentWorkspaceModel` dimensions.
    learning_rate, clip_norm : float
        Identical optimizer settings for exact BPTT and pp-prop.
    trace_decay : float, default=0.9
        Production pp-prop eligibility decay.
    model_seed, split_seed, train_episode_seed, validation_episode_seed : int
        Independent deterministic seeds.  ``split_seed`` defines an affine
        permutation of the complete mapping catalog; training and validation
        consume disjoint contiguous ranges of that permutation.
    gradient_chunk_size : int, default=1
        Finite pp-prop oracle window.  It must be shorter than the sequence.
    sparse_backend : str, optional
        Optional sparse execution backend forwarded to the legacy model.

    Examples
    --------
    .. code-block:: python

        >>> config = BindingControlConfig.smoke_config()
        >>> (config.symbol_count, config.sequence_length)
        (4, 6)
    """

    training_updates: int = 10000
    batch_size: int = 64
    validation_episodes: int = 512
    gap_steps: int = 1
    neuron_count: int = 2048
    recurrent_edges: int = 16384
    readout_width: int = 128
    color_rank: int = 16
    input_gain: float = 4.0
    recurrent_gain: float = 0.8
    learning_rate: float = 3e-3
    clip_norm: float = 1.0
    trace_decay: float = 0.9
    model_seed: int = 2108
    split_seed: int = 20260817
    train_episode_seed: int = 31021
    validation_episode_seed: int = 91021
    gradient_chunk_size: int = 1
    sparse_backend: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "training_updates",
            "batch_size",
            "validation_episodes",
            "gap_steps",
            "neuron_count",
            "recurrent_edges",
            "readout_width",
            "color_rank",
            "gradient_chunk_size",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in (
            "model_seed",
            "split_seed",
            "train_episode_seed",
            "validation_episode_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
                raise TypeError(f"{name} must be a nonnegative integer")
            if value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        for name in (
            "input_gain",
            "recurrent_gain",
            "learning_rate",
            "clip_norm",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))
        if isinstance(self.trace_decay, (bool, np.bool_)) or not isinstance(
            self.trace_decay, (int, float)
        ):
            raise TypeError("trace_decay must be a finite scalar in [0, 1)")
        decay = float(self.trace_decay)
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("trace_decay must be a finite scalar in [0, 1)")
        object.__setattr__(self, "trace_decay", decay)
        if self.neuron_count % 64:
            raise ValueError("neuron_count must be divisible by 64")
        if self.recurrent_edges > self.neuron_count * (self.neuron_count - 1):
            raise ValueError("recurrent_edges exceeds the no-self-edge capacity")
        if self.training_episode_count + self.validation_episodes > _MAPPING_COUNT:
            raise ValueError("requested episodes exceed the unique K=4 mapping catalog")
        if self.validation_episodes > self.training_episode_count:
            raise ValueError(
                "validation_episodes cannot exceed the fixed training-probe pool"
            )
        if self.gradient_chunk_size >= self.sequence_length:
            raise ValueError("gradient_chunk_size must be shorter than the sequence")
        if self.sparse_backend is not None and not isinstance(self.sparse_backend, str):
            raise TypeError("sparse_backend must be a string or None")

    @property
    def symbol_count(self) -> int:
        """Return the fixed number of one-cell binding symbols."""

        return SYMBOL_COUNT

    @property
    def row_config(self) -> RowEventConfig:
        """Return the exact one-row ARC event layout used by the control."""

        return RowEventConfig(
            max_demonstrations=SYMBOL_COUNT,
            max_grid_size=1,
        )

    @property
    def sequence_length(self) -> int:
        """Return four demonstrations, one query, and the latent gap."""

        return SYMBOL_COUNT + 1 + self.gap_steps

    @property
    def training_episode_count(self) -> int:
        """Return the number of unique training bijections."""

        return self.training_updates * self.batch_size

    @property
    def configuration_scale(self) -> str:
        """Classify the topology as production or reduced smoke scale."""

        production = (
            self.neuron_count == 2048
            and self.recurrent_edges == 16384
            and self.readout_width == 128
            and self.color_rank == 16
            and self.input_gain == 4.0
            and self.recurrent_gain == 0.8
            and self.trace_decay == 0.9
        )
        return "production_topology" if production else "reduced_smoke"

    @classmethod
    def smoke_config(cls) -> "BindingControlConfig":
        """Return a small deterministic configuration for tests.

        Returns
        -------
        BindingControlConfig
            Two updates of four episodes with the real 64-neuron reservoir.
        """

        return cls(
            training_updates=2,
            batch_size=4,
            validation_episodes=8,
            neuron_count=64,
            recurrent_edges=64,
            readout_width=8,
            color_rank=1,
            sparse_backend="jax_raw",
        )


@dataclass(frozen=True)
class BindingData:
    """Hold one shared training schedule and matched validation interventions.

    Parameters
    ----------
    training_events : numpy.ndarray
        ``(updates, time, batch, event_width)`` real row-event streams.
    training_targets, training_mapping_ids : numpy.ndarray
        Integer arrays shaped ``(updates, batch)``.
    validation_intact, validation_shuffled, validation_no_context : numpy.ndarray
        Timing-matched arrays shaped ``(time, validation, event_width)``.
    validation_targets, validation_mapping_ids : numpy.ndarray
        Held-out integer target and unique mapping identifiers.
    training_query_indices, validation_query_indices : numpy.ndarray
        Which of the four demonstrated inputs was queried in each episode.
    """

    training_events: np.ndarray
    training_targets: np.ndarray
    training_mapping_ids: np.ndarray
    training_query_indices: np.ndarray
    validation_intact: np.ndarray
    validation_shuffled: np.ndarray
    validation_no_context: np.ndarray
    validation_targets: np.ndarray
    validation_mapping_ids: np.ndarray
    validation_query_indices: np.ndarray


def _affine_mapping_ids(count: int, *, seed: int) -> np.ndarray:
    rng = brainstate.random.RandomState(seed)
    offset = int(np.asarray(rng.randint(0, _MAPPING_COUNT)))
    multiplier = int(np.asarray(rng.randint(1, _MAPPING_COUNT)))
    while math.gcd(multiplier, _MAPPING_COUNT) != 1:
        multiplier = multiplier % (_MAPPING_COUNT - 1) + 1
    positions = np.arange(count, dtype=np.int64)
    return (offset + multiplier * positions) % _MAPPING_COUNT


def _decode_mapping(mapping_id: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    input_index, output_index = divmod(int(mapping_id), len(_OUTPUT_ASSIGNMENTS))
    return _INPUT_SETS[input_index], _OUTPUT_ASSIGNMENTS[output_index]


def _episode_choices(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = brainstate.random.RandomState(seed)
    order_scores = np.asarray(rng.rand(count, SYMBOL_COUNT), dtype=np.float32)
    demonstration_orders = np.argsort(order_scores, axis=1).astype(np.int32)
    query_indices = np.asarray(rng.randint(0, SYMBOL_COUNT, size=count), dtype=np.int32)
    return demonstration_orders, query_indices


def _one_cell(color: int) -> ArcGrid:
    return ArcGrid(((int(color),),))


def _encode_mapping_episodes(
    mapping_ids: np.ndarray,
    *,
    seed: int,
    config: BindingControlConfig,
    controls: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    count = int(mapping_ids.size)
    rows = config.row_config
    orders, queries = _episode_choices(count, seed)
    intact = np.zeros(
        (count, config.sequence_length, rows.input_width), dtype=np.float32
    )
    shuffled = np.zeros_like(intact) if controls else None
    targets = np.zeros((count,), dtype=np.int32)

    for episode_index, mapping_id in enumerate(mapping_ids):
        input_colors, output_colors = _decode_mapping(int(mapping_id))
        order = orders[episode_index]
        demonstrations = tuple(
            ArcPair(_one_cell(input_colors[index]), _one_cell(output_colors[index]))
            for index in order
        )
        query_index = int(queries[episode_index])
        query = ArcPair(
            _one_cell(input_colors[query_index]),
            _one_cell(output_colors[query_index]),
        )
        task = ArcTask(train=demonstrations, test=(query,))
        encoded = encode_query_episode(task, 0, rows)
        intact[episode_index, : rows.max_events] = encoded.events
        targets[episode_index] = output_colors[query_index]

        if shuffled is not None:
            rotated = tuple(
                ArcPair(pair.input, demonstrations[(index + 1) % SYMBOL_COUNT].output)
                for index, pair in enumerate(demonstrations)
            )
            shuffled_task = ArcTask(train=rotated, test=(query,))
            shuffled[episode_index, : rows.max_events] = encode_query_episode(
                shuffled_task, 0, rows
            ).events

    no_context = None
    if controls:
        assert shuffled is not None
        no_context = np.array(intact, copy=True)
        no_context[:, :SYMBOL_COUNT] = 0.0
    return intact, targets, shuffled, no_context, queries


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def build_binding_data(config: BindingControlConfig) -> BindingData:
    """Generate the shared, unique, disjoint binding schedule.

    Parameters
    ----------
    config : BindingControlConfig
        Dataset sizes and deterministic split seeds.

    Returns
    -------
    BindingData
        Real row-event streams for training and all validation interventions.
    """

    if not isinstance(config, BindingControlConfig):
        raise TypeError("config must be a BindingControlConfig")
    total = config.training_episode_count + config.validation_episodes
    all_ids = _affine_mapping_ids(total, seed=config.split_seed)
    train_ids = all_ids[: config.training_episode_count]
    validation_ids = all_ids[config.training_episode_count :]
    if np.intersect1d(train_ids, validation_ids).size:
        raise RuntimeError("training and validation mapping ranges overlap")
    train, train_targets, _, _, train_queries = _encode_mapping_episodes(
        train_ids,
        seed=config.train_episode_seed,
        config=config,
        controls=False,
    )
    intact, validation_targets, shuffled, no_context, validation_queries = (
        _encode_mapping_episodes(
            validation_ids,
            seed=config.validation_episode_seed,
            config=config,
            controls=True,
        )
    )
    assert shuffled is not None and no_context is not None
    updates = config.training_updates
    batch = config.batch_size
    training_events = train.reshape(
        updates, batch, config.sequence_length, config.row_config.input_width
    ).transpose(0, 2, 1, 3)
    training_targets = train_targets.reshape(updates, batch)
    training_ids = train_ids.reshape(updates, batch)
    training_queries = train_queries.reshape(updates, batch)
    return BindingData(
        training_events=_readonly(training_events),
        training_targets=_readonly(training_targets),
        training_mapping_ids=_readonly(training_ids),
        training_query_indices=_readonly(training_queries),
        validation_intact=_readonly(intact.transpose(1, 0, 2)),
        validation_shuffled=_readonly(shuffled.transpose(1, 0, 2)),
        validation_no_context=_readonly(no_context.transpose(1, 0, 2)),
        validation_targets=_readonly(validation_targets),
        validation_mapping_ids=_readonly(validation_ids),
        validation_query_indices=_readonly(validation_queries),
    )


def _model_config(config: BindingControlConfig, *, batch_size: int) -> ModelConfig:
    return ModelConfig(
        input_width=config.row_config.input_width,
        batch_size=batch_size,
        neuron_count=config.neuron_count,
        recurrent_edges=config.recurrent_edges,
        max_latent_steps=config.gap_steps,
        readout_width=config.readout_width,
        color_rank=config.color_rank,
        input_gain=config.input_gain,
        recurrent_gain=config.recurrent_gain,
        trace_decay=config.trace_decay,
        seed=config.model_seed,
        sparse_backend=config.sparse_backend,
    )


def _parameter_values(model: brainstate.nn.Module) -> dict[str, Any]:
    return {
        "/".join(map(str, path)): jax.tree.map(
            lambda leaf: jnp.array(leaf, copy=True), state.value
        )
        for path, state in model.states(brainstate.ParamState).items()
    }


def _array_digest(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode("utf-8"))
        for leaf in jax.tree.leaves(values[key]):
            array = np.ascontiguousarray(np.asarray(leaf))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _copy_parameters(
    source: brainstate.nn.Module, target: brainstate.nn.Module
) -> None:
    source_states = source.states(brainstate.ParamState)
    target_states = target.states(brainstate.ParamState)
    if tuple(source_states.keys()) != tuple(target_states.keys()):
        raise ValueError("parameter paths differ")
    for source_state, target_state in zip(
        source_states.values(), target_states.values(), strict=True
    ):
        target_state.value = jax.tree.map(
            lambda leaf: jnp.array(leaf, copy=True), source_state.value
        )


def _color_logits(compact: jax.Array, color_rank: int) -> jax.Array:
    return expand_compact_logits(compact, color_rank).colors[:, 0, 0, :]


def _classification_loss(
    compact: jax.Array, targets: jax.Array, color_rank: int
) -> jax.Array:
    return braintools.metric.softmax_cross_entropy_with_integer_labels(
        _color_logits(compact, color_rank), targets
    ).mean()


def _path_group(label: str) -> str:
    if "ff_syn/comm" in label:
        return "feedforward"
    if "rec_syn/comm" in label:
        return "recurrent"
    if "readout_projection" in label:
        return "readout_projection"
    if "color_factor_head" in label:
        return "color_decoder"
    if "height_head" in label or "width_head" in label:
        return "unused_shape_heads"
    return "other"


def _parameter_movement(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, dict[str, float | int]]:
    totals: dict[str, list[float]] = {}
    for path in before:
        if path not in after:
            raise ValueError(f"parameter disappeared during training: {path}")
        group = _path_group(path)
        bucket = totals.setdefault(group, [0.0, 0.0])
        for old, new in zip(
            jax.tree.leaves(before[path]), jax.tree.leaves(after[path]), strict=True
        ):
            delta = np.asarray(new, dtype=np.float64) - np.asarray(
                old, dtype=np.float64
            )
            bucket[0] += float(np.sum(delta * delta))
            bucket[1] += float(delta.size)
    return {
        group: {"l2_delta": math.sqrt(values[0]), "parameter_count": int(values[1])}
        for group, values in sorted(totals.items())
    }


def _train_bptt(
    model: LatentWorkspaceModel,
    data: BindingData,
    config: BindingControlConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parameters = model.states(brainstate.ParamState)
    optimizer = braintools.optim.Adam(lr=config.learning_rate)
    optimizer.register_trainable_weights(parameters)
    advance = jnp.ones((config.batch_size,), dtype=jnp.bool_)
    rank = config.color_rank

    @brainstate.transform.jit
    def train_all(events: jax.Array, targets: jax.Array) -> jax.Array:
        def train_one(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
            sequence, target = inputs
            model.reset_state()

            def objective() -> jax.Array:
                outputs = brainstate.transform.for_loop(
                    lambda event: model.update(event, advance), sequence
                )
                return _classification_loss(outputs[-1], target, rank)

            gradients, loss = brainstate.transform.grad(
                objective, parameters, return_value=True
            )()
            optimizer.update(brainstate.nn.clip_grad_norm(gradients, config.clip_norm))
            return loss

        return brainstate.transform.for_loop(train_one, (events, targets))

    before = _parameter_values(model)
    start = time.perf_counter()
    losses = train_all(
        jnp.asarray(data.training_events), jnp.asarray(data.training_targets)
    )
    losses = np.asarray(jax.block_until_ready(losses), dtype=np.float64)
    elapsed = time.perf_counter() - start
    after = _parameter_values(model)
    return (
        {
            "algorithm": "exact_bptt",
            "loss_definition": "terminal one-cell color cross_entropy",
            "losses": losses.tolist(),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "tail_64_mean_loss": float(losses[-min(64, losses.size) :].mean()),
            "cold_compile_and_train_seconds": elapsed,
            "parameter_sha256_before": _array_digest(before),
            "parameter_sha256_after": _array_digest(after),
            "parameter_movement": _parameter_movement(before, after),
        },
        after,
    )


def _compiler_summary(learner: Any) -> dict[str, Any]:
    report = getattr(learner, "report", None)
    if report is None:
        return {"available": False, "diagnostics": []}
    diagnostics = []
    for record in report.diagnostics:
        item: dict[str, Any] = {
            "kind": str(getattr(record.kind, "value", record.kind)),
            "level": str(getattr(record.level, "value", record.level)),
            "message": str(record.message),
        }
        weight_path = getattr(record, "weight_path", None)
        if weight_path is not None:
            item["weight_path"] = "/".join(map(str, weight_path))
        diagnostics.append(item)
    return {
        "available": True,
        "diagnostics": diagnostics,
        "compiled_parameter_paths": [
            "/".join(map(str, path))
            for path in getattr(learner, "param_states", {}).keys()
        ],
    }


def _train_pp_prop(
    model: LatentWorkspaceModel,
    data: BindingData,
    config: BindingControlConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        learner = compile_pp_prop(model)
    optimizer = braintools.optim.Adam(lr=config.learning_rate)
    optimizer.register_trainable_weights(learner.param_states)
    advance = jnp.ones((config.sequence_length, config.batch_size), dtype=jnp.bool_)
    mask = jnp.zeros((config.sequence_length,), dtype=jnp.float32).at[-1].set(1.0)
    rank = config.color_rank

    @brainstate.transform.jit
    def train_all(events: jax.Array, targets: jax.Array) -> jax.Array:
        def train_one(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
            sequence, target = inputs
            model.reset_state()
            learner.reset_state(batch_size=config.batch_size)

            def step_loss(event: jax.Array, gate: jax.Array) -> jax.Array:
                return _classification_loss(learner(event, gate), target, rank)

            gradients, loss = learner.etrace_grad(
                sequence,
                advance,
                step_fn=step_loss,
                mask=mask,
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            optimizer.update(brainstate.nn.clip_grad_norm(gradients, config.clip_norm))
            return loss

        return brainstate.transform.for_loop(train_one, (events, targets))

    before = _parameter_values(model)
    start = time.perf_counter()
    losses = train_all(
        jnp.asarray(data.training_events), jnp.asarray(data.training_targets)
    )
    losses = np.asarray(jax.block_until_ready(losses), dtype=np.float64)
    elapsed = time.perf_counter() - start
    after = _parameter_values(model)
    return (
        {
            "algorithm": "production_pp_prop",
            "etrace_config": dataclasses.asdict(model.etrace_config()),
            "loss_definition": "terminal one-cell color cross_entropy",
            "losses": losses.tolist(),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "tail_64_mean_loss": float(losses[-min(64, losses.size) :].mean()),
            "cold_compile_and_train_seconds": elapsed,
            "parameter_sha256_before": _array_digest(before),
            "parameter_sha256_after": _array_digest(after),
            "parameter_movement": _parameter_movement(before, after),
            "compiler": _compiler_summary(learner),
            "compile_warnings": [str(item.message) for item in caught],
        },
        after,
    )


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count <= 0:
        raise ValueError("count must be positive")
    z = 1.959963984540054
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / count + z * z / (4 * count * count))
    radius /= denominator
    return center - radius, center + radius


def _accuracy(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    correct = int(np.sum(predictions == targets))
    lower, upper = _wilson_interval(correct, int(targets.size))
    return {
        "correct": correct,
        "count": int(targets.size),
        "accuracy": correct / targets.size,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "prediction_histogram": np.bincount(
            predictions, minlength=COLOR_COUNT
        ).tolist(),
        "prediction_sha256": _digest_arrays(predictions),
    }


def _training_probe(
    data: BindingData, config: BindingControlConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = config.validation_episodes
    episode_major = data.training_events.transpose(0, 2, 1, 3).reshape(
        config.training_episode_count,
        config.sequence_length,
        config.row_config.input_width,
    )
    intact = np.array(episode_major[:count].transpose(1, 0, 2), copy=True)
    shuffled = np.array(intact, copy=True)
    rows = config.row_config
    output_indices = np.concatenate(
        (
            np.asarray([rows.side_valid_slice.start + 1], dtype=np.int32),
            np.arange(
                rows.normalized_slice.start + 3,
                rows.normalized_slice.start + 5,
                dtype=np.int32,
            ),
            np.arange(
                rows.output_height_slice.start,
                rows.output_height_slice.stop,
                dtype=np.int32,
            ),
            np.arange(
                rows.output_width_slice.start,
                rows.output_width_slice.stop,
                dtype=np.int32,
            ),
            np.arange(
                rows.output_mask_slice.start,
                rows.output_mask_slice.stop,
                dtype=np.int32,
            ),
            np.arange(
                rows.output_color_slice.start,
                rows.output_color_slice.stop,
                dtype=np.int32,
            ),
        )
    )
    shuffled[:SYMBOL_COUNT, :, output_indices] = np.roll(
        intact[:SYMBOL_COUNT, :, output_indices], shift=-1, axis=0
    )
    no_context = np.array(intact, copy=True)
    no_context[:SYMBOL_COUNT] = 0.0
    targets = np.array(data.training_targets.reshape(-1)[:count], copy=True)
    return intact, shuffled, no_context, targets


def _evaluate_model(
    trained_model: LatentWorkspaceModel,
    data: BindingData,
    config: BindingControlConfig,
) -> dict[str, Any]:
    model = LatentWorkspaceModel(
        _model_config(config, batch_size=config.validation_episodes)
    )
    _copy_parameters(trained_model, model)
    advance = jnp.ones((config.validation_episodes,), dtype=jnp.bool_)
    rank = config.color_rank

    @brainstate.transform.jit
    def predict(events: jax.Array) -> jax.Array:
        model.reset_state()
        outputs = brainstate.transform.for_loop(
            lambda event: model.update(event, advance), events
        )
        return jnp.argmax(_color_logits(outputs[-1], rank), axis=-1)

    start = time.perf_counter()
    intact = predict(jnp.asarray(data.validation_intact))
    shuffled = predict(jnp.asarray(data.validation_shuffled))
    no_context = predict(jnp.asarray(data.validation_no_context))
    (
        train_intact_events,
        train_shuffled_events,
        train_no_context_events,
        train_targets,
    ) = _training_probe(data, config)
    train_intact = predict(jnp.asarray(train_intact_events))
    train_shuffled = predict(jnp.asarray(train_shuffled_events))
    train_no_context = predict(jnp.asarray(train_no_context_events))
    (
        intact,
        shuffled,
        no_context,
        train_intact,
        train_shuffled,
        train_no_context,
    ) = jax.block_until_ready(
        (
            intact,
            shuffled,
            no_context,
            train_intact,
            train_shuffled,
            train_no_context,
        )
    )
    elapsed = time.perf_counter() - start
    targets = data.validation_targets
    intact_metric = _accuracy(np.asarray(intact), targets)
    shuffled_metric = _accuracy(np.asarray(shuffled), targets)
    no_context_metric = _accuracy(np.asarray(no_context), targets)
    train_intact_metric = _accuracy(np.asarray(train_intact), train_targets)
    train_shuffled_metric = _accuracy(np.asarray(train_shuffled), train_targets)
    train_no_context_metric = _accuracy(np.asarray(train_no_context), train_targets)
    return {
        "intact": intact_metric,
        "shuffled": shuffled_metric,
        "no_context": no_context_metric,
        "intact_minus_shuffled": (
            intact_metric["accuracy"] - shuffled_metric["accuracy"]
        ),
        "intact_minus_no_context": (
            intact_metric["accuracy"] - no_context_metric["accuracy"]
        ),
        "pairing_chance": 1.0 / SYMBOL_COUNT,
        "unconditional_color_chance": 1.0 / COLOR_COUNT,
        "training_probe": {
            "intact": train_intact_metric,
            "shuffled": train_shuffled_metric,
            "no_context": train_no_context_metric,
            "intact_minus_shuffled": (
                train_intact_metric["accuracy"] - train_shuffled_metric["accuracy"]
            ),
            "intact_minus_no_context": (
                train_intact_metric["accuracy"] - train_no_context_metric["accuracy"]
            ),
            "episode_count": int(train_targets.size),
            "source": "first held-fixed episodes from the shared training schedule",
        },
        "cold_compile_and_six_arm_eval_seconds": elapsed,
    }


class _TerminalResidualModel(brainstate.nn.Module):
    def __init__(self, config: BindingControlConfig):
        super().__init__()
        self.config = config
        self.reservoir = LatentWorkspaceModel(_model_config(config, batch_size=1))

    def update(self, packed: jax.Array) -> jax.Array:
        width = self.config.row_config.input_width
        event = packed[:, :width]
        advance = packed[:, width] > 0.5
        target = packed[:, width + 1 : width + 1 + COLOR_COUNT]
        loss_gate = packed[:, width + 1 + COLOR_COUNT :]
        logits = _color_logits(
            self.reservoir.update(event, advance), self.config.color_rank
        )
        return loss_gate * (logits - target)


def _oracle_inputs(data: BindingData, config: BindingControlConfig) -> jax.Array:
    width = config.row_config.input_width
    packed = np.zeros(
        (config.sequence_length, 1, width + 1 + COLOR_COUNT + 1),
        dtype=np.float32,
    )
    packed[:, 0, :width] = data.validation_intact[:, 0]
    packed[:, 0, width] = 1.0
    target = int(data.validation_targets[0])
    packed[-1, 0, width + 1 + target] = 1.0
    packed[-1, 0, -1] = 1.0
    return jnp.asarray(packed)


def _gradient_vector(leaves: dict[str, jax.Array], labels: Sequence[str]) -> np.ndarray:
    if not labels:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(
        [np.asarray(leaves[label], dtype=np.float64).reshape(-1) for label in labels]
    )


def _vector_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual_norm = float(np.linalg.norm(actual))
    expected_norm = float(np.linalg.norm(expected))
    difference = float(np.linalg.norm(actual - expected))
    cosine = None
    if actual_norm > 0.0 and expected_norm > 0.0:
        cosine = float(np.dot(actual, expected) / (actual_norm * expected_norm))
    deviation = None
    if expected_norm > 0.0:
        deviation = difference / expected_norm
    elif difference == 0.0:
        deviation = 0.0
    return {
        "bptt_norm": expected_norm,
        "pp_prop_norm": actual_norm,
        "cosine": cosine,
        "relative_deviation": deviation,
    }


def _gradient_oracle(data: BindingData, config: BindingControlConfig) -> dict[str, Any]:
    inputs = _oracle_inputs(data, config)

    def factory() -> _TerminalResidualModel:
        return _TerminalResidualModel(config)

    trace_config = braintrace.ETraceConfig(
        trace_factorization="io_factorized",
        recurrence_scope="diagonal",
        decay=config.trace_decay,
    )

    def algorithm_factory(model: brainstate.nn.Module) -> braintrace.ETraceAlgorithm:
        return braintrace.pp_prop(
            model,
            decay_or_rank=config.trace_decay,
            vjp_method="multi-step",
            config=trace_config,
        )

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        bptt = bptt_param_gradients(factory, inputs)
        pp_prop = chunked_online_param_gradients(
            factory,
            inputs,
            algo_factory=algorithm_factory,
            chunk_size=config.gradient_chunk_size,
        )
    elapsed = time.perf_counter() - start
    bptt_flat = flat_gradient_leaves(bptt)
    pp_flat = flat_gradient_leaves(pp_prop)
    if set(bptt_flat) != set(pp_flat):
        raise RuntimeError("BPTT and pp-prop gradient structures differ")
    groups = sorted({_path_group(label) for label in bptt_flat})
    by_group = {}
    for group in groups:
        labels = sorted(label for label in bptt_flat if _path_group(label) == group)
        by_group[group] = _vector_metrics(
            _gradient_vector(pp_flat, labels),
            _gradient_vector(bptt_flat, labels),
        )
        by_group[group]["leaf_labels"] = labels
    return {
        "loss_definition": "terminal sum_squared(color_logits - one_hot_target)",
        "sequence_length": config.sequence_length,
        "chunk_size": config.gradient_chunk_size,
        "finite_window": config.gradient_chunk_size < config.sequence_length,
        "bptt_norm": gradient_norm(bptt),
        "pp_prop_norm": gradient_norm(pp_prop),
        "relative_deviation": relative_deviation(pp_prop, bptt),
        "by_parameter_group": by_group,
        "cold_compile_and_measure_seconds": elapsed,
        "warnings": [str(item.message) for item in caught],
    }


def _data_report(data: BindingData, config: BindingControlConfig) -> dict[str, Any]:
    train_ids = data.training_mapping_ids.reshape(-1)
    validation_ids = data.validation_mapping_ids
    target_histogram = np.bincount(data.validation_targets, minlength=COLOR_COUNT)
    return {
        "mapping_catalog_size": _MAPPING_COUNT,
        "mapping_split": "seeded affine permutation; disjoint contiguous ranges",
        "training_episode_count": int(train_ids.size),
        "validation_episode_count": int(validation_ids.size),
        "training_mappings_unique": int(np.unique(train_ids).size) == train_ids.size,
        "validation_mappings_unique": (
            int(np.unique(validation_ids).size) == validation_ids.size
        ),
        "train_validation_mapping_overlap": int(
            np.intersect1d(train_ids, validation_ids).size
        ),
        "training_mapping_ids_sha256": _digest_arrays(train_ids),
        "validation_mapping_ids_sha256": _digest_arrays(validation_ids),
        "training_schedule_sha256": _digest_arrays(
            data.training_events,
            data.training_targets,
            data.training_mapping_ids,
        ),
        "validation_schedule_sha256": _digest_arrays(
            data.validation_intact,
            data.validation_targets,
            data.validation_mapping_ids,
        ),
        "validation_target_histogram": target_histogram.tolist(),
        "validation_best_fixed_color_accuracy": float(
            target_histogram.max() / target_histogram.sum()
        ),
        "intact_shuffled_timing_matched": (
            data.validation_intact.shape == data.validation_shuffled.shape
        ),
        "intact_no_context_timing_matched": (
            data.validation_intact.shape == data.validation_no_context.shape
        ),
        "sequence_length": config.sequence_length,
        "event_width": config.row_config.input_width,
        "external_gap_nonzero_count": int(
            np.count_nonzero(data.validation_intact[-config.gap_steps :])
        ),
    }


def _source_report() -> dict[str, Any]:
    commit = os.environ.get("BRAINTRACE_SOURCE_COMMIT", "").strip()
    dirty_env = os.environ.get("BRAINTRACE_SOURCE_DIRTY", "").strip()
    if not commit:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unavailable"
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        dirty = dirty_env.lower() not in {"", "0", "false", "no"}
    return {"commit": commit, "dirty": dirty}


def _interpretation(
    bptt: dict[str, Any], pp_prop: dict[str, Any], config: BindingControlConfig
) -> str:
    def binds(result: dict[str, Any]) -> bool:
        return bool(
            result["intact"]["accuracy"] >= 0.8
            and result["intact_minus_shuffled"] >= 0.25
        )

    bptt_binds = binds(bptt)
    pp_binds = binds(pp_prop)
    if config.configuration_scale != "production_topology":
        return "reduced_smoke_only_no_architecture_conclusion"
    if not bptt_binds and pp_binds:
        return "invalid_control_pp_prop_binds_while_bptt_fails"
    if not bptt_binds:
        return "legacy_architecture_necessary_bptt_also_fails_binding"
    if not pp_binds:
        return "pp_prop_truncation_blocker_bptt_binds"
    return "both_bind_increase_only_the_preregistered_gap"


def run_binding_control(config: BindingControlConfig) -> dict[str, Any]:
    """Run exact BPTT, production pp-prop, interventions, and gradient oracle.

    Parameters
    ----------
    config : BindingControlConfig
        Complete deterministic experiment configuration.

    Returns
    -------
    dict
        JSON-compatible scientific artifact including losses, accuracies,
        parameter movement, finite-window gradients, timing, and digests.
    """

    if not isinstance(config, BindingControlConfig):
        raise TypeError("config must be a BindingControlConfig")
    total_start = time.perf_counter()
    data_start = time.perf_counter()
    data = build_binding_data(config)
    data_seconds = time.perf_counter() - data_start

    bptt_model = LatentWorkspaceModel(
        _model_config(config, batch_size=config.batch_size)
    )
    pp_model = LatentWorkspaceModel(_model_config(config, batch_size=config.batch_size))
    bptt_initial = _parameter_values(bptt_model)
    pp_initial = _parameter_values(pp_model)
    bptt_initial_digest = _array_digest(bptt_initial)
    pp_initial_digest = _array_digest(pp_initial)
    if bptt_initial_digest != pp_initial_digest:
        raise RuntimeError("BPTT and pp-prop initializations are not byte-identical")

    bptt_training, _ = _train_bptt(bptt_model, data, config)
    pp_training, _ = _train_pp_prop(pp_model, data, config)
    bptt_evaluation = _evaluate_model(bptt_model, data, config)
    pp_evaluation = _evaluate_model(pp_model, data, config)
    gradient = _gradient_oracle(data, config)
    result = {
        "schema_version": 1,
        "control": "example21_legacy_reservoir_binding_bptt_vs_pp_prop",
        "source": _source_report(),
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "config": {
            **dataclasses.asdict(config),
            "configuration_scale": config.configuration_scale,
        },
        "data": {**_data_report(data, config), "generation_seconds": data_seconds},
        "initialization": {
            "byte_identical": True,
            "bptt_parameter_sha256": bptt_initial_digest,
            "pp_prop_parameter_sha256": pp_initial_digest,
        },
        "training": {"bptt": bptt_training, "pp_prop": pp_training},
        "evaluation": {"bptt": bptt_evaluation, "pp_prop": pp_evaluation},
        "gradient_oracle": gradient,
    }
    result["interpretation"] = _interpretation(bptt_evaluation, pp_evaluation, config)
    result["total_wall_seconds"] = time.perf_counter() - total_start
    return result


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_artifact(result: dict[str, Any], path: str | Path) -> Path:
    """Write one standards-compliant JSON artifact atomically.

    Parameters
    ----------
    result : dict
        Result returned by :func:`run_binding_control`.
    path : str or pathlib.Path
        Destination JSON path.

    Returns
    -------
    pathlib.Path
        Resolved artifact path.
    """

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = json.dumps(_json_ready(result), indent=2, sort_keys=True, allow_nan=False)
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/example21-binding-control/control.json"),
    )
    parser.add_argument("--training-updates", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-episodes", type=int, default=512)
    parser.add_argument("--gap-steps", type=int, default=1)
    parser.add_argument("--neuron-count", type=int, default=2048)
    parser.add_argument("--recurrent-edges", type=int, default=16384)
    parser.add_argument("--readout-width", type=int, default=128)
    parser.add_argument("--color-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--gradient-chunk-size", type=int, default=1)
    parser.add_argument("--sparse-backend", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the control and write its JSON artifact.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero after a complete artifact is written.
    """

    args = _parser().parse_args(argv)
    if args.smoke:
        config = BindingControlConfig.smoke_config()
    else:
        config = BindingControlConfig(
            training_updates=args.training_updates,
            batch_size=args.batch_size,
            validation_episodes=args.validation_episodes,
            gap_steps=args.gap_steps,
            neuron_count=args.neuron_count,
            recurrent_edges=args.recurrent_edges,
            readout_width=args.readout_width,
            color_rank=args.color_rank,
            learning_rate=args.learning_rate,
            gradient_chunk_size=args.gradient_chunk_size,
            sparse_backend=args.sparse_backend,
        )
    result = run_binding_control(config)
    output = write_artifact(result, args.output)
    print(
        json.dumps(
            {
                "artifact": str(output),
                "interpretation": result["interpretation"],
                "bptt_intact": result["evaluation"]["bptt"]["intact"]["accuracy"],
                "pp_prop_intact": result["evaluation"]["pp_prop"]["intact"]["accuracy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
