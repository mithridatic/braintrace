"""Preregistered demonstrated-depth Gate B for Example 21."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np

from examples.pp_prop import latent_workspace_binding_control as legacy
from examples.pp_prop import latent_workspace_binding_gate as gate
from examples.pp_prop.latent_workspace_model import (
    LatentWorkspaceModel,
    ModelConfig,
    compile_pp_prop,
)
from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    RowEventConfig,
    associative_memory_feature_indices,
    encode_query_episode,
)


QUALIFYING_EFFORTS = (1, 2, 4, 8)
TEN_CYCLE_CATALOG_SIZE = math.factorial(9)
STAGING_CHUNK_UPDATES = 128
STAGING_CHUNK_COUNT = 32
GATE_B_SCHEMA_VERSION = 1
GATE_B_CONTROL = "example21_demonstrated_depth_gate_b"
GATE_B_INITIALIZATION_CONTROL = "example21_gate_b_initialization_admission"

_GATE_A_RESULT_SHA256 = (
    "3a585e739715b31757082b50fe57b98ca50107891f7c79edaa7e5e54c90ad632"
)
_GATE_A_MANIFEST_SHA256 = (
    "69d690daa5023f5b3ce22b0e65ea09a1a6706687d792e998651422f6d6ea15cf"
)
_GATE_A_SOURCE_COMMIT = "4737e9172b1c6ca99347af5b2c83fc795a294a16"
_MODEL_SOURCE_SHA256 = (
    "467022c79123b976dd5cebc8d5ae5da37d1373bc46477133003b0b263abd8216"
)
_TASK_SOURCE_SHA256 = (
    "cfaec054bd42f6dccf9fb24c5fbec0cd703fdef17ba8d3b6dd68bf78366de18b"
)
_PRODUCTION_ENCODED_GLOBAL_SHA256 = {
    "events": "a1937b7f8d5d4da5f30216847cc63d022d9ec46d5cf152b25f5a30a59a1eb84f",
    "targets": "4082d2fd1440e9d14b0c81c754158f05b8056137a9116aee667f8d112312184c",
    "loss_weights": "044616bf9dd86cbdc1d472184ede8027bf9ff65d65834b15ec619bf3095d2e31",
    "advance_masks": "2fc1b2acd9f73e567684d2a85f44c4009c5941ce262a527589066117ec27a4cc",
    "mapping_ids": "78c2d8aaa9e874dbcc1c25363875ff8aec0356a711d2426e09f2e79c76c72cb7",
    "efforts": "c7ca75132501bda8e6b5695a48a1ae5cde22da587f4658f7721bd4e3adcd58e6",
    "query_colors": "38b4cecef323dce16b0478fdd3874c9383804c913c39aaf017ce34554dcd37cb",
    "presentation_orders": (
        "0650be382b381d7ab14b642c6fcdb16ae410e70a4c5821b10643bce41e3f7ca5"
    ),
}
_PRODUCTION_CHUNK_SHA256_MANIFEST = {
    "events": "c73cb65b9b774f07e617000f531511a561c2672cd5c5dc0bfb5c95328bded771",
    "targets": "e969705e4d9ab68581cde897f5f8fd780a716964d9a96a842b3e9850db294f3e",
    "loss_weights": "968eed06f7bc37d52d68300f2ef803666320c2ffaa094843876f4c3b39101b9c",
    "advance_masks": "226ee64d8c3ffbe5eec66bb8cc01c7030b2d3c700eb2c75da91b5391e659cfd8",
    "mapping_ids": "d198e47d766ad69c2691243fd759835d4a3d5d79377f8d47cc9bbf8488306a84",
    "efforts": "267aef30ba47f5e67c17a36d299ffe2a41acfe57c92a078f4d04cc8f7cea903b",
    "query_colors": "f67148579356209f6846d681eeaafa3e00ccf9206023f59a7837d1c351a58da9",
    "presentation_orders": (
        "60b52825e54e2b2b5efbc9f2cbaea65ef3737577b7b4c2e51b924ea9e907a8ef"
    ),
}
_PRODUCTION_VALIDATION_SHA256 = {
    "mapping_ids": "b036444e228c60116b8bfd9c10399261bcf6645d7b69d27d4b391460fae83cd8",
    "query_colors": "c7e70f56cca66d920d5d690a902b9943f2fcfdff7003fa4bbb3580070738d67e",
    "presentation_orders": (
        "0bab5cfe3c2c87109909d36c59f88ef04983322aacbce469fea6221aa4ac37b0"
    ),
    "shuffled_shifts": (
        "15af1f04589cc523d89b66d2f07027158d69068901d786eecfd259a156f2f2d0"
    ),
    "intact": "5683aa84aa2ef8a1ff623e5e0b60afb3451e617728f0363d3ad84f2ea52dacde",
    "shuffled": "abd5eb4ab2e2a685faeb8f6bf785ad2deb97721b00e8d194b4f65d4995516be3",
    "no_context": "45fd14d3faefad83b0ce6d908456320afa67944b361159cfe503fdfab591162d",
    "targets_by_depth": (
        "a438d64347dc4ec5cfc639342d8b142c785e497ddf06728eb03f8ccfb42d3cd6"
    ),
    "advance_masks": (
        "b88b3593d9df51260fbafa4a937159c3da3f56fc33335a30993c0ff8a7462ac8"
    ),
}


@dataclass(frozen=True, slots=True)
class DepthGateConfig:
    """Fixed configuration for demonstrated-depth Gate B.

    Parameters
    ----------
    training_updates
        Number of pp-prop optimizer updates.
    batch_size
        Episodes in every optimizer update.
    validation_episodes
        Held-out episodes used for every declared effort.
    neuron_count, recurrent_edges, readout_width, color_rank
        Production latent-workspace topology.
    context_memory_width, memory_decay
        Associative-memory width and persistence.
    trace_decay, learning_rate, gradient_clip_norm
        Fixed pp-prop and optimizer settings.
    model_seed, catalog_seed, train_episode_seed, validation_episode_seed
        Frozen model, catalog, and presentation seeds.
    staging_chunk_updates
        Updates staged into each fixed-shape compiled call.
    row_config
        One-cell row-event layout with capacity for ten demonstrations.
    """

    training_updates: int = 4_096
    batch_size: int = 64
    validation_episodes: int = 512
    gap_steps: int = 8
    neuron_count: int = 2_048
    recurrent_edges: int = 16_384
    readout_width: int = 128
    color_rank: int = 16
    context_memory_width: int = 32
    memory_decay: float = 1.0
    trace_decay: float = 0.9
    learning_rate: float = 0.003
    clip_norm: float = 1.0
    input_gain: float = 4.0
    recurrent_gain: float = 0.8
    model_seed: int = 2_108
    catalog_seed: int = 20_260_818
    train_episode_seed: int = 32_021
    validation_episode_seed: int = 92_021
    staging_chunk_updates: int = STAGING_CHUNK_UPDATES
    row_config: RowEventConfig = field(
        default_factory=lambda: RowEventConfig(
            max_demonstrations=10,
            max_grid_size=1,
        )
    )

    def __post_init__(self) -> None:
        integer_fields = (
            "training_updates",
            "batch_size",
            "validation_episodes",
            "gap_steps",
            "neuron_count",
            "recurrent_edges",
            "readout_width",
            "color_rank",
            "context_memory_width",
            "model_seed",
            "catalog_seed",
            "train_episode_seed",
            "validation_episode_seed",
            "staging_chunk_updates",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"DepthGateConfig.{name} must be an integer")
            object.__setattr__(self, name, int(value))
        real_fields = (
            "memory_decay",
            "trace_decay",
            "learning_rate",
            "clip_norm",
            "input_gain",
            "recurrent_gain",
        )
        for name in real_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"DepthGateConfig.{name} must be a real number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"DepthGateConfig.{name} must be finite")
            object.__setattr__(self, name, value)
        if self.training_updates <= 0 or self.batch_size <= 0:
            raise ValueError("training_updates and batch_size must be positive")
        if self.validation_episodes <= 0:
            raise ValueError("validation_episodes must be positive")
        if self.gap_steps != 8:
            raise ValueError("Gate B requires eight latent steps")
        if self.staging_chunk_updates <= 0:
            raise ValueError("staging_chunk_updates must be positive")
        if (
            self.training_updates == 4_096
            and self.training_updates % self.staging_chunk_updates
        ):
            raise ValueError("the production schedule must divide into fixed chunks")
        if self.training_episode_count + self.validation_episodes > TEN_CYCLE_CATALOG_SIZE:
            raise ValueError("Gate B episode count exceeds the finite 10-cycle catalog")
        if self.row_config.max_demonstrations != 10:
            raise ValueError("Gate B requires ten demonstration bindings")
        if self.row_config.max_grid_size != 1:
            raise ValueError("Gate B requires one-cell grids")

    @property
    def sequence_length(self) -> int:
        """Return the fixed ten-demo, query, and H1--H8 length."""

        return 10 + 1 + self.gap_steps

    @property
    def training_episode_count(self) -> int:
        """Return the number of unique training cycles consumed."""

        return self.training_updates * self.batch_size

    @property
    def staging_chunk_count(self) -> int:
        """Return the number of fixed staging chunks."""

        return math.ceil(self.training_updates / self.staging_chunk_updates)

    @property
    def qualification_regime(self) -> str:
        """Return whether this configuration is the preregistered full regime."""

        actual = (
            self.training_updates,
            self.batch_size,
            self.validation_episodes,
            self.gap_steps,
            self.neuron_count,
            self.recurrent_edges,
            self.readout_width,
            self.color_rank,
            self.context_memory_width,
            self.memory_decay,
            self.trace_decay,
            self.learning_rate,
            self.clip_norm,
            self.input_gain,
            self.recurrent_gain,
            self.model_seed,
            self.catalog_seed,
            self.train_episode_seed,
            self.validation_episode_seed,
            self.staging_chunk_updates,
            self.row_config,
        )
        expected = (
            4_096,
            64,
            512,
            8,
            2_048,
            16_384,
            128,
            16,
            32,
            1.0,
            0.9,
            0.003,
            1.0,
            4.0,
            0.8,
            2_108,
            20_260_818,
            32_021,
            92_021,
            STAGING_CHUNK_UPDATES,
            RowEventConfig(max_demonstrations=10, max_grid_size=1),
        )
        return "preregistered_full" if actual == expected else "nonqualifying_abbreviated"

    @property
    def gate_a_result_sha256(self) -> str:
        """Return the pinned authenticated Gate A result digest."""

        return _GATE_A_RESULT_SHA256

    @property
    def gate_a_manifest_sha256(self) -> str:
        """Return the pinned authenticated Gate A manifest digest."""

        return _GATE_A_MANIFEST_SHA256

    @property
    def gate_a_source_commit(self) -> str:
        """Return the pinned authenticated Gate A source revision."""

        return _GATE_A_SOURCE_COMMIT

    @property
    def model_source_sha256(self) -> str:
        """Return the pinned latent-workspace model source digest."""

        return _MODEL_SOURCE_SHA256

    @property
    def task_source_sha256(self) -> str:
        """Return the pinned row-event task source digest."""

        return _TASK_SOURCE_SHA256


@dataclass(frozen=True, slots=True)
class DepthSchedule:
    """Deterministic mapping, effort, query, and presentation schedule.

    Parameters
    ----------
    training_mapping_ids, validation_mapping_ids
        Disjoint ten-cycle catalog IDs for the two splits.
    training_efforts
        Effort assigned to each optimizer update.
    training_query_colors, validation_query_colors
        Query colors in deterministic global schedule order.
    training_presentation_orders, validation_presentation_orders
        Per-episode permutations of the ten demonstration colors.
    """

    training_mapping_ids: np.ndarray
    validation_mapping_ids: np.ndarray
    training_efforts: np.ndarray
    training_query_colors: np.ndarray
    validation_query_colors: np.ndarray
    training_presentation_orders: np.ndarray
    validation_presentation_orders: np.ndarray


@dataclass(frozen=True, slots=True)
class DepthScheduleChunk:
    """One fixed-shape host staging chunk from a depth schedule.

    Parameters
    ----------
    training_mapping_ids
        Catalog IDs with shape ``(updates, batch)``.
    training_efforts
        One effort per update.
    training_query_colors
        Query colors with shape ``(updates, batch)``.
    training_presentation_orders
        Demonstration permutations with shape ``(updates, batch, 10)``.
    """

    training_mapping_ids: np.ndarray
    training_efforts: np.ndarray
    training_query_colors: np.ndarray
    training_presentation_orders: np.ndarray


@dataclass(frozen=True, slots=True)
class DepthTrainingChunk:
    """One encoded fixed-shape training chunk.

    Parameters
    ----------
    events
        Row events with shape ``(updates, 19, batch, input_width)``.
    targets
        Per-checkpoint color targets with shape ``(updates, 19, batch)``.
    loss_weights
        Per-update temporal masks with shape ``(updates, 19)``.
    advance_masks
        Per-example physical-advance gates with shape ``(updates, 19, batch)``.
    mapping_ids, efforts, query_colors, presentation_orders
        Schedule identity copied into the encoded chunk.
    """

    events: np.ndarray
    targets: np.ndarray
    loss_weights: np.ndarray
    advance_masks: np.ndarray
    mapping_ids: np.ndarray
    efforts: np.ndarray
    query_colors: np.ndarray
    presentation_orders: np.ndarray


@dataclass(frozen=True, slots=True)
class DepthValidationData:
    """Held-out intact and matched-control Gate B streams.

    Parameters
    ----------
    intact, shuffled, no_context
        Time-major evaluation streams with shape ``(19, episodes, input_width)``.
    targets_by_depth
        Intact targets for H0 through H8.
    advance_masks
        Time-major physical-advance gates shared by all arms.
    mapping_ids, query_colors, presentation_orders
        Held-out schedule identity.
    shuffled_shifts
        First valid output rotation selected for each control episode.
    """

    intact: np.ndarray
    shuffled: np.ndarray
    no_context: np.ndarray
    targets_by_depth: np.ndarray
    advance_masks: np.ndarray
    mapping_ids: np.ndarray
    query_colors: np.ndarray
    presentation_orders: np.ndarray
    shuffled_shifts: np.ndarray


@dataclass(frozen=True, slots=True)
class CheckpointContract:
    """Targets, weights, and advances for one effort-specific episode.

    Parameters
    ----------
    targets
        Fixed-length color target sequence.
    loss_weights
        Normalized H0-through-HR supervision weights.
    advance_mask
        True semantic prefix followed by a frozen suffix.
    active_length
        Exclusive end of the semantic prefix.
    """

    targets: np.ndarray
    loss_weights: np.ndarray
    advance_mask: np.ndarray
    active_length: int


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    array.setflags(write=False)
    return array


def _affine_mapping_ids(count: int, *, seed: int) -> np.ndarray:
    rng = brainstate.random.RandomState(seed)
    offset = int(np.asarray(rng.randint(0, TEN_CYCLE_CATALOG_SIZE)))
    multiplier = int(np.asarray(rng.randint(1, TEN_CYCLE_CATALOG_SIZE)))
    while math.gcd(multiplier, TEN_CYCLE_CATALOG_SIZE) != 1:
        multiplier = multiplier % (TEN_CYCLE_CATALOG_SIZE - 1) + 1
    positions = np.arange(count, dtype=np.int64)
    return (offset + multiplier * positions) % TEN_CYCLE_CATALOG_SIZE


def _presentation_orders(count: int, seed: int) -> np.ndarray:
    rng = brainstate.random.RandomState(seed)
    scores = np.asarray(rng.rand(count, 10), dtype=np.float32)
    return np.argsort(scores, axis=1, kind="stable").astype(np.int32)


def _build_schedule(config: DepthGateConfig) -> DepthSchedule:
    total = config.training_episode_count + config.validation_episodes
    mapping_ids = _affine_mapping_ids(total, seed=config.catalog_seed)
    training_count = config.training_episode_count
    training_mapping_ids = mapping_ids[:training_count].reshape(
        config.training_updates,
        config.batch_size,
    )
    validation_mapping_ids = mapping_ids[training_count:]
    training_queries = np.arange(training_count, dtype=np.int64) % 10
    validation_queries = (
        training_count + np.arange(config.validation_episodes, dtype=np.int64)
    ) % 10
    return DepthSchedule(
        training_mapping_ids=_readonly(training_mapping_ids.astype(np.int64)),
        validation_mapping_ids=_readonly(validation_mapping_ids.astype(np.int64)),
        training_efforts=_readonly(
            np.resize(np.asarray(QUALIFYING_EFFORTS, dtype=np.int32), config.training_updates)
        ),
        training_query_colors=_readonly(
            training_queries.reshape(config.training_updates, config.batch_size).astype(
                np.int32
            )
        ),
        validation_query_colors=_readonly(validation_queries.astype(np.int32)),
        training_presentation_orders=_readonly(
            _presentation_orders(training_count, config.train_episode_seed).reshape(
                config.training_updates,
                config.batch_size,
                10,
            )
        ),
        validation_presentation_orders=_readonly(
            _presentation_orders(config.validation_episodes, config.validation_episode_seed)
        ),
    )


def _iter_schedule_chunks(
    schedule: DepthSchedule,
    config: DepthGateConfig,
) -> Iterator[DepthScheduleChunk]:
    for start in range(0, config.training_updates, config.staging_chunk_updates):
        stop = start + config.staging_chunk_updates
        yield DepthScheduleChunk(
            training_mapping_ids=schedule.training_mapping_ids[start:stop],
            training_efforts=schedule.training_efforts[start:stop],
            training_query_colors=schedule.training_query_colors[start:stop],
            training_presentation_orders=schedule.training_presentation_orders[start:stop],
        )


def unrank_ten_cycle(mapping_id: int) -> np.ndarray:
    """Decode a catalog ID into its anchored single 10-cycle mapping.

    Parameters
    ----------
    mapping_id
        Lexicographic Lehmer rank in ``[0, 9!)``.

    Returns
    -------
    numpy.ndarray
        Integer successor mapping for colors zero through nine.
    """

    if isinstance(mapping_id, bool) or not isinstance(mapping_id, (int, np.integer)):
        raise ValueError("mapping ID must be an integer in the 10-cycle catalog")
    rank = int(mapping_id)
    if not 0 <= rank < TEN_CYCLE_CATALOG_SIZE:
        raise ValueError("mapping ID is outside the 10-cycle catalog")
    remaining = list(range(1, 10))
    order = [0]
    for width in range(9, 0, -1):
        divisor = math.factorial(width - 1)
        index, rank = divmod(rank, divisor)
        order.append(remaining.pop(index))
    mapping = np.empty((10,), dtype=np.int32)
    for source, target in zip(order, order[1:] + order[:1], strict=True):
        mapping[source] = target
    return mapping


def rank_ten_cycle(mapping: np.ndarray) -> int:
    """Return the lexicographic catalog rank of one anchored 10-cycle.

    Parameters
    ----------
    mapping
        Successor mapping containing one cycle over all ten colors.

    Returns
    -------
    int
        Catalog ID in ``[0, 9!)``.
    """

    array = np.asarray(mapping)
    if array.shape != (10,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("mapping must be an integer successor array of shape (10,)")
    if sorted(array.astype(int).tolist()) != list(range(10)):
        raise ValueError("mapping must be a permutation of all ten colors")
    order = [0]
    value = 0
    for _ in range(9):
        value = int(array[value])
        if value == 0 or value in order:
            raise ValueError("mapping must contain one single 10-cycle")
        order.append(value)
    if int(array[order[-1]]) != 0:
        raise ValueError("mapping must return to zero after ten colors")
    remaining = list(range(1, 10))
    rank = 0
    for position, value in enumerate(order[1:]):
        index = remaining.index(value)
        rank += index * math.factorial(8 - position)
        remaining.pop(index)
    return rank


def _iterate_mapping(mapping: np.ndarray, color: int, applications: int) -> int:
    value = int(color)
    for _ in range(applications):
        value = int(mapping[value])
    return value


def _select_shuffled_rotation(
    mapping: np.ndarray,
    query_color: int,
) -> tuple[int, np.ndarray]:
    array = np.asarray(mapping, dtype=np.int32)
    if array.shape != (10,):
        raise ValueError("mapping must have shape (10,)")
    if isinstance(query_color, bool) or not isinstance(query_color, (int, np.integer)):
        raise ValueError("query color must be an integer")
    query = int(query_color)
    if not 0 <= query < 10:
        raise ValueError("query color must be in 0..9")
    for shift in range(1, 10):
        shuffled = (array + shift) % 10
        if not np.all(shuffled != array):
            continue
        if all(
            _iterate_mapping(shuffled, query, effort + 1)
            != _iterate_mapping(array, query, effort + 1)
            for effort in QUALIFYING_EFFORTS
        ):
            return shift, shuffled.astype(np.int32)
    raise ValueError("no shuffled rotation breaks every declared depth")


def _checkpoint_contract(
    mapping: np.ndarray,
    query_color: int,
    effort: int,
) -> CheckpointContract:
    if effort not in QUALIFYING_EFFORTS:
        raise ValueError("effort must be one of 1, 2, 4, or 8")
    array = np.asarray(mapping, dtype=np.int32)
    if array.shape != (10,):
        raise ValueError("mapping must have shape (10,)")
    if isinstance(query_color, bool) or not isinstance(query_color, (int, np.integer)):
        raise ValueError("query color must be an integer")
    query = int(query_color)
    if not 0 <= query < 10:
        raise ValueError("query color must be in 0..9")
    active_length = 11 + effort
    targets = np.zeros((19,), dtype=np.int32)
    weights = np.zeros((19,), dtype=np.float64)
    advances = np.zeros((19,), dtype=np.bool_)
    advances[:active_length] = True
    for depth in range(effort + 1):
        targets[10 + depth] = _iterate_mapping(array, query, depth + 1)
    weights[10:active_length] = 1.0 / (effort + 1)
    return CheckpointContract(
        targets=_readonly(targets),
        loss_weights=_readonly(weights),
        advance_mask=_readonly(advances),
        active_length=active_length,
    )


def _one_cell(color: int) -> ArcGrid:
    return ArcGrid(((int(color),),))


def _encode_cycle_episode(
    mapping: np.ndarray,
    query_color: int,
    presentation_order: np.ndarray,
    row_config: RowEventConfig,
) -> np.ndarray:
    demonstrations = tuple(
        ArcPair(_one_cell(color), _one_cell(int(mapping[color])))
        for color in np.asarray(presentation_order, dtype=np.int32)
    )
    task = ArcTask(
        train=demonstrations,
        test=(
            ArcPair(
                _one_cell(query_color),
                _one_cell(int(mapping[query_color])),
            ),
        ),
    )
    return np.asarray(encode_query_episode(task, 0, row_config).events)


def _encode_training_chunk(
    schedule_chunk: DepthScheduleChunk,
    config: DepthGateConfig,
) -> DepthTrainingChunk:
    mapping_ids = np.asarray(schedule_chunk.training_mapping_ids)
    efforts = np.asarray(schedule_chunk.training_efforts)
    query_colors = np.asarray(schedule_chunk.training_query_colors)
    presentation_orders = np.asarray(schedule_chunk.training_presentation_orders)
    if mapping_ids.ndim != 2:
        raise ValueError("training mapping IDs must have shape (updates, batch)")
    update_count, batch_size = mapping_ids.shape
    if batch_size != config.batch_size:
        raise ValueError("training chunk batch size differs from the configuration")
    if efforts.shape != (update_count,):
        raise ValueError("training efforts must have shape (updates,)")
    if query_colors.shape != mapping_ids.shape:
        raise ValueError("training query colors must match mapping ID shape")
    if presentation_orders.shape != (update_count, batch_size, 10):
        raise ValueError("training presentation orders have the wrong shape")

    events = np.zeros(
        (update_count, config.sequence_length, batch_size, config.row_config.input_width),
        dtype=np.float32,
    )
    targets = np.zeros(
        (update_count, config.sequence_length, batch_size), dtype=np.int32
    )
    weights = np.zeros((update_count, config.sequence_length), dtype=np.float64)
    advances = np.zeros(
        (update_count, config.sequence_length, batch_size), dtype=np.bool_
    )
    for update_index in range(update_count):
        effort = int(efforts[update_index])
        reference_weights: np.ndarray | None = None
        for batch_index in range(batch_size):
            mapping = unrank_ten_cycle(int(mapping_ids[update_index, batch_index]))
            query = int(query_colors[update_index, batch_index])
            encoded = _encode_cycle_episode(
                mapping,
                query,
                presentation_orders[update_index, batch_index],
                config.row_config,
            )
            if encoded.shape != (11, config.row_config.input_width):
                raise ValueError("encoded Gate B episode must contain exactly 11 rows")
            events[update_index, :11, batch_index] = encoded
            contract = _checkpoint_contract(mapping, query, effort)
            targets[update_index, :, batch_index] = contract.targets
            advances[update_index, :, batch_index] = contract.advance_mask
            if reference_weights is None:
                reference_weights = np.asarray(contract.loss_weights)
            elif not np.array_equal(reference_weights, contract.loss_weights):
                raise ValueError("all examples in an update must share one effort mask")
        assert reference_weights is not None
        weights[update_index] = reference_weights
    return DepthTrainingChunk(
        events=_readonly(events),
        targets=_readonly(targets),
        loss_weights=_readonly(weights),
        advance_masks=_readonly(advances),
        mapping_ids=_readonly(mapping_ids),
        efforts=_readonly(efforts),
        query_colors=_readonly(query_colors),
        presentation_orders=_readonly(presentation_orders),
    )


def _encode_validation_data(
    schedule: DepthSchedule,
    config: DepthGateConfig,
) -> DepthValidationData:
    mapping_ids = np.asarray(schedule.validation_mapping_ids)
    query_colors = np.asarray(schedule.validation_query_colors)
    presentation_orders = np.asarray(schedule.validation_presentation_orders)
    count = config.validation_episodes
    if mapping_ids.shape != (count,) or query_colors.shape != (count,):
        raise ValueError("validation IDs and queries must match validation count")
    if presentation_orders.shape != (count, 10):
        raise ValueError("validation presentation orders have the wrong shape")
    shape = (config.sequence_length, count, config.row_config.input_width)
    intact = np.zeros(shape, dtype=np.float32)
    shuffled = np.zeros(shape, dtype=np.float32)
    no_context = np.zeros(shape, dtype=np.float32)
    targets = np.zeros((config.gap_steps + 1, count), dtype=np.int32)
    shifts = np.zeros((count,), dtype=np.int32)
    for episode_index in range(count):
        mapping = unrank_ten_cycle(int(mapping_ids[episode_index]))
        query = int(query_colors[episode_index])
        order = presentation_orders[episode_index]
        shift, shuffled_mapping = _select_shuffled_rotation(mapping, query)
        intact[:11, episode_index] = _encode_cycle_episode(
            mapping, query, order, config.row_config
        )
        shuffled[:11, episode_index] = _encode_cycle_episode(
            shuffled_mapping, query, order, config.row_config
        )
        no_context[10, episode_index] = intact[10, episode_index]
        shifts[episode_index] = shift
        for depth_index in range(config.gap_steps + 1):
            targets[depth_index, episode_index] = _iterate_mapping(
                mapping, query, depth_index + 1
            )
        if any(
            _iterate_mapping(shuffled_mapping, query, effort + 1)
            == targets[effort, episode_index]
            for effort in QUALIFYING_EFFORTS
        ):
            raise ValueError("shuffled control retains an intact qualifying answer")
    advances = np.ones((config.sequence_length, count), dtype=np.bool_)
    return DepthValidationData(
        intact=_readonly(intact),
        shuffled=_readonly(shuffled),
        no_context=_readonly(no_context),
        targets_by_depth=_readonly(targets),
        advance_masks=_readonly(advances),
        mapping_ids=_readonly(mapping_ids),
        query_colors=_readonly(query_colors),
        presentation_orders=_readonly(presentation_orders),
        shuffled_shifts=_readonly(shifts),
    )


_ENCODED_CHUNK_FIELDS = (
    "events",
    "targets",
    "loss_weights",
    "advance_masks",
    "mapping_ids",
    "efforts",
    "query_colors",
    "presentation_orders",
)
_VALIDATION_IDENTITY_FIELDS = (
    "mapping_ids",
    "query_colors",
    "presentation_orders",
    "shuffled_shifts",
    "intact",
    "shuffled",
    "no_context",
    "targets_by_depth",
    "advance_masks",
)
_TELEMETRY_CATEGORIES = (
    "logits",
    "model_states",
    "gradients",
    "pp_prop_traces",
    "adam",
    "parameters",
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _ordered_json_list_sha256(values: list[str]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _strict_json_sha256(value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validation_data_report(data: DepthValidationData) -> dict[str, Any]:
    return {
        "episode_count": int(np.asarray(data.mapping_ids).size),
        "sha256": {
            field: _array_sha256(np.asarray(getattr(data, field)))
            for field in _VALIDATION_IDENTITY_FIELDS
        },
    }


@dataclass(slots=True)
class _EncodedScheduleHashState:
    global_digests: dict[str, Any]
    chunk_digests: dict[str, list[str]]
    encoded_updates: int = 0
    chunk_count: int = 0


def _new_encoded_schedule_hash_state() -> _EncodedScheduleHashState:
    return _EncodedScheduleHashState(
        global_digests={
            field: hashlib.sha256() for field in _ENCODED_CHUNK_FIELDS
        },
        chunk_digests={field: [] for field in _ENCODED_CHUNK_FIELDS},
    )


def _update_encoded_schedule_hash_state(
    state: _EncodedScheduleHashState,
    encoded: DepthTrainingChunk,
    config: DepthGateConfig,
) -> None:
    chunk_updates = int(np.asarray(encoded.efforts).shape[0])
    if chunk_updates <= 0:
        raise ValueError("encoded staging chunks must contain at least one update")
    if state.chunk_count == 0:
        for field in _ENCODED_CHUNK_FIELDS:
            array = np.ascontiguousarray(getattr(encoded, field))
            logical_shape = (config.training_updates, *array.shape[1:])
            state.global_digests[field].update(array.dtype.str.encode("ascii"))
            state.global_digests[field].update(str(logical_shape).encode("ascii"))
    for field in _ENCODED_CHUNK_FIELDS:
        array = np.ascontiguousarray(getattr(encoded, field))
        state.global_digests[field].update(array.tobytes())
        state.chunk_digests[field].append(_array_sha256(array))
    state.encoded_updates += chunk_updates
    state.chunk_count += 1


def _finish_encoded_schedule_report(
    state: _EncodedScheduleHashState,
    config: DepthGateConfig,
) -> dict[str, Any]:
    if state.encoded_updates != config.training_updates:
        raise ValueError("encoded chunks do not cover every scheduled update")
    if state.chunk_count != config.staging_chunk_count:
        raise ValueError("encoded chunk count differs from the configuration")
    return {
        "chunk_count": state.chunk_count,
        "chunk_updates": config.staging_chunk_updates,
        "training_updates": config.training_updates,
        "training_episode_count": config.training_episode_count,
        "global_sha256": {
            field: digest.hexdigest()
            for field, digest in state.global_digests.items()
        },
        "chunk_sha256_manifest": {
            field: _ordered_json_list_sha256(values)
            for field, values in state.chunk_digests.items()
        },
        "chunk_sha256": state.chunk_digests,
    }


def _encoded_schedule_report(
    schedule: DepthSchedule,
    config: DepthGateConfig,
) -> dict[str, Any]:
    state = _new_encoded_schedule_hash_state()
    for schedule_chunk in _iter_schedule_chunks(schedule, config):
        encoded = _encode_training_chunk(schedule_chunk, config)
        _update_encoded_schedule_hash_state(state, encoded, config)
    return _finish_encoded_schedule_report(state, config)


def _model_config(config: DepthGateConfig, *, batch_size: int) -> ModelConfig:
    indices = associative_memory_feature_indices(config.row_config)
    rows = config.row_config
    return ModelConfig(
        input_width=rows.input_width,
        batch_size=batch_size,
        neuron_count=config.neuron_count,
        recurrent_edges=config.recurrent_edges,
        max_latent_steps=config.gap_steps,
        readout_width=config.readout_width,
        color_rank=config.color_rank,
        input_gain=config.input_gain,
        recurrent_gain=config.recurrent_gain,
        trace_decay=config.trace_decay,
        event_valid_index=rows.valid_slice.start,
        context_memory_width=config.context_memory_width,
        memory_decay=config.memory_decay,
        demonstration_phase_index=rows.phase_slice.start,
        query_phase_index=rows.phase_slice.start + 1,
        input_side_valid_index=rows.side_valid_slice.start,
        output_side_valid_index=rows.side_valid_slice.start + 1,
        memory_key_indices=indices.key_indices,
        memory_value_indices=indices.value_indices,
        seed=config.model_seed,
    )


def _initialization_report(
    model: LatentWorkspaceModel,
    config: DepthGateConfig,
) -> dict[str, Any]:
    parameters = legacy._parameter_values(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        learner = compile_pp_prop(model)
    return {
        "fresh_model": True,
        "model_seed": config.model_seed,
        "configuration": asdict(config),
        "parameter_sha256": legacy._array_digest(parameters),
        "parameter_count": int(
            sum(
                np.asarray(leaf).size
                for value in parameters.values()
                for leaf in jax.tree.leaves(value)
            )
        ),
        "parameters_finite": bool(
            all(
                np.isfinite(np.asarray(leaf)).all()
                for value in parameters.values()
                for leaf in jax.tree.leaves(value)
            )
        ),
        "compiler": gate._compiler_report(learner),
        "compile_warnings": [str(item.message) for item in caught],
    }


@dataclass(slots=True)
class _DepthPPPropTrainer:
    learner: Any
    optimizer: Any
    compiler: dict[str, Any]
    compile_warnings: list[str]
    train_chunk: Any
    algorithm: str = "production_pp_prop"


def _tree_telemetry(value: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
    leaves = tuple(jnp.asarray(leaf) for leaf in jax.tree.leaves(value))
    if not leaves:
        raise RuntimeError("telemetry subject has no array leaves")
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
    maximum = jnp.max(
        jnp.stack(
            [
                jnp.max(jnp.abs(leaf.astype(jnp.float32)), initial=0.0)
                for leaf in leaves
            ]
        )
    )
    count = jnp.asarray(sum(leaf.size for leaf in leaves), dtype=jnp.int32)
    return finite, maximum, count


def _make_pp_prop_trainer(
    model: LatentWorkspaceModel,
    config: DepthGateConfig,
) -> _DepthPPPropTrainer:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        learner = compile_pp_prop(model)
    compiler = gate._compiler_report(learner)
    optimizer = braintools.optim.Adam(lr=config.learning_rate)
    optimizer.register_trainable_weights(learner.param_states)
    rank = config.color_rank
    parameter_keys = {
        gate._path(path): path for path in learner.param_states.keys()
    }
    missing_trace_paths = set(gate._STAGE21_TRACE_PATHS) - set(parameter_keys)
    if missing_trace_paths:
        raise RuntimeError(
            f"trainer is missing pp-prop trace paths: {sorted(missing_trace_paths)}"
        )
    model_states = tuple(
        state
        for state in model.states().values()
        if not isinstance(state, brainstate.ParamState)
    )

    @brainstate.transform.jit
    def train_chunk(
        events: jax.Array,
        targets: jax.Array,
        loss_weights: jax.Array,
        advance_masks: jax.Array,
    ) -> dict[str, jax.Array]:
        def train_one(
            inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> dict[str, jax.Array]:
            sequence, target_sequence, weights, advances = inputs
            model.reset_state()
            learner.reset_state(batch_size=config.batch_size)

            def step_loss(
                event: jax.Array,
                advance: jax.Array,
                target: jax.Array,
            ) -> jax.Array:
                return legacy._classification_loss(
                    learner(event, advance), target, rank
                )

            gradients, loss = learner.etrace_grad(
                sequence,
                advances,
                target_sequence,
                step_fn=step_loss,
                mask=weights.astype(jnp.float32),
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            compact = model.compact_readout()
            trace_factors = tuple(
                learner.get_etrace_of(parameter_keys[label])
                for label in gate._STAGE21_TRACE_PATHS
            )
            optimizer.update(brainstate.nn.clip_grad_norm(gradients, config.clip_norm))
            measurements = {
                "logits": _tree_telemetry(
                    legacy._color_logits(compact, config.color_rank)
                ),
                "model_states": _tree_telemetry(
                    tuple(state.value for state in model_states)
                ),
                "gradients": _tree_telemetry(gradients),
                "pp_prop_traces": _tree_telemetry(trace_factors),
                "adam": _tree_telemetry(optimizer.opt_state.value),
                "parameters": _tree_telemetry(
                    tuple(state.value for state in learner.param_states.values())
                ),
            }
            return {
                "loss": loss,
                "finite": {key: value[0] for key, value in measurements.items()},
                "max_abs": {key: value[1] for key, value in measurements.items()},
                "value_count": {
                    key: value[2] for key, value in measurements.items()
                },
            }

        return brainstate.transform.for_loop(
            train_one,
            (events, targets, loss_weights, advance_masks),
        )

    return _DepthPPPropTrainer(
        learner=learner,
        optimizer=optimizer,
        compiler=compiler,
        compile_warnings=[str(item.message) for item in caught],
        train_chunk=train_chunk,
    )


def _train_depth_gate(
    model: LatentWorkspaceModel,
    schedule: DepthSchedule,
    config: DepthGateConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trainer = _make_pp_prop_trainer(model, config)
    initial_parameters = legacy._parameter_values(model)
    initial_sha = legacy._array_digest(initial_parameters)
    hash_state = _new_encoded_schedule_hash_state()
    telemetry_chunks: list[Mapping[str, Any]] = []
    for schedule_chunk in _iter_schedule_chunks(schedule, config):
        encoded = _encode_training_chunk(schedule_chunk, config)
        _update_encoded_schedule_hash_state(hash_state, encoded, config)
        telemetry_chunks.append(
            jax.device_get(
                jax.block_until_ready(
                    trainer.train_chunk(
                        encoded.events,
                        encoded.targets,
                        encoded.loss_weights,
                        encoded.advance_masks,
                    )
                )
            )
        )
    schedule_report = _finish_encoded_schedule_report(hash_state, config)
    losses = np.concatenate(
        [np.asarray(item["loss"]).reshape(-1) for item in telemetry_chunks]
    )
    finite = {
        category: bool(
            all(
                np.asarray(item["finite"][category], dtype=np.bool_).all()
                for item in telemetry_chunks
            )
        )
        for category in _TELEMETRY_CATEGORIES
    }
    maxima = {
        category: float(
            max(
                np.max(np.asarray(item["max_abs"][category], dtype=np.float64))
                for item in telemetry_chunks
            )
        )
        for category in _TELEMETRY_CATEGORIES
    }
    value_counts = {
        category: int(
            sum(
                int(np.sum(np.asarray(item["value_count"][category], dtype=np.int64)))
                for item in telemetry_chunks
            )
        )
        for category in _TELEMETRY_CATEGORIES
    }
    effort_counts = {
        str(effort): int(
            np.count_nonzero(np.asarray(schedule.training_efforts) == effort)
        )
        for effort in QUALIFYING_EFFORTS
    }
    final_sha = legacy._array_digest(legacy._parameter_values(model))
    training = {
        "algorithm": trainer.algorithm,
        "executed_updates": int(losses.size),
        "batch_size": config.batch_size,
        "chunk_count": len(telemetry_chunks),
        "chunk_updates": config.staging_chunk_updates,
        "effort_update_counts": effort_counts,
        "initialization_parameter_sha256": initial_sha,
        "final_parameter_sha256": final_sha,
        "losses": losses.tolist(),
        "finite": finite,
        "max_abs": maxima,
        "value_count": value_counts,
        "compiler": trainer.compiler,
        "compile_warnings": trainer.compile_warnings,
    }
    return training, schedule_report


def _metric(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    checkpoint: int,
) -> dict[str, Any]:
    return {
        **legacy._accuracy(predictions, targets),
        "checkpoint": checkpoint,
    }


def _evaluate_model(
    trained_model: LatentWorkspaceModel,
    data: DepthValidationData,
    config: DepthGateConfig,
) -> dict[str, Any]:
    model = LatentWorkspaceModel(
        _model_config(config, batch_size=config.validation_episodes)
    )
    legacy._copy_parameters(trained_model, model)
    advances = jnp.asarray(data.advance_masks)
    model_states = tuple(model.states().values())

    @brainstate.transform.jit
    def evaluate_arm(
        events: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        model.reset_state()

        def step(inputs: tuple[jax.Array, jax.Array]):
            event, advance = inputs
            compact = model.update(event, advance)
            state_finite = jnp.all(
                jnp.stack(
                    [
                        jnp.all(jnp.isfinite(jnp.asarray(leaf)))
                        for state in model_states
                        for leaf in jax.tree.leaves(state.value)
                    ]
                )
            )
            return compact, state_finite

        compact, state_finite = brainstate.transform.for_loop(
            step, (events, advances)
        )
        return compact[10:], state_finite

    intact_values = jax.block_until_ready(evaluate_arm(jnp.asarray(data.intact)))
    shuffled_values = jax.block_until_ready(evaluate_arm(jnp.asarray(data.shuffled)))
    no_context_values = jax.block_until_ready(evaluate_arm(jnp.asarray(data.no_context)))
    arms = {
        "intact": tuple(np.asarray(value) for value in intact_values),
        "shuffled": tuple(np.asarray(value) for value in shuffled_values),
        "no_context": tuple(np.asarray(value) for value in no_context_values),
    }
    finite = all(
        bool(np.isfinite(value).all())
        for values in arms.values()
        for value in values
    ) and all(bool(np.asarray(values[1]).all()) for values in arms.values())
    logits: dict[str, list[np.ndarray]] = {name: [] for name in arms}
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in arms}
    for name, values in arms.items():
        for depth_index in range(config.gap_steps + 1):
            compact = values[0][depth_index]
            color_logits = np.asarray(
                legacy._color_logits(jnp.asarray(compact), config.color_rank)
            )
            finite = finite and bool(np.isfinite(color_logits).all())
            logits[name].append(color_logits)
            predictions[name].append(np.argmax(color_logits, axis=-1))

    targets = np.asarray(data.targets_by_depth)
    depth_reports = {
        str(depth_index): {
            arm: _metric(
                predictions[arm][depth_index],
                targets[depth_index],
                checkpoint=depth_index,
            )
            for arm in ("intact", "shuffled", "no_context")
        }
        for depth_index in range(config.gap_steps + 1)
    }
    h0_predictions = predictions["intact"][0]
    h0_proper = depth_reports["0"]["intact"]
    efforts: dict[str, dict[str, Any]] = {}
    for effort in QUALIFYING_EFFORTS:
        intact = depth_reports[str(effort)]["intact"]
        shuffled = depth_reports[str(effort)]["shuffled"]
        no_context = depth_reports[str(effort)]["no_context"]
        h0_final = _metric(h0_predictions, targets[effort], checkpoint=0)
        efforts[str(effort)] = {
            "intact": intact,
            "shuffled": shuffled,
            "no_context": no_context,
            "h0_final_target": h0_final,
            "intact_minus_h0": intact["accuracy"] - h0_final["accuracy"],
            "intact_minus_shuffled": intact["accuracy"] - shuffled["accuracy"],
        }
    return {
        "finite": finite,
        "h0_proper": h0_proper,
        "depths": depth_reports,
        "efforts": efforts,
    }


_QUALIFICATION_CRITERIA = (
    "schema_and_control",
    "preregistered_configuration",
    "gate_a_prerequisite_authenticated",
    "gate_b_initialization_authenticated",
    "cycle_catalog_and_schedule_complete",
    "checkpoint_targets_and_controls_complete",
    "training_complete_and_finite",
    "evaluation_complete_and_finite",
    "matching_depth_above_chance_at_every_effort",
    "at_least_two_depths_improve_over_h0",
    "intact_exceeds_shuffled_at_every_effort",
    "controls_not_demonstrably_above_chance",
    "h0_one_step_above_chance",
    "held_out_invariants_complete",
)
_INITIALIZATION_QUALIFICATION_CRITERIA = (
    "schema_and_control",
    "preregistered_configuration",
    "gate_a_prerequisite_authenticated",
    "source_and_gpu_authenticated",
    "initialization_fresh_and_finite",
    "compiler_paths_complete",
)


def _sha256_complete(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _compiler_evidence_complete(compiler: Mapping[str, Any]) -> bool:
    required = {
        "memory_write_scale",
        "workspace_query_projection/weight",
        "memory_read_projection/weight",
    }
    diagnostics = compiler.get("diagnostics")
    compiled_paths = compiler.get("compiled_parameter_paths")
    direct_paths = compiler.get("required_direct_paths")
    direct_status = compiler.get("direct_path_status")
    return bool(
        compiler.get("available") is True
        and isinstance(diagnostics, list)
        and not any(
            isinstance(item, Mapping) and item.get("level") == "error"
            for item in diagnostics
        )
        and isinstance(compiled_paths, list)
        and required <= set(compiled_paths)
        and isinstance(direct_paths, list)
        and set(direct_paths) == required
        and isinstance(direct_status, Mapping)
        and set(direct_status) == required
        and all(direct_status[path] is True for path in required)
        and compiler.get("all_required_direct") is True
        and compiler.get("context_memory_isolated_from_workspace_lif") is True
        and gate._compiler_topology_complete(compiler)
    )


def _schedule_evidence_complete(schedule: Mapping[str, Any]) -> bool:
    if set(schedule) != {
        "chunk_count",
        "chunk_updates",
        "training_updates",
        "training_episode_count",
        "global_sha256",
        "chunk_sha256_manifest",
        "chunk_sha256",
    }:
        return False
    integer_values = {
        "chunk_count": 32,
        "chunk_updates": 128,
        "training_updates": 4_096,
        "training_episode_count": 262_144,
    }
    if any(
        not gate._is_integer(schedule[name])
        or int(schedule[name]) != expected
        for name, expected in integer_values.items()
    ):
        return False
    if not gate._json_exact(
        schedule["global_sha256"], _PRODUCTION_ENCODED_GLOBAL_SHA256
    ):
        return False
    chunk_digests = schedule["chunk_sha256"]
    manifests = schedule["chunk_sha256_manifest"]
    return bool(
        isinstance(chunk_digests, Mapping)
        and set(chunk_digests) == set(_ENCODED_CHUNK_FIELDS)
        and gate._json_exact(manifests, _PRODUCTION_CHUNK_SHA256_MANIFEST)
        and all(
            isinstance(chunk_digests[field], list)
            and len(chunk_digests[field]) == 32
            and all(_sha256_complete(value) for value in chunk_digests[field])
            and _ordered_json_list_sha256(chunk_digests[field])
            == manifests[field]
            for field in _ENCODED_CHUNK_FIELDS
        )
    )


def _validation_evidence_complete(validation: Mapping[str, Any]) -> bool:
    return bool(
        set(validation) == {"episode_count", "sha256"}
        and gate._is_integer(validation["episode_count"])
        and int(validation["episode_count"]) == 512
        and gate._json_exact(
            validation["sha256"], _PRODUCTION_VALIDATION_SHA256
        )
    )


def _target_and_control_evidence_complete(data: Mapping[str, Any]) -> bool:
    expected_target = {
        "efforts": [1, 2, 4, 8],
        "sequence_length": 19,
        "h0_index": 10,
        "active_lengths": {"1": 12, "2": 13, "4": 15, "8": 19},
        "target_rule": "H_r=f^(r+1)(x)",
        "supervision_weights": {
            "1": 0.5,
            "2": 1.0 / 3.0,
            "4": 0.2,
            "8": 1.0 / 9.0,
        },
        "suffix_advance_false": True,
        "suffix_loss_weight_zero": True,
        "padded_compact_objective_equal": True,
        "padded_compact_pp_prop_gradients_equal": True,
        "finite_window_chunk_size": 1,
    }
    expected_controls = {
        "rotation_candidates": list(range(1, 10)),
        "all_validation_rotations_valid": True,
        "all_shuffled_answers_differ_at_qualifying_depths": True,
        "exact_input_output_marginals": True,
        "no_context_demonstrations_zero": True,
        "event_timing_identical": True,
    }
    return gate._json_exact(data["target_contract"], expected_target) and gate._json_exact(
        data["controls"], expected_controls
    )


def _validated_initialization_admission(
    prerequisite: Mapping[str, Any],
    config: DepthGateConfig,
    *,
    source_start: Mapping[str, Any],
    environment: Mapping[str, Any],
    require_pass: bool,
) -> Mapping[str, Any]:
    expected_keys = {
        "target",
        "source_head",
        "image_digest",
        "bundle_sha256",
        "manifest_sha256",
        "preflight_sha256",
        "result_sha256",
        "admission",
    }
    if not isinstance(prerequisite, Mapping) or set(prerequisite) != expected_keys:
        raise ValueError("Gate B initialization prerequisite is not authenticated")
    source_head = prerequisite["source_head"]
    image_digest = prerequisite["image_digest"]
    if (
        prerequisite["target"] != "gate_b_init"
        or not isinstance(source_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        or not isinstance(image_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest) is None
        or not all(
            _sha256_complete(prerequisite[name])
            for name in (
                "bundle_sha256",
                "manifest_sha256",
                "preflight_sha256",
                "result_sha256",
            )
        )
    ):
        raise ValueError("Gate B initialization provenance fields are invalid")
    admission = prerequisite["admission"]
    if not isinstance(admission, Mapping):
        raise ValueError("Gate B initialization admission is missing")
    if set(admission) != {
        "schema_version",
        "control",
        "qualification_regime",
        "config",
        "prerequisites",
        "initialization",
        "source_start",
        "source_end",
        "source_files",
        "environment",
        "qualification",
    }:
        raise ValueError("Gate B initialization admission schema is invalid")
    if _strict_json_sha256(admission) != prerequisite["result_sha256"]:
        raise ValueError("Gate B initialization result digest is invalid")
    expected_bundle = hashlib.sha256(
        (
            "example21-launch-bundle-v1\0gate_b_init\0"
            f"{source_head}\0{prerequisite['preflight_sha256']}\0"
            f"{prerequisite['result_sha256']}"
        ).encode()
    ).hexdigest()
    if prerequisite["bundle_sha256"] != expected_bundle:
        raise ValueError("Gate B initialization bundle digest is invalid")
    qualification = _gate_b_initialization_qualification(admission, config)
    if not gate._json_exact(admission["qualification"], qualification):
        raise ValueError("Gate B initialization qualification is stale")
    if require_pass and qualification["passed"] is not True:
        raise ValueError("Gate B initialization admission did not pass")
    if (
        admission["source_start"]["commit"] != source_head
        or admission["source_end"]["commit"] != source_head
        or admission["environment"]["image_digest"] != image_digest
        or source_start["commit"] != source_head
        or environment["image_digest"] != image_digest
    ):
        raise ValueError("Gate B initialization source or image differs")
    return admission


def _initialization_evidence_complete(
    report: Mapping[str, Any],
    config: DepthGateConfig,
) -> bool:
    try:
        admission = _validated_initialization_admission(
            report["prerequisites"]["gate_b_initialization"],
            config,
            source_start=report["source_start"],
            environment=report["environment"],
            require_pass=True,
        )
        initialization = admission["initialization"]
        parameter_sha = initialization["parameter_sha256"]
        parameter_count = initialization["parameter_count"]
        return bool(
            initialization["fresh_model"] is True
            and initialization["parameters_finite"] is True
            and gate._is_integer(initialization["model_seed"])
            and int(initialization["model_seed"]) == config.model_seed
            and gate._json_exact(initialization["configuration"], asdict(config))
            and _sha256_complete(parameter_sha)
            and gate._is_integer(parameter_count)
            and int(parameter_count) > 0
            and isinstance(initialization["compile_warnings"], list)
            and _compiler_evidence_complete(initialization["compiler"])
            and gate._source_evidence_clean(report["source_start"])
            and gate._gpu_environment_verified(report["environment"])
            and report["training"]["initialization_parameter_sha256"]
            == parameter_sha
            and report["evaluation"]["initialization_parameter_sha256"]
            == parameter_sha
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _training_evidence_complete(
    report: Mapping[str, Any],
    config: DepthGateConfig,
) -> bool:
    training = report["training"]
    initialization = report["prerequisites"]["gate_b_initialization"][
        "admission"
    ]["initialization"]
    losses = gate._real_array(training["losses"])
    categories = {
        "logits",
        "model_states",
        "gradients",
        "pp_prop_traces",
        "adam",
        "parameters",
    }
    if (
        losses.shape != (config.training_updates,)
        or not np.isfinite(losses).all()
        or training["algorithm"] != "production_pp_prop"
        or not gate._is_integer(training["executed_updates"])
        or int(training["executed_updates"]) != config.training_updates
        or not gate._is_integer(training["batch_size"])
        or int(training["batch_size"]) != config.batch_size
        or not gate._is_integer(training["chunk_count"])
        or int(training["chunk_count"]) != 32
        or not gate._is_integer(training["chunk_updates"])
        or int(training["chunk_updates"]) != 128
        or not gate._json_exact(
            training["effort_update_counts"],
            {"1": 1_024, "2": 1_024, "4": 1_024, "8": 1_024},
        )
        or not _sha256_complete(training["initialization_parameter_sha256"])
        or not _sha256_complete(training["final_parameter_sha256"])
        or training["initialization_parameter_sha256"]
        == training["final_parameter_sha256"]
        or not isinstance(training["compile_warnings"], list)
        or not _compiler_evidence_complete(training["compiler"])
        or not gate._json_exact(
            training["compiler"], initialization["compiler"]
        )
        or not gate._json_exact(
            training["compile_warnings"], initialization["compile_warnings"]
        )
    ):
        return False
    finite = training["finite"]
    maxima = training["max_abs"]
    counts = training["value_count"]
    if not all(
        isinstance(section, Mapping) and set(section) == categories
        for section in (finite, maxima, counts)
    ):
        return False
    if not all(finite[name] is True for name in categories):
        return False
    if not all(
        gate._is_finite_real(maxima[name]) and float(maxima[name]) >= 0.0
        for name in categories
    ):
        return False
    if not all(
        gate._is_integer(counts[name]) and int(counts[name]) > 0
        for name in categories
    ):
        return False
    source_start = report["source_start"]
    source_end = report["source_end"]
    return bool(
        gate._source_evidence_clean(source_end)
        and source_start["commit"] == source_end["commit"]
    )


def _depth_metric_complete(
    metric: Mapping[str, Any],
    *,
    checkpoint: int,
    count: int,
) -> bool:
    if set(metric) != {
        "correct",
        "count",
        "accuracy",
        "wilson_95_lower",
        "wilson_95_upper",
        "prediction_histogram",
        "prediction_sha256",
        "checkpoint",
    }:
        return False
    return bool(
        gate._is_integer(metric["checkpoint"])
        and int(metric["checkpoint"]) == checkpoint
        and _sha256_complete(metric["prediction_sha256"])
        and gate._accuracy_evidence_complete(metric, count)
        and max(map(int, metric["prediction_histogram"])) < count
    )


def _evaluation_evidence_complete(
    evaluation: Mapping[str, Any],
    config: DepthGateConfig,
) -> bool:
    count = config.validation_episodes
    if evaluation["finite"] is not True:
        return False
    depths = evaluation["depths"]
    if not isinstance(depths, Mapping) or set(depths) != {
        str(index) for index in range(config.gap_steps + 1)
    }:
        return False
    for depth_index in range(config.gap_steps + 1):
        arms = depths[str(depth_index)]
        if not isinstance(arms, Mapping) or set(arms) != {
            "intact",
            "shuffled",
            "no_context",
        }:
            return False
        if not all(
            _depth_metric_complete(
                arms[arm], checkpoint=depth_index, count=count
            )
            for arm in ("intact", "shuffled", "no_context")
        ):
            return False
    if not gate._json_exact(evaluation["h0_proper"], depths["0"]["intact"]):
        return False
    efforts = evaluation["efforts"]
    if not isinstance(efforts, Mapping) or set(efforts) != {
        str(effort) for effort in QUALIFYING_EFFORTS
    }:
        return False
    h0 = evaluation["h0_proper"]
    for effort in QUALIFYING_EFFORTS:
        evidence = efforts[str(effort)]
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "intact",
            "shuffled",
            "no_context",
            "h0_final_target",
            "intact_minus_h0",
            "intact_minus_shuffled",
        }:
            return False
        matching = depths[str(effort)]
        if not all(
            gate._json_exact(evidence[arm], matching[arm])
            for arm in ("intact", "shuffled", "no_context")
        ):
            return False
        h0_final = evidence["h0_final_target"]
        if (
            not _depth_metric_complete(h0_final, checkpoint=0, count=count)
            or h0_final["prediction_sha256"] != h0["prediction_sha256"]
            or h0_final["prediction_histogram"] != h0["prediction_histogram"]
        ):
            return False
        intact_gap = (
            evidence["intact"]["accuracy"] - h0_final["accuracy"]
        )
        shuffled_gap = (
            evidence["intact"]["accuracy"] - evidence["shuffled"]["accuracy"]
        )
        if not all(
            gate._is_finite_real(evidence[name])
            for name in ("intact_minus_h0", "intact_minus_shuffled")
        ):
            return False
        if not math.isclose(
            float(evidence["intact_minus_h0"]),
            intact_gap,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(evidence["intact_minus_shuffled"]),
            shuffled_gap,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
    return True


def _held_out_invariants_complete(data: Mapping[str, Any]) -> bool:
    expected = {
        "validation_episode_count": 512,
        "distinct_training_cycle_count": 262_144,
        "distinct_validation_cycle_count": 512,
        "training_validation_overlap_count": 0,
        "balanced_queries": True,
        "target_trajectories_exact": True,
        "no_copy_shortcut_at_every_effort": True,
        "cross_effort_h0_prediction_identity": True,
        "cross_effort_h0_identity_count": 512,
    }
    return gate._json_exact(data["held_out_invariants"], expected)


def _gate_a_prerequisite_complete(
    gate_a: Mapping[str, Any],
    config: DepthGateConfig,
) -> bool:
    return bool(
        gate_a["qualification_passed"] is True
        and gate_a["result_sha256"] == config.gate_a_result_sha256
        and gate_a["manifest_sha256"] == config.gate_a_manifest_sha256
        and gate_a["source_commit"] == config.gate_a_source_commit
    )


def _gate_b_initialization_qualification(
    report: Mapping[str, Any],
    config: DepthGateConfig,
) -> dict[str, Any]:
    criteria = {
        name: False for name in _INITIALIZATION_QUALIFICATION_CRITERIA
    }
    if config.qualification_regime != "preregistered_full":
        return {
            "passed": False,
            "criteria": criteria,
            "interpretation": "nonqualifying_abbreviated_no_capability_conclusion",
        }
    try:
        initialization = report["initialization"]
        criteria["schema_and_control"] = bool(
            set(report) - {"qualification"}
            == {
                "schema_version",
                "control",
                "qualification_regime",
                "config",
                "prerequisites",
                "initialization",
                "source_start",
                "source_end",
                "source_files",
                "environment",
            }
            and gate._is_integer(report["schema_version"])
            and int(report["schema_version"]) == GATE_B_SCHEMA_VERSION
            and report["control"] == GATE_B_INITIALIZATION_CONTROL
            and report["qualification_regime"] == "preregistered_full"
        )
        criteria["preregistered_configuration"] = gate._json_exact(
            report["config"], asdict(config)
        )
        criteria["gate_a_prerequisite_authenticated"] = (
            _gate_a_prerequisite_complete(
                report["prerequisites"]["gate_a"], config
            )
        )
        source_start = report["source_start"]
        source_end = report["source_end"]
        criteria["source_and_gpu_authenticated"] = bool(
            gate._source_evidence_clean(source_start)
            and gate._source_evidence_clean(source_end)
            and source_start["commit"] == source_end["commit"]
            and gate._gpu_environment_verified(report["environment"])
            and gate._json_exact(
                report["source_files"],
                {
                    "latent_workspace_model.py": config.model_source_sha256,
                    "latent_workspace_task.py": config.task_source_sha256,
                },
            )
        )
        criteria["initialization_fresh_and_finite"] = bool(
            initialization["fresh_model"] is True
            and gate._is_integer(initialization["model_seed"])
            and int(initialization["model_seed"]) == config.model_seed
            and gate._json_exact(initialization["configuration"], asdict(config))
            and _sha256_complete(initialization["parameter_sha256"])
            and gate._is_integer(initialization["parameter_count"])
            and int(initialization["parameter_count"]) > 0
            and initialization["parameters_finite"] is True
        )
        criteria["compiler_paths_complete"] = _compiler_evidence_complete(
            initialization["compiler"]
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    passed = bool(all(criteria.values()))
    return {
        "passed": passed,
        "criteria": criteria,
        "interpretation": (
            "gate_b_initialization_admission_passed"
            if passed
            else "gate_b_initialization_admission_failed_stop"
        ),
    }


def _gate_b_initialization_report(
    model: LatentWorkspaceModel,
    config: DepthGateConfig,
    *,
    gate_a: Mapping[str, Any],
    source_start: Mapping[str, Any],
    source_end_reporter: Callable[[], Mapping[str, Any]],
    source_files: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    if not callable(source_end_reporter):
        raise TypeError("source_end_reporter must be callable")
    initialization = _initialization_report(model, config)
    source_end = source_end_reporter()
    if not isinstance(source_end, Mapping):
        raise TypeError("source_end_reporter must return a mapping")
    report = {
        "schema_version": GATE_B_SCHEMA_VERSION,
        "control": GATE_B_INITIALIZATION_CONTROL,
        "qualification_regime": config.qualification_regime,
        "config": asdict(config),
        "prerequisites": {"gate_a": dict(gate_a)},
        "initialization": initialization,
        "source_start": dict(source_start),
        "source_end": dict(source_end),
        "source_files": dict(source_files),
        "environment": dict(environment),
    }
    report["qualification"] = _gate_b_initialization_qualification(report, config)
    return report


def _target_contract_report() -> dict[str, Any]:
    return {
        "efforts": list(QUALIFYING_EFFORTS),
        "sequence_length": 19,
        "h0_index": 10,
        "active_lengths": {"1": 12, "2": 13, "4": 15, "8": 19},
        "target_rule": "H_r=f^(r+1)(x)",
        "supervision_weights": {
            "1": 0.5,
            "2": 1.0 / 3.0,
            "4": 0.2,
            "8": 1.0 / 9.0,
        },
        "suffix_advance_false": True,
        "suffix_loss_weight_zero": True,
        "padded_compact_objective_equal": True,
        "padded_compact_pp_prop_gradients_equal": True,
        "finite_window_chunk_size": 1,
    }


def _control_report(
    data: DepthValidationData,
    config: DepthGateConfig,
) -> dict[str, Any]:
    indices = associative_memory_feature_indices(config.row_config)
    keys = np.asarray(indices.key_indices, dtype=np.int32)
    values = np.asarray(indices.value_indices, dtype=np.int32)
    intact_demo = np.asarray(data.intact[:10])
    shuffled_demo = np.asarray(data.shuffled[:10])
    return {
        "rotation_candidates": list(range(1, 10)),
        "all_validation_rotations_valid": bool(
            np.all((np.asarray(data.shuffled_shifts) >= 1) & (np.asarray(data.shuffled_shifts) <= 9))
        ),
        "all_shuffled_answers_differ_at_qualifying_depths": bool(
            all(
                _iterate_mapping(
                    (unrank_ten_cycle(int(mapping_id)) + int(shift)) % 10,
                    int(query),
                    effort + 1,
                )
                != int(np.asarray(data.targets_by_depth)[effort, episode_index])
                for episode_index, (mapping_id, query, shift) in enumerate(
                    zip(data.mapping_ids, data.query_colors, data.shuffled_shifts, strict=True)
                )
                for effort in QUALIFYING_EFFORTS
            )
        ),
        "exact_input_output_marginals": bool(
            np.array_equal(intact_demo[..., keys], shuffled_demo[..., keys])
            and np.array_equal(
                intact_demo[..., values].sum(axis=0),
                shuffled_demo[..., values].sum(axis=0),
            )
        ),
        "no_context_demonstrations_zero": bool(
            np.count_nonzero(np.asarray(data.no_context[:10])) == 0
        ),
        "event_timing_identical": bool(
            np.asarray(data.advance_masks).shape
            == (config.sequence_length, config.validation_episodes)
            and np.asarray(data.advance_masks).all()
        ),
    }


def _held_out_invariants_report(
    schedule: DepthSchedule,
    data: DepthValidationData,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    training_ids = np.asarray(schedule.training_mapping_ids).reshape(-1)
    validation_ids = np.asarray(schedule.validation_mapping_ids)
    query_histogram = np.bincount(np.asarray(data.query_colors), minlength=10)
    h0_hash = evaluation["h0_proper"]["prediction_sha256"]
    return {
        "validation_episode_count": int(validation_ids.size),
        "distinct_training_cycle_count": int(np.unique(training_ids).size),
        "distinct_validation_cycle_count": int(np.unique(validation_ids).size),
        "training_validation_overlap_count": int(
            np.intersect1d(training_ids, validation_ids).size
        ),
        "balanced_queries": bool(query_histogram.max() - query_histogram.min() <= 1),
        "target_trajectories_exact": bool(
            all(
                int(data.targets_by_depth[depth_index, episode_index])
                == _iterate_mapping(
                    unrank_ten_cycle(int(mapping_id)),
                    int(query),
                    depth_index + 1,
                )
                for episode_index, (mapping_id, query) in enumerate(
                    zip(data.mapping_ids, data.query_colors, strict=True)
                )
                for depth_index in range(9)
            )
        ),
        "no_copy_shortcut_at_every_effort": bool(
            all(
                int(data.targets_by_depth[effort, episode_index])
                != int(data.targets_by_depth[0, episode_index])
                for episode_index in range(validation_ids.size)
                for effort in QUALIFYING_EFFORTS
            )
        ),
        "cross_effort_h0_prediction_identity": bool(
            all(
                evaluation["efforts"][str(effort)]["h0_final_target"][
                    "prediction_sha256"
                ]
                == h0_hash
                for effort in QUALIFYING_EFFORTS
            )
        ),
        "cross_effort_h0_identity_count": int(validation_ids.size),
    }


def _learner_report(config: DepthGateConfig) -> dict[str, Any]:
    return {
        "algorithm": "production_pp_prop",
        "optimizer": "Adam",
        "trace_factorization": "io_factorized",
        "recurrence_scope": "diagonal",
        "trace_decay": config.trace_decay,
        "vjp_method": "multi-step",
    }


def _qualification_report(
    report: Mapping[str, Any],
    *,
    config: DepthGateConfig,
) -> dict[str, Any]:
    criteria = {name: False for name in _QUALIFICATION_CRITERIA}
    if config.qualification_regime != "preregistered_full":
        return {
            "passed": False,
            "criteria": criteria,
            "interpretation": "nonqualifying_abbreviated_no_capability_conclusion",
        }
    try:
        learner = report["learner"]
        criteria["schema_and_control"] = bool(
            gate._is_integer(report["schema_version"])
            and int(report["schema_version"]) == GATE_B_SCHEMA_VERSION
            and report["control"] == GATE_B_CONTROL
            and report["qualification_regime"] == "preregistered_full"
            and learner["algorithm"] == "production_pp_prop"
        )
        criteria["preregistered_configuration"] = gate._json_exact(
            report["config"], asdict(config)
        )
        gate_a = report["prerequisites"]["gate_a"]
        criteria["gate_a_prerequisite_authenticated"] = bool(
            gate_a["qualification_passed"] is True
            and gate_a["result_sha256"] == config.gate_a_result_sha256
            and gate_a["manifest_sha256"] == config.gate_a_manifest_sha256
            and gate_a["source_commit"] == config.gate_a_source_commit
            and gate._json_exact(
                report["source_files"],
                {
                    "latent_workspace_model.py": config.model_source_sha256,
                    "latent_workspace_task.py": config.task_source_sha256,
                },
            )
        )
        criteria["gate_b_initialization_authenticated"] = (
            _initialization_evidence_complete(report, config)
        )
        data = report["data"]
        criteria["cycle_catalog_and_schedule_complete"] = (
            _schedule_evidence_complete(data["schedule"])
            and _validation_evidence_complete(data["validation"])
        )
        criteria["checkpoint_targets_and_controls_complete"] = (
            _target_and_control_evidence_complete(data)
        )
        criteria["training_complete_and_finite"] = bool(
            gate._json_exact(
                learner,
                {
                    "algorithm": "production_pp_prop",
                    "optimizer": "Adam",
                    "trace_factorization": "io_factorized",
                    "recurrence_scope": "diagonal",
                    "trace_decay": config.trace_decay,
                    "vjp_method": "multi-step",
                },
            )
            and _training_evidence_complete(report, config)
        )
        evaluation = report["evaluation"]
        criteria["evaluation_complete_and_finite"] = (
            _evaluation_evidence_complete(evaluation, config)
        )
        efforts = evaluation["efforts"]
        criteria["matching_depth_above_chance_at_every_effort"] = all(
            gate._is_finite_real(efforts[str(effort)]["intact"]["wilson_95_lower"])
            and float(efforts[str(effort)]["intact"]["wilson_95_lower"]) > 1.0 / 8.0
            for effort in QUALIFYING_EFFORTS
        )
        improvements = [
            float(efforts[str(effort)]["intact"]["accuracy"])
            - float(efforts[str(effort)]["h0_final_target"]["accuracy"])
            for effort in QUALIFYING_EFFORTS
        ]
        criteria["at_least_two_depths_improve_over_h0"] = (
            sum(value >= 0.15 for value in improvements) >= 2
        )
        criteria["intact_exceeds_shuffled_at_every_effort"] = all(
            float(efforts[str(effort)]["intact"]["accuracy"])
            - float(efforts[str(effort)]["shuffled"]["accuracy"])
            >= 0.15
            for effort in QUALIFYING_EFFORTS
        )
        criteria["controls_not_demonstrably_above_chance"] = all(
            gate._is_finite_real(
                efforts[str(effort)][arm]["wilson_95_lower"]
            )
            and float(efforts[str(effort)][arm]["wilson_95_lower"]) <= 1.0 / 8.0
            for effort in QUALIFYING_EFFORTS
            for arm in ("shuffled", "no_context")
        )
        h0_lower = evaluation["h0_proper"]["wilson_95_lower"]
        criteria["h0_one_step_above_chance"] = bool(
            gate._is_finite_real(h0_lower) and float(h0_lower) > 1.0 / 8.0
        )
        criteria["held_out_invariants_complete"] = (
            _held_out_invariants_complete(data)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    passed = bool(all(criteria.values()))
    return {
        "passed": passed,
        "criteria": criteria,
        "interpretation": (
            "gate_b_passed_demonstrated_depth_application"
            if passed
            else "gate_b_failed_stop_no_capability_conclusion"
        ),
    }


def run_depth_gate(
    config: DepthGateConfig,
    *,
    prerequisites: Mapping[str, Any],
    source_start: Mapping[str, Any],
    source_end_reporter: Callable[[], Mapping[str, Any]],
    source_files: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the demonstrated-depth pp-prop experiment.

    Parameters
    ----------
    config
        Frozen production configuration or an explicitly nonqualifying smoke.
    prerequisites
        Authenticated Gate A and Gate B initialization evidence.
    source_start
        Clean live-Git report captured before the run.
    source_end_reporter
        Zero-argument callback that captures live Git after evaluation.
    source_files
        Frozen model and task source digests.
    environment
        Authenticated accelerator and immutable-image evidence.

    Returns
    -------
    dict
        Strict Gate B report with a recomputed fail-closed qualification.
    """

    if not callable(source_end_reporter):
        raise TypeError("source_end_reporter must be callable")
    start = time.perf_counter()
    try:
        initialization_prerequisite = prerequisites["gate_b_initialization"]
        initialization_admission = _validated_initialization_admission(
            initialization_prerequisite,
            config,
            source_start=source_start,
            environment=environment,
            require_pass=config.qualification_regime == "preregistered_full",
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "Gate B initialization admission authentication failed"
        ) from error
    gate_b_initialization = initialization_admission["initialization"]
    model = LatentWorkspaceModel(
        _model_config(config, batch_size=config.batch_size)
    )
    initial_parameters = legacy._parameter_values(model)
    initial_sha = legacy._array_digest(initial_parameters)
    initial_count = int(
        sum(
            np.asarray(leaf).size
            for value in initial_parameters.values()
            for leaf in jax.tree.leaves(value)
        )
    )
    initial_finite = bool(
        all(
            np.isfinite(np.asarray(leaf)).all()
            for value in initial_parameters.values()
            for leaf in jax.tree.leaves(value)
        )
    )
    try:
        runtime_initialization_matches = bool(
            gate_b_initialization["fresh_model"] is True
            and gate_b_initialization["parameters_finite"] is True
            and gate._is_integer(gate_b_initialization["model_seed"])
            and int(gate_b_initialization["model_seed"]) == config.model_seed
            and gate._json_exact(
                gate_b_initialization["configuration"], asdict(config)
            )
            and gate_b_initialization["parameter_sha256"] == initial_sha
            and gate._is_integer(gate_b_initialization["parameter_count"])
            and int(gate_b_initialization["parameter_count"]) == initial_count
            and _compiler_evidence_complete(gate_b_initialization["compiler"])
            and initial_finite
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "Gate B runtime initialization differs from admission"
        ) from error
    if not runtime_initialization_matches:
        raise RuntimeError("Gate B runtime initialization differs from admission")
    if config.qualification_regime == "preregistered_full" and not (
        _gate_a_prerequisite_complete(prerequisites["gate_a"], config)
        and gate._source_evidence_clean(source_start)
        and gate._gpu_environment_verified(environment)
        and gate._json_exact(
            source_files,
            {
                "latent_workspace_model.py": config.model_source_sha256,
                "latent_workspace_task.py": config.task_source_sha256,
            },
        )
        and _compiler_evidence_complete(gate_b_initialization["compiler"])
    ):
        raise RuntimeError("Gate B authentication failed before training")

    schedule = _build_schedule(config)
    training, schedule_report = _train_depth_gate(model, schedule, config)
    validation = _encode_validation_data(schedule, config)
    evaluation = _evaluate_model(model, validation, config)
    source_end = source_end_reporter()
    if not isinstance(source_end, Mapping):
        raise TypeError("source_end_reporter must return a mapping")
    evaluation["initialization_parameter_sha256"] = initial_sha
    data = {
        "schedule": schedule_report,
        "validation": _validation_data_report(validation),
        "target_contract": _target_contract_report(),
        "controls": _control_report(validation, config),
        "held_out_invariants": _held_out_invariants_report(
            schedule, validation, evaluation
        ),
    }
    report = {
        "schema_version": GATE_B_SCHEMA_VERSION,
        "control": GATE_B_CONTROL,
        "qualification_regime": config.qualification_regime,
        "learner": _learner_report(config),
        "config": asdict(config),
        "prerequisites": {
            "gate_a": dict(prerequisites["gate_a"]),
            "gate_b_initialization": dict(initialization_prerequisite),
        },
        "data": data,
        "training": training,
        "evaluation": evaluation,
        "source_start": dict(source_start),
        "source_end": dict(source_end),
        "source_files": dict(source_files),
        "environment": dict(environment),
    }
    report["qualification"] = _qualification_report(report, config=config)
    report["total_wall_seconds"] = time.perf_counter() - start
    return report


def _source_files_report(config: DepthGateConfig) -> dict[str, str]:
    source_directory = Path(__file__).resolve().parent
    paths = {
        "latent_workspace_model.py": source_directory / "latent_workspace_model.py",
        "latent_workspace_task.py": source_directory / "latent_workspace_task.py",
    }
    result = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    expected = {
        "latent_workspace_model.py": config.model_source_sha256,
        "latent_workspace_task.py": config.task_source_sha256,
    }
    if config.qualification_regime == "preregistered_full" and result != expected:
        raise RuntimeError("Gate B model or task source digest changed")
    return result


def write_artifact(value: Mapping[str, Any], path: str | Path) -> Path:
    """Write one deterministic, standards-compliant Gate B artifact.

    Parameters
    ----------
    value
        JSON-compatible top-level mapping. NaN and infinity are rejected.
    path
        Final artifact path.

    Returns
    -------
    pathlib.Path
        Final artifact path after an atomic replacement.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("gate_b_init", "formal_gate_b"),
        required=True,
    )
    parser.add_argument("--gate-a-result", type=Path, required=True)
    parser.add_argument("--gate-a-manifest", type=Path, required=True)
    parser.add_argument("--gate-b-init-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one fixed authenticated Gate B target.

    Parameters
    ----------
    argv
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero after a complete artifact is written. Scientific failure remains
        encoded in the artifact for the authenticated launcher to sign.
    """

    from examples.pp_prop import latent_workspace_binding_gate_launcher as launcher

    args = _parser().parse_args(argv)
    if args.target == "gate_b_init" and args.gate_b_init_manifest is not None:
        raise ValueError("gate_b_init target rejects an initialization manifest")
    if args.target == "formal_gate_b" and args.gate_b_init_manifest is None:
        raise ValueError("formal_gate_b target requires an initialization manifest")

    repo_root = Path(__file__).resolve().parents[2]
    launch_config = launcher.LaunchConfig(
        target=args.target,
        repo_root=repo_root,
        output_dir=args.output.resolve().parent,
    )
    gate_a_paths = launcher._gate_a_artifact_paths(launch_config)
    if (
        args.gate_a_result.resolve() != gate_a_paths.result.resolve()
        or args.gate_a_manifest.resolve() != gate_a_paths.manifest.resolve()
    ):
        raise ValueError("Gate B target requires the fixed Gate A artifact paths")

    source_start = gate._source_report()
    environment = gate._environment_report()
    gate._require_authenticated_gpu_launch(source_start, environment)
    head = str(source_start["commit"])
    expected_paths = launcher.target_paths(launch_config, head, args.target)
    if args.output.resolve() != expected_paths.result.resolve():
        raise ValueError("Gate B target requires the fixed output path")
    if args.target == "formal_gate_b":
        expected_init = launcher.target_paths(
            launch_config, head, "gate_b_init"
        ).manifest
        if args.gate_b_init_manifest.resolve() != expected_init.resolve():
            raise ValueError("formal_gate_b requires the fixed initialization manifest")

    gate_a = launcher._load_gate_a_prerequisite(launch_config)
    config = DepthGateConfig()
    source_files = _source_files_report(config)
    if args.target == "gate_b_init":
        model = LatentWorkspaceModel(
            _model_config(config, batch_size=config.batch_size)
        )
        result = _gate_b_initialization_report(
            model,
            config,
            gate_a=gate_a,
            source_start=source_start,
            source_end_reporter=gate._source_report,
            source_files=source_files,
            environment=environment,
        )
    else:
        initialization = launcher._load_gate_b_init_manifest(
            launch_config,
            head=head,
            image_id=str(environment["image_digest"]),
        )
        result = run_depth_gate(
            config,
            prerequisites={
                "gate_a": gate_a,
                "gate_b_initialization": initialization,
            },
            source_start=source_start,
            source_end_reporter=gate._source_report,
            source_files=source_files,
            environment=environment,
        )
    destination = write_artifact(result, args.output)
    print(destination)
    print(json.dumps(result["qualification"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
