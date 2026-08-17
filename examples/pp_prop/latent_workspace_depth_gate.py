"""Preregistered demonstrated-depth Gate B for Example 21."""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Mapping

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


def _qualification_report(
    report: Mapping[str, Any],
    *,
    config: DepthGateConfig,
) -> dict[str, Any]:
    criteria = {name: False for name in _QUALIFICATION_CRITERIA}
    return {
        "passed": False,
        "criteria": criteria,
        "interpretation": "gate_b_failed_stop_no_capability_conclusion",
    }
