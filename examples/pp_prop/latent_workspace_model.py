"""Recurrent spiking ARC workspace for Example 21.

The model deliberately contains only repository-native mechanisms: BrainPy
LIF neurons, exponential current synapses, BrainTrace dense and sparse linear
operators, and a pp-prop eligibility-trace coordinate.  ARC row events drive a
slow feed-forward synapse.  After the query, exactly-zero event vectors leave
the recurrent LIF population to evolve for up to 32 latent steps.

The color head is a compact CP factorization.  It emits row, column, and color
factors while the network is running and expands them to independent
``30 x 30 x 10`` logits only at a requested checkpoint.  This avoids a dense
``readout_width x 9000`` parameter matrix and avoids materializing 9,000 logits
at every context row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, NamedTuple

import brainpy.state as bpstate
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import braintrace

MAX_GRID_SIZE = 30
COLOR_COUNT = 10
NEURONS_PER_SLOT = 64


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive non-boolean integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive non-boolean integer")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative non-boolean integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative non-boolean integer")
    return result


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return result


@dataclass(frozen=True)
class ModelConfig:
    """Configure the ARC recurrent spiking workspace.

    Parameters
    ----------
    input_width : int
        Width of one encoded ARC row event.
    batch_size : int, default=1
        Native batch dimension used by the BrainState model.
    neuron_count : int, default=2048
        Number of physical LIF neurons.  It must be divisible by 64 so slot
        controls always refer to exact 64-neuron ranges.
    recurrent_edges : int, default=16384
        Exact number of directed non-self recurrent edges.
    max_latent_steps : int, default=32
        Maximum allowed zero-input recurrent steps.
    readout_width : int, default=128
        Shared low-rank readout bottleneck.
    color_rank : int, default=16
        CP rank of the spatial color-logit tensor.
    membrane_tau_ms, feedforward_tau_ms, recurrent_tau_ms : float
        Physical time constants for neurons and both synapses.
    time_step_ms : float, default=1.0
        Duration represented by one event or latent step.
    input_gain, recurrent_gain : float
        Initialization gains for the feed-forward and recurrent projections.
    trace_decay : float, default=0.9
        pp-prop eligibility-trace decay in ``[0, 1)``.
    event_valid_index : int, default=0
        Row-event channel whose one means a context row advances state.  Latent
        steps use a separate advance gate, keeping their external vector
        exactly zero.
    seed : int, default=2108
        Seed for all topology and parameter randomness.  Random values are
        drawn exclusively through :mod:`brainstate.random`.
    sparse_backend : str, optional
        Optional ``brainevent.CSR`` execution backend.

    Examples
    --------
    .. code-block:: python

        >>> config = ModelConfig(input_width=828)
        >>> (config.neuron_count, config.recurrent_edges, config.slot_count)
        (2048, 16384, 32)
    """

    input_width: int
    batch_size: int = 1
    neuron_count: int = 2048
    recurrent_edges: int = 16384
    max_latent_steps: int = 32
    readout_width: int = 128
    color_rank: int = 16
    membrane_tau_ms: float = 20.0
    feedforward_tau_ms: float = 40.0
    recurrent_tau_ms: float = 10.0
    time_step_ms: float = 1.0
    input_gain: float = 4.0
    recurrent_gain: float = 0.8
    trace_decay: float = 0.9
    event_valid_index: int = 0
    seed: int = 2108
    sparse_backend: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_width",
            "batch_size",
            "neuron_count",
            "recurrent_edges",
            "max_latent_steps",
            "readout_width",
            "color_rank",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        object.__setattr__(
            self,
            "event_valid_index",
            _nonnegative_integer(self.event_valid_index, "event_valid_index"),
        )
        if self.event_valid_index >= self.input_width:
            raise ValueError("event_valid_index must be smaller than input_width")
        for name in (
            "membrane_tau_ms",
            "feedforward_tau_ms",
            "recurrent_tau_ms",
            "time_step_ms",
            "input_gain",
            "recurrent_gain",
        ):
            object.__setattr__(self, name, _positive_real(getattr(self, name), name))
        if self.neuron_count % NEURONS_PER_SLOT:
            raise ValueError(
                f"neuron_count must be divisible by {NEURONS_PER_SLOT} for exact "
                "slot ablation"
            )
        capacity = self.neuron_count * (self.neuron_count - 1)
        if self.recurrent_edges > capacity:
            raise ValueError(
                f"recurrent_edges {self.recurrent_edges} exceeds no-self capacity "
                f"{capacity}"
            )
        if isinstance(self.trace_decay, (bool, np.bool_)) or not isinstance(
            self.trace_decay, Real
        ):
            raise TypeError("trace_decay must be a finite real scalar in [0, 1)")
        trace_decay = float(self.trace_decay)
        if not math.isfinite(trace_decay) or not 0.0 <= trace_decay < 1.0:
            raise ValueError("trace_decay must be a finite real scalar in [0, 1)")
        object.__setattr__(self, "trace_decay", trace_decay)
        if self.sparse_backend is not None and not isinstance(self.sparse_backend, str):
            raise TypeError("sparse_backend must be a string or None")

    @property
    def slot_count(self) -> int:
        """Return the number of exact 64-neuron analysis slots."""
        return self.neuron_count // NEURONS_PER_SLOT

    @property
    def compact_output_width(self) -> int:
        """Return the width of the factorized ARC output vector."""
        return compact_output_width(self.color_rank)


@dataclass(frozen=True)
class SparseTopology:
    """Hold one deterministic directed sparse topology.

    Parameters
    ----------
    rows, columns : numpy.ndarray
        Int32 post- and presynaptic endpoint arrays.
    values : numpy.ndarray
        Float32 initial edge values before physical current units are attached.
    neuron_count : int
        Square adjacency dimension.

    Examples
    --------
    .. code-block:: python

        >>> topology = build_sparse_topology(64, 128, seed=1)
        >>> (topology.edge_count, bool((topology.rows == topology.columns).any()))
        (128, False)
    """

    rows: NDArray[np.int32]
    columns: NDArray[np.int32]
    values: NDArray[np.float32]
    neuron_count: int

    @property
    def edge_count(self) -> int:
        """Return the exact number of stored directed edges."""
        return int(self.values.size)


@dataclass(frozen=True)
class ArcLogits:
    """Hold expanded ARC shape and cell-color logits.

    Parameters
    ----------
    height, width : jax.Array
        Logits whose final dimension has length 30 and indexes sizes 1--30.
    colors : jax.Array
        Color logits with trailing shape ``(30, 30, 10)``.
    """

    height: jax.Array
    width: jax.Array
    colors: jax.Array


@dataclass(frozen=True)
class ModelStateSnapshot:
    """Hold an exact copy of every non-parameter model state.

    Parameters
    ----------
    entries : tuple
        Pairs of BrainState graph paths and copied state pytrees.
    batch_size, neuron_count : int
        Shape identity used to reject restoration into another configuration.
    """

    entries: tuple[tuple[tuple[Any, ...], Any], ...]
    batch_size: int
    neuron_count: int


@dataclass(frozen=True)
class ContextCheckpoint:
    """Describe the query-terminal state before latent computation.

    Parameters
    ----------
    compact_logits : jax.Array
        Compact output shaped ``(batch, compact_output_width)``.
    spikes, voltage : jax.Array
        Query-terminal spikes and voltage shaped ``(batch, neurons)``.  Voltage
        is stored as the numeric value in millivolts.
    feedforward_current, recurrent_current : jax.Array
        Separate Expon synaptic-current states shaped ``(batch, neurons)`` and
        stored as numeric values in milliamps.
    snapshot : ModelStateSnapshot
        Restorable state at checkpoint zero.
    context_steps : int
        Number of valid ARC row events executed.
    """

    compact_logits: jax.Array
    spikes: jax.Array
    voltage: jax.Array
    feedforward_current: jax.Array
    recurrent_current: jax.Array
    snapshot: ModelStateSnapshot
    context_steps: int


@dataclass(frozen=True)
class ModelTrajectory:
    """Hold checkpoint zero and a continuous zero-input latent trajectory.

    Parameters
    ----------
    compact_logits : jax.Array
        Factorized outputs shaped ``(steps + 1, batch, compact_width)``.
    spikes, voltage : jax.Array
        Physical trajectories shaped ``(steps + 1, batch, neurons)``.
    feedforward_current, recurrent_current : jax.Array
        Separate Expon current trajectories shaped
        ``(steps + 1, batch, neurons)`` in milliamps.
    zero_inputs : jax.Array
        Exactly-zero external inputs used for latent updates, shaped
        ``(steps, batch, input_width)``.
    color_rank : int
        Rank needed to expand compact color factors.
    """

    compact_logits: jax.Array
    spikes: jax.Array
    voltage: jax.Array
    feedforward_current: jax.Array
    recurrent_current: jax.Array
    zero_inputs: jax.Array
    color_rank: int

    @property
    def latent_steps(self) -> int:
        """Return the number of recurrent updates after checkpoint zero."""
        return int(self.compact_logits.shape[0] - 1)

    @property
    def expanded(self) -> ArcLogits:
        """Expand every checkpoint to full ARC logits."""
        return expand_compact_logits(self.compact_logits, self.color_rank)

    def at_effort(self, effort: int) -> ArcLogits:
        """Return expanded logits at one latent-effort checkpoint.

        Parameters
        ----------
        effort : int
            Number of recurrent latent updates, from zero through
            :attr:`latent_steps`.

        Returns
        -------
        ArcLogits
            Batched logits at the selected checkpoint.
        """
        effort = _nonnegative_integer(effort, "effort")
        if effort > self.latent_steps:
            raise ValueError(
                f"effort {effort} exceeds trajectory length {self.latent_steps}"
            )
        return expand_compact_logits(self.compact_logits[effort], self.color_rank)


@dataclass(frozen=True)
class PackedTrajectory:
    """Hold outputs recorded after every tick of one fixed packed stream.

    Parameters
    ----------
    compact_logits : jax.Array
        Compact outputs shaped ``(time, batch, compact_width)``.
    spikes, voltage : jax.Array
        Physical states shaped ``(time, batch, neurons)``.
    feedforward_current, recurrent_current : jax.Array
        Separate Expon current states shaped ``(time, batch, neurons)`` and
        represented numerically in milliamps.
    color_rank : int
        Rank needed to expand compact color factors.
    """

    compact_logits: jax.Array
    spikes: jax.Array
    voltage: jax.Array
    feedforward_current: jax.Array
    recurrent_current: jax.Array
    color_rank: int

    @property
    def expanded(self) -> ArcLogits:
        """Expand all packed ticks to full ARC logits."""
        return expand_compact_logits(self.compact_logits, self.color_rank)


@dataclass(frozen=True)
class SelectedPackedTrajectory:
    """Hold only requested per-example checkpoints from a packed stream.

    Parameters
    ----------
    selected_indices : jax.Array
        Strictly increasing stream indices shaped ``(checkpoints, batch)``.
    compact_logits : jax.Array
        Compact outputs shaped ``(checkpoints, batch, compact_width)``.
    spikes, voltage : jax.Array
        Selected physical states shaped ``(checkpoints, batch, neurons)``.
    feedforward_current, recurrent_current : jax.Array
        Selected Expon current states shaped
        ``(checkpoints, batch, neurons)`` in milliamps.
    color_rank : int
        Rank needed to expand compact color factors.
    """

    selected_indices: jax.Array
    compact_logits: jax.Array
    spikes: jax.Array
    voltage: jax.Array
    feedforward_current: jax.Array
    recurrent_current: jax.Array
    color_rank: int

    @property
    def expanded(self) -> ArcLogits:
        """Expand selected compact checkpoints to full ARC logits."""
        return expand_compact_logits(self.compact_logits, self.color_rank)


@dataclass(frozen=True)
class SequenceResult:
    """Pair a context checkpoint with its continuous latent trajectory.

    Parameters
    ----------
    context : ContextCheckpoint
        Query-terminal checkpoint before recurrent latent updates.
    trajectory : ModelTrajectory
        Checkpoint zero followed by all zero-input updates.
    """

    context: ContextCheckpoint
    trajectory: ModelTrajectory

    @property
    def feedforward_current(self) -> jax.Array:
        """Return checkpoint-zero-through-latent feed-forward current."""
        return self.trajectory.feedforward_current

    @property
    def recurrent_current(self) -> jax.Array:
        """Return checkpoint-zero-through-latent recurrent current."""
        return self.trajectory.recurrent_current


class ArcLossComponents(NamedTuple):
    """Hold scalar total, height, width, and valid-cell color losses.

    Parameters
    ----------
    total, height, width, colors : jax.Array
        Scalar mean losses.  ``total`` contains the configured weighted sum.
    """

    total: jax.Array
    height: jax.Array
    width: jax.Array
    colors: jax.Array


def compact_output_width(color_rank: int) -> int:
    """Return compact output width for a CP color rank.

    Parameters
    ----------
    color_rank : int
        Number of color-tensor factors.

    Returns
    -------
    int
        Two 30-way shape heads plus rank times 30 rows, 30 columns, and 10
        colors.
    """
    color_rank = _positive_integer(color_rank, "color_rank")
    return 2 * MAX_GRID_SIZE + color_rank * (2 * MAX_GRID_SIZE + COLOR_COUNT)


def build_sparse_topology(
    neuron_count: int,
    edge_count: int,
    *,
    seed: int,
    recurrent_gain: float = 0.8,
) -> SparseTopology:
    """Build a deterministic exact-edge, no-self sparse topology.

    Each row receives either ``floor(E/N)`` or ``ceil(E/N)`` edges.  A random
    affine permutation of the ``N - 1`` legal offsets selects distinct
    presynaptic endpoints in that row.  Starts, multipliers, and weights are
    sampled with :class:`brainstate.random.RandomState`; NumPy is used only for
    static host-side indexing and validation.

    Parameters
    ----------
    neuron_count : int
        Number of recurrent neurons, at least two.
    edge_count : int
        Exact directed edge count, no larger than ``N * (N - 1)``.
    seed : int
        Nonnegative BrainState random seed.
    recurrent_gain : float, default=0.8
        Standard-deviation gain relative to square-root mean degree.

    Returns
    -------
    SparseTopology
        Sorted CSR-compatible endpoints and float32 values.
    """
    neuron_count = _positive_integer(neuron_count, "neuron_count")
    edge_count = _positive_integer(edge_count, "edge_count")
    seed = _nonnegative_integer(seed, "seed")
    recurrent_gain = _positive_real(recurrent_gain, "recurrent_gain")
    if neuron_count < 2:
        raise ValueError("neuron_count must be at least 2 for no-self topology")
    capacity = neuron_count * (neuron_count - 1)
    if edge_count > capacity:
        raise ValueError(f"edge_count {edge_count} exceeds no-self capacity {capacity}")

    random = brainstate.random.RandomState(seed)
    modulus = neuron_count - 1
    starts = np.asarray(
        random.randint(0, modulus, size=(neuron_count,), dtype=jnp.int32),
        dtype=np.int64,
    )
    if modulus == 1:
        multipliers = np.ones(neuron_count, dtype=np.int64)
    else:
        multipliers = np.asarray(
            random.randint(1, modulus, size=(neuron_count,), dtype=jnp.int32),
            dtype=np.int64,
        )
        # Project every draw onto the finite set of units modulo N-1.  This is
        # deterministic post-processing of BrainState randomness, not a second
        # random source.
        while_indices = np.flatnonzero(np.gcd(multipliers, modulus) != 1)
        while while_indices.size:
            multipliers[while_indices] = (
                multipliers[while_indices] % (modulus - 1)
            ) + 1
            while_indices = np.flatnonzero(np.gcd(multipliers, modulus) != 1)

    base_degree, remainder = divmod(edge_count, neuron_count)
    degrees = np.full(neuron_count, base_degree, dtype=np.int64)
    degrees[:remainder] += 1
    maximum_degree = int(degrees.max())
    positions = np.arange(maximum_degree, dtype=np.int64)[None, :]
    offsets = (starts[:, None] + multipliers[:, None] * positions) % modulus + 1
    rows_matrix = np.broadcast_to(
        np.arange(neuron_count, dtype=np.int64)[:, None], offsets.shape
    )
    valid = positions < degrees[:, None]
    rows = rows_matrix[valid]
    columns = (rows_matrix + offsets)[valid] % neuron_count
    order = np.lexsort((columns, rows))
    rows = rows[order].astype(np.int32, copy=False)
    columns = columns[order].astype(np.int32, copy=False)
    value_scale = recurrent_gain / math.sqrt(edge_count / neuron_count)
    values = np.asarray(random.randn(edge_count), dtype=np.float32) * value_scale
    values = values[order].astype(np.float32, copy=False)

    if rows.size != edge_count or np.any(rows == columns):
        raise RuntimeError("sparse topology construction violated its exact contract")
    flat = rows.astype(np.int64) * neuron_count + columns.astype(np.int64)
    if np.unique(flat).size != edge_count:
        raise RuntimeError("sparse topology construction produced duplicate edges")
    for array in (rows, columns, values):
        array.setflags(write=False)
    return SparseTopology(rows, columns, values, neuron_count)


def _indptr_from_rows(rows: NDArray[np.int32], neuron_count: int) -> NDArray[np.int32]:
    indptr = np.zeros(neuron_count + 1, dtype=np.int64)
    np.add.at(indptr, rows.astype(np.int64) + 1, 1)
    return np.cumsum(indptr).astype(np.int32)


def _topology_to_csr(topology: SparseTopology, backend: str | None) -> Any:
    import brainevent

    return brainevent.CSR(
        jnp.asarray(topology.values),
        jnp.asarray(topology.columns),
        jnp.asarray(_indptr_from_rows(topology.rows, topology.neuron_count)),
        shape=(topology.neuron_count, topology.neuron_count),
        backend=backend,
    )


def _split_compact_logits(
    compact: jax.Array, color_rank: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    expected = compact_output_width(color_rank)
    if compact.ndim < 1 or compact.shape[-1] != expected:
        raise ValueError(
            f"compact logits must have final width {expected}, got {compact.shape}"
        )
    cursor = 0
    height = compact[..., cursor : cursor + MAX_GRID_SIZE]
    cursor += MAX_GRID_SIZE
    width = compact[..., cursor : cursor + MAX_GRID_SIZE]
    cursor += MAX_GRID_SIZE
    row_count = color_rank * MAX_GRID_SIZE
    row = compact[..., cursor : cursor + row_count].reshape(
        *compact.shape[:-1], color_rank, MAX_GRID_SIZE
    )
    cursor += row_count
    column = compact[..., cursor : cursor + row_count].reshape(
        *compact.shape[:-1], color_rank, MAX_GRID_SIZE
    )
    cursor += row_count
    color = compact[..., cursor:].reshape(*compact.shape[:-1], color_rank, COLOR_COUNT)
    return height, width, row, column, color


def expand_compact_logits(compact: jax.Array, color_rank: int) -> ArcLogits:
    """Expand factorized outputs to full ARC logits.

    Parameters
    ----------
    compact : jax.Array
        Array with any leading dimensions and final width given by
        :func:`compact_output_width`.
    color_rank : int
        CP rank encoded by ``compact``.

    Returns
    -------
    ArcLogits
        Height and width logits plus a color array with trailing shape
        ``(30, 30, 10)``.
    """
    color_rank = _positive_integer(color_rank, "color_rank")
    compact = jnp.asarray(compact)
    height, width, row, column, color = _split_compact_logits(compact, color_rank)
    color_logits = jnp.einsum(
        "...ri,...rj,...rc->...ijc", row, column, color
    ) / math.sqrt(color_rank)
    return ArcLogits(height=height, width=width, colors=color_logits)


def arc_loss_components(
    compact_logits: jax.Array,
    target_height: jax.Array,
    target_width: jax.Array,
    target_colors: jax.Array,
    *,
    color_rank: int,
    shape_weight: float = 1.0,
    color_weight: float = 1.0,
) -> ArcLossComponents:
    """Compute terminal ARC height, width, and valid-cell cross entropy.

    Parameters
    ----------
    compact_logits : jax.Array
        Batched compact model output shaped ``(batch, compact_width)``.
    target_height, target_width : jax.Array
        Integer sizes in 1--30, shaped ``(batch,)``.
    target_colors : jax.Array
        Integer padded color grid shaped ``(batch, 30, 30)``.  Only cells
        inside each target shape contribute.
    color_rank : int
        CP rank of ``compact_logits``.
    shape_weight, color_weight : float, default=1.0
        Nonnegative component weights.

    Returns
    -------
    ArcLossComponents
        Scalar mean losses suitable for a terminal pp-prop step loss.
    """
    total, height_loss, width_loss, color_loss = _arc_loss_vectors(
        compact_logits,
        target_height,
        target_width,
        target_colors,
        color_rank=color_rank,
        shape_weight=shape_weight,
        color_weight=color_weight,
    )
    return ArcLossComponents(
        total.mean(), height_loss.mean(), width_loss.mean(), color_loss.mean()
    )


def arc_loss_per_example(
    compact_logits: jax.Array,
    target_height: jax.Array,
    target_width: jax.Array,
    target_colors: jax.Array,
    *,
    color_rank: int,
    shape_weight: float = 1.0,
    color_weight: float = 1.0,
) -> jax.Array:
    """Return one terminal ARC loss for each batch element.

    This form lets a packed pp-prop stream multiply losses by a per-time,
    per-example terminal gate before reducing.  Each example's cell loss is
    normalized by its own target area, so large grids do not receive more
    weight merely because they contain more cells.

    Parameters
    ----------
    compact_logits : jax.Array
        Batched compact model output shaped ``(batch, compact_width)``.
    target_height, target_width : jax.Array
        Integer sizes in 1--30, shaped ``(batch,)``.
    target_colors : jax.Array
        Integer padded colors shaped ``(batch, 30, 30)``.
    color_rank : int
        CP rank encoded by ``compact_logits``.
    shape_weight, color_weight : float, default=1.0
        Nonnegative component weights.

    Returns
    -------
    jax.Array
        Loss vector shaped ``(batch,)``.
    """
    total, _, _, _ = _arc_loss_vectors(
        compact_logits,
        target_height,
        target_width,
        target_colors,
        color_rank=color_rank,
        shape_weight=shape_weight,
        color_weight=color_weight,
    )
    return total


def _arc_loss_vectors(
    compact_logits: jax.Array,
    target_height: jax.Array,
    target_width: jax.Array,
    target_colors: jax.Array,
    *,
    color_rank: int,
    shape_weight: float,
    color_weight: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    compact_logits = jnp.asarray(compact_logits)
    if compact_logits.ndim != 2:
        raise ValueError(
            "compact_logits must have shape (batch, compact_width), got "
            f"{compact_logits.shape}"
        )
    batch_size = compact_logits.shape[0]
    target_height = jnp.asarray(target_height)
    target_width = jnp.asarray(target_width)
    target_colors = jnp.asarray(target_colors)
    if target_height.shape != (batch_size,) or target_width.shape != (batch_size,):
        raise ValueError("target_height and target_width must each have shape (batch,)")
    if target_colors.shape != (batch_size, MAX_GRID_SIZE, MAX_GRID_SIZE):
        raise ValueError(
            "target_colors must have shape "
            f"({batch_size}, {MAX_GRID_SIZE}, {MAX_GRID_SIZE})"
        )
    shape_weight = float(shape_weight)
    color_weight = float(color_weight)
    if not math.isfinite(shape_weight) or shape_weight < 0.0:
        raise ValueError("shape_weight must be finite and nonnegative")
    if not math.isfinite(color_weight) or color_weight < 0.0:
        raise ValueError("color_weight must be finite and nonnegative")

    logits = expand_compact_logits(compact_logits, color_rank)
    height_indices = target_height.astype(jnp.int32) - 1
    width_indices = target_width.astype(jnp.int32) - 1
    height_loss = -jnp.take_along_axis(
        jax.nn.log_softmax(logits.height, axis=-1),
        height_indices[:, None],
        axis=-1,
    )[:, 0]
    width_loss = -jnp.take_along_axis(
        jax.nn.log_softmax(logits.width, axis=-1),
        width_indices[:, None],
        axis=-1,
    )[:, 0]
    color_nll = -jnp.take_along_axis(
        jax.nn.log_softmax(logits.colors, axis=-1),
        target_colors.astype(jnp.int32)[..., None],
        axis=-1,
    )[..., 0]
    rows = jnp.arange(MAX_GRID_SIZE)[None, :, None]
    columns = jnp.arange(MAX_GRID_SIZE)[None, None, :]
    valid = (rows < target_height[:, None, None]) & (
        columns < target_width[:, None, None]
    )
    color_loss = jnp.sum(jnp.where(valid, color_nll, 0.0), axis=(1, 2)) / jnp.maximum(
        jnp.sum(valid, axis=(1, 2)), 1
    )
    total = shape_weight * (height_loss + width_loss) + color_weight * color_loss
    return total, height_loss, width_loss, color_loss


def terminal_arc_loss(
    compact_logits: jax.Array,
    target_height: jax.Array,
    target_width: jax.Array,
    target_colors: jax.Array,
    *,
    color_rank: int,
    shape_weight: float = 1.0,
    color_weight: float = 1.0,
) -> jax.Array:
    """Return scalar terminal ARC loss for pp-prop supervision.

    Parameters
    ----------
    compact_logits : jax.Array
        Batched compact model output shaped ``(batch, compact_width)``.
    target_height, target_width : jax.Array
        Integer sizes in 1--30, shaped ``(batch,)``.
    target_colors : jax.Array
        Integer padded colors shaped ``(batch, 30, 30)``.
    color_rank : int
        CP rank encoded by ``compact_logits``.
    shape_weight, color_weight : float, default=1.0
        Nonnegative component weights.

    Returns
    -------
    jax.Array
        Scalar weighted cross entropy.
    """
    return arc_loss_components(
        compact_logits,
        target_height,
        target_width,
        target_colors,
        color_rank=color_rank,
        shape_weight=shape_weight,
        color_weight=color_weight,
    ).total


def _copy_tree(value: Any) -> Any:
    return jax.tree.map(lambda leaf: jnp.array(leaf, copy=True), value)


class LatentWorkspaceModel(brainstate.nn.Module):
    """BrainPy LIF network with sparse recurrent ARC computation.

    Parameters
    ----------
    config : ModelConfig
        Physical, sparse, readout, and batching configuration.

    Notes
    -----
    ``cell_step`` advances physical state and returns spikes.  ``update`` adds
    the compact ARC head and is therefore the callable compiled by BrainTrace.
    Inference context loops call ``cell_step`` and run the head only at the
    query-terminal checkpoint.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        self.config = config
        self.topology = build_sparse_topology(
            config.neuron_count,
            config.recurrent_edges,
            seed=config.seed,
            recurrent_gain=config.recurrent_gain,
        )
        random = brainstate.random.RandomState(config.seed + 1)

        self.neu = bpstate.LIF(
            config.neuron_count,
            R=1.0 * u.ohm,
            tau=config.membrane_tau_ms * u.ms,
            V_th=1.0 * u.mV,
            V_reset=0.0 * u.mV,
            V_rest=0.0 * u.mV,
            V_initializer=braintools.init.ZeroInit(unit=u.mV),
        )
        input_weights = random.randn(config.input_width, config.neuron_count)
        input_weights = (
            input_weights * (config.input_gain / math.sqrt(config.input_width)) * u.mA
        )
        recurrent_linear = braintrace.nn.SparseLinear(
            _topology_to_csr(self.topology, config.sparse_backend), b_init=None
        )
        recurrent_parameters = dict(recurrent_linear.weight.value)
        recurrent_parameters["weight"] = recurrent_parameters["weight"] * u.mA
        recurrent_linear.weight.value = recurrent_parameters

        self.ff_syn = bpstate.AlignPostProj(
            comm=braintrace.nn.Linear(
                config.input_width,
                config.neuron_count,
                w_init=input_weights,
                b_init=None,
            ),
            syn=bpstate.Expon(
                config.neuron_count,
                tau=config.feedforward_tau_ms * u.ms,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=bpstate.CUBA(scale=1.0),
            post=self.neu,
        )
        self.rec_syn = bpstate.AlignPostProj(
            comm=recurrent_linear,
            syn=bpstate.Expon(
                config.neuron_count,
                tau=config.recurrent_tau_ms * u.ms,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=bpstate.CUBA(scale=1.0),
            post=self.neu,
        )

        bottleneck_weights = random.randn(config.neuron_count, config.readout_width)
        bottleneck_weights = bottleneck_weights / math.sqrt(config.neuron_count)
        shape_weights = random.randn(config.readout_width, MAX_GRID_SIZE)
        shape_weights = shape_weights / math.sqrt(config.readout_width)
        factor_width = config.color_rank * (2 * MAX_GRID_SIZE + COLOR_COUNT)
        factor_weights = random.randn(config.readout_width, factor_width)
        factor_weights = factor_weights / math.sqrt(config.readout_width)
        self.readout_projection = braintrace.nn.Linear(
            config.neuron_count,
            config.readout_width,
            w_init=bottleneck_weights,
            b_init=braintools.init.ZeroInit(),
        )
        self.height_head = braintrace.nn.Linear(
            config.readout_width,
            MAX_GRID_SIZE,
            w_init=shape_weights,
            b_init=braintools.init.ZeroInit(),
        )
        self.width_head = braintrace.nn.Linear(
            config.readout_width,
            MAX_GRID_SIZE,
            w_init=random.randn(config.readout_width, MAX_GRID_SIZE)
            / math.sqrt(config.readout_width),
            b_init=braintools.init.ZeroInit(),
        )
        self.color_factor_head = braintrace.nn.Linear(
            config.readout_width,
            factor_width,
            w_init=factor_weights,
            b_init=braintools.init.ZeroInit(),
        )
        brainstate.nn.init_all_states(self, batch_size=config.batch_size)

    @property
    def neuron_count(self) -> int:
        """Return the physical LIF population size."""
        return self.config.neuron_count

    @property
    def recurrent_edge_count(self) -> int:
        """Return the instantiated recurrent edge count."""
        return self.topology.edge_count

    @property
    def slot_count(self) -> int:
        """Return the number of exact 64-neuron slots."""
        return self.config.slot_count

    @property
    def spikes(self) -> jax.Array:
        """Return current binary LIF spikes shaped ``(batch, neurons)``."""
        return jnp.asarray(self.neu.get_spike())

    @property
    def voltage(self) -> jax.Array:
        """Return current membrane voltage numeric values in millivolts."""
        return jnp.asarray(self.neu.V.value.to_decimal(u.mV))

    @property
    def feedforward_current(self) -> jax.Array:
        """Return the feed-forward Expon state numerically in milliamps."""
        return jnp.asarray(self.ff_syn.syn.g.value.to_decimal(u.mA))

    @property
    def recurrent_current(self) -> jax.Array:
        """Return the recurrent Expon state numerically in milliamps."""
        return jnp.asarray(self.rec_syn.syn.g.value.to_decimal(u.mA))

    def reset_state(self, batch_size: int | None = None, **_: object) -> None:
        """Reset every inference state without modifying any parameter.

        Parameters
        ----------
        batch_size : int, optional
            If supplied, it must match the configured native batch size.
        **_ : object
            Extra framework reset options, accepted and ignored.
        """
        if batch_size is not None and batch_size != self.config.batch_size:
            raise ValueError(
                f"batch_size {batch_size} does not match configured "
                f"{self.config.batch_size}"
            )
        self.neu.reset_state(batch_size=self.config.batch_size)
        self.ff_syn.syn.reset_state(batch_size=self.config.batch_size)
        self.rec_syn.syn.reset_state(batch_size=self.config.batch_size)

    def snapshot_state(self) -> ModelStateSnapshot:
        """Copy every hidden state while excluding all parameters.

        Returns
        -------
        ModelStateSnapshot
            Exact restorable state keyed by BrainState graph path.
        """
        entries = tuple(
            (tuple(path), _copy_tree(state.value))
            for path, state in self.states(brainstate.HiddenState).items()
        )
        return ModelStateSnapshot(
            entries=entries,
            batch_size=self.config.batch_size,
            neuron_count=self.config.neuron_count,
        )

    def restore_state(self, snapshot: ModelStateSnapshot) -> None:
        """Restore a compatible hidden-state snapshot exactly.

        Parameters
        ----------
        snapshot : ModelStateSnapshot
            Snapshot produced by this model configuration.
        """
        if not isinstance(snapshot, ModelStateSnapshot):
            raise TypeError("snapshot must be a ModelStateSnapshot")
        if (
            snapshot.batch_size != self.config.batch_size
            or snapshot.neuron_count != self.config.neuron_count
        ):
            raise ValueError("snapshot configuration does not match this model")
        states = self.states(brainstate.HiddenState)
        expected_paths = tuple(tuple(path) for path in states)
        actual_paths = tuple(path for path, _ in snapshot.entries)
        if actual_paths != expected_paths:
            raise ValueError("snapshot hidden-state paths do not match this model")
        for (path, value), state in zip(snapshot.entries, states.values(), strict=True):
            if tuple(path) not in expected_paths:
                raise ValueError("snapshot contains an unknown state path")
            state.value = _copy_tree(value)

    def ablate_slot(self, slot_index: int) -> tuple[jax.Array, jax.Array]:
        """Zero one exact 64-neuron voltage/spike slice without changing weights.

        Parameters
        ----------
        slot_index : int
            Zero-based slot index in ``[0, slot_count)``.
        Returns
        -------
        tuple of jax.Array
            Post-ablation voltage and derived spikes.  The selected 64-neuron
            slice is exactly zero in both arrays because BrainPy LIF computes
            spikes directly from its voltage state.
        """
        slot_index = _nonnegative_integer(slot_index, "slot_index")
        if slot_index >= self.slot_count:
            raise ValueError(f"slot_index {slot_index} outside [0, {self.slot_count})")
        slots = jnp.full((self.config.batch_size,), slot_index, dtype=jnp.int32)
        enabled = jnp.ones((self.config.batch_size,), dtype=jnp.bool_)
        return self.mask_slots(slots, enabled)

    def mask_slots(
        self, slot_indices: jax.Array, enabled: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Apply a jittable per-example 64-neuron state-ablation gate.

        Parameters
        ----------
        slot_indices : jax.Array
            Integer slot index for each batch element, shaped ``(batch,)``.
        enabled : jax.Array
            Boolean gate per batch element.  False elements remain unchanged.

        Returns
        -------
        tuple of jax.Array
            Post-mask numeric voltage and derived spikes, each shaped
            ``(batch, neurons)``.

        Notes
        -----
        :class:`brainpy.state.LIF` has no separate spike state:
        ``get_spike()`` derives spikes from ``V`` on demand.  Masking ``V``
        therefore masks the exact spike vector consumed by ``rec_syn`` on the
        next step.  Callers that accept traced indices must validate their
        range before entering a compiled loop; :meth:`ablate_slot` provides the
        checked scalar interface.
        """
        slot_indices = jnp.asarray(slot_indices, dtype=jnp.int32)
        enabled = jnp.asarray(enabled, dtype=jnp.bool_)
        expected = (self.config.batch_size,)
        if slot_indices.shape != expected or enabled.shape != expected:
            raise ValueError(
                f"slot_indices and enabled must each have shape {expected}"
            )
        neuron_slots = jnp.arange(self.config.neuron_count) // NEURONS_PER_SLOT
        selected = neuron_slots[None, :] == slot_indices[:, None]
        keep = ~(selected & enabled[:, None])
        self.neu.V.value = self.neu.V.value * keep
        return self.voltage, self.spikes

    def etrace_config(self) -> braintrace.ETraceConfig:
        """Return the explicit IO-factorized pp-prop coordinate.

        Returns
        -------
        braintrace.ETraceConfig
            Diagonal recurrent pp-prop with the configured scalar trace decay.
        """
        return braintrace.ETraceConfig(
            trace_factorization="io_factorized",
            recurrence_scope="diagonal",
            decay=self.config.trace_decay,
        )

    def compact_readout(self, spikes: jax.Array | None = None) -> jax.Array:
        """Map spikes to compact ARC shape and CP color factors.

        Parameters
        ----------
        spikes : jax.Array, optional
            Spikes shaped ``(batch, neurons)``.  Current LIF spikes are used
            when omitted.

        Returns
        -------
        jax.Array
            Compact logits shaped ``(batch, compact_output_width)``.
        """
        if spikes is None:
            spikes = self.spikes
        if spikes.shape != (self.config.batch_size, self.config.neuron_count):
            raise ValueError(
                "spikes must have shape "
                f"({self.config.batch_size}, {self.config.neuron_count}), got "
                f"{spikes.shape}"
            )
        hidden = jax.nn.gelu(self.readout_projection(spikes))
        return jnp.concatenate(
            (
                self.height_head(hidden),
                self.width_head(hidden),
                self.color_factor_head(hidden),
            ),
            axis=-1,
        )

    def cell_step(
        self, event: jax.Array, advance: jax.Array | None = None
    ) -> jax.Array:
        """Advance one physical row-event or zero-input latent step.

        Parameters
        ----------
        event : jax.Array
            Native batched external input shaped ``(batch, input_width)``.
        advance : jax.Array, optional
            Boolean state-advance gate shaped ``(batch,)``.  When omitted it
            is read from ``event[:, event_valid_index]``.  False rows leave
            voltage and both synaptic currents byte-identical to their prior
            values.  Latent callers pass true while keeping ``event`` zero.

        Returns
        -------
        jax.Array
            Current binary spikes shaped ``(batch, neurons)``.
        """
        expected = (self.config.batch_size, self.config.input_width)
        if event.shape != expected:
            raise ValueError(f"event must have shape {expected}, got {event.shape}")
        if advance is None:
            advance = event[:, self.config.event_valid_index] > 0.5
        else:
            advance = jnp.asarray(advance, dtype=jnp.bool_)
        if advance.shape != (self.config.batch_size,):
            raise ValueError(
                f"advance must have shape ({self.config.batch_size},), got "
                f"{advance.shape}"
            )
        previous_voltage = self.neu.V.value
        previous_feedforward = self.ff_syn.syn.g.value
        previous_recurrent = self.rec_syn.syn.g.value
        with brainstate.environ.context(dt=self.config.time_step_ms * u.ms):
            self.ff_syn(event)
            self.rec_syn(self.neu.get_spike())
            self.neu(0.0 * u.mA)
        gate = advance[:, None]
        self.neu.V.value = u.math.where(gate, self.neu.V.value, previous_voltage)
        self.ff_syn.syn.g.value = u.math.where(
            gate, self.ff_syn.syn.g.value, previous_feedforward
        )
        self.rec_syn.syn.g.value = u.math.where(
            gate, self.rec_syn.syn.g.value, previous_recurrent
        )
        return self.spikes

    def update(self, event: jax.Array, advance: jax.Array | None = None) -> jax.Array:
        """Advance one step and return the compact BrainTrace training output.

        Parameters
        ----------
        event : jax.Array
            Native batched external input shaped ``(batch, input_width)``.
        advance : jax.Array, optional
            Per-example state-advance gate.  See :meth:`cell_step`.

        Returns
        -------
        jax.Array
            Compact ARC logits shaped ``(batch, compact_output_width)``.
        """
        return self.compact_readout(self.cell_step(event, advance))


def _batched_events(model: LatentWorkspaceModel, events: jax.Array) -> jax.Array:
    events = jnp.asarray(events, dtype=jnp.float32)
    if events.ndim == 2:
        if model.config.batch_size != 1:
            raise ValueError(
                "rank-two events are supported only when configured batch_size is 1"
            )
        events = events[:, None, :]
    expected_tail = (model.config.batch_size, model.config.input_width)
    if events.ndim != 3 or events.shape[1:] != expected_tail:
        raise ValueError(
            f"events must have shape (time, {expected_tail[0]}, "
            f"{expected_tail[1]}), got {events.shape}"
        )
    if events.shape[0] < 1:
        raise ValueError("context events must contain at least one valid row")
    return events


def run_context(
    model: LatentWorkspaceModel,
    events: jax.Array,
    *,
    reset: bool = True,
) -> ContextCheckpoint:
    """Execute valid ARC row events through one BrainState compiled loop.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model whose physical state is advanced.
    events : jax.Array
        Valid, padding-trimmed events shaped ``(time, input_width)`` for batch
        one or ``(time, batch, input_width)`` generally.
    reset : bool, default=True
        Reset all inference state before the context loop.

    Returns
    -------
    ContextCheckpoint
        Query-terminal state, compact output, and exact restorable snapshot.
    """
    packed = _batched_events(model, events)
    if reset:
        model.reset_state()

    advance = packed[..., model.config.event_valid_index] > 0.5

    def context_step(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
        event, gate = inputs
        model.cell_step(event, gate)
        return jnp.asarray(0, dtype=jnp.int8)

    brainstate.transform.for_loop(context_step, (packed, advance))
    spikes = model.spikes
    return ContextCheckpoint(
        compact_logits=model.compact_readout(spikes),
        spikes=spikes,
        voltage=model.voltage,
        feedforward_current=model.feedforward_current,
        recurrent_current=model.recurrent_current,
        snapshot=model.snapshot_state(),
        context_steps=int(packed.shape[0]),
    )


def run_packed_stream(
    model: LatentWorkspaceModel,
    events: jax.Array,
    *,
    reset: bool = True,
    advance_gates: jax.Array | None = None,
    ablation_slots: jax.Array | None = None,
    ablation_gates: jax.Array | None = None,
) -> PackedTrajectory:
    """Run a fixed packed context-plus-latent stream and record every tick.

    This is the integration path for batches whose valid contexts have
    different lengths.  The caller packs each sample's valid context, zero
    tail, and 32 additional zero rows into one static tensor, then gathers
    checkpoint zero and later efforts using that sample's own ``query_stop``.
    The recurrent driver is one :func:`brainstate.transform.for_loop`.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model to reset and execute.
    events : jax.Array
        Fixed tensor shaped ``(time, batch, input_width)``.  Rank two is
        accepted for native batch one.
    reset : bool, default=True
        Reset all inference state before execution.
    advance_gates : jax.Array, optional
        Boolean gates shaped ``(time, batch)``.  Omitted gates are derived from
        the row-event valid channel, which freezes context padding.  A packed
        stream containing zero-input latent rows must supply true gates for
        those rows so recurrent state advances while external input stays
        exactly zero.
    ablation_slots : jax.Array, optional
        One checked slot index per batch element, shaped ``(batch,)``.
    ablation_gates : jax.Array, optional
        Boolean pre-step ablation gates shaped ``(time, batch)``.  A gate at
        ``query_stop`` zeros the selected voltage/spike slice after checkpoint
        zero and before the first latent update.

    Returns
    -------
    PackedTrajectory
        Compact outputs, spikes, and voltage after every input tick.
    """
    packed = _batched_events(model, events)
    if reset:
        model.reset_state()
    if advance_gates is None:
        advance = packed[..., model.config.event_valid_index] > 0.5
    else:
        advance = jnp.asarray(advance_gates, dtype=jnp.bool_)
        if advance.shape != (packed.shape[0], model.config.batch_size):
            raise ValueError(
                "advance_gates must have shape "
                f"({packed.shape[0]}, {model.config.batch_size})"
            )
    controlled = ablation_slots is not None or ablation_gates is not None
    if controlled:
        if ablation_slots is None or ablation_gates is None:
            raise ValueError(
                "ablation_slots and ablation_gates must be supplied together"
            )
        raw_slots = np.asarray(ablation_slots)
        if raw_slots.shape != (model.config.batch_size,):
            raise ValueError(
                f"ablation_slots must have shape ({model.config.batch_size},)"
            )
        if not np.issubdtype(raw_slots.dtype, np.integer):
            raise ValueError("ablation_slots must contain integers")
        if np.any(raw_slots < 0) or np.any(raw_slots >= model.slot_count):
            raise ValueError(f"ablation_slots must lie in [0, {model.slot_count})")
        slots = jnp.asarray(raw_slots, dtype=jnp.int32)
        gates = jnp.asarray(ablation_gates, dtype=jnp.bool_)
        if gates.shape != (packed.shape[0], model.config.batch_size):
            raise ValueError(
                "ablation_gates must have shape "
                f"({packed.shape[0]}, {model.config.batch_size})"
            )

        def controlled_step(
            inputs: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            event, advance_gate, ablation_gate = inputs
            model.mask_slots(slots, ablation_gate)
            spikes = model.cell_step(event, advance_gate)
            return (
                model.compact_readout(spikes),
                spikes,
                model.voltage,
                model.feedforward_current,
                model.recurrent_current,
            )

        compact, spikes, voltage, feedforward_current, recurrent_current = (
            brainstate.transform.for_loop(controlled_step, (packed, advance, gates))
        )
    else:

        def packed_step(
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            event, gate = inputs
            spikes = model.cell_step(event, gate)
            return (
                model.compact_readout(spikes),
                spikes,
                model.voltage,
                model.feedforward_current,
                model.recurrent_current,
            )

        compact, spikes, voltage, feedforward_current, recurrent_current = (
            brainstate.transform.for_loop(packed_step, (packed, advance))
        )
    return PackedTrajectory(
        compact_logits=compact,
        spikes=spikes,
        voltage=voltage,
        feedforward_current=feedforward_current,
        recurrent_current=recurrent_current,
        color_rank=model.config.color_rank,
    )


def run_selected_packed_stream(
    model: LatentWorkspaceModel,
    events: jax.Array,
    selected_indices: jax.Array,
    *,
    reset: bool = True,
    advance_gates: jax.Array | None = None,
    ablation_slots: jax.Array | None = None,
    ablation_gates: jax.Array | None = None,
) -> SelectedPackedTrajectory:
    """Run a packed stream while retaining only selected per-example ticks.

    A BrainState scan carries fixed ``(checkpoints, batch, ...)`` buffers and a
    cursor for each batch element.  Every model step still executes in order,
    including advance and ablation controls, but scan outputs only one byte per
    tick instead of stacking neuron-sized states.  Evaluation memory therefore
    scales with the number of requested checkpoints rather than packed-stream
    length.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model to reset and execute.
    events : jax.Array
        Fixed stream shaped ``(time, batch, input_width)``.  Rank two is
        accepted for native batch one.
    selected_indices : jax.Array
        Integer stream indices shaped ``(checkpoints, batch)``.  Each batch
        column must be strictly increasing and lie in ``[0, time)``.  For the
        full experiment these are ``query_stop - 1 + arange(33)``.
    reset : bool, default=True
        Reset inference state before scanning.
    advance_gates : jax.Array, optional
        Boolean state-advance gates shaped ``(time, batch)``.  Omitted gates
        are derived from the event-valid channel.
    ablation_slots : jax.Array, optional
        One integer 64-neuron slot per batch element.
    ablation_gates : jax.Array, optional
        Boolean pre-step ablation gates shaped ``(time, batch)``.  Both
        ablation arguments must be supplied together.

    Returns
    -------
    SelectedPackedTrajectory
        Only the requested compact outputs and physical states.
    """
    packed = _batched_events(model, events)
    raw_indices = np.asarray(selected_indices)
    if raw_indices.ndim != 2 or raw_indices.shape[1] != model.config.batch_size:
        raise ValueError(
            f"selected_indices must have shape (checkpoints, {model.config.batch_size})"
        )
    if raw_indices.shape[0] < 1:
        raise ValueError("selected_indices must contain at least one checkpoint")
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise ValueError("selected_indices must contain integers")
    if np.any(raw_indices < 0) or np.any(raw_indices >= packed.shape[0]):
        raise ValueError(f"selected_indices must lie in [0, {packed.shape[0]})")
    if np.any(np.diff(raw_indices.astype(np.int64), axis=0) <= 0):
        raise ValueError(
            "selected_indices must be strictly increasing in each batch column"
        )
    indices = jnp.asarray(raw_indices, dtype=jnp.int32)
    checkpoint_count = int(raw_indices.shape[0])
    batch_size = model.config.batch_size

    if advance_gates is None:
        advance = packed[..., model.config.event_valid_index] > 0.5
    else:
        advance = jnp.asarray(advance_gates, dtype=jnp.bool_)
        if advance.shape != (packed.shape[0], batch_size):
            raise ValueError(
                f"advance_gates must have shape ({packed.shape[0]}, {batch_size})"
            )

    controlled = ablation_slots is not None or ablation_gates is not None
    if controlled:
        if ablation_slots is None or ablation_gates is None:
            raise ValueError(
                "ablation_slots and ablation_gates must be supplied together"
            )
        raw_slots = np.asarray(ablation_slots)
        if raw_slots.shape != (batch_size,):
            raise ValueError(f"ablation_slots must have shape ({batch_size},)")
        if not np.issubdtype(raw_slots.dtype, np.integer):
            raise ValueError("ablation_slots must contain integers")
        if np.any(raw_slots < 0) or np.any(raw_slots >= model.slot_count):
            raise ValueError(f"ablation_slots must lie in [0, {model.slot_count})")
        slots = jnp.asarray(raw_slots, dtype=jnp.int32)
        ablations = jnp.asarray(ablation_gates, dtype=jnp.bool_)
        if ablations.shape != (packed.shape[0], batch_size):
            raise ValueError(
                f"ablation_gates must have shape ({packed.shape[0]}, {batch_size})"
            )
    else:
        slots = jnp.zeros((batch_size,), dtype=jnp.int32)
        ablations = jnp.zeros((packed.shape[0], batch_size), dtype=jnp.bool_)

    if reset:
        model.reset_state()
    compact_buffer = jnp.zeros(
        (checkpoint_count, batch_size, model.config.compact_output_width),
        dtype=jnp.float32,
    )
    state_shape = (checkpoint_count, batch_size, model.config.neuron_count)
    spikes_buffer = jnp.zeros(state_shape, dtype=jnp.float32)
    voltage_buffer = jnp.zeros(state_shape, dtype=model.voltage.dtype)
    feedforward_buffer = jnp.zeros(state_shape, dtype=model.feedforward_current.dtype)
    recurrent_buffer = jnp.zeros(state_shape, dtype=model.recurrent_current.dtype)
    batch_indices = jnp.arange(batch_size, dtype=jnp.int32)

    initial_carry = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((batch_size,), dtype=jnp.int32),
        compact_buffer,
        spikes_buffer,
        voltage_buffer,
        feedforward_buffer,
        recurrent_buffer,
    )

    def scan_step(
        carry: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        jax.Array,
    ]:
        (
            time_index,
            cursors,
            compact_values,
            spike_values,
            voltage_values,
            feedforward_values,
            recurrent_values,
        ) = carry
        event, advance_gate, ablation_gate = inputs
        model.mask_slots(slots, ablation_gate)
        spikes = model.cell_step(event, advance_gate)
        compact = model.compact_readout(spikes)
        voltage = model.voltage
        feedforward = model.feedforward_current
        recurrent = model.recurrent_current

        safe_cursors = jnp.minimum(cursors, checkpoint_count - 1)
        targets = indices[safe_cursors, batch_indices]
        matched = (cursors < checkpoint_count) & (time_index == targets)

        def record(buffer: jax.Array, value: jax.Array) -> jax.Array:
            previous = buffer[safe_cursors, batch_indices]
            selected = jnp.where(matched[:, None], value, previous)
            return buffer.at[safe_cursors, batch_indices].set(selected)

        compact_values = record(compact_values, compact)
        spike_values = record(spike_values, spikes)
        voltage_values = record(voltage_values, voltage)
        feedforward_values = record(feedforward_values, feedforward)
        recurrent_values = record(recurrent_values, recurrent)
        next_carry = (
            time_index + 1,
            cursors + matched.astype(jnp.int32),
            compact_values,
            spike_values,
            voltage_values,
            feedforward_values,
            recurrent_values,
        )
        return next_carry, jnp.asarray(0, dtype=jnp.int8)

    final_carry, _ = brainstate.transform.scan(
        scan_step, initial_carry, (packed, advance, ablations)
    )
    (
        _,
        _,
        compact_buffer,
        spikes_buffer,
        voltage_buffer,
        feedforward_buffer,
        recurrent_buffer,
    ) = final_carry
    return SelectedPackedTrajectory(
        selected_indices=indices,
        compact_logits=compact_buffer,
        spikes=spikes_buffer,
        voltage=voltage_buffer,
        feedforward_current=feedforward_buffer,
        recurrent_current=recurrent_buffer,
        color_rank=model.config.color_rank,
    )


def run_latent_trajectory(
    model: LatentWorkspaceModel,
    *,
    steps: int | None = None,
) -> ModelTrajectory:
    """Continue current state through an exactly-zero compiled latent rollout.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Context-conditioned model.  This function does not reset it.
    steps : int, optional
        Number of recurrent updates.  The configured maximum is used when
        omitted.

    Returns
    -------
    ModelTrajectory
        Checkpoint zero followed by every latent update.
    """
    if steps is None:
        steps = model.config.max_latent_steps
    else:
        steps = _nonnegative_integer(steps, "steps")
    if steps > model.config.max_latent_steps:
        raise ValueError(
            f"steps {steps} exceeds configured maximum {model.config.max_latent_steps}"
        )

    initial_spikes = model.spikes
    initial_compact = model.compact_readout(initial_spikes)
    initial_voltage = model.voltage
    initial_feedforward_current = model.feedforward_current
    initial_recurrent_current = model.recurrent_current
    zero_inputs = jnp.zeros(
        (steps, model.config.batch_size, model.config.input_width),
        dtype=jnp.float32,
    )

    if steps:
        advance = jnp.ones((steps, model.config.batch_size), dtype=jnp.bool_)

        def latent_step(
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            event, gate = inputs
            spikes = model.cell_step(event, gate)
            return (
                model.compact_readout(spikes),
                spikes,
                model.voltage,
                model.feedforward_current,
                model.recurrent_current,
            )

        compact, spikes, voltage, feedforward_current, recurrent_current = (
            brainstate.transform.for_loop(latent_step, (zero_inputs, advance))
        )
    else:
        compact = jnp.zeros(
            (0, model.config.batch_size, model.config.compact_output_width),
            dtype=initial_compact.dtype,
        )
        spikes = jnp.zeros(
            (0, model.config.batch_size, model.config.neuron_count),
            dtype=initial_spikes.dtype,
        )
        voltage = jnp.zeros(
            (0, model.config.batch_size, model.config.neuron_count),
            dtype=initial_voltage.dtype,
        )
        feedforward_current = jnp.zeros(
            (0, model.config.batch_size, model.config.neuron_count),
            dtype=initial_feedforward_current.dtype,
        )
        recurrent_current = jnp.zeros(
            (0, model.config.batch_size, model.config.neuron_count),
            dtype=initial_recurrent_current.dtype,
        )
    return ModelTrajectory(
        compact_logits=jnp.concatenate((initial_compact[None], compact), axis=0),
        spikes=jnp.concatenate((initial_spikes[None], spikes), axis=0),
        voltage=jnp.concatenate((initial_voltage[None], voltage), axis=0),
        feedforward_current=jnp.concatenate(
            (initial_feedforward_current[None], feedforward_current), axis=0
        ),
        recurrent_current=jnp.concatenate(
            (initial_recurrent_current[None], recurrent_current), axis=0
        ),
        zero_inputs=zero_inputs,
        color_rank=model.config.color_rank,
    )


def run_sequence(
    model: LatentWorkspaceModel,
    events: jax.Array,
    *,
    latent_steps: int | None = None,
) -> SequenceResult:
    """Reset, execute ARC context, and collect one continuous latent path.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model to execute.
    events : jax.Array
        Valid ARC row events accepted by :func:`run_context`.
    latent_steps : int, optional
        Recurrent effort after checkpoint zero.

    Returns
    -------
    SequenceResult
        Query-terminal checkpoint and its zero-input latent trajectory.
    """
    context = run_context(model, events, reset=True)
    trajectory = run_latent_trajectory(model, steps=latent_steps)
    return SequenceResult(context=context, trajectory=trajectory)


def compile_pp_prop(
    model: LatentWorkspaceModel,
    *,
    verbose: int = 0,
) -> Any:
    """Compile the model with its explicit pp-prop eligibility coordinate.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model whose parameter objects are shared across all effort lengths.
    verbose : int, default=0
        BrainTrace compiler verbosity.

    Returns
    -------
    object
        BrainTrace learner exposing ``etrace_grad`` and ``etrace_evolve``.
    """
    sample = jnp.zeros(
        (model.config.batch_size, model.config.input_width), dtype=jnp.float32
    )
    sample_advance = jnp.ones((model.config.batch_size,), dtype=jnp.bool_)
    return braintrace.compile(
        model,
        model.etrace_config(),
        sample,
        sample_advance,
        batch_size=model.config.batch_size,
        vmap=False,
        verbose=verbose,
    )


def parameter_snapshot(model: LatentWorkspaceModel) -> dict[str, Any]:
    """Copy every trainable parameter for immutability checks.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model whose parameters are copied.

    Returns
    -------
    dict of str to object
        Path-keyed copied parameter pytrees.  Sparse weights remain dictionaries
        containing their value arrays.
    """
    return {
        ".".join(map(str, path)): _copy_tree(state.value)
        for path, state in model.states(brainstate.ParamState).items()
    }
