"""Memory-equipped latent workspace model for Example 21."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Literal

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import braintrace

try:
    from .latent_workspace_task import TaskConfig, build_codebook
except ImportError:
    from latent_workspace_task import TaskConfig, build_codebook

WriteMode = Literal["fixed_random", "learned"]
_LATENT_THRESHOLD = 1.0


def _validated_integral_scalar(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-boolean integer")
    result = int(value)
    if result < minimum:
        qualifier = "nonnegative" if minimum == 0 else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _validated_real_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-boolean real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite non-boolean real scalar")
    return result


def _surrogate_spike(value: jax.Array) -> jax.Array:
    hard = (value >= 0.0).astype(value.dtype)
    soft = jax.nn.sigmoid(value)
    return soft + jax.lax.stop_gradient(hard - soft)


@dataclass(frozen=True)
class ModelConfig:
    """Configure the latent-workspace network.

    Parameters
    ----------
    task : TaskConfig
        Episode dimensions and flat input layout.
    batch_size : int
        Native leading batch dimension used by every model call.
    latent_width : int
        Width of the value rows, key rows, and latent workspace.
    ingestion_tau_ms : float
        Membrane constant of the distinct input-ingestion population.
    latent_tau_ms : float
        Membrane constant of the zero-input latent recurrence.
    time_step_ms : float
        Duration represented by one task tick.
    latent_spectral_radius : float
        Requested spectral radius for the recurrent workspace matrix.
    latent_connectivity : float
        Connection probability in the positive sparse recurrent reservoir.
    max_jacobian_elements : int
        Explicit safety ceiling for the intentional coupled hidden Jacobian.
    write_mode : {"fixed_random", "learned"}
        Whether an outer training driver should optimize ``Wk`` and ``Wv``.
        Both remain :class:`brainstate.ParamState` objects so BrainTrace can
        compile their ETP relations. The release default is honestly labelled
        ``"fixed_random"`` because the feasibility spike did not establish
        learning of the write path.
    seed : int
        Experiment seed for recurrent initialization through
        ``brainstate.random``.
    projection_seed : int
        Fixed seed for the key, value, and readout projections. Keeping this
        separate from ``seed`` holds the contextual-memory basis constant
        across recurrent experiment seeds.
    """

    task: TaskConfig = field(default_factory=TaskConfig)
    batch_size: int = 1
    latent_width: int = 32
    ingestion_tau_ms: float = 20.0
    latent_tau_ms: float = 160.0
    time_step_ms: float = 1.0
    latent_spectral_radius: float = 0.9
    latent_connectivity: float = 0.75
    max_jacobian_elements: int = 1 << 26
    write_mode: WriteMode = "fixed_random"
    seed: int = 2108
    projection_seed: int = 210848

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "batch_size",
            _validated_integral_scalar(self.batch_size, "batch_size", minimum=1),
        )
        object.__setattr__(
            self,
            "latent_width",
            _validated_integral_scalar(self.latent_width, "latent_width", minimum=1),
        )
        object.__setattr__(
            self,
            "seed",
            _validated_integral_scalar(self.seed, "seed", minimum=0),
        )
        object.__setattr__(
            self,
            "projection_seed",
            _validated_integral_scalar(
                self.projection_seed, "projection_seed", minimum=0
            ),
        )
        object.__setattr__(
            self,
            "max_jacobian_elements",
            _validated_integral_scalar(
                self.max_jacobian_elements,
                "max_jacobian_elements",
                minimum=1,
            ),
        )
        for name in (
            "ingestion_tau_ms",
            "latent_tau_ms",
            "time_step_ms",
            "latent_spectral_radius",
            "latent_connectivity",
        ):
            object.__setattr__(
                self, name, _validated_real_scalar(getattr(self, name), name)
            )
        if self.ingestion_tau_ms <= 0.0:
            raise ValueError("ingestion_tau_ms must be finite and positive")
        if self.latent_tau_ms <= 0.0:
            raise ValueError("latent_tau_ms must be finite and positive")
        if self.time_step_ms <= 0.0:
            raise ValueError("time_step_ms must be finite and positive")
        if self.latent_spectral_radius <= 0.0:
            raise ValueError("latent_spectral_radius must be finite and positive")
        if not 0.0 < self.latent_connectivity <= 1.0:
            raise ValueError("latent_connectivity must be in (0, 1]")
        if self.write_mode not in ("fixed_random", "learned"):
            raise ValueError(
                "write_mode must be 'fixed_random' or 'learned', "
                f"got {self.write_mode!r}"
            )


@dataclass(frozen=True)
class SequenceResult:
    """Hold outputs collected while executing one sequence.

    Parameters
    ----------
    logits : jax.Array
        Per-tick class logits shaped ``(time, batch, symbol_count)``.
    workspace : jax.Array
        Per-tick binary LIF workspace states shaped
        ``(time, batch, latent_width)``. The final query state is ``H0`` and
        latent ticks expose ``H1`` onward. The separate analog ``memory_read``
        controls query-phase logits without replacing this recurrent state.
    memory_read : jax.Array
        Pure contextual read ``A @ (B.T @ q)`` computed after the query from
        the accumulated query encoding, shaped ``(batch, latent_width)``.
    query_encoding : jax.Array
        Query projection accumulated without workspace feedback, shaped
        ``(batch, latent_width)``.
    memory_values, memory_keys : jax.Array
        Final factor rows, each shaped ``(batch, slots, latent_width)``.
    """

    logits: jax.Array
    workspace: jax.Array
    memory_read: jax.Array
    query_encoding: jax.Array
    memory_values: jax.Array
    memory_keys: jax.Array

    @property
    def terminal_logits(self) -> jax.Array:
        """Return logits from the final query or latent tick."""
        return self.logits[-1]

    @property
    def memory_factors(self) -> tuple[jax.Array, jax.Array]:
        """Return final value and key factor rows as ``(A, B)``."""
        return self.memory_values, self.memory_keys


def phase_masks(
    model_inputs: jax.Array, task: TaskConfig
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Extract the three arithmetic phase gates from a flat input axis.

    Parameters
    ----------
    model_inputs : jax.Array
        One tick or a sequence whose last dimension is ``task.input_width``.
    task : TaskConfig
        Input layout defining the phase-vector slice.

    Returns
    -------
    tuple of jax.Array
        Demonstration, query, and latent masks, each retaining a singleton
        final dimension for broadcast arithmetic.
    """
    if model_inputs.shape[-1] != task.input_width:
        raise ValueError(
            "model_inputs final dimension must equal input_width "
            f"{task.input_width}, got shape {model_inputs.shape}"
        )
    phases = model_inputs[..., task.phase_slice]
    return phases[..., 0:1], phases[..., 1:2], phases[..., 2:3]


def factored_memory_read(
    values: jax.Array, keys: jax.Array, query: jax.Array
) -> jax.Array:
    """Read an outer-product memory without materializing its dense matrix.

    This computes ``A @ (B.T @ query)`` with exactly two hidden-only
    contractions. Factor rows have shape ``(batch, slots, width)``.

    Parameters
    ----------
    values, keys : jax.Array
        Value and key factor rows with identical rank-three shapes.
    query : jax.Array
        Query vectors shaped ``(batch, width)``.

    Returns
    -------
    jax.Array
        Read vectors shaped ``(batch, width)``.
    """
    if values.ndim != 3 or keys.shape != values.shape:
        raise ValueError(
            "values and keys must have one identical (batch, slots, width) shape, "
            f"got {values.shape} and {keys.shape}"
        )
    expected_query_shape = (values.shape[0], values.shape[2])
    if query.shape != expected_query_shape:
        raise ValueError(
            "query shape must match factor batch and width "
            f"{expected_query_shape}, got {query.shape}"
        )
    slot_scores = jnp.einsum("bmd,bd->bm", keys, query)
    return jnp.einsum("bmd,bm->bd", values, slot_scores)


def shuffled_memory_factors(
    values: jax.Array, keys: jax.Array, permutation: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Permute value slots relative to unchanged key slots.

    Parameters
    ----------
    values, keys : jax.Array
        Memory factors shaped ``(batch, slots, width)``.
    permutation : jax.Array
        Permutation of ``range(slots)``.

    Returns
    -------
    tuple of jax.Array
        Permuted values and unchanged keys. Shapes and global magnitudes are
        preserved while stored associations are mismatched.
    """
    if values.ndim != 3 or keys.shape != values.shape:
        raise ValueError(
            "values and keys must have one identical (batch, slots, width) shape, "
            f"got {values.shape} and {keys.shape}"
        )
    raw_permutation = np.asarray(permutation)
    if raw_permutation.shape != (values.shape[1],):
        raise ValueError(
            "permutation shape must be "
            f"({values.shape[1]},), got {raw_permutation.shape}"
        )
    if not np.issubdtype(raw_permutation.dtype, np.integer):
        raise ValueError("permutation must contain integer slot indices")
    permutation_values = raw_permutation.astype(np.int64, copy=False)
    expected = np.arange(values.shape[1], dtype=np.int64)
    if not np.array_equal(np.sort(permutation_values), expected):
        raise ValueError("permutation must be a bijection over all memory slots")
    if np.array_equal(permutation_values, expected):
        raise ValueError("permutation must mismatch at least one memory association")
    validated = jnp.asarray(permutation_values, dtype=jnp.int32)
    return jnp.take(values, validated, axis=1), keys


def occupied_slot_derangement(slot_count: int, occupied_count: int) -> jax.Array:
    """Return a cyclic derangement of occupied slots with unused slots fixed.

    Parameters
    ----------
    slot_count : int
        Total contextual-memory capacity.
    occupied_count : int
        Number of leading slots populated by demonstrations; at least two.

    Returns
    -------
    jax.Array
        Integer permutation shaped ``(slot_count,)``. Every occupied slot moves
        and every unused slot retains its position.
    """
    slot_count = _validated_integral_scalar(slot_count, "slot_count", minimum=1)
    occupied_count = _validated_integral_scalar(
        occupied_count, "occupied_count", minimum=1
    )
    if occupied_count < 2:
        raise ValueError("occupied_count must be at least 2 for a derangement")
    if occupied_count > slot_count:
        raise ValueError(
            f"occupied_count {occupied_count} exceeds slot_count {slot_count}"
        )
    permutation = np.arange(slot_count, dtype=np.int32)
    permutation[:occupied_count] = np.roll(permutation[:occupied_count], -1)
    return jnp.asarray(permutation)


def _normalize_rows(value: jax.Array) -> jax.Array:
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


def _state_path(path: tuple[object, ...]) -> str:
    return ".".join(str(part) for part in path)


def parameter_snapshot(model: LatentWorkspaceModel) -> dict[str, NDArray[np.generic]]:
    """Copy every trainable parameter for a before/after mutation audit.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Model whose :class:`brainstate.ParamState` leaves are copied.

    Returns
    -------
    dict
        Parameter-path strings mapped to independent NumPy copies.
    """
    return {
        _state_path(path): np.asarray(state.value).copy()
        for path, state in model.states(brainstate.ParamState).items()
    }


class LatentWorkspaceModel(brainstate.nn.Module):
    """Run demonstration writes, query encoding, and latent computation.

    The contextual value rows, key rows, and workspace occupy one physical
    ``HiddenState`` of shape ``(batch * (2 * slots + 1), latent_width)``.
    Parameter projections consume that same flat grouped row axis, which keeps
    native batched ETP dispatch and avoids a generic ``vmap`` decomposition.

    Parameters
    ----------
    config : ModelConfig
        Network, task, and initialization settings.
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.slot_count = config.task.slot_capacity
        self.width = config.latent_width
        self.state_rows = 2 * self.slot_count + 1
        code_width = config.task.code_width

        code_rates = jnp.mean(
            jnp.asarray(build_codebook(config.task), dtype=jnp.float32), axis=1
        )
        projection_random = brainstate.random.RandomState(config.projection_seed)
        scale = 1.0 / math.sqrt(code_width)
        raw_key = projection_random.randn(code_width, self.width) * scale
        raw_value = projection_random.randn(code_width, self.width) * scale
        key_codes = _normalize_rows(jnp.maximum(code_rates @ raw_key, 0.0))
        value_codes = _normalize_rows(jnp.maximum(code_rates @ raw_value, 0.0))
        codebook_pinv = jnp.linalg.pinv(code_rates)

        self.Wk = brainstate.ParamState(codebook_pinv @ key_codes)
        self.Wv = brainstate.ParamState(codebook_pinv @ value_codes)
        recurrent_random = brainstate.random.RandomState(config.seed)
        recurrent = recurrent_random.uniform(0.0, 1.0, size=(self.width, self.width))
        recurrent = recurrent * recurrent_random.bernoulli(
            config.latent_connectivity, size=(self.width, self.width)
        )
        recurrent = recurrent + 0.01 * jnp.eye(self.width, dtype=jnp.float32)
        recurrent_radius = jnp.max(jnp.abs(jnp.linalg.eigvals(recurrent)))
        recurrent = recurrent * (config.latent_spectral_radius / recurrent_radius)
        self.Wf = brainstate.ParamState(recurrent.astype(jnp.float32))
        self.Wo = brainstate.ParamState(value_codes.T)
        self.init_state()

    @property
    def input_width(self) -> int:
        """Return the complete flat input width expected by ``update``."""
        return self.config.task.input_width

    @property
    def memory_storage_elements(self) -> int:
        """Return contextual-memory storage excluding the workspace row."""
        return 2 * self.slot_count * self.width

    @property
    def write_projections_trainable(self) -> bool:
        """Return whether an outer optimizer should update ``Wk`` and ``Wv``."""
        return self.config.write_mode == "learned"

    def trainable_parameters(
        self,
    ) -> dict[tuple[object, ...], brainstate.ParamState]:
        """Return the parameter mapping an outer optimizer must register.

        Returns
        -------
        dict
            Parameter paths mapped to states. The fixed-random release mode
            excludes ``Wk`` and ``Wv`` while retaining their compiler-visible
            :class:`brainstate.ParamState` representation.
        """
        parameters = self.states(brainstate.ParamState)
        if self.write_projections_trainable:
            return dict(parameters.items())
        fixed_paths = {("Wk",), ("Wv",)}
        return {
            path: state for path, state in parameters.items() if path not in fixed_paths
        }

    @property
    def workspace(self) -> jax.Array:
        """Return the logical workspace view shaped ``(batch, width)``."""
        state = self.grouped_state.value.reshape(
            self.batch_size, self.state_rows, self.width
        )
        return state[:, -1]

    @property
    def latent_voltage_view(self) -> jax.Array:
        """Return the logical latent voltage shaped ``(batch, width)``.

        Returns
        -------
        jax.Array
            The workspace row of the flat compiler-aligned voltage state.
        """
        state = self.latent_voltage.value.reshape(
            self.batch_size, self.state_rows, self.width
        )
        return state[:, -1]

    @property
    def query_encoding_view(self) -> jax.Array:
        """Return the feedback-free query encoding shaped ``(batch, width)``.

        Returns
        -------
        jax.Array
            The workspace row of the flat compiler-aligned query state.
        """
        state = self.query_encoding.value.reshape(
            self.batch_size, self.state_rows, self.width
        )
        return state[:, -1]

    def init_state(self, batch_size: int | None = None, **_: object) -> None:
        """Initialize grouped memory/workspace and distinct ingestion states.

        Parameters
        ----------
        batch_size : int, optional
            If supplied, it must equal the native batch size in ``config``.
        **_ : object
            Extra framework initialization options, accepted and ignored.
        """
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError(
                f"batch_size {batch_size} does not match configured {self.batch_size}"
            )
        self.grouped_state = brainstate.HiddenState(
            jnp.zeros(
                (self.batch_size * self.state_rows, self.width), dtype=jnp.float32
            )
        )
        self.ingestion_state = brainstate.HiddenState(
            jnp.zeros(
                (self.batch_size * 2, self.config.task.code_width),
                dtype=jnp.float32,
            )
        )
        self.latent_voltage = brainstate.HiddenState(
            jnp.zeros(
                (self.batch_size * self.state_rows, self.width), dtype=jnp.float32
            )
        )
        self.query_encoding = brainstate.HiddenState(
            jnp.zeros(
                (self.batch_size * self.state_rows, self.width), dtype=jnp.float32
            )
        )

    def reset_state(self, batch_size: int | None = None, **_: object) -> None:
        """Reset all inference-time state without touching a parameter.

        Parameters
        ----------
        batch_size : int, optional
            If supplied, it must equal the native batch size in ``config``.
        **_ : object
            Extra framework reset options, accepted and ignored.
        """
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError(
                f"batch_size {batch_size} does not match configured {self.batch_size}"
            )
        self.grouped_state.value = jnp.zeros_like(self.grouped_state.value)
        self.ingestion_state.value = jnp.zeros_like(self.ingestion_state.value)
        self.latent_voltage.value = jnp.zeros_like(self.latent_voltage.value)
        self.query_encoding.value = jnp.zeros_like(self.query_encoding.value)

    def memory_factors(self) -> tuple[jax.Array, jax.Array]:
        """Return logical value and key rows from the grouped hidden state.

        Returns
        -------
        tuple of jax.Array
            ``(values, keys)``, each shaped ``(batch, slots, width)``.
        """
        state = self.grouped_state.value.reshape(
            self.batch_size, self.state_rows, self.width
        )
        return state[:, : self.slot_count], state[:, self.slot_count : -1]

    def memory_read(self, query: jax.Array) -> jax.Array:
        """Read the current contextual memory for query vectors.

        Parameters
        ----------
        query : jax.Array
            Query representations shaped ``(batch, latent_width)``.

        Returns
        -------
        jax.Array
            Memory-read vectors with the same shape as ``query``.
        """
        values, keys = self.memory_factors()
        return factored_memory_read(values, keys, query)

    def shuffle_memory(self, permutation: jax.Array) -> None:
        """Apply the value-slot permutation used by the shuffled control.

        Parameters
        ----------
        permutation : jax.Array
            Permutation of ``range(slot_capacity)``.
        """
        values, keys = self.memory_factors()
        shuffled_values, unchanged_keys = shuffled_memory_factors(
            values, keys, permutation
        )
        state = jnp.concatenate(
            (shuffled_values, unchanged_keys, self.workspace[:, None, :]), axis=1
        )
        self.grouped_state.value = state.reshape(
            self.batch_size * self.state_rows, self.width
        )

    def etrace_config(self) -> braintrace.ETraceConfig:
        """Return the explicit coupled trace configuration for this model.

        Returns
        -------
        braintrace.ETraceConfig
            IO-factorized coupled recurrence configuration.
        """
        return braintrace.ETraceConfig(
            trace_factorization="io_factorized",
            recurrence_scope="coupled",
            decay=0.9,
        )

    def compile_options(self) -> dict[str, int]:
        """Return compiler options required by the supported configuration.

        Returns
        -------
        dict of str to int
            Maximum materialized hidden-Jacobian size. The explicit ceiling
            admits the native batch-four, eight-slot, width-32 release model.
        """
        return {
            "snap_max_jacobian_elements": self.config.max_jacobian_elements,
        }

    def _advance(self, packed: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Advance one tick and return logits plus its latent representation.

        Parameters
        ----------
        packed : jax.Array
            Native batched input shaped ``(batch, task.input_width)``.

        Returns
        -------
        tuple of jax.Array
            Current class logits and the internal binary workspace for this
            tick. Query logits use the analog pure-memory read, but the second
            return value remains the homogeneous recurrent LIF trajectory.
        """
        expected = (self.batch_size, self.input_width)
        if packed.shape != expected:
            raise ValueError(
                f"packed input shape must be {expected}, got {packed.shape}"
            )

        task = self.config.task
        key_input = packed[:, task.key_slice]
        value_input = packed[:, task.value_slice]
        slot = packed[:, task.slot_slice]
        demo, query, latent = phase_masks(packed, task)

        ingestion = self.ingestion_state.value.reshape(
            self.batch_size, 2, task.code_width
        )
        ingestion_input = jnp.stack(
            ((demo + query) * key_input, demo * value_input), axis=1
        )
        ingestion_decay = math.exp(
            -self.config.time_step_ms / self.config.ingestion_tau_ms
        )
        ingestion_voltage = ingestion_decay * ingestion + ingestion_input
        ingestion_spikes = _surrogate_spike(ingestion_voltage - _LATENT_THRESHOLD)
        ingestion_next = ingestion_voltage - ingestion_spikes
        key_rate = ingestion_spikes[:, 0]
        value_rate = ingestion_spikes[:, 1]

        zero_slots = jnp.zeros_like(slot)
        workspace_one = jnp.ones((self.batch_size, 1), dtype=slot.dtype)
        key_mask = jnp.concatenate(
            (zero_slots, demo * slot, jnp.zeros_like(workspace_one)), axis=1
        )
        value_mask = jnp.concatenate(
            (demo * slot, zero_slots, jnp.zeros_like(workspace_one)), axis=1
        )
        workspace_mask = jnp.concatenate(
            (zero_slots, zero_slots, workspace_one), axis=1
        )
        active = query + latent
        active_rows = (active[:, :, None] * workspace_mask[:, :, None]).reshape(
            self.batch_size * self.state_rows, 1
        )

        presentation_scale = 1.0 / task.symbol_ticks
        key_projection_input = (
            presentation_scale * key_mask[:, :, None] * key_rate[:, None, :]
        ).reshape(self.batch_size * self.state_rows, task.code_width)
        value_projection_input = (
            presentation_scale * value_mask[:, :, None] * value_rate[:, None, :]
        ).reshape(self.batch_size * self.state_rows, task.code_width)
        query_mask = jnp.concatenate(
            (zero_slots, zero_slots, query * workspace_one), axis=1
        )
        query_projection_input = (
            presentation_scale * query_mask[:, :, None] * key_rate[:, None, :]
        ).reshape(self.batch_size * self.state_rows, task.code_width)
        key_rows = braintrace.matmul(key_projection_input, self.Wk.value)
        value_rows = braintrace.matmul(value_projection_input, self.Wv.value)
        query_drive = braintrace.matmul(query_projection_input, self.Wk.value)

        previous = self.grouped_state.value.reshape(
            self.batch_size, self.state_rows, self.width
        )
        values = previous[:, : self.slot_count]
        keys = previous[:, self.slot_count : -1]
        workspace_previous = previous[:, -1]
        query_encoding_previous = self.query_encoding.value.reshape(
            self.batch_size, self.state_rows, self.width
        )[:, -1]
        query_encoding_next = self.query_encoding.value + query_drive
        read_vector = query * query_encoding_previous + latent * workspace_previous
        dynamics_read = factored_memory_read(values, keys, read_vector)
        dynamics_read_rows = (
            workspace_mask[:, :, None] * dynamics_read[:, None, :]
        ).reshape(self.batch_size * self.state_rows, self.width)

        workspace_activity_rows = (
            active[:, :, None]
            * workspace_mask[:, :, None]
            * workspace_previous[:, None, :]
        ).reshape(self.batch_size * self.state_rows, self.width)
        recurrent_rows = braintrace.matmul(workspace_activity_rows, self.Wf.value)
        latent_decay = math.exp(-self.config.time_step_ms / self.config.latent_tau_ms)
        parameter_drive = recurrent_rows + query_drive
        active_voltage = active_rows * self.latent_voltage.value
        voltage_candidate = (
            latent_decay * active_voltage + dynamics_read_rows + parameter_drive
        )
        spike_voltage = (
            latent_decay * active_voltage
            + dynamics_read_rows
            + jax.lax.stop_gradient(parameter_drive)
        )
        spikes_candidate = _surrogate_spike(spike_voltage - _LATENT_THRESHOLD)
        voltage_next = voltage_candidate - (
            jax.lax.stop_gradient(spikes_candidate) * _LATENT_THRESHOLD
        )
        workspace_delta_rows = spikes_candidate - workspace_activity_rows
        flat_next = (
            self.grouped_state.value + key_rows + value_rows + workspace_delta_rows
        )
        state_next = flat_next.reshape(self.batch_size, self.state_rows, self.width)
        query_encoding_logical = query_encoding_next.reshape(
            self.batch_size, self.state_rows, self.width
        )[:, -1]
        pure_query_read = factored_memory_read(values, keys, query_encoding_logical)
        representation = query * pure_query_read + (1.0 - query) * state_next[:, -1]
        readout_input = (
            workspace_mask[:, :, None] * representation[:, None, :]
        ).reshape(self.batch_size * self.state_rows, self.width)
        logit_rows = jnp.matmul(readout_input, self.Wo.value)

        self.ingestion_state.value = ingestion_next.reshape(
            self.batch_size * 2, task.code_width
        )
        self.grouped_state.value = flat_next
        self.latent_voltage.value = voltage_next
        self.query_encoding.value = query_encoding_next
        logits = logit_rows.reshape(
            self.batch_size, self.state_rows, task.symbol_count
        )[:, -1]
        return logits, state_next[:, -1]

    def update(self, packed: jax.Array) -> jax.Array:
        """Advance one arithmetically phase-gated model tick.

        Parameters
        ----------
        packed : jax.Array
            Native batched input shaped ``(batch, task.input_width)``.

        Returns
        -------
        jax.Array
            Current class logits shaped ``(batch, symbol_count)``.
        """
        logits, _ = self._advance(packed)
        return logits


def run_sequence(
    model: LatentWorkspaceModel, model_inputs: jax.Array
) -> SequenceResult:
    """Run a complete episode with one compiled BrainState loop.

    Parameters
    ----------
    model : LatentWorkspaceModel
        Stateful model. Call ``reset_state`` before this function when starting
        an independent episode.
    model_inputs : jax.Array
        Inputs shaped ``(time, input_width)`` for batch size one or
        ``(time, batch, input_width)`` for native batched execution.

    Returns
    -------
    SequenceResult
        Per-tick logits and workspace states. ``R = 0`` is valid because the
        query span still supplies the terminal tick.
    """
    raw_sequence = np.asarray(model_inputs)
    sequence = jnp.asarray(raw_sequence, dtype=jnp.float32)
    if sequence.ndim == 2:
        if model.batch_size != 1:
            raise ValueError(
                "two-dimensional model_inputs require configured batch_size 1"
            )
        sequence = sequence[:, None, :]
    expected_tail = (model.batch_size, model.input_width)
    if sequence.ndim != 3 or sequence.shape[1:] != expected_tail:
        raise ValueError(
            "model_inputs shape must be (time, batch, input_width) with tail "
            f"{expected_tail}, got {sequence.shape}"
        )
    phase_values = np.asarray(sequence[..., model.config.task.phase_slice])
    if (
        not np.all(np.isfinite(phase_values))
        or not np.all((phase_values == 0.0) | (phase_values == 1.0))
        or not np.all(np.sum(phase_values, axis=-1) == 1.0)
    ):
        raise ValueError("phase values must be finite binary one-hot vectors")
    if not np.all(np.any(phase_values[..., 1] == 1.0, axis=0)):
        raise ValueError("query phase must be present for every batch element")

    def step(one_tick: jax.Array) -> tuple[jax.Array, jax.Array]:
        return model._advance(one_tick)

    logits, workspace = brainstate.transform.for_loop(step, sequence)
    memory_values, memory_keys = model.memory_factors()
    query_encoding = model.query_encoding_view
    pure_query_read = factored_memory_read(memory_values, memory_keys, query_encoding)
    return SequenceResult(
        logits=logits,
        workspace=workspace,
        memory_read=pure_query_read,
        query_encoding=query_encoding,
        memory_values=memory_values,
        memory_keys=memory_keys,
    )
