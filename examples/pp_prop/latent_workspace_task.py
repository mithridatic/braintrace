"""Episode construction for Example 21's latent-workspace task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import brainstate
import numpy as np
from numpy.typing import ArrayLike, NDArray

Condition = Literal["supported", "short"]
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]
_MAX_CODEBOOK_ATTEMPTS = 128


def _validated_integral_scalar(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-boolean integer scalar")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        try:
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite non-boolean integer scalar")
        except (OverflowError, TypeError, ValueError):
            raise ValueError(
                f"{name} must be a finite non-boolean integer scalar"
            ) from None
    raise ValueError(f"{name} must be a finite non-boolean integer scalar")


def _validated_real_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-boolean real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must be a finite non-boolean real scalar") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite non-boolean real scalar")
    return result


@dataclass(frozen=True)
class TaskConfig:
    """Configure one latent-workspace episode.

    Attributes
    ----------
    symbol_count : int
        Number of symbols in each freshly drawn bijection.
    binding_count : int
        Number of demonstration bindings in the episode.
    slot_capacity : int
        Maximum number of demonstration bindings held by the model.
    latent_steps : int
        Number of zero-external-input ticks following the query.
    code_width : int
        Number of channels in each parallel symbol bank.
    spike_rate : float
        Bernoulli probability for each channel in each symbol tick. The
        realized fraction is measured from the fixed sampled codebook.
    symbol_ticks : int
        Number of ticks used to present a demonstration or query symbol.
    clock_width : int
        Width of the latent clock bank. Successive latent ticks carry distinct
        external drive so the latent recurrence is not an autonomous map. Must
        be positive and even, because the bank holds sine/cosine pairs.
    codebook_seed : int
        Nonnegative seed for the local BrainState codebook stream.
    """

    symbol_count: int = 10
    binding_count: int = 4
    slot_capacity: int = 8
    latent_steps: int = 4
    code_width: int = 24
    spike_rate: float = 0.25
    symbol_ticks: int = 4
    clock_width: int = 4
    codebook_seed: int = 313320

    def __post_init__(self) -> None:
        for name in (
            "symbol_count",
            "binding_count",
            "slot_capacity",
            "latent_steps",
            "code_width",
            "symbol_ticks",
            "clock_width",
            "codebook_seed",
        ):
            object.__setattr__(
                self,
                name,
                _validated_integral_scalar(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "spike_rate",
            _validated_real_scalar(self.spike_rate, "spike_rate"),
        )
        if self.symbol_count < 10:
            raise ValueError("symbol_count must be at least 10")
        if self.binding_count < 1:
            raise ValueError("binding_count must be positive")
        if self.slot_capacity < 1:
            raise ValueError("slot_capacity must be positive")
        if self.binding_count > self.slot_capacity:
            raise ValueError(
                "binding_count "
                f"{self.binding_count} exceeds slot_capacity {self.slot_capacity}"
            )
        if self.symbol_count < self.binding_count + 2:
            raise ValueError(
                "symbol_count must be at least binding_count + 2 "
                f"({self.symbol_count} < {self.binding_count} + 2)"
            )
        if self.latent_steps < 0:
            raise ValueError("latent_steps must be nonnegative")
        if self.code_width < self.symbol_count:
            raise ValueError(
                "code_width must be at least symbol_count "
                f"({self.code_width} < {self.symbol_count})"
            )
        if self.symbol_ticks < 1:
            raise ValueError("symbol_ticks must be positive")
        if self.clock_width < 1:
            raise ValueError("clock_width must be positive")
        if self.clock_width % 2:
            raise ValueError("clock_width must be even to hold sine/cosine pairs")
        if self.codebook_seed < 0:
            raise ValueError("codebook_seed must be nonnegative")
        if not math.isfinite(self.spike_rate) or not 0.0 < self.spike_rate < 1.0:
            raise ValueError("spike_rate must be finite and strictly between 0 and 1")

    @property
    def key_slice(self) -> slice:
        """Return the key-bank slice in a model input row."""
        return slice(0, self.code_width)

    @property
    def value_slice(self) -> slice:
        """Return the value-bank slice in a model input row."""
        return slice(self.code_width, 2 * self.code_width)

    @property
    def slot_slice(self) -> slice:
        """Return the memory-slot slice in a model input row."""
        start = 2 * self.code_width
        return slice(start, start + self.slot_capacity)

    @property
    def phase_slice(self) -> slice:
        """Return the demonstration/query/latent-seed/latent phase slice.

        The latent span is split across two one-hot channels. The first latent
        tick uses the seed channel, which initializes ``H_0`` from contextual
        memory; every later latent tick uses the plain latent channel. The
        split keeps the phase vector one-hot while giving the model a
        state-free signal for the first latent tick.
        """
        start = 2 * self.code_width + self.slot_capacity
        return slice(start, start + 4)

    @property
    def clock_slice(self) -> slice:
        """Return the latent-clock slice in a model input row.

        The bank carries a Fourier phase code of the latent tick index. Its
        width is independent of ``latent_steps``, so a model trained at one
        latent depth is evaluated at another without an input-shape change.
        """
        start = self.phase_slice.stop
        return slice(start, start + self.clock_width)

    @property
    def input_width(self) -> int:
        """Return the complete per-tick model input width."""
        return 2 * self.code_width + self.slot_capacity + 4 + self.clock_width

    @property
    def demonstration_steps(self) -> int:
        """Return the number of demonstration ticks."""
        return self.binding_count * self.symbol_ticks

    @property
    def query_slice(self) -> slice:
        """Return the query span on the episode time axis."""
        start = self.demonstration_steps
        return slice(start, start + self.symbol_ticks)

    @property
    def latent_slice(self) -> slice:
        """Return the latent span on the episode time axis."""
        start = self.query_slice.stop
        return slice(start, start + self.latent_steps)

    @property
    def total_steps(self) -> int:
        """Return the total number of ticks in one episode."""
        return self.latent_slice.stop


@dataclass(frozen=True)
class Episode:
    """Hold one encoded episode and non-input audit metadata.

    Attributes
    ----------
    config : TaskConfig
        Dimensions and encoding used for the episode.
    model_inputs : numpy.ndarray
        Direct model input tensor shaped ``(time, input_width)``.
    rule : numpy.ndarray
        Episode-specific symbol bijection, retained only as audit metadata.
    demonstration_keys, demonstration_values : numpy.ndarray
        Demonstration bindings retained only as audit metadata.
    query_symbol : int
        Held-out symbol presented during the query phase.
    terminal_target : int
        Oracle answer supervised at the terminal tick.
    condition : {"supported", "short"}
        Context-support intervention represented by the episode.
    codebook : numpy.ndarray
        Distributed spike encodings used to build ``model_inputs``.
    """

    config: TaskConfig
    model_inputs: FloatArray
    rule: IntArray
    demonstration_keys: IntArray
    demonstration_values: IntArray
    query_symbol: int
    terminal_target: int
    condition: Condition
    codebook: FloatArray

    @property
    def demonstration_pairs(self) -> tuple[tuple[int, int], ...]:
        """Return demonstration bindings as immutable integer pairs."""
        return tuple(
            zip(
                self.demonstration_keys.tolist(),
                self.demonstration_values.tolist(),
                strict=True,
            )
        )

    @property
    def target(self) -> int:
        """Return the query's terminal class target."""
        return self.terminal_target

    @property
    def terminal_index(self) -> int:
        """Return the only time index at which the target is supervised."""
        return self.model_inputs.shape[0] - 1

    @property
    def query_inputs(self) -> FloatArray:
        """Return the query-phase model inputs."""
        return self.model_inputs[self.config.query_slice]

    @property
    def latent_inputs(self) -> FloatArray:
        """Return the latent-phase model inputs."""
        return self.model_inputs[self.config.latent_slice]


@dataclass(frozen=True)
class MatchedEpisodes:
    """Hold supported and short views of the same base episode.

    Attributes
    ----------
    supported : Episode
        View whose demonstrations contain the queried binding.
    short : Episode
        Matched view whose demonstrations omit the query symbol.
    """

    supported: Episode
    short: Episode


def build_codebook(config: TaskConfig) -> FloatArray:
    """Build the deterministic, distributed spike codebook.

    A local :class:`brainstate.random.RandomState` samples independent Bernoulli
    channels, so construction does not consume the global BrainState stream.
    The configured rate is a probability; each accepted codebook records a
    realized rate rather than enforcing equal-weight symbol words. If the first
    draw is not unique and full rank after a bias column is appended, generation
    retries deterministically with ``codebook_seed + attempt``.

    Parameters
    ----------
    config : TaskConfig
        Encoding dimensions and rate.

    Returns
    -------
    numpy.ndarray
        Read-only float32 binary array shaped
        ``(symbol_count, symbol_ticks, code_width)``.

    Raises
    ------
    ValueError
        If no unique, augmented-full-rank codebook is found within the bounded
        retry budget.
    """
    shape = (config.symbol_count, config.symbol_ticks, config.code_width)
    for attempt in range(_MAX_CODEBOOK_ATTEMPTS):
        rng = brainstate.random.RandomState(config.codebook_seed + attempt)
        codebook = np.asarray(
            rng.bernoulli(config.spike_rate, size=shape),
            dtype=np.float32,
        )
        flattened = codebook.reshape(config.symbol_count, -1)
        unique = np.unique(flattened, axis=0).shape[0] == config.symbol_count
        augmented = np.column_stack(
            (flattened, np.ones(config.symbol_count, dtype=np.float32))
        )
        full_rank = np.linalg.matrix_rank(augmented) == config.symbol_count
        if unique and full_rank:
            codebook.setflags(write=False)
            return codebook
    raise ValueError(
        f"codebook_seed {config.codebook_seed} exhausted "
        f"{_MAX_CODEBOOK_ATTEMPTS} attempts without producing unique symbols "
        "and a full-rank augmented design"
    )


def latent_clock_code(latent_steps: int, clock_width: int) -> FloatArray:
    """Build the per-latent-tick Fourier phase code.

    Successive latent ticks must differ in their external drive; otherwise the
    latent recurrence applies one identical map at every depth and converges to
    a fixed point no matter how the recurrent weights are signed. This code
    supplies that difference without a counter hidden state, so the hidden
    Jacobian does not grow.

    Parameters
    ----------
    latent_steps : int
        Number of latent ticks in the episode. Zero yields an empty code.
    clock_width : int
        Positive even bank width. The leading half holds sines and the
        trailing half cosines, at periods ``2, 4, 8, ...`` ticks.

    Returns
    -------
    numpy.ndarray
        Float32 array shaped ``(latent_steps, clock_width)``.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> code = latent_clock_code(4, 2)
        >>> code.shape
        (4, 2)
        >>> bool(np.allclose(code[0], code[2]))
        True
        >>> bool(np.allclose(code[0], code[1]))
        False
    """
    latent_steps = _validated_integral_scalar(latent_steps, "latent_steps")
    clock_width = _validated_integral_scalar(clock_width, "clock_width")
    if latent_steps < 0:
        raise ValueError("latent_steps must be nonnegative")
    if clock_width < 1:
        raise ValueError("clock_width must be positive")
    if clock_width % 2:
        raise ValueError("clock_width must be even to hold sine/cosine pairs")
    indices = np.arange(latent_steps, dtype=np.float32)[:, None]
    periods = np.exp2(np.arange(1, clock_width // 2 + 1, dtype=np.float32))
    angles = 2.0 * np.pi * indices / periods[None, :]
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=1).astype(np.float32)


def draw_rule(symbol_count: int, rng: brainstate.random.RandomState) -> IntArray:
    """Draw a fresh symbol bijection using only ``brainstate.random``.

    Parameters
    ----------
    symbol_count : int
        Size of the bijection.
    rng : brainstate.random.RandomState
        Stateful random stream owned by the caller.

    Returns
    -------
    numpy.ndarray
        Permutation where element ``x`` is the mapped symbol ``rule(x)``.
    """
    symbol_count = _validated_integral_scalar(symbol_count, "symbol_count")
    if symbol_count < 1:
        raise ValueError("symbol_count must be positive")
    return np.asarray(rng.permutation(symbol_count), dtype=np.int32)


def oracle_answer(rule: ArrayLike, query_symbol: int) -> int:
    """Apply a validated bijection to one query symbol.

    Parameters
    ----------
    rule : array-like
        One-dimensional permutation of ``range(C)``.
    query_symbol : int
        Symbol to map.

    Returns
    -------
    int
        The mapped symbol.
    """
    raw = np.asarray(rule)
    if raw.ndim != 1:
        raise ValueError(f"rule shape must be one-dimensional, got {raw.shape}")
    values = _validated_symbols(raw, "rule", raw.size, raw.size)
    expected = np.arange(values.size)
    if not np.array_equal(np.sort(values), expected):
        raise ValueError("rule must be a bijection over its symbol_count")
    query_symbol = _validated_integral_scalar(query_symbol, "query_symbol")
    if not 0 <= query_symbol < values.size:
        raise ValueError(
            f"query_symbol {query_symbol} outside symbol_count {values.size}"
        )
    return int(values[query_symbol])


def _validated_symbols(
    values: ArrayLike, name: str, count: int, symbol_count: int
) -> IntArray:
    raw = np.asarray(values)
    if raw.shape != (count,):
        raise ValueError(f"{name} shape must be ({count},), got {raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain integer symbols")
    if np.any(raw < 0) or np.any(raw >= symbol_count):
        raise ValueError(
            f"{name} contains a symbol outside symbol_count {symbol_count}"
        )
    result = raw.astype(np.int32, copy=True)
    result.setflags(write=False)
    return result


def build_episode(
    config: TaskConfig,
    *,
    rule: ArrayLike,
    demonstration_keys: ArrayLike,
    demonstration_values: ArrayLike,
    query_symbol: int,
    terminal_target: int,
    condition: Condition,
) -> Episode:
    """Validate metadata and encode one episode onto a flat time axis.

    Parameters
    ----------
    config : TaskConfig
        Episode dimensions.
    rule : array-like
        Episode-specific bijection.
    demonstration_keys, demonstration_values : array-like
        The ``K`` observed bindings.
    query_symbol, terminal_target : int
        Held-out query and its oracle answer.
    condition : {"supported", "short"}
        Whether the queried binding appears in the demonstrations.

    Returns
    -------
    Episode
        Validated metadata and the direct model input tensor.
    """
    if condition not in ("supported", "short"):
        raise ValueError(f"condition must be 'supported' or 'short', got {condition!r}")
    query_symbol = _validated_integral_scalar(query_symbol, "query_symbol")
    terminal_target = _validated_integral_scalar(terminal_target, "terminal_target")
    if not 0 <= query_symbol < config.symbol_count:
        raise ValueError(
            f"query_symbol {query_symbol} outside symbol_count {config.symbol_count}"
        )
    if not 0 <= terminal_target < config.symbol_count:
        raise ValueError(
            f"terminal_target {terminal_target} outside symbol_count "
            f"{config.symbol_count}"
        )
    rule_array = _validated_symbols(
        rule, "rule", config.symbol_count, config.symbol_count
    )
    if np.unique(rule_array).size != config.symbol_count:
        raise ValueError("rule must be a bijection over symbol_count")
    keys = _validated_symbols(
        demonstration_keys,
        "demonstration_keys",
        config.binding_count,
        config.symbol_count,
    )
    values = _validated_symbols(
        demonstration_values,
        "demonstration_values",
        config.binding_count,
        config.symbol_count,
    )
    if np.unique(keys).size != keys.size:
        raise ValueError("demonstration_keys must contain distinct bindings")
    if not np.array_equal(values, rule_array[keys]):
        raise ValueError("demonstration_values disagree with rule")
    expected_target = oracle_answer(rule_array, query_symbol)
    if terminal_target != expected_target:
        raise ValueError(
            "terminal_target disagrees with rule: "
            f"{terminal_target} != {expected_target}"
        )
    query_hits = (keys == query_symbol) & (values == terminal_target)
    if condition == "supported" and int(query_hits.sum()) != 1:
        raise ValueError("supported condition must contain the queried binding once")
    if condition == "short" and (
        np.any(keys == query_symbol) or np.any(values == query_symbol)
    ):
        raise ValueError(
            "short condition must omit query_symbol on both sides of the mapping"
        )

    codebook = build_codebook(config)
    inputs = np.zeros((config.total_steps, config.input_width), dtype=np.float32)
    for slot, (key, value) in enumerate(zip(keys, values, strict=True)):
        span = slice(slot * config.symbol_ticks, (slot + 1) * config.symbol_ticks)
        inputs[span, config.key_slice] = codebook[key]
        inputs[span, config.value_slice] = codebook[value]
        inputs[span, config.slot_slice.start + slot] = 1.0
        inputs[span, config.phase_slice.start] = 1.0
    inputs[config.query_slice, config.key_slice] = codebook[query_symbol]
    inputs[config.query_slice, config.phase_slice.start + 1] = 1.0
    inputs[config.latent_slice, config.phase_slice.start + 3] = 1.0
    if config.latent_steps:
        inputs[config.latent_slice.start, config.phase_slice.start + 3] = 0.0
        inputs[config.latent_slice.start, config.phase_slice.start + 2] = 1.0
        inputs[config.latent_slice, config.clock_slice] = latent_clock_code(
            config.latent_steps, config.clock_width
        )
    inputs.setflags(write=False)
    rule_array.setflags(write=False)
    return Episode(
        config=config,
        model_inputs=inputs,
        rule=rule_array,
        demonstration_keys=keys,
        demonstration_values=values,
        query_symbol=query_symbol,
        terminal_target=terminal_target,
        condition=condition,
        codebook=codebook,
    )


def generate_matched_episodes(
    config: TaskConfig, rng: brainstate.random.RandomState
) -> MatchedEpisodes:
    """Generate supported and short views of one base episode.

    The two views share a rule, query, terminal target, and byte-identical query
    and latent inputs. They differ in one demonstration slot: the supported
    view contains the queried binding there, while the short view contains an
    unrelated binding and contains the query symbol on neither mapping side.

    Parameters
    ----------
    config : TaskConfig
        Episode dimensions and encoding.
    rng : brainstate.random.RandomState
        Stateful random stream owned by the caller.

    Returns
    -------
    MatchedEpisodes
        The two controlled views.
    """
    rule = draw_rule(config.symbol_count, rng)
    query_symbol = int(np.asarray(rng.randint(config.symbol_count)))
    terminal_target = oracle_answer(rule, query_symbol)
    candidates = np.flatnonzero(
        (np.arange(config.symbol_count) != query_symbol) & (rule != query_symbol)
    )
    candidates = np.asarray(rng.permutation(candidates), dtype=np.int32)
    if candidates.size < config.binding_count:
        raise ValueError(
            "symbol_count does not leave binding_count short-context candidates: "
            f"{config.symbol_count} symbols, {config.binding_count} bindings"
        )
    short_keys = candidates[: config.binding_count].copy()
    supported_keys = short_keys.copy()
    support_slot = int(np.asarray(rng.randint(config.binding_count)))
    supported_keys[support_slot] = query_symbol
    short_values = rule[short_keys]
    supported_values = rule[supported_keys]
    kwargs = {"config": config, "rule": rule, "query_symbol": query_symbol}
    supported = build_episode(
        **kwargs,
        demonstration_keys=supported_keys,
        demonstration_values=supported_values,
        terminal_target=terminal_target,
        condition="supported",
    )
    short = build_episode(
        **kwargs,
        demonstration_keys=short_keys,
        demonstration_values=short_values,
        terminal_target=terminal_target,
        condition="short",
    )
    return MatchedEpisodes(supported=supported, short=short)


def generate_episode(
    config: TaskConfig,
    rng: brainstate.random.RandomState,
    *,
    condition: Condition = "supported",
) -> Episode:
    """Generate one fresh-rule episode in the requested context condition.

    Parameters
    ----------
    config : TaskConfig
        Episode dimensions and encoding.
    rng : brainstate.random.RandomState
        Stateful random stream owned by the caller.
    condition : {"supported", "short"}
        Context-support intervention to return.

    Returns
    -------
    Episode
        One encoded episode.
    """
    if condition not in ("supported", "short"):
        raise ValueError(f"condition must be 'supported' or 'short', got {condition!r}")
    pair = generate_matched_episodes(config, rng)
    if condition == "supported":
        return pair.supported
    return pair.short
