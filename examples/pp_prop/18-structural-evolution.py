"""18 · Multi-task structural evolution under pp-prop.

One sparse recurrent LIF network learns interleaved delayed-cue temporal
"tricks" continually while its recurrent synapses are pruned, regrown, and
re-budgeted once per round. The question is whether the wiring
self-organizes into task-specialized sections, whether parts stay shared,
whether the network keeps doing every trick at the same quality — and how
few synapses it actually needs to do so.

Two arms run on identical initial weights, identical topology (1024
irregular CSR edges), and identical trial seeds. The control arm rebuilds
the same frozen topology every round so the optimizer reset at each rebuild
is matched and topology is the only asymmetry. The evolving arm runs an
adaptive budget controller after each round's evaluation (the v1 behavior —
prune the weakest 5% by ``|w|`` and respawn the same count on a constant
budget — remains available via ``--fixed-budget``):

- weakest trick below ``--target-accuracy`` (default 0.95) → GROW: add
  activity-biased edges (endpoints proportional to per-neuron mean spike
  rate on probe trials plus a small floor, no self-loops, no within-row
  duplicates, values drawn from the initial value distribution), growing
  the budget by ``--growth-factor`` (default 1.1x) up to ``--max-edges``
  (default 1048576);
- every trick at or above target → SHRINK: prune the weakest 10% by ``|w|``
  without replacement, never below ``--min-edges`` (default 64).

The expected trace is a sawtooth — shrink until the weakest trick dips
below target, grow back — settling near the minimal sufficient budget. The
edge count over rounds is part of the story: the right PNG panel draws it
in grey behind the accuracy curves, and the plain-English report narrates
the budget journey with per-round wall times. At this scale edge count
barely moves round time (fixed per-step costs and the dense input
projection dominate the sparse recurrent edges), and the report says so
rather than claiming speedups.

Surviving edge values, the dense input projection, and the readout weights
carry over across every rebuild; new edges start with zero attribution
mass. Per-edge task attribution is the accumulated per-task
absolute-gradient mass on the CSR values during training; an edge is
task-leaning when its top task holds more than 60% of its total mass,
otherwise it is shared.

The run ends with the report and one PNG (matplotlib Agg, ``--plot-output``,
default ``structural_evolution.png``): the left panel scatters the evolved
final adjacency colored by attribution (one color per trick, grey =
shared); the right panel shows per-task accuracy over rounds (solid =
evolving, dashed = control, chance line at 1/N) with the evolving arm's
synapse count on a grey twin axis.

Task styles (``--task-style``, default ``simple``):

- ``simple``: detect-then-respond trials — cue on input units 0-31 (fetch)
  or 32-63 (roll over) at ticks 0-4, respond at ticks 10-14 / 30-34 with
  the slower example-09 time constants.
- ``temporal-credit``: example 17's delayed-cue recall task (data encoding
  imported from ``temporal_benchmark_data``; no benchmark governance is
  involved). Each trick gets an 8-channel cue ensemble presented at ticks
  0-3 with physical-rate encoding (200 Hz, ``p = 1 - exp(-rate*dt)``), and
  a common go channel fires identically in every response window, so
  response inputs are label-independent and only the active trick's window
  is supervised. Example 17's fast time constants (membrane 0.5 ms,
  feed-forward synapse 0.5 ms, recurrent synapse 3 ms, readout 0.5 ms)
  kill passive membrane traces, so the task is solvable only by
  regenerating activity through the recurrent edges — which is what makes
  the budget controller's GROW path fire organically when the budget gets
  too small. The pp-prop trace decay becomes half-life 20 (example 17's
  selected medium-horizon value) and recurrent init values use example
  17's ``gain / sqrt(degree)`` scale with gain 0.8.

``--num-tricks`` (default 2, temporal-credit style only for more) scales
the number of tricks competing for the same neurons: trick ``k`` gets cue
channels ``8k..8k+7``, the go channel sits at index ``8N``, and an N-unit
softmax readout is supervised only in trick ``k``'s response window. Two
tricks keep the v3 geometry (respond at 6-9 and 26-29, example 17's short
and medium horizons); three or more space windows every 7 ticks (6-9,
13-16, 20-23, 27-30 for four), never exceeding the medium horizon. Chance
accuracy is 1/N.

``--task-style context`` (v7) makes the tricks context-dependent: the same
cue demands a different answer depending on what came earlier, so passive
detection cannot work and input channels cannot be privately partitioned
per trick. Two cue ensembles (A = units 0-7, B = units 8-15) and one
context ensemble (X = units 16-23) share the channel budget, and the go
channel (unit 24) fires in all four response windows. X fires at ticks 0-3
on half the trials, the cue at ticks 5-8, and the four conditions — A
alone (respond 12-15), A with context (16-19), B alone (20-23), B with
context (24-27) — each supervise their own window with label = condition
(chance 25%). Cue-detection wiring is forced to be shared, since both
conditions of a cue share its channels, so only context-keeping circuitry
can specialize; and X must be remembered ~20 ticks past its extinction, at
the edge of pp-prop's medium horizon. Time constants, encoding, trace
decay, and gain follow the temporal-credit style.

Optimization follows example 17's per-group policy by default (v5): one
Adam instance per parameter group — readout 3e-3, feed-forward 1e-3,
recurrent 3e-4 — with per-group gradient clipping at 1.0, stepped in that
order.

Growth is paced against consolidation capacity (v6): the budget grows by
1.1x per round (was 1.5x) and each round trains 800 trials (was 400).
Growth spurts inject fresh random edges and each rebuild resets the Adam
moments, so at example 17's slow recurrent rate (3e-4) a 1.5x spurt
outpaced what 400 trials could consolidate — per-trick accuracies churned
and the growing arm fell behind its own control. The gentler step and the
doubled round budget give each new layer of synapses time to integrate.

Growth can also aim at the error signal instead of raw activity (v8):
``--grow-rule gradient`` scores candidate edges by their accumulated
per-edge gradient marginals — post-synaptic demand times pre-synaptic
supply, a rank-one estimate of where a dense gradient would concentrate —
rather than by spike rate. pp-prop only differentiates edges that exist,
so the marginals are the cheap proxy; the activity floor keeps every free
position reachable, and zero mass everywhere falls back to a uniform draw.

Topology rebuilds used to discard all optimizer state: each round's new
CSR meant fresh Adam moments, and the growing arm — rebuilding every round
— paid the miscalibration tax every round. Rebuilds now carry Adam's
``mu``/``nu`` and step counts (v9, default): dense groups copy over
unchanged, and the recurrent group's per-edge moments follow edge
identity, so surviving synapses keep their moments and newborn synapses
cold-start at zero.

Claims are illustrative example output, not benchmark evidence: the only
gates are finite losses and structural invariants. A full default CPU run
(1024 neurons, 5 rounds, 800 trials per round, both arms) finishes in about
two minutes. Use ``--smoke`` (32 neurons, 32 edges, 1 round, 8 trials) for
a fast iteration check.

Spec: ``docs/specs/2026-08-10-structural-evolution-example.md``.
"""

import argparse
import importlib.util
import math
import pathlib
import sys
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braintrace


def _load_sparse_example():
    path = pathlib.Path(__file__).resolve().with_name("09-operator-sparse.py")
    spec = importlib.util.spec_from_file_location("_pp_prop_sparse_operator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load sparse pp-prop operators from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SPARSE = _load_sparse_example()

FETCH = 0
ROLL_OVER = 1
SHARED = 2
TASK_NAMES = {FETCH: "fetch", ROLL_OVER: "roll over"}

_CUE_UNITS = 32
_CUE_TICKS = 5
_RESPONSE_TICKS = 5
_RESPONSE_START = {FETCH: 10, ROLL_OVER: 30}

_SIMPLE_STYLE = "simple"
_TEMPORAL_STYLE = "temporal-credit"
_CONTEXT_STYLE = "context"
_TASK_STYLES = (_SIMPLE_STYLE, _TEMPORAL_STYLE, _CONTEXT_STYLE)

_TRICK_NAMES = ("fetch", "roll over", "sit", "stay")
_CONTEXT_NAMES = ("A alone", "A with context", "B alone", "B with context")
_TRICK_COLORS = ("tab:red", "tab:blue", "tab:green", "tab:orange")
_SHARED_COLOR = "0.6"

# Example 17's horizons: short is (cue 4, silence 2, response 4), medium is
# (cue 4, silence 22, response 4). Two tricks keep the short and medium
# horizons; more tricks space response windows every 7 ticks, so the latest
# onset (27 for four tricks) stays within one tick of the medium onset (26)
# — the longest delay pp-prop demonstrably learns directly (its selected
# medium trace half-life is 20 steps). The long horizon (silence 92) sits
# beyond pp-prop's eligibility horizon without a curriculum and is not used.
_TEMPORAL_CUE_TICKS = 4
_TEMPORAL_RESPONSE_TICKS = 4
_TEMPORAL_WINDOW_STRIDE = 7
_TEMPORAL_MAX_RESPONSE_START = 27


def _load_temporal_data():
    """Import example 17's trial-encoding module (pure NumPy, no governance).

    Only ``temporal_benchmark_data`` (and its constants-only config import)
    is loaded — manifest, sealing, and runner machinery are untouched.
    """
    example_dir = pathlib.Path(__file__).resolve().parent
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))
    import temporal_benchmark_data

    return temporal_benchmark_data


@dataclass(frozen=True)
class _TaskLayout:
    """Concrete trial geometry for one task style and trick count."""

    n_in: int
    n_step: int
    cue_channels: int
    cue_ticks: int
    response_ticks: int
    response_start: Dict[int, int]
    go_channel: Optional[int]
    rate_based: bool
    cue_start_tick: int = 0
    cue_offset: Optional[Dict[int, int]] = None
    context_channels: Optional[Tuple[int, int]] = None
    context_window: Optional[Tuple[int, int]] = None


def _layout(config: "_EvolutionConfig") -> _TaskLayout:
    """Resolve the trial geometry implied by ``config.task_style``."""
    if config.task_style == _CONTEXT_STYLE:
        return _TaskLayout(
            n_in=25,
            n_step=28,
            cue_channels=8,
            cue_ticks=4,
            response_ticks=4,
            response_start={0: 12, 1: 16, 2: 20, 3: 24},
            go_channel=24,
            rate_based=True,
            cue_start_tick=5,
            cue_offset={0: 0, 1: 0, 2: 8, 3: 8},
            context_channels=(16, 24),
            context_window=(0, 4),
        )
    if config.task_style == _TEMPORAL_STYLE:
        data = _load_temporal_data()
        if config.num_tricks == 2:
            response_start = {FETCH: 6, ROLL_OVER: 26}
            n_step = 30
        else:
            response_start = {
                trick: 6 + _TEMPORAL_WINDOW_STRIDE * trick
                for trick in range(config.num_tricks)
            }
            n_step = 9 + _TEMPORAL_WINDOW_STRIDE * (config.num_tricks - 1) + 1
        return _TaskLayout(
            n_in=data.CLASS_CHANNELS * config.num_tricks + 1,
            n_step=n_step,
            cue_channels=data.CLASS_CHANNELS,
            cue_ticks=_TEMPORAL_CUE_TICKS,
            response_ticks=_TEMPORAL_RESPONSE_TICKS,
            response_start=response_start,
            go_channel=data.CLASS_CHANNELS * config.num_tricks,
            rate_based=True,
        )
    return _TaskLayout(
        n_in=config.n_in,
        n_step=config.n_step,
        cue_channels=_CUE_UNITS,
        cue_ticks=_CUE_TICKS,
        response_ticks=_RESPONSE_TICKS,
        response_start=dict(_RESPONSE_START),
        go_channel=None,
        rate_based=False,
    )


@dataclass(frozen=True)
class _EvolutionConfig:
    """Configuration for one structural-evolution run (both arms)."""

    seed: int = 0
    n_rec: int = 1024
    n_edges: int = 1024
    n_in: int = 64
    n_step: int = 40
    n_rounds: int = 5
    trials_per_round: int = 800
    eval_trials_per_task: int = 16
    rate_probe_trials: int = 8
    prune_fraction: float = 0.05
    activity_floor: float = 1e-3
    share_threshold: float = 0.6
    readout_learning_rate: float = 3e-3
    feedforward_learning_rate: float = 1e-3
    recurrent_learning_rate: float = 3e-4
    decay_or_rank: float = 0.95
    clip_norm: float = 1.0
    cue_probability: float = 0.5
    recurrent_scale: float = 1.0
    sparse_backend: Optional[str] = "jax_raw"
    fixed_budget: bool = False
    target_accuracy: float = 0.95
    max_edges: int = 1_048_576
    min_edges: int = 64
    growth_factor: float = 1.1
    shrink_fraction: float = 0.1
    grow_rule: str = "activity"
    max_growth_events: Optional[int] = None
    carry_optimizer_state: bool = True
    task_style: str = "simple"
    cue_rate_hz: float = 200.0
    go_rate_hz: float = 200.0
    dt_seconds: float = 0.001
    trace_half_life: float = 20.0
    recurrent_gain: float = 0.8
    num_tricks: int = 2

    def __post_init__(self) -> None:
        positive = {
            "n_rec": self.n_rec,
            "n_edges": self.n_edges,
            "n_in": self.n_in,
            "n_step": self.n_step,
            "n_rounds": self.n_rounds,
            "trials_per_round": self.trials_per_round,
            "eval_trials_per_task": self.eval_trials_per_task,
            "rate_probe_trials": self.rate_probe_trials,
            "readout_learning_rate": self.readout_learning_rate,
            "feedforward_learning_rate": self.feedforward_learning_rate,
            "recurrent_learning_rate": self.recurrent_learning_rate,
            "clip_norm": self.clip_norm,
            "activity_floor": self.activity_floor,
            "min_edges": self.min_edges,
            "max_edges": self.max_edges,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"{', '.join(invalid)} must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not 0.0 < self.prune_fraction < 1.0:
            raise ValueError("prune_fraction must be in (0, 1)")
        if not 0.5 < self.share_threshold < 1.0:
            raise ValueError("share_threshold must be in (0.5, 1)")
        if not 0.0 < self.cue_probability <= 1.0:
            raise ValueError("cue_probability must be in (0, 1]")
        if self.recurrent_scale <= 0:
            raise ValueError("recurrent_scale must be positive")
        if self.task_style not in _TASK_STYLES:
            raise ValueError(f"task_style must be one of {_TASK_STYLES}")
        if self.task_style == _SIMPLE_STYLE:
            if self.n_in < 2 * _CUE_UNITS:
                raise ValueError(f"n_in must be at least {2 * _CUE_UNITS}")
            latest_response = max(_RESPONSE_START.values()) + _RESPONSE_TICKS
            if self.n_step < latest_response:
                raise ValueError(f"n_step must be at least {latest_response}")
        for name in ("cue_rate_hz", "go_rate_hz", "dt_seconds", "trace_half_life"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.recurrent_gain) or self.recurrent_gain <= 0:
            raise ValueError("recurrent_gain must be finite and positive")
        if isinstance(self.num_tricks, bool) or not 2 <= self.num_tricks <= len(
            _TRICK_NAMES
        ):
            raise ValueError(f"num_tricks must be between 2 and {len(_TRICK_NAMES)}")
        if self.num_tricks > 2 and self.task_style == _SIMPLE_STYLE:
            raise ValueError("num_tricks above two requires a temporal task style")
        if self.task_style == _CONTEXT_STYLE and self.num_tricks != 4:
            raise ValueError(
                "the context style has exactly four conditions (num_tricks=4)"
            )
        if self.task_style != _SIMPLE_STYLE:
            last_start = max(_layout(self).response_start.values())
            if last_start > _TEMPORAL_MAX_RESPONSE_START:
                raise ValueError(
                    "response windows must stay at or inside the medium horizon"
                )
        if self.n_edges > self.n_rec * (self.n_rec - 1):
            raise ValueError("n_edges exceeds the off-diagonal positions")
        if isinstance(self.decay_or_rank, bool):
            raise ValueError("decay_or_rank must be a float decay or integer rank")
        if isinstance(self.decay_or_rank, int):
            if self.decay_or_rank < 1:
                raise ValueError("integer decay_or_rank must be at least one")
        elif isinstance(self.decay_or_rank, float):
            valid_decay = math.isfinite(self.decay_or_rank) and (
                0.0 <= self.decay_or_rank < 1.0
            )
            if not valid_decay:
                raise ValueError("float decay_or_rank must be in [0, 1)")
        else:
            raise ValueError("decay_or_rank must be a float decay or integer rank")
        if self.sparse_backend == "":
            raise ValueError("sparse_backend must be non-empty or None")
        # Values outside [0, 1] are deliberately legal: 0.0 forces shrink and
        # anything above 1.0 forces growth, which the tests and demos exploit.
        if not math.isfinite(self.target_accuracy) or self.target_accuracy < 0:
            raise ValueError("target_accuracy must be finite and non-negative")
        if self.max_edges < self.min_edges:
            raise ValueError("max_edges must be at least min_edges")
        if not self.growth_factor > 1.0:
            raise ValueError("growth_factor must exceed one")
        if self.grow_rule not in ("activity", "gradient"):
            raise ValueError("grow_rule must be 'activity' or 'gradient'")
        if self.max_growth_events is not None and (
            isinstance(self.max_growth_events, bool) or self.max_growth_events < 0
        ):
            raise ValueError("max_growth_events must be a non-negative int or None")
        if not 0.0 < self.shrink_fraction < 1.0:
            raise ValueError("shrink_fraction must be in (0, 1)")

    @classmethod
    def smoke(cls) -> "_EvolutionConfig":
        """Tiny fast configuration for tests and ``--smoke`` runs."""
        return cls(
            n_rec=32,
            n_edges=32,
            n_rounds=1,
            trials_per_round=8,
            eval_trials_per_task=4,
            rate_probe_trials=4,
            min_edges=16,
            max_edges=64,
        )

    def prune_count(self) -> int:
        """Fixed-budget mode: edges pruned (and respawned) each round."""
        return max(1, round(self.n_edges * self.prune_fraction))


# --- Topology helpers (pure NumPy, unit-tested) ----------------------------


def _draw_free_pairs(
    n_rec: int,
    count: int,
    row_weight: Optional[np.ndarray],
    col_weight: Optional[np.ndarray],
    keep_rows: np.ndarray,
    keep_cols: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``count`` distinct free off-diagonal pairs without dense matrices.

    Rows are drawn in proportion to ``row_weight`` and columns to
    ``col_weight`` (uniform when the weight is ``None``), so a candidate
    ``(i, j)`` lands with probability proportional to
    ``row_weight[i] * col_weight[j]`` — the same product distribution the
    dense joint-matrix draws produced, but sampled by independent endpoint
    draws with rejection of self-loops, pairs already among
    ``(keep_rows, keep_cols)``, and duplicates. Memory stays O(n_rec +
    n_edges) at any ``n_rec``.

    Weighted placement is a preference, not a constraint: if the weighted
    draws stop finding free pairs even at the maximum batch (the high-mass
    region is saturated with existing edges), the draw degrades to uniform
    over the remaining free pairs instead of failing.

    Parameters
    ----------
    n_rec : int
        Number of recurrent neurons; rows and columns both index it.
    count : int
        Number of new pairs to draw.
    row_weight, col_weight : np.ndarray or None
        Per-neuron sampling weights for the two endpoints; ``None`` means
        uniform. Weights must be non-negative with positive total.
    keep_rows, keep_cols : np.ndarray
        Endpoints of the existing edges; drawn pairs never duplicate them.
    rng : np.random.Generator
        Host-side randomness for the draw.

    Returns
    -------
    np.ndarray
        int64 flat indices ``row * n_rec + col`` of the drawn pairs.

    Raises
    ------
    ValueError
        If fewer than ``count`` free off-diagonal positions exist, a weight
        vector has no positive mass, or the draw fails to converge.
    """
    capacity = n_rec * (n_rec - 1) - keep_rows.size
    if count > capacity:
        raise ValueError("no free off-diagonal positions left to draw from")
    blocked = keep_rows.astype(np.int64) * n_rec + keep_cols.astype(np.int64)

    def _probabilities(weight: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if weight is None:
            return None
        total = weight.sum()
        if total <= 0.0:
            raise ValueError("sampling weights carry no positive mass")
        return weight / total

    p_row = _probabilities(row_weight)
    p_col = _probabilities(col_weight)
    accepted = np.empty(0, dtype=np.int64)
    batch = max(64, 2 * count)
    plans = [(p_row, p_col)]
    if p_row is not None or p_col is not None:
        plans.append((None, None))
    for plan_row, plan_col in plans:
        for _ in range(1000):
            if accepted.size >= count:
                return rng.permutation(accepted)[:count]
            rows = (
                rng.integers(0, n_rec, size=batch)
                if plan_row is None
                else rng.choice(n_rec, size=batch, replace=True, p=plan_row)
            )
            cols = (
                rng.integers(0, n_rec, size=batch)
                if plan_col is None
                else rng.choice(n_rec, size=batch, replace=True, p=plan_col)
            )
            flat = rows.astype(np.int64) * n_rec + cols.astype(np.int64)
            free = (rows != cols) & ~np.isin(flat, blocked)
            grown = np.unique(np.concatenate([accepted, flat[free]]))
            if grown.size == accepted.size:
                if batch >= 1 << 22:
                    break
                batch = min(batch * 16, 1 << 22)
            else:
                batch = max(64, 2 * (count - grown.size))
            accepted = grown
        batch = max(64, 2 * (count - accepted.size))
    raise ValueError("pair draw failed to converge on free positions")


def _sample_irregular_topology(
    n_rec: int, n_edges: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``n_edges`` distinct off-diagonal (row, col) pairs uniformly.

    Parameters
    ----------
    n_rec : int
        Number of recurrent neurons; rows and columns both index it.
    n_edges : int
        Number of edges to draw; may not exceed ``n_rec * (n_rec - 1)``.
    rng : np.random.Generator
        Host-side randomness for the draw.

    Returns
    -------
    tuple of np.ndarray
        ``(rows, cols)`` int64 arrays sorted lexicographically by row then
        column, with no self-loops and no duplicate pairs.
    """
    empty = np.empty(0, dtype=np.int64)
    flat = _draw_free_pairs(n_rec, n_edges, None, None, empty, empty, rng)
    rows, cols = flat // n_rec, flat % n_rec
    order = np.lexsort((cols, rows))
    return rows[order].astype(np.int64), cols[order].astype(np.int64)


def _sort_edges(
    rows: np.ndarray, cols: np.ndarray, *arrays: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, ...]]:
    """Sort edges by (row, col), applying the same order to ``arrays``."""
    order = np.lexsort((cols, rows))
    return (
        rows[order],
        cols[order],
        tuple(array[order] for array in arrays),
    )


def _indptr_from_rows(rows: np.ndarray, n_rec: int) -> np.ndarray:
    """Build the CSR row-pointer array from sorted row indices."""
    indptr = np.zeros(n_rec + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    return np.cumsum(indptr).astype(np.int32)


def _rows_from_indptr(indptr: np.ndarray) -> np.ndarray:
    """Recover sorted row indices from a CSR row-pointer array."""
    counts = np.diff(np.asarray(indptr, dtype=np.int64))
    return np.repeat(np.arange(counts.size, dtype=np.int64), counts)


def _to_csr(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    n_rec: int,
    backend: Optional[str],
):
    """Assemble sorted (row, col, value) edges into a ``brainevent.CSR``."""
    import brainevent

    rows, cols, (values,) = _sort_edges(rows, cols, values)
    return brainevent.CSR(
        jnp.asarray(values, dtype=jnp.float32),
        jnp.asarray(cols, dtype=jnp.int32),
        jnp.asarray(_indptr_from_rows(rows, n_rec)),
        shape=(n_rec, n_rec),
        backend=backend,
    )


def _prune_survivors(values: np.ndarray, prune_count: int) -> np.ndarray:
    """Indices of the edges that survive pruning of the weakest ``|w|``.

    Parameters
    ----------
    values : np.ndarray
        Current CSR edge values, one per edge.
    prune_count : int
        Number of edges to remove; must be smaller than ``values.size``.

    Returns
    -------
    np.ndarray
        Sorted indices of the surviving edges; the removed ones are exactly
        the ``prune_count`` entries with the smallest absolute value (ties
        broken by stable sort order).
    """
    if not 0 < prune_count < values.size:
        raise ValueError("prune_count must leave at least one survivor")
    order = np.argsort(np.abs(values), kind="stable")
    return np.sort(order[prune_count:])


def _respawn_endpoints(
    n_rec: int,
    count: int,
    rates: np.ndarray,
    floor: float,
    keep_rows: np.ndarray,
    keep_cols: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``count`` new edges with endpoints biased toward active neurons.

    The sampling weight of candidate edge ``(i, j)`` is
    ``(rates[i] + floor) * (rates[j] + floor)``; self-loops and edges that
    already exist among ``(keep_rows, keep_cols)`` get zero weight, so the
    floor alone keeps every free off-diagonal position reachable even when
    the network is silent.

    Parameters
    ----------
    n_rec : int
        Number of recurrent neurons.
    count : int
        Number of new edges to draw (without replacement).
    rates : np.ndarray
        Per-neuron mean spike rate measured on probe trials.
    floor : float
        Activity floor added to every rate before sampling.
    keep_rows, keep_cols : np.ndarray
        Endpoints of the surviving edges; respawned edges may not duplicate
        any of these pairs.
    rng : np.random.Generator
        Host-side randomness for the draw.

    Returns
    -------
    tuple of np.ndarray
        ``(new_rows, new_cols)`` int64 arrays of length ``count``.
    """
    activity = np.asarray(rates, dtype=np.float64) + floor
    flat = _draw_free_pairs(
        n_rec, count, activity, activity, keep_rows, keep_cols, rng
    )
    return (flat // n_rec).astype(np.int64), (flat % n_rec).astype(np.int64)


def _gradient_endpoints(
    n_rec: int,
    count: int,
    grad_mass: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    floor: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``count`` new edges where the error signal is loudest (v8).

    pp-prop only differentiates edges that exist, so the dense candidate
    gradient is unavailable. Instead each neuron's gradient marginals come
    from the accumulated per-edge mass: a post-synaptic neuron's *demand* is
    the mass summed over its incoming edges, a pre-synaptic neuron's
    *supply* is the mass summed over its outgoing edges, and candidate
    ``(row, col)`` scores ``(demand[row] + floor) * (supply[col] + floor)`` —
    a rank-one estimate of where a dense gradient would concentrate. The
    floor keeps every free off-diagonal position reachable; with zero mass
    everywhere the draw falls back to uniform.

    Parameters
    ----------
    n_rec : int
        Number of recurrent neurons.
    count : int
        Number of new edges to draw (without replacement).
    grad_mass : np.ndarray
        Per-edge accumulated absolute-gradient mass, aligned with
        ``rows``/``cols``.
    rows, cols : np.ndarray
        Endpoints of the existing edges; candidates may not duplicate them.
    floor : float
        Additive floor on both marginals before taking the product.
    rng : np.random.Generator
        Host-side randomness for the draw.

    Returns
    -------
    tuple of np.ndarray
        ``(new_rows, new_cols)`` int64 arrays of length ``count``.
    """
    if grad_mass.shape != rows.shape or rows.shape != cols.shape:
        raise ValueError("grad_mass, rows and cols must align")
    demand = np.zeros(n_rec, dtype=np.float64)
    supply = np.zeros(n_rec, dtype=np.float64)
    np.add.at(demand, rows, grad_mass)
    np.add.at(supply, cols, grad_mass)
    flat = _draw_free_pairs(
        n_rec, count, demand + floor, supply + floor, rows, cols, rng
    )
    return (flat // n_rec).astype(np.int64), (flat % n_rec).astype(np.int64)


def _classify_attribution(
    *masses: np.ndarray, threshold: float = 0.6
) -> np.ndarray:
    """Label each edge by task or as shared, from per-task gradient mass.

    An edge leans toward a task when that task holds more than ``threshold``
    of the edge's total accumulated mass; everything else — including edges
    with no recorded mass at all — is shared. Works for any task count;
    the shared label equals the number of tasks (``SHARED`` for two).

    Parameters
    ----------
    *masses : np.ndarray
        One per-edge accumulated absolute-gradient mass array per task.
    threshold : float
        Share above which an edge counts as task-leaning.

    Returns
    -------
    np.ndarray
        int32 labels: the task index for task-leaning edges, or
        ``len(masses)`` for shared edges.
    """
    stacked = np.stack([np.asarray(mass, dtype=np.float64) for mass in masses])
    total = stacked.sum(axis=0)
    top = stacked.argmax(axis=0)
    top_share = np.divide(
        stacked.max(axis=0), total, out=np.full_like(total, 0.5), where=total > 0
    )
    labels = np.full(total.shape, len(masses), dtype=np.int32)
    labels[top_share > threshold] = top[top_share > threshold]
    return labels


def _next_budget(current: int, min_accuracy: float, config: _EvolutionConfig) -> int:
    """Next round's edge budget under the adaptive controller.

    Grow when the weakest trick is below ``target_accuracy`` — by
    ``growth_factor`` (at least one edge), never above ``max_edges`` or the
    off-diagonal capacity. Otherwise shrink by ``shrink_fraction`` (at least
    one edge), never below ``min_edges`` and never growing through the floor
    when the current budget already sits under it.
    """
    capacity = config.n_rec * (config.n_rec - 1)
    if min_accuracy < config.target_accuracy:
        grown = max(current + 1, round(current * config.growth_factor))
        return min(grown, config.max_edges, capacity)
    pruned = current - max(1, round(current * config.shrink_fraction))
    return max(min(config.min_edges, current), pruned)


# --- Model -----------------------------------------------------------------


class _Net(brainstate.nn.Module):
    """LIF recurrent net on a supplied CSR with dense input and leaky readout."""

    def __init__(self, config: _EvolutionConfig, csr: Any):
        super().__init__()
        import brainpy.state

        layout = _layout(config)
        temporal = config.task_style != _SIMPLE_STYLE
        # Example 17's fast constants (membrane 0.5 ms, ff synapse 0.5 ms,
        # recurrent synapse 3 ms, readout 0.5 ms) leave no passive trace to
        # read out, so memory must be regenerated through the recurrence;
        # the simple style keeps the slower example-09 constants.
        membrane_tau = 0.5 if temporal else 20.0
        ff_tau = 0.5 if temporal else 10.0
        rec_tau = 3.0 if temporal else 10.0
        self.neu = brainpy.state.LIF(
            config.n_rec,
            R=1.0 * u.ohm,
            tau=membrane_tau * u.ms,
            V_th=1.0 * u.mV,
            V_reset=0.0 * u.mV,
            V_rest=0.0 * u.mV,
            V_initializer=braintools.init.ZeroInit(unit=u.mV),
        )
        if temporal:
            ff_w = brainstate.random.randn(layout.n_in, config.n_rec)
            ff_w = ff_w * (6.0 / layout.n_in**0.5) * u.mA
            ff_bias = None
        else:
            ff_w = braintools.init.KaimingNormal(6.0, unit=u.mA)(
                (layout.n_in, config.n_rec)
            )
            ff_bias = braintools.init.ZeroInit(unit=u.mA)
        rec_linear = braintrace.nn.SparseLinear(csr, b_init=None)
        rec_params = dict(rec_linear.weight.value)
        rec_params["weight"] = rec_params["weight"] * u.mA
        rec_linear.weight.value = rec_params
        self.ff_syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                layout.n_in,
                config.n_rec,
                w_init=ff_w,
                b_init=ff_bias,
            ),
            syn=brainpy.state.Expon(
                config.n_rec,
                tau=ff_tau * u.ms,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=brainpy.state.CUBA(scale=1.0),
            post=self.neu,
        )
        self.rec_syn = brainpy.state.AlignPostProj(
            comm=rec_linear,
            syn=brainpy.state.Expon(
                config.n_rec,
                tau=rec_tau * u.ms,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=brainpy.state.CUBA(scale=1.0),
            post=self.neu,
        )
        if temporal:
            readout_w = brainstate.random.randn(config.n_rec, config.num_tricks)
            readout_w = readout_w / config.n_rec**0.5
            self.readout = braintrace.nn.LeakyRateReadout(
                in_size=config.n_rec,
                out_size=config.num_tricks,
                tau=0.5 * u.ms,
                w_init=readout_w,
            )
        else:
            self.readout = _SPARSE._shared.LeakyReadout(
                n_rec=config.n_rec, n_out=config.num_tricks
            )

    def cell_step(self, x):
        self.ff_syn(x)
        self.rec_syn(self.neu.get_spike())
        self.neu(0.0 * u.mA)
        return self.neu.get_spike()

    def update(self, x):
        return self.readout(self.cell_step(x))


# --- Trials ------------------------------------------------------------------


def _rate_template(task: int, config: _EvolutionConfig) -> np.ndarray:
    """Physical-rate template ``(n_step, n_in)`` for one rate-based trial.

    The condition's cue ensemble fires for ``cue_ticks`` from
    ``cue_start_tick``; the context style additionally fires the X ensemble
    on its context window when the condition carries context, and the go
    channel fires identically in every response window so response inputs
    are label-independent.
    """
    layout = _layout(config)
    rates = np.zeros((layout.n_step, layout.n_in), dtype=np.float64)
    if layout.cue_offset is not None:
        cue0 = layout.cue_offset[task]
    else:
        cue0 = task * layout.cue_channels
    cue_end = layout.cue_start_tick + layout.cue_ticks
    rates[layout.cue_start_tick : cue_end, cue0 : cue0 + layout.cue_channels] = (
        config.cue_rate_hz
    )
    if layout.context_channels is not None and task % 2 == 1:
        t0, t1 = layout.context_window
        stop = layout.context_channels[1]
        rates[t0:t1, layout.context_channels[0] : stop] = config.cue_rate_hz
    for start in layout.response_start.values():
        rates[start : start + layout.response_ticks, layout.go_channel] = (
            config.go_rate_hz
        )
    return rates


def _make_trial(
    task: int, seed: int, config: _EvolutionConfig
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """One delayed-cue trial: spikes ``(n_step, 1, n_in)``, label, loss mask.

    Rate-based styles follow example 17 exactly: Bernoulli spikes with
    probability ``1 - exp(-rate * dt)`` from :func:`_rate_template`; only
    the active condition's response window is supervised.
    """
    layout = _layout(config)
    rng = np.random.default_rng(seed)
    if layout.rate_based:
        probabilities = -np.expm1(-_rate_template(task, config) * config.dt_seconds)
        spikes = (rng.random(probabilities.shape) < probabilities).astype(np.float32)
        spikes = spikes[:, None, :]
    else:
        spikes = np.zeros((layout.n_step, 1, layout.n_in), dtype=np.float32)
        cue = (
            rng.random((layout.cue_ticks, 1, layout.cue_channels))
            < config.cue_probability
        ).astype(np.float32)
        unit0 = task * layout.cue_channels
        spikes[: layout.cue_ticks, :, unit0 : unit0 + layout.cue_channels] = cue
    mask = np.zeros(layout.n_step, dtype=np.float32)
    start = layout.response_start[task]
    mask[start : start + layout.response_ticks] = 1.0
    labels = np.asarray([task], dtype=np.int32)
    return jnp.asarray(spikes), jnp.asarray(labels), jnp.asarray(mask)


def _trial_seed(config: _EvolutionConfig, round_index: int, trial_index: int) -> int:
    """Deterministic per-trial data seed, identical across the two arms."""
    return config.seed * 10_000_019 + round_index * 100_003 + trial_index


def _probe_seed(task: int, index: int) -> int:
    """Fixed accuracy-probe seed, identical across rounds and arms."""
    return 900_000_011 + task * 10_007 + index


def _rate_seed(index: int) -> int:
    """Fixed spike-rate-probe seed, identical across rounds and arms."""
    return 800_000_009 + index


# --- Experiment --------------------------------------------------------------


@dataclass
class _Experiment:
    """A compiled model plus its per-edge, per-task attribution accumulators."""

    model: _Net
    learner: Any
    optimizers: Dict[str, Any]
    groups: Dict[str, Tuple[Any, ...]]
    rec_key: Any
    task_mass: np.ndarray


def _parameter_group(path: Tuple[Any, ...]) -> str:
    """Map a parameter path onto example 17's optimizer group names."""
    names = {str(part) for part in path}
    if "readout" in names:
        return "readout"
    if "ff_syn" in names:
        return "feedforward"
    if "rec_syn" in names:
        return "recurrent"
    raise ValueError(f"parameter path {path} matches no optimizer group")


def _carry_params(old_model: _Net, new_model: _Net) -> None:
    """Copy every parameter from ``old_model`` except the recurrent CSR values.

    The recurrent values are defined by the rebuilt CSR itself; the dense
    input projection and the readout weights carry over unchanged.
    """
    rec_new = new_model.rec_syn.comm.weight
    old_states = old_model.states(brainstate.ParamState)
    for key, state in new_model.states(brainstate.ParamState).items():
        if state is not rec_new:
            state.value = old_states[key].value


def _build_experiment(
    config: _EvolutionConfig,
    csr: Any,
    donor: Optional[_Experiment] = None,
    task_mass: Optional[np.ndarray] = None,
) -> _Experiment:
    """Build and compile one arm state, optionally inheriting a donor's weights."""
    with brainstate.random.seed_context(config.seed + 1):
        model = _Net(config, csr)
    if donor is not None:
        _carry_params(donor.model, model)
    weights = model.states(brainstate.ParamState)
    brainstate.nn.init_all_states(model, batch_size=1)
    n_in = _layout(config).n_in
    # Example 17's selected medium-horizon trace half-life is 20 steps; the
    # simple style keeps example 15's decay.
    decay = (
        2.0 ** (-1.0 / config.trace_half_life)
        if config.task_style != _SIMPLE_STYLE
        else config.decay_or_rank
    )
    learner = braintrace.compile(
        model,
        braintrace.pp_prop,
        jnp.zeros((1, n_in), dtype=jnp.float32),
        batch_size=1,
        vmap=False,
        decay_or_rank=decay,
        vjp_method="single-step",
    )
    # Example 17's per-group optimizer policy: one Adam per parameter group,
    # stepped in order, with per-group clipping applied at update time.
    group_rates = {
        "readout": config.readout_learning_rate,
        "feedforward": config.feedforward_learning_rate,
        "recurrent": config.recurrent_learning_rate,
    }
    groups = {
        name: tuple(
            path for path in weights if _parameter_group(path) == name
        )
        for name in group_rates
    }
    optimizers = {}
    for name, keys in groups.items():
        optimizer = braintools.optim.Adam(lr=group_rates[name])
        optimizer.register_trainable_weights({key: weights[key] for key in keys})
        optimizers[name] = optimizer
    rec_key = next(
        key for key, state in weights.items() if state is model.rec_syn.comm.weight
    )
    n_edges = int(csr.data.shape[0])
    if task_mass is None:
        task_mass = np.zeros((config.num_tricks, n_edges), dtype=np.float64)
    return _Experiment(
        model=model,
        learner=learner,
        optimizers=optimizers,
        groups=groups,
        rec_key=rec_key,
        task_mass=task_mass,
    )


def _initial_experiment(config: _EvolutionConfig) -> _Experiment:
    """The shared starting point: one irregular random graph, one weight draw."""
    rng = np.random.default_rng(config.seed + 77)
    rows, cols = _sample_irregular_topology(config.n_rec, config.n_edges, rng)
    values = brainstate.random.RandomState(config.seed).randn(config.n_edges)
    values = np.asarray(values) * _value_std(config)
    csr = _to_csr(rows, cols, values, config.n_rec, config.sparse_backend)
    return _build_experiment(config, csr)


def _value_std(config: _EvolutionConfig) -> float:
    """Standard deviation of the initial (and respawned) edge values.

    The temporal-credit style uses example 17's ``gain / sqrt(degree)`` scale
    with the initial average degree ``n_edges / n_rec``; the simple style
    keeps example 09's ``(recurrent_scale / n_rec) ** 0.5``. The scale is
    computed from the initial budget, so respawned edges always draw from
    the same distribution no matter how the budget evolves.
    """
    if config.task_style != _SIMPLE_STYLE:
        return config.recurrent_gain * (config.n_rec / config.n_edges) ** 0.5
    return (config.recurrent_scale / config.n_rec) ** 0.5


def _current_topology(
    experiment: _Experiment,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (rows, cols, values) from the experiment's recurrent CSR."""
    linear = experiment.model.rec_syn.comm
    rows = _rows_from_indptr(np.asarray(linear.spar_mat.indptr))
    cols = np.asarray(linear.spar_mat.indices, dtype=np.int64)
    values = np.asarray(
        u.get_mantissa(linear.weight.value["weight"]), dtype=np.float64
    )
    return rows, cols, values


def _remap_edge_array(
    old_array: np.ndarray,
    old_rows: np.ndarray,
    old_cols: np.ndarray,
    new_rows: np.ndarray,
    new_cols: np.ndarray,
) -> np.ndarray:
    """Re-express a per-edge array on a new edge list, zero-filling new edges.

    Edges carry no identity beyond their ``(row, col)`` pair, which is
    unique by construction, so surviving entries land on their pair and
    edges born in the rebuild start at zero.
    """
    index = {
        (int(r), int(c)): i for i, (r, c) in enumerate(zip(old_rows, old_cols))
    }
    out = np.zeros(new_rows.shape[0], dtype=np.float64)
    for j, (r, c) in enumerate(zip(new_rows, new_cols)):
        i = index.get((int(r), int(c)))
        if i is not None:
            out[j] = float(old_array[i])
    return out


def _remap_edge_leaf(
    donor_leaf: Any,
    old_rows: np.ndarray,
    old_cols: np.ndarray,
    new_rows: np.ndarray,
    new_cols: np.ndarray,
) -> Any:
    """Remap one per-edge optimizer leaf, preserving any brainunit unit."""
    mantissa = _remap_edge_array(
        np.asarray(u.get_mantissa(donor_leaf), dtype=np.float64),
        old_rows,
        old_cols,
        new_rows,
        new_cols,
    )
    array = jnp.asarray(mantissa, dtype=jnp.float32)
    unit = getattr(donor_leaf, "unit", None)
    return array * unit if unit is not None else array


def _carry_optimizer_moments(
    donor: _Experiment,
    experiment: _Experiment,
    old_rows: np.ndarray,
    old_cols: np.ndarray,
) -> None:
    """Carry Adam moments and step counts across a topology rebuild (v9).

    Readout and feed-forward groups copy wholesale — their shapes never
    change. The recurrent group's per-edge ``mu``/``nu`` are remapped by
    edge identity: surviving edges keep their moments, newborn edges
    cold-start at zero. Without this, every rebuild silently restarted
    Adam, and the growing arm — rebuilding into a new shape every round —
    paid the miscalibration tax every round.
    """
    new_rows, new_cols, _ = _current_topology(experiment)
    for name, optimizer in experiment.optimizers.items():
        donor_state = donor.optimizers[name].opt_state.value
        if not isinstance(donor_state, tuple):
            continue  # donor never updated; nothing to carry
        adam_index = next(i for i, s in enumerate(donor_state) if hasattr(s, "mu"))
        donor_adam = donor_state[adam_index]
        mu_map: Dict[Any, Any] = {}
        nu_map: Dict[Any, Any] = {}
        for key, donor_mu in donor_adam.mu.items():
            donor_nu = donor_adam.nu[key]
            if name == "recurrent":
                mu_map[key] = {
                    "weight": _remap_edge_leaf(
                        donor_mu["weight"], old_rows, old_cols, new_rows, new_cols
                    )
                }
                nu_map[key] = {
                    "weight": _remap_edge_leaf(
                        donor_nu["weight"], old_rows, old_cols, new_rows, new_cols
                    )
                }
            else:
                mu_map[key] = donor_mu
                nu_map[key] = donor_nu
        new_adam = donor_adam._replace(mu=mu_map, nu=nu_map)
        optimizer.opt_state.value = tuple(
            new_adam if index == adam_index else element
            for index, element in enumerate(donor_state)
        )
        optimizer.step_count.value = donor.optimizers[name].step_count.value


def _rebuild_experiment(
    config: _EvolutionConfig,
    experiment: _Experiment,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    task_mass: np.ndarray,
) -> _Experiment:
    """Rebuild the compiled experiment on new edges, carrying weights and mass."""
    old_rows, old_cols, _ = _current_topology(experiment)
    rows, cols, (values, task_mass) = _sort_edges(
        rows, cols, values, task_mass.T
    )
    task_mass = task_mass.T
    csr = _to_csr(rows, cols, values, config.n_rec, config.sparse_backend)
    rebuilt = _build_experiment(config, csr, donor=experiment, task_mass=task_mass)
    if config.carry_optimizer_state:
        _carry_optimizer_moments(experiment, rebuilt, old_rows, old_cols)
    return rebuilt


def _respawn_values(count: int, config: _EvolutionConfig, round_index: int) -> np.ndarray:
    """Fresh values for respawned edges from the initial value distribution."""
    rand = brainstate.random.RandomState(config.seed * 7_919 + round_index)
    return np.asarray(rand.randn(count)) * _value_std(config)


def _evolve_topology(
    experiment: _Experiment,
    config: _EvolutionConfig,
    rates: np.ndarray,
    round_index: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-budget mode: prune the weakest edges, respawn the same count."""
    rows, cols, values = _current_topology(experiment)
    prune_count = config.prune_count()
    survivors = _prune_survivors(values, prune_count)
    new_rows, new_cols = _respawn_endpoints(
        config.n_rec,
        prune_count,
        rates,
        config.activity_floor,
        rows[survivors],
        cols[survivors],
        rng,
    )
    n_tasks = experiment.task_mass.shape[0]
    return (
        np.concatenate([rows[survivors], new_rows]),
        np.concatenate([cols[survivors], new_cols]),
        np.concatenate([values[survivors], _respawn_values(prune_count, config, round_index)]),
        np.concatenate(
            [experiment.task_mass[:, survivors], np.zeros((n_tasks, prune_count))],
            axis=1,
        ),
    )


def _grow_edges(
    experiment: _Experiment,
    config: _EvolutionConfig,
    add_count: int,
    round_index: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add ``add_count`` edges with init-distribution values.

    Endpoint placement follows ``config.grow_rule``: ``activity`` samples
    between the busiest neurons, ``gradient`` samples where the accumulated
    gradient marginals are largest.
    """
    rows, cols, values = _current_topology(experiment)
    if config.grow_rule == "gradient":
        new_rows, new_cols = _gradient_endpoints(
            config.n_rec,
            add_count,
            experiment.task_mass.sum(axis=0),
            rows,
            cols,
            config.activity_floor,
            rng,
        )
    else:
        rates = _measure_rates(experiment, config)
        new_rows, new_cols = _respawn_endpoints(
            config.n_rec,
            add_count,
            rates,
            config.activity_floor,
            rows,
            cols,
            rng,
        )
    n_tasks = experiment.task_mass.shape[0]
    return (
        np.concatenate([rows, new_rows]),
        np.concatenate([cols, new_cols]),
        np.concatenate([values, _respawn_values(add_count, config, round_index)]),
        np.concatenate(
            [experiment.task_mass, np.zeros((n_tasks, add_count))], axis=1
        ),
    )


def _shrink_edges(
    experiment: _Experiment, prune_count: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove the ``prune_count`` weakest-``|w|`` edges without replacement."""
    rows, cols, values = _current_topology(experiment)
    survivors = _prune_survivors(values, prune_count)
    return (
        rows[survivors],
        cols[survivors],
        values[survivors],
        experiment.task_mass[:, survivors],
    )


def _adaptive_edges(
    experiment: _Experiment,
    config: _EvolutionConfig,
    new_budget: int,
    round_index: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grow or shrink toward ``new_budget``; hold when it already matches."""
    current = int(experiment.task_mass.shape[1])
    if new_budget > current:
        return _grow_edges(experiment, config, new_budget - current, round_index, rng)
    if new_budget < current:
        return _shrink_edges(experiment, current - new_budget)
    rows, cols, values = _current_topology(experiment)
    return rows, cols, values, experiment.task_mass


# --- Compiled step factories ----------------------------------------------------


def _reset(experiment: _Experiment) -> None:
    brainstate.nn.reset_all_states(experiment.model, batch_size=1)
    experiment.learner.reset_state(batch_size=1)


def _make_train_trial(experiment: _Experiment, config: _EvolutionConfig):
    """One jitted per-trial pp-prop update returning (loss, recurrent gradient)."""

    @brainstate.transform.jit
    def train_trial(spikes, labels, mask):
        _reset(experiment)

        def step_loss(step_spikes):
            logits = experiment.learner(step_spikes)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(
                logits, labels
            ).mean()

        gradients, objective = experiment.learner.etrace_grad(
            spikes,
            step_fn=step_loss,
            mask=mask,
            reduction="mean",
            loss_output="scalar",
            return_value=True,
        )
        for name, optimizer in experiment.optimizers.items():
            keys = experiment.groups[name]
            group_grads = {key: gradients[key] for key in keys}
            clipped = brainstate.nn.clip_grad_norm(group_grads, config.clip_norm)
            optimizer.update(clipped)
        return objective, gradients[experiment.rec_key]["weight"]

    return train_trial


def _make_eval_trial(experiment: _Experiment):
    """One jitted forward-only trial returning stacked readout outputs."""

    @brainstate.transform.jit
    def eval_trial(spikes):
        _reset(experiment)
        return experiment.learner.etrace_evolve(spikes, return_outputs=True)

    return eval_trial


def _make_rate_probe(experiment: _Experiment):
    """One jitted forward-only trial returning per-step recurrent spikes."""

    @brainstate.transform.jit
    def rate_probe(spikes):
        brainstate.nn.reset_all_states(experiment.model, batch_size=1)
        return brainstate.transform.for_loop(experiment.model.cell_step, spikes)

    return rate_probe


# --- Evaluation and measurement --------------------------------------------


def _evaluate(
    experiment: _Experiment, config: _EvolutionConfig
) -> Tuple[float, ...]:
    """Per-trick accuracy on fixed probe trials, one entry per trick."""
    layout = _layout(config)
    eval_trial = _make_eval_trial(experiment)
    accuracies = []
    for task in range(config.num_tricks):
        start = layout.response_start[task]
        window = np.arange(start, start + layout.response_ticks)
        correct = 0
        for index in range(config.eval_trials_per_task):
            spikes, _, _ = _make_trial(task, _probe_seed(task, index), config)
            outputs = np.asarray(eval_trial(spikes))
            logits = outputs[window].mean(axis=0)
            correct += int(np.argmax(logits[0]) == task)
        accuracies.append(correct / config.eval_trials_per_task)
    return tuple(float(accuracy) for accuracy in accuracies)


def _measure_rates(experiment: _Experiment, config: _EvolutionConfig) -> np.ndarray:
    """Per-neuron mean spike rate over fixed probe trials of every trick."""
    rate_probe = _make_rate_probe(experiment)
    total = np.zeros(config.n_rec, dtype=np.float64)
    for index in range(config.rate_probe_trials):
        spikes, _, _ = _make_trial(index % config.num_tricks, _rate_seed(index), config)
        total += np.asarray(rate_probe(spikes)).mean(axis=(0, 1))
    return total / config.rate_probe_trials


# --- Arms ---------------------------------------------------------------------


def _train_round(
    experiment: _Experiment, config: _EvolutionConfig, round_index: int
) -> float:
    """One round of interleaved per-trial updates; returns the mean trial loss."""
    train_trial = _make_train_trial(experiment, config)
    losses = []
    for trial_index in range(config.trials_per_round):
        task = trial_index % config.num_tricks
        seed = _trial_seed(config, round_index, trial_index)
        spikes, labels, mask = _make_trial(task, seed, config)
        loss, rec_grad = train_trial(spikes, labels, mask)
        losses.append(float(loss))
        mass = np.abs(np.asarray(u.get_mantissa(rec_grad), dtype=np.float64))
        experiment.task_mass[task] += mass
    return float(np.mean(losses))


def _run_arm(config: _EvolutionConfig, evolve: bool) -> Dict[str, Any]:
    """Run one arm (evolving or control) and collect its full history."""
    arm = "evolve" if evolve else "control"
    experiment = _initial_experiment(config)
    topo_rng = np.random.default_rng(config.seed + 2_026)
    if config.task_style == _CONTEXT_STYLE:
        names = list(_CONTEXT_NAMES)
    else:
        names = list(_TRICK_NAMES[: config.num_tricks])
    accuracies = [[] for _ in range(config.num_tricks)]
    round_losses, round_seconds, events = [], [], []
    growth_events = 0
    acc = _evaluate(experiment, config)
    for task, value in enumerate(acc):
        accuracies[task].append(value)
    edge_counts = [int(experiment.task_mass.shape[1])]
    for round_index in range(config.n_rounds):
        started = time.perf_counter()
        round_losses.append(_train_round(experiment, config, round_index))
        acc = _evaluate(experiment, config)
        for task, value in enumerate(acc):
            accuracies[task].append(value)
        current = int(experiment.task_mass.shape[1])
        if evolve and not config.fixed_budget:
            new_budget = _next_budget(current, min(acc), config)
            if (
                new_budget > current
                and config.max_growth_events is not None
                and growth_events >= config.max_growth_events
            ):
                new_budget = current
            if new_budget != current:
                if new_budget > current:
                    growth_events += 1
                events.append(
                    {
                        "round": round_index + 1,
                        "kind": "grow" if new_budget > current else "shrink",
                        "from": current,
                        "to": new_budget,
                        "min_accuracy": float(min(acc)),
                        "bottleneck": names[int(np.argmin(acc))],
                    }
                )
            edges = _adaptive_edges(
                experiment, config, new_budget, round_index, topo_rng
            )
        elif evolve:
            rates = _measure_rates(experiment, config)
            edges = _evolve_topology(
                experiment, config, rates, round_index, topo_rng
            )
        else:
            rows, cols, values = _current_topology(experiment)
            edges = (rows, cols, values, experiment.task_mass)
        experiment = _rebuild_experiment(config, experiment, *edges)
        edge_counts.append(int(experiment.task_mass.shape[1]))
        round_seconds.append(time.perf_counter() - started)
        scoreboard = " ".join(f"{name}={value:.3f}" for name, value in zip(names, acc))
        print(
            f"[18-evolution] arm={arm} round={round_index + 1}/{config.n_rounds} "
            f"loss={round_losses[-1]:.4f} {scoreboard} "
            f"edges={current}->{edge_counts[-1]} "
            f"({round_seconds[-1]:.1f}s)",
            flush=True,
        )
    labels = _classify_attribution(
        *experiment.task_mass, threshold=config.share_threshold
    )
    rows, cols, _ = _current_topology(experiment)
    result = {
        "arm": arm,
        "trick_names": list(names),
        "round_losses": round_losses,
        "round_seconds": round_seconds,
        "accuracies": accuracies,
        "edge_counts": edge_counts,
        "events": events,
        "attribution": labels,
        "split": tuple(
            float(np.mean(labels == label))
            for label in range(config.num_tricks + 1)
        ),
        "rows": rows,
        "cols": cols,
    }
    result["fetch_accuracy"] = accuracies[FETCH]
    result["roll_over_accuracy"] = accuracies[ROLL_OVER]
    return result


# --- Reporting ------------------------------------------------------------------


def _works_phrase(name: str, accuracy: float, chance: float) -> str:
    """Plain-English verdict on whether one trick works."""
    if accuracy >= 0.6:
        verdict = "works"
    elif accuracy < chance + 0.05:
        verdict = "is at chance"
    else:
        verdict = "is marginal"
    return f"{name} {verdict}: {accuracy:.0%} correct (chance {chance:.0%})"


def _interference_line(accs: Tuple[float, ...], names, chance: float) -> str:
    """Plain-English verdict on whether any trick wrecked another."""
    gap = max(accs) - min(accs)
    worst = min(accs)
    detail = ", ".join(
        f"{name} {acc:.0%}" for name, acc in zip(names, accs)
    )
    if gap <= 0.15 or worst >= 0.65:
        return f"No trick wrecked another ({detail})."
    if worst <= chance + 0.05 and gap > 0.25:
        losers = ", ".join(
            name for name, acc in zip(names, accs) if acc <= chance + 0.05
        )
        return f"Some tricks were sacrificed: {losers} stayed at chance ({detail})."
    return f"The tricks partially interfered ({detail})."


def _split_phrase(split: Tuple[float, ...], names) -> str:
    """Format per-trick plus shared fractions as a plain-English list."""
    parts = [
        f"{fraction:.0%} of synapses care only about {name}"
        if index == 0
        else f"{fraction:.0%} only about {name}"
        for index, (fraction, name) in enumerate(zip(split[:-1], names))
    ]
    return ", ".join(parts) + f", {split[-1]:.0%} are shared"


def _split_compact(split: Tuple[float, ...], names) -> str:
    """Compact form of per-trick plus shared fractions."""
    parts = [
        f"{fraction:.0%} {name}-only"
        for fraction, name in zip(split[:-1], names)
    ]
    return ", ".join(parts) + f", {split[-1]:.0%} shared"


def _budget_journey_lines(
    evolve: Dict[str, Any], control: Dict[str, Any], config: _EvolutionConfig
) -> List[str]:
    """Plain-English narration of the evolving arm's edge-budget journey."""
    counts = evolve["edge_counts"]
    seconds = evolve["round_seconds"]
    final_accs = [history[-1] for history in evolve["accuracies"]]
    names = evolve["trick_names"]
    if config.fixed_budget:
        return [
            f"- It kept a fixed budget of {counts[0]} synapses "
            f"(fixed-budget mode); the control froze at "
            f"{control['edge_counts'][-1]}."
        ]
    lines = [f"- It started with {counts[0]} synapses."]
    grows = [event for event in evolve["events"] if event["kind"] == "grow"]
    shrinks = [event for event in evolve["events"] if event["kind"] == "shrink"]
    shown = grows[:3]
    for event in shown:
        lines.append(
            f"  {event['bottleneck'].capitalize()} was only "
            f"{event['min_accuracy']:.0%} after round {event['round']}, "
            f"so it grew to {event['to']}."
        )
    if len(grows) > len(shown):
        lines.append(f"  ... and {len(grows) - len(shown)} more growth rounds.")
    if shrinks and grows:
        lines.append(
            f"  Once every trick passed {config.target_accuracy:.0%} it "
            "slimmed down."
        )
    elif shrinks:
        lines.append(
            f"  Every trick stayed at or above {config.target_accuracy:.0%}, "
            "so it slimmed down whenever it could."
        )
    final_detail = ", ".join(
        f"{name} at {acc:.0%}" for name, acc in zip(names, final_accs)
    )
    warm = seconds[1:] or seconds
    lines.append(
        f"  It settled at {counts[-1]} synapses — vs "
        f"{control['edge_counts'][-1]} frozen for the control — with "
        f"{final_detail}, in "
        f"{sum(seconds):.0f} total seconds ({seconds[-1]:.1f}s per round at "
        f"the end vs {warm[0]:.1f}s at the start; the first round carries "
        "one-time compile)."
    )
    control_seconds = control["round_seconds"]
    control_warm = control_seconds[1:] or control_seconds
    evolve_early = float(np.median(seconds[1:3] or seconds))
    evolve_late = float(np.median(seconds[-2:]))
    control_early = float(np.median(control_warm[:2]))
    control_late = float(np.median(control_seconds[-2:]))
    improvement = (evolve_early - evolve_late) - (control_early - control_late)
    faster = (
        len(seconds) > 2
        and improvement > 0.2
        and improvement / max(evolve_early, 1e-9) > 0.30
        and evolve_late <= control_late * 1.05
    )
    if faster:
        lines.append(
            "  Slimming down made rounds visibly faster (control rounds, "
            "frozen at the initial budget, barely changed)."
        )
    elif len(seconds) > 2:
        lines.append(
            "  Edge count barely moved round time at this scale: fixed "
            "per-step costs and the dense input projection dominate, not the "
            "sparse recurrent edges (control rounds showed the same drift "
            "with a frozen budget)."
        )
    return lines


def _plain_english_report(
    evolve: Dict[str, Any],
    control: Dict[str, Any],
    plot_path: pathlib.Path,
    config: _EvolutionConfig,
) -> str:
    """The user-facing plain-English summary mandated by the spec."""
    names = evolve["trick_names"]
    chance = 1.0 / config.num_tricks
    final_accs = tuple(history[-1] for history in evolve["accuracies"])
    specialized = sum(evolve["split"][:-1])
    control_specialized = sum(control["split"][:-1])
    if control_specialized < specialized:
        comparison = "the split was flatter"
    else:
        comparison = "the split was no flatter"
    works_line = " ".join(
        f"{_works_phrase(name.capitalize(), acc, chance)}."
        for name, acc in zip(names, final_accs)
    )
    lines = [
        "=== What happened, in plain English ===",
        f"- {works_line}",
        f"- {_interference_line(final_accs, names, chance)}",
        *_budget_journey_lines(evolve, control, config),
        f"- The brain grew sections: {_split_phrase(evolve['split'], names)}.",
        f"- With evolution switched off, {comparison}: "
        f"{_split_compact(control['split'], names)} "
        f"(vs {specialized:.0%} specialized with evolution, "
        f"{control_specialized:.0%} without).",
        f"- Picture saved to {plot_path}: one color per trick's synapses, "
        "grey = shared; right panel = accuracy per trick over rounds with "
        "the evolving arm's synapse count in grey.",
    ]
    return "\n".join(lines)


def _plot(
    evolve: Dict[str, Any], control: Dict[str, Any], plot_path: pathlib.Path
) -> None:
    """Save the two-panel PNG: colored adjacency + accuracy and budget over rounds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = evolve["trick_names"]
    n_tricks = len(names)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, name in enumerate(names):
        picked = evolve["attribution"] == label
        left.scatter(
            evolve["cols"][picked],
            evolve["rows"][picked],
            s=4,
            c=_TRICK_COLORS[label],
            label=f"{name}-only ({int(picked.sum())})",
            linewidths=0,
        )
    picked = evolve["attribution"] == n_tricks
    left.scatter(
        evolve["cols"][picked],
        evolve["rows"][picked],
        s=4,
        c=_SHARED_COLOR,
        label=f"shared ({int(picked.sum())})",
        linewidths=0,
    )
    left.set_xlabel("pre-synaptic neuron")
    left.set_ylabel("post-synaptic neuron")
    left.set_title("Evolved recurrent wiring")
    left.legend(markerscale=3, fontsize=8)

    rounds = np.arange(len(evolve["accuracies"][0]))
    edge_axis = right.twinx()
    edge_axis.plot(
        rounds,
        evolve["edge_counts"],
        color="0.75",
        marker=".",
        linestyle="-",
        linewidth=1.5,
        label="synapses (evolving)",
        zorder=1,
    )
    edge_axis.set_ylabel("recurrent synapses (evolving)", color="0.45")
    edge_axis.tick_params(axis="y", colors="0.45")
    edge_axis.set_ylim(bottom=0)
    edge_axis.set_zorder(right.get_zorder() - 1)
    right.patch.set_visible(False)

    markers = ("o", "s", "^", "d")
    for label, name in enumerate(names):
        right.plot(
            rounds,
            evolve["accuracies"][label],
            marker=markers[label],
            linestyle="-",
            c=_TRICK_COLORS[label],
            label=f"{name} (evolving)",
        )
        right.plot(
            rounds,
            control["accuracies"][label],
            marker=markers[label],
            linestyle="--",
            c=_TRICK_COLORS[label],
            alpha=0.5,
            label=f"{name} (control)",
        )
    right.axhline(1.0 / n_tricks, color="k", linestyle=":", label="chance")
    right.set_xlabel("round (0 = before training)")
    right.set_ylabel("accuracy")
    right.set_ylim(0.0, 1.0)
    right.set_title("Accuracy per trick over rounds")
    handles, labels = right.get_legend_handles_labels()
    edge_handles, edge_labels = edge_axis.get_legend_handles_labels()
    right.legend(handles + edge_handles, labels + edge_labels, fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


# --- Entry point ------------------------------------------------------------------


def run(config: _EvolutionConfig, plot_output: pathlib.Path) -> Dict[str, Any]:
    """Run both arms, save the picture, print and return the plain-English report."""
    started = time.perf_counter()
    with brainstate.environ.context(dt=1.0 * u.ms):
        evolve = _run_arm(config, evolve=True)
        control = _run_arm(config, evolve=False)
    plot_path = plot_output.resolve()
    _plot(evolve, control, plot_path)
    report = _plain_english_report(evolve, control, plot_path, config)
    print(report, flush=True)
    return {
        "config": config,
        "evolve": evolve,
        "control": control,
        "report": report,
        "plot_path": str(plot_path),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(argv: Optional[list] = None) -> Dict[str, Any]:
    """Parse arguments and run the structural-evolution example."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny fast configuration (32 neurons, 32 edges, 1 round, 8 trials)",
    )
    parser.add_argument(
        "--plot-output",
        type=pathlib.Path,
        default=pathlib.Path("structural_evolution.png"),
        help="where to save the two-panel PNG (default: structural_evolution.png)",
    )
    parser.add_argument(
        "--fixed-budget",
        action="store_true",
        help="evolving arm keeps the constant initial budget (v1 prune + respawn)",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=None,
        help="adaptive controller target for the weakest trick (default 0.95)",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=None,
        help="adaptive budget cap (default 1048576)",
    )
    parser.add_argument(
        "--min-edges",
        type=int,
        default=None,
        help="adaptive budget floor (default 64)",
    )
    parser.add_argument(
        "--growth-factor",
        type=float,
        default=None,
        help="adaptive budget growth multiplier per round (default 1.1)",
    )
    parser.add_argument(
        "--grow-rule",
        choices=("activity", "gradient"),
        default=None,
        help="where new synapses sprout: between the busiest neurons "
        "(default) or where gradient marginals are largest",
    )
    parser.add_argument(
        "--task-style",
        choices=_TASK_STYLES,
        default=None,
        help="task family: 'simple' detect-then-respond (default) or "
        "'temporal-credit' (example 17's delayed-cue recall)",
    )
    parser.add_argument(
        "--num-tricks",
        type=int,
        default=None,
        choices=range(2, len(_TRICK_NAMES) + 1),
        help="number of interleaved tricks (default 2; above 2 requires "
        "--task-style temporal-credit)",
    )
    args = parser.parse_args(argv)
    base = _EvolutionConfig.smoke() if args.smoke else _EvolutionConfig()
    overrides: Dict[str, Any] = {"fixed_budget": args.fixed_budget}
    for name in (
        "target_accuracy",
        "max_edges",
        "min_edges",
        "growth_factor",
        "grow_rule",
        "task_style",
        "num_tricks",
    ):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if overrides.get("task_style") == _CONTEXT_STYLE and "num_tricks" not in overrides:
        overrides["num_tricks"] = 4
    return run(replace(base, **overrides), args.plot_output)


if __name__ == "__main__":
    main()
