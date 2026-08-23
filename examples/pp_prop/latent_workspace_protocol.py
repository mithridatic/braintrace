"""Explicit phase scheduling for the Example 21 protocol-v2 experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

PROTOCOL_VERSION = 2
DEFAULT_EFFORTS = (0, 30, 60)
DEFAULT_DECODER_ROWS = 30
CANDIDATE_POLICY = "latest_checkpoint_factorized_global_top2_v2"


def _boolean_gate(value: Any, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim not in (1, 2):
        raise ValueError(f"{name} must have time or time-by-batch shape. Ensure {name} has time or time-by-batch shape.")
    return array.astype(jnp.bool_)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StepGates:
    """Carry explicit, JAX-compatible protocol phase gates.

    Parameters
    ----------
    advance_physics, latent_update, decode_row, answer_feedback, recurrent_enabled
        Boolean arrays with identical ``(time, batch)`` or ``(batch,)`` shape.

    Raises
    ------
    ValueError
        If shapes differ, decoder work overlaps physical work, latent updates
        occur without physical advancement, or protocol-v2 answer feedback is
        enabled.
    """

    advance_physics: jax.Array
    latent_update: jax.Array
    decode_row: jax.Array
    answer_feedback: jax.Array
    recurrent_enabled: jax.Array

    def __post_init__(self) -> None:
        names = (
            "advance_physics",
            "latent_update",
            "decode_row",
            "answer_feedback",
            "recurrent_enabled",
        )
        arrays = tuple(_boolean_gate(getattr(self, name), name) for name in names)
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("All StepGates leaves must have the same shape. Ensure All StepGates leaves has the same shape.")
        for name, array in zip(names, arrays, strict=True):
            object.__setattr__(self, name, array)
        if not any(isinstance(array, jax.core.Tracer) for array in arrays):
            if bool(np.asarray(jnp.any(arrays[3]))):
                raise ValueError("Protocol v2 forbids answer feedback. Fix the input condition named in the error, then rerun the operation.")
            if bool(np.asarray(jnp.any(arrays[1] & ~arrays[0]))):
                raise ValueError("latent_update requires advance_physics. Provide the required value for latent_update.")
            if bool(np.asarray(jnp.any(arrays[2] & (arrays[0] | arrays[1])))):
                raise ValueError("Decoder rows must preserve physical and latent state. Set Decoder rows to preserve physical and latent state.")

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], None]:
        """Return JAX pytree leaves and no static auxiliary data."""

        return (
            (
                self.advance_physics,
                self.latent_update,
                self.decode_row,
                self.answer_feedback,
                self.recurrent_enabled,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls, auxiliary: None, children: tuple[jax.Array, ...]
    ) -> StepGates:
        """Rebuild gates during a JAX tree transformation."""

        del auxiliary
        instance = object.__new__(cls)
        for name, child in zip(
            (
                "advance_physics",
                "latent_update",
                "decode_row",
                "answer_feedback",
                "recurrent_enabled",
            ),
            children,
            strict=True,
        ):
            object.__setattr__(instance, name, child)
        return instance


@dataclass(frozen=True)
class ArmStream:
    """Validated events, phase gates, boundaries, and metadata for one arm.

    Parameters
    ----------
    events
        Float event tensor shaped ``(time, batch, feature)``.
    Gates
        Explicit gates shaped ``(time, batch)``.
    Boundaries
        Monotonic named half-open phase boundaries.
    Metadata
        JSON-oriented protocol metadata.
    """

    events: jax.Array
    gates: StepGates
    boundaries: Mapping[str, int]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        events = jnp.asarray(self.events, dtype=jnp.float32)
        if events.ndim != 3:
            raise ValueError("Events must have time, batch, and feature axes. Ensure Events has time, batch, and feature axes.")
        if self.gates.advance_physics.shape != events.shape[:2]:
            raise ValueError("Gate shape must match the event time and batch axes. Make Gate shape match the event time and batch axes.")
        if not isinstance(self.boundaries, Mapping) or not self.boundaries:
            raise ValueError("Boundaries must be a nonempty mapping. Set Boundaries to a nonempty mapping.")
        normalized: dict[str, int] = {}
        previous = 0
        for name, raw_value in self.boundaries.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Boundary names must be nonempty strings. Set Boundary names to nonempty strings.")
            if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
                raw_value, Integral
            ):
                raise TypeError("Boundary values must be integers. Set Boundary values to integers.")
            value = int(raw_value)
            if value < previous or value > events.shape[0]:
                raise ValueError("Boundaries must be monotonic and within the stream. Set Boundaries to monotonic and within the stream.")
            normalized[name] = value
            previous = value
        if normalized.get("total_steps") != events.shape[0]:
            raise ValueError("total_steps boundary must equal the stream length. Set total_steps boundary to equal the stream length.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("Metadata must be a mapping. Set Metadata to a mapping.")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "boundaries", MappingProxyType(normalized))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer. Set {name} to a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer. Set {name} to a positive integer.")
    return result


def _effort_schedule(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("Efforts must be a sequence. Set Efforts to a sequence.")
    efforts: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError("Efforts must be nonnegative integers. Use nonnegative integers.")
        efforts.append(int(value))
    result = tuple(efforts)
    if not result or result[0] != 0 or any(value < 0 for value in result):
        raise ValueError("Efforts must begin at zero and be nonnegative. Set Efforts to begin at zero and be nonnegative.")
    if tuple(sorted(set(result))) != result:
        raise ValueError("Efforts must be strictly increasing. Set Efforts to strictly increasing.")
    return result


def build_protocol_v2_arm(
    context_events: Any,
    *,
    query_start: int,
    query_stop: int,
    decoder_rows: int = DEFAULT_DECODER_ROWS,
    efforts: Sequence[int] = DEFAULT_EFFORTS,
    no_context: bool = False,
    control: Literal["intact", "state_hold", "recurrent_lesion"] = "intact",
) -> ArmStream:
    """Construct one equal-effort protocol-v2 arm.

    Parameters
    ----------
    context_events
        Context and query events shaped ``(time, batch, feature)``.
    query_start, query_stop
        Half-open query interval inside ``context_events``.
    decoder_rows
        Fixed row count used at every effort checkpoint.
    Efforts
        Strictly increasing recurrent tick counts beginning at zero.
    no_context
        Zero demonstration content before ``query_start`` while preserving its
        physical schedule.
    Control
        Intact, frozen-state, or recurrent-lesion causal arm.

    Returns
    -------
    ArmStream
        Validated event stream with explicit gates and phase boundaries.
    """

    context = np.asarray(context_events, dtype=np.float32)
    if context.ndim != 3 or not context.shape[0] or not context.shape[1]:
        raise ValueError("context_events must have nonempty time, batch, feature axes. Ensure context_events has nonempty time, batch, feature axes.")
    if not 0 <= query_start < query_stop <= context.shape[0]:
        raise ValueError("Query boundaries must be a nonempty interval in context. Set Query boundaries to a nonempty interval in context.")
    rows = _positive_integer(decoder_rows, "decoder_rows")
    checkpoints = _effort_schedule(efforts)
    if control not in ("intact", "state_hold", "recurrent_lesion"):
        raise ValueError("Control is not a protocol-v2 control. Fix the input condition named in the error, then rerun the operation.")

    context = context.copy()
    if no_context:
        context[:query_start] = 0.0
    event_parts: list[np.ndarray] = [context]
    phase_parts: list[tuple[str, int, int]] = []
    boundaries: dict[str, int] = {
        "context_start": 0,
        "query_start": int(query_start),
        "query_stop": int(query_stop),
        "context_stop": int(context.shape[0]),
    }
    cursor = int(context.shape[0])
    for index, effort in enumerate(checkpoints):
        if index:
            ticks = effort - checkpoints[index - 1]
            phase_parts.append(("reason", cursor, ticks))
            event_parts.append(
                np.zeros((ticks, context.shape[1], context.shape[2]), dtype=np.float32)
            )
            boundaries[f"reason_r{checkpoints[index - 1]}_to_r{effort}_stop"] = (
                cursor + ticks
            )
            cursor += ticks
        boundaries[f"decode_r{effort}_start"] = cursor
        phase_parts.append(("decode", cursor, rows))
        event_parts.append(
            np.zeros((rows, context.shape[1], context.shape[2]), dtype=np.float32)
        )
        cursor += rows
        boundaries[f"decode_r{effort}_stop"] = cursor
    boundaries["total_steps"] = cursor

    events = np.concatenate(event_parts, axis=0)
    shape = events.shape[:2]
    advance = np.zeros(shape, dtype=np.bool_)
    advance[: context.shape[0]] = True
    latent = np.zeros(shape, dtype=np.bool_)
    decode = np.zeros(shape, dtype=np.bool_)
    recurrent = np.zeros(shape, dtype=np.bool_)
    recurrent[: context.shape[0]] = True
    for kind, start, length in phase_parts:
        stop = start + length
        if kind == "reason" and control != "state_hold":
            advance[start:stop] = True
            latent[start:stop] = True
            recurrent[start:stop] = control == "intact"
        elif kind == "decode":
            decode[start:stop] = True

    gates = StepGates(
        advance_physics=advance,
        latent_update=latent,
        decode_row=decode,
        answer_feedback=np.zeros(shape, dtype=np.bool_),
        recurrent_enabled=recurrent,
    )
    return ArmStream(
        events=events,
        gates=gates,
        boundaries=boundaries,
        metadata={
            "protocol_version": PROTOCOL_VERSION,
            "efforts": list(checkpoints),
            "decoder_rows_per_effort": rows,
            "no_context": bool(no_context),
            "control": control,
            "candidate_policy": CANDIDATE_POLICY,
        },
    )


def build_batched_protocol_v2_arm(
    context_events: Any,
    context_advance: Any,
    query_stops: Any,
    *,
    decoder_rows: int = DEFAULT_DECODER_ROWS,
    efforts: Sequence[int] = DEFAULT_EFFORTS,
    control: Literal["intact", "state_hold", "recurrent_lesion"] = "intact",
) -> ArmStream:
    """Pack per-example query boundaries into one static protocol-v2 stream.

    Parameters
    ----------
    context_events
        Padded events shaped ``(time, batch, feature)``.
    context_advance
        Matched physical schedule shaped ``(time, batch)``.
    query_stops
        Exclusive context/query terminal for every batch lane.
    decoder_rows, efforts, control
        Protocol schedule and causal-control selection.

    Returns
    -------
    ArmStream
        Static-shape batched stream. Per-example phase boundaries and decoder
        checkpoint indices are retained in metadata.
    """

    context = np.asarray(context_events, dtype=np.float32)
    advance = np.asarray(context_advance, dtype=np.bool_)
    stops = np.asarray(query_stops)
    if (
        context.ndim != 3
        or advance.ndim != 2
        or advance.shape[1] != context.shape[1]
    ):
        raise ValueError("Context events and advance schedule batches must align. Set Context events and advance schedule batches to align.")
    if stops.shape != (context.shape[1],) or not np.issubdtype(stops.dtype, np.integer):
        raise ValueError("query_stops must contain one integer per batch lane. Add one integer per batch lane to query_stops.")
    stops = stops.astype(np.int32)
    if np.any(stops <= 0) or np.any(stops > min(context.shape[0], advance.shape[0])):
        raise ValueError("query_stops must lie inside the context stream. Set query_stops to lie inside the context stream.")
    rows = _positive_integer(decoder_rows, "decoder_rows")
    checkpoints = _effort_schedule(efforts)
    if control not in ("intact", "state_hold", "recurrent_lesion"):
        raise ValueError("Control is not a protocol-v2 control. Fix the input condition named in the error, then rerun the operation.")
    tail = rows * len(checkpoints) + checkpoints[-1]
    total = int(np.max(stops)) + tail
    events = np.zeros((total, context.shape[1], context.shape[2]), dtype=np.float32)
    physics = np.zeros((total, context.shape[1]), dtype=np.bool_)
    latent = np.zeros_like(physics)
    decode = np.zeros_like(physics)
    recurrent = np.zeros_like(physics)
    checkpoint_indices = np.zeros((len(checkpoints), context.shape[1]), dtype=np.int32)
    per_example: list[dict[str, int]] = []
    for batch, stop in enumerate(stops.tolist()):
        events[:stop, batch] = context[:stop, batch]
        physics[:stop, batch] = advance[:stop, batch]
        recurrent[:stop, batch] = advance[:stop, batch]
        cursor = stop
        boundaries: dict[str, int] = {"query_stop": stop}
        for index, effort in enumerate(checkpoints):
            if index:
                ticks = effort - checkpoints[index - 1]
                if control != "state_hold":
                    physics[cursor : cursor + ticks, batch] = True
                    latent[cursor : cursor + ticks, batch] = True
                    recurrent[cursor : cursor + ticks, batch] = control == "intact"
                cursor += ticks
            boundaries[f"decode_r{effort}_start"] = cursor
            decode[cursor : cursor + rows, batch] = True
            cursor += rows
            boundaries[f"decode_r{effort}_stop"] = cursor
            checkpoint_indices[index, batch] = cursor - 1
        per_example.append(boundaries)
    gates = StepGates(
        physics,
        latent,
        decode,
        np.zeros_like(physics),
        recurrent,
    )
    return ArmStream(
        events,
        gates,
        {"stream_start": 0, "total_steps": total},
        {
            "protocol_version": PROTOCOL_VERSION,
            "efforts": list(checkpoints),
            "decoder_rows_per_effort": rows,
            "control": control,
            "candidate_policy": CANDIDATE_POLICY,
            "per_example_boundaries": per_example,
            "checkpoint_indices": checkpoint_indices.tolist(),
        },
    )


def normalized_episode_weights(valid: Any) -> jax.Array:
    """Normalize arbitrary decoder masks to equal per-episode batch weight.

    Parameters
    ----------
    valid
        Boolean mask shaped ``(batch, ...)`` with at least one valid element
        per episode.

    Returns
    -------
    jax.Array
        Float weights of the same shape. Every episode sums to ``1 / batch``.
    """

    mask = jnp.asarray(valid, dtype=jnp.bool_)
    if mask.ndim < 2 or mask.shape[0] < 1:
        raise ValueError("Valid must have a nonempty batch and value axes. Ensure Valid has a nonempty batch and value axes.")
    axes = tuple(range(1, mask.ndim))
    counts = jnp.sum(mask, axis=axes, keepdims=True)
    if bool(np.asarray(jnp.any(counts == 0))):
        raise ValueError("Every episode must contain at least one valid value. Add at least one valid value to Every episode.")
    return mask.astype(jnp.float32) / counts / float(mask.shape[0])
