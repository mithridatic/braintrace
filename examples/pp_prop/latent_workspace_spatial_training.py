"""Compiled PP-prop training and evaluation for the V22 Conv-LIF model."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from numbers import Integral, Real
import pathlib

import brainstate
import braintools
import jax
import jax.numpy as jnp
import msgspec
import numpy as np
import brainunit as u

import braintrace

from examples.pp_prop.latent_workspace_direct_generation import (
    first_prediction_bytes,
    strict_task_pass_at_1,
)
from examples.pp_prop.latent_workspace_online_model import (
    MAX_GRID_SIZE,
    OUTPUT_WIDTH,
    split_step_logits,
)
from examples.pp_prop.latent_workspace_online_training import (
    OnlineEpisode,
    OnlineTrainingChunk,
    online_step_loss,
    stack_online_episodes,
)
from examples.pp_prop.latent_workspace_spatial_model import (
    ANSWER_HEAD_VERSION,
    SpatialARCConvLIF,
    SpatialModelConfig,
)


def _group(path: tuple[object, ...]) -> str:
    root = str(path[0])
    groups = {
        "input_conv": "input",
        "recurrent_conv": "recurrent",
        "color_head": "color",
        "height_head": "height",
        "width_head": "width",
    }
    try:
        return groups[root]
    except KeyError as error:
        raise ValueError(f"Spatial parameter path {path!r} has no group.") from error


def _array(value: object) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(u.get_mantissa(value)))


def _parameter_tree(value: object):
    return jax.tree.flatten(value, is_leaf=lambda leaf: isinstance(leaf, u.Quantity))


def spatial_parameter_arrays(model: SpatialARCConvLIF) -> dict[str, np.ndarray]:
    """Return deterministic checkpoint bytes grouped by spatial role.

    Parameters
    ----------
    model : SpatialARCConvLIF
        Spatial model whose leaves are grouped.

    Returns
    -------
    dict
        One contiguous ``uint8`` array per parameter group.
    """

    if not isinstance(model, SpatialARCConvLIF):
        raise TypeError("model must be a SpatialARCConvLIF instance.")
    chunks = {name: [] for name in ("input", "recurrent", "color", "height", "width")}
    for path, state in model.states(brainstate.ParamState).items():
        chunks[_group(path)].extend(
            _array(leaf).tobytes() for leaf in _parameter_tree(state.value)[0]
        )
    return {
        name: np.frombuffer(b"".join(values), dtype=np.uint8).copy()
        for name, values in chunks.items()
    }


def spatial_parameter_digest(model: SpatialARCConvLIF) -> str:
    """Hash ordered spatial parameter paths, units, metadata, and bytes.

    Parameters
    ----------
    model : SpatialARCConvLIF
        Model whose trainable leaves are bound.

    Returns
    -------
    str
        Lowercase SHA-256 digest.
    """

    if not isinstance(model, SpatialARCConvLIF):
        raise TypeError("model must be a SpatialARCConvLIF instance.")
    digest = hashlib.sha256()
    for path, state in model.states(brainstate.ParamState).items():
        digest.update(".".join(map(str, path)).encode("utf-8"))
        for leaf in _parameter_tree(state.value)[0]:
            array = _array(leaf)
            digest.update(bytes((isinstance(leaf, u.Quantity),)))
            digest.update(str(u.get_unit(leaf)).encode("utf-8"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


class SpatialPPPropTrainer:
    """Train the Conv-LIF canvas with compiled single-step PP-prop.

    Parameters
    ----------
    model : SpatialARCConvLIF
        Model updated in place.
    batch_size : int
        Static batch size for neuron and eligibility states.
    learning_rate : float, default=0.001
        Adam learning rate.
    trace_decay : float, default=2 ** (-1 / 40)
        Eligibility decay in ``(0, 1]``.
    """

    algorithm = "pp_prop"
    vjp_method = "single-step"
    loss_version = "target_balanced_color_v21"

    def __init__(
        self,
        model: SpatialARCConvLIF,
        *,
        batch_size: int,
        learning_rate: float = 0.001,
        trace_decay: float = 2.0 ** (-1.0 / 40.0),
        ):
        if not isinstance(model, SpatialARCConvLIF):
            raise TypeError("model must be a SpatialARCConvLIF instance.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise TypeError("batch_size must be a positive integer.")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        for name, raw in (("learning_rate", learning_rate), ("trace_decay", trace_decay)):
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise TypeError(f"{name} must be a positive finite real.")
            value = float(raw)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite real.")
            if name == "trace_decay" and value > 1.0:
                raise ValueError("trace_decay must be at most 1.0.")
            setattr(self, name, value)
        self.model = model
        self.weights = model.states(brainstate.ParamState)
        brainstate.nn.init_all_states(model, batch_size=self.batch_size)
        with brainstate.environ.context(dt=1.0 * u.ms):
            self.learner = braintrace.compile(
                model,
                braintrace.pp_prop,
                jnp.zeros(
                    (self.batch_size, model.config.input_width), dtype=jnp.float32
                ),
                batch_size=self.batch_size,
                vmap=False,
                decay_or_rank=self.trace_decay,
                vjp_method=self.vjp_method,
            )
        self.optimizer = braintools.optim.Adam(lr=self.learning_rate)
        self.optimizer.register_trainable_weights(self.weights)
        grouped = {name: [] for name in ("input", "recurrent", "color", "height", "width")}
        for path in self.weights:
            grouped[_group(path)].append(path)
        self.groups = {name: tuple(paths) for name, paths in grouped.items()}
        if any(not paths for paths in self.groups.values()):
            raise ValueError("Every spatial parameter group must be nonempty.")
        self._train_many = self._make_train_many()

    def _reset(self) -> None:
        brainstate.nn.reset_all_states(self.model, batch_size=self.batch_size)
        self.learner.reset_state(batch_size=self.batch_size)

    def _make_train_many(self):
        learner = self.learner
        optimizer = self.optimizer
        groups = self.groups

        def norm(gradients, paths):
            leaves = [
                u.get_mantissa(leaf)
                for path in paths
                for leaf in jax.tree.leaves(gradients[path])
            ]
            return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))

        def train_one(events, rows, masks, weights, heights, widths, decode_mask):
            self._reset()

            def step_loss(event, row, mask, class_weights, height, width):
                return online_step_loss(
                    learner(event), row, mask, height, width, class_weights
                )

            gradients, objective = learner.etrace_grad(
                events,
                rows,
                masks,
                weights,
                heights,
                widths,
                step_fn=step_loss,
                mask=decode_mask,
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            norms = jnp.stack([norm(gradients, groups[name]) for name in groups])
            optimizer.update(brainstate.nn.clip_grad_norm(gradients, 1.0))
            return objective, norms

        @brainstate.transform.jit
        def train_many(events, rows, masks, weights, heights, widths, decode_mask):
            def body(values):
                return train_one(*values, decode_mask)

            return brainstate.transform.for_loop(
                body, (events, rows, masks, weights, heights, widths)
            )

        return train_many

    def train_chunk(
        self, chunk: OnlineTrainingChunk
    ) -> tuple[jax.Array, dict[str, float]]:
        """Run a compiled update chunk and return group gradient maxima.

        Parameters
        ----------
        chunk : OnlineTrainingChunk
            Update-major target-isolated arrays.

        Returns
        -------
        losses : jax.Array
            One objective per update.
        gradient_norms : dict
            Maximum finite norm for each parameter group.
        """

        if not isinstance(chunk, OnlineTrainingChunk):
            raise TypeError("chunk must be an OnlineTrainingChunk instance.")
        if chunk.events.shape[2] != self.batch_size:
            raise ValueError("chunk batch axis must equal trainer batch_size.")
        with brainstate.environ.context(dt=1.0 * u.ms):
            losses, norms = self._train_many(
                jnp.asarray(chunk.events),
                jnp.asarray(chunk.target_rows),
                jnp.asarray(chunk.target_cell_mask),
                jnp.asarray(chunk.class_weights),
                jnp.asarray(chunk.target_heights),
                jnp.asarray(chunk.target_widths),
                jnp.asarray(chunk.decode_mask),
            )
        observed = np.asarray(norms)
        return losses, {
            name: float(np.max(observed[:, index]))
            for index, name in enumerate(self.groups)
        }


def decode_spatial_outputs(
    outputs: np.ndarray,
    decode_mask: np.ndarray,
    dependencies: tuple[str, ...],
) -> dict[str, object]:
    """Decode 30 spatial neural rows into one greedy candidate.

    Parameters
    ----------
    outputs : numpy.ndarray
        Per-step fixed-layout logits.
    decode_mask : numpy.ndarray
        Mask selecting exactly 30 decode steps.
    dependencies : tuple of str
        Ordered checkpoint paths.

    Returns
    -------
    dict
        Fixed greedy first candidate with V22 provenance.
    """

    values = np.asarray(outputs)
    mask = np.asarray(decode_mask, dtype=np.bool_)
    if values.ndim != 2 or values.shape[1] != OUTPUT_WIDTH or mask.shape != values.shape[:1]:
        raise ValueError("outputs and decode_mask shapes are incompatible.")
    selected = values[mask]
    if selected.shape[0] != MAX_GRID_SIZE or not np.all(np.isfinite(selected)):
        raise ValueError("decode_mask must select 30 finite output steps.")
    row, height_logits, width_logits = split_step_logits(jnp.asarray(selected))
    colors = np.asarray(jnp.argmax(row, axis=-1), dtype=np.int32)
    height = int(np.argmax(np.asarray(height_logits).mean(axis=0))) + 1
    width = int(np.argmax(np.asarray(width_logits).mean(axis=0))) + 1
    return {
        "rank": 1,
        "height": height,
        "width": width,
        "grid": colors[:height, :width].tolist(),
        "provenance": "model",
        "dependency_class": "model_checkpoint",
        "parameter_dependencies": list(dependencies),
        "proposal_source": "spatial_model_logits",
        "selection_role": "greedy_argmax",
        "ranking_source": "none_single_greedy_candidate",
        "answer_head_version": ANSWER_HEAD_VERSION,
    }


def evaluate_spatial_model(
    model: SpatialARCConvLIF,
    episodes: tuple[OnlineEpisode, ...],
    *,
    trace_decay: float = 2.0 ** (-1.0 / 40.0),
    batch_size: int = 10,
) -> dict[str, object]:
    """Execute target-free Conv-LIF evolution and strict scoring.

    Parameters
    ----------
    model : SpatialARCConvLIF
        Frozen spatial model.
    episodes : tuple of OnlineEpisode
        Ordered inputs with scorer-side targets.
    trace_decay : float
        Eligibility decay used only by the forward learner wrapper.
    batch_size : int
        Positive static evaluation batch size.

    Returns
    -------
    dict
        Exact score, memberships, candidates, and candidate digest.
    """

    if not isinstance(model, SpatialARCConvLIF):
        raise TypeError("model must be a SpatialARCConvLIF instance.")
    if not isinstance(episodes, tuple) or not episodes:
        raise ValueError("episodes must be a nonempty tuple.")
    static_batch = min(int(batch_size), len(episodes))
    if static_batch < 1:
        raise ValueError("batch_size must be positive.")
    padding = (-len(episodes)) % static_batch
    padded = episodes + (episodes[-1],) * padding
    batches = tuple(
        stack_online_episodes(padded[index : index + static_batch])
        for index in range(0, len(padded), static_batch)
    )
    brainstate.nn.init_all_states(model, batch_size=static_batch)
    with brainstate.environ.context(dt=1.0 * u.ms):
        learner = braintrace.compile(
            model,
            braintrace.pp_prop,
            jnp.zeros((static_batch, model.config.input_width), dtype=jnp.float32),
            batch_size=static_batch,
            vmap=False,
            decay_or_rank=trace_decay,
            vjp_method="single-step",
        )

        def run_batch(events):
            brainstate.nn.reset_all_states(model, batch_size=static_batch)
            learner.reset_state(batch_size=static_batch)
            return learner.etrace_evolve(events, return_outputs=True)

        @brainstate.transform.jit
        def run_all(events):
            return brainstate.transform.for_loop(run_batch, events)

        raw = np.asarray(
            run_all(jnp.asarray(np.stack([batch.events for batch in batches])))
        )
    ordered = raw.transpose(0, 2, 1, 3).reshape(-1, raw.shape[1], raw.shape[3])[
        : len(episodes)
    ]
    dependencies = tuple(
        ".".join(map(str, path)) for path in model.states(brainstate.ParamState)
    )
    candidates = []
    predictions = []
    targets = []
    task_ids = []
    for episode, outputs in zip(episodes, ordered, strict=True):
        candidate = decode_spatial_outputs(outputs, episode.decode_mask, dependencies)
        candidates.append(candidate)
        predictions.append(candidate["grid"])
        rows = episode.target_rows[episode.decode_mask.astype(np.bool_)]
        height = int(episode.target_heights[-1]) + 1
        width = int(episode.target_widths[-1]) + 1
        targets.append(rows[:height, :width].tolist())
        task_ids.append(episode.task_id)
    score = strict_task_pass_at_1(predictions, targets, task_ids)
    candidate_bytes = first_prediction_bytes(candidates)
    return {
        **score,
        "query_count": len(episodes),
        "task_count": len(set(task_ids)),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_bytes_size": len(candidate_bytes),
        "candidates": candidates,
    }


def save_spatial_checkpoint(model: SpatialARCConvLIF, path: pathlib.Path) -> str:
    """Save a unit-aware exact-schema spatial checkpoint.

    Parameters
    ----------
    model : SpatialARCConvLIF
        Model whose trainable leaves are saved.
    path : pathlib.Path
        Destination ``.npz`` path.

    Returns
    -------
    str
        Ordered spatial parameter digest.
    """

    if not isinstance(model, SpatialARCConvLIF):
        raise TypeError("model must be a SpatialARCConvLIF instance.")
    path = pathlib.Path(path)
    arrays = {}
    metadata_leaves = []
    cursor = 0
    for state_path, state in model.states(brainstate.ParamState).items():
        state_name = ".".join(map(str, state_path))
        for index, leaf in enumerate(_parameter_tree(state.value)[0]):
            key = f"leaf_{cursor:04d}"
            array = _array(leaf)
            arrays[key] = array
            metadata_leaves.append(
                {
                    "key": key,
                    "state_path": state_name,
                    "tree_leaf_index": index,
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "quantity": isinstance(leaf, u.Quantity),
                    "unit": str(u.get_unit(leaf)),
                }
            )
            cursor += 1
    digest = spatial_parameter_digest(model)
    metadata = {
        "schema_version": 2,
        "architecture": asdict(model.config),
        "parameter_sha256": digest,
        "leaves": metadata_leaves,
    }
    arrays["__metadata__"] = np.frombuffer(
        msgspec.json.encode(metadata, order="sorted"), dtype=np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez(stream, **arrays)
    return digest


def load_spatial_checkpoint(
    path: pathlib.Path,
) -> tuple[SpatialARCConvLIF, dict[str, object]]:
    """Load a unit-aware exact-schema spatial checkpoint.

    Parameters
    ----------
    path : pathlib.Path
        Existing checkpoint path.

    Returns
    -------
    model : SpatialARCConvLIF
        Restored model.
    metadata : dict
        Validated checkpoint metadata.
    """

    path = pathlib.Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            if "__metadata__" not in names:
                raise ValueError("spatial checkpoint metadata is missing.")
            metadata = msgspec.json.decode(bytes(archive["__metadata__"]))
            if not isinstance(metadata, dict) or metadata.get("schema_version") != 2:
                raise ValueError("spatial checkpoint schema is unsupported.")
            architecture = metadata.get("architecture")
            leaves_metadata = metadata.get("leaves")
            if not isinstance(architecture, dict) or not isinstance(leaves_metadata, list):
                raise ValueError("spatial checkpoint metadata is invalid.")
            model = SpatialARCConvLIF(SpatialModelConfig(**architecture))
            states = model.states(brainstate.ParamState)
            expected = {item.get("key") for item in leaves_metadata if isinstance(item, dict)}
            if names != expected | {"__metadata__"}:
                raise ValueError("spatial checkpoint leaf set is invalid.")
            cursor = 0
            for state_path, state in states.items():
                state_name = ".".join(map(str, state_path))
                originals, structure = _parameter_tree(state.value)
                restored = []
                for index, original in enumerate(originals):
                    item = leaves_metadata[cursor]
                    key = item.get("key")
                    array = np.asarray(archive[key])
                    if (
                        item.get("state_path") != state_name
                        or item.get("tree_leaf_index") != index
                        or item.get("shape") != list(_array(original).shape)
                        or item.get("dtype") != _array(original).dtype.str
                        or item.get("quantity") != isinstance(original, u.Quantity)
                        or item.get("unit") != str(u.get_unit(original))
                        or not np.all(np.isfinite(array))
                    ):
                        raise ValueError("spatial checkpoint leaf schema is invalid.")
                    value = jnp.asarray(array)
                    if isinstance(original, u.Quantity):
                        value = u.Quantity(value, unit=u.get_unit(original))
                    restored.append(value)
                    cursor += 1
                state.value = jax.tree.unflatten(structure, restored)
            if cursor != len(leaves_metadata):
                raise ValueError("spatial checkpoint has unexpected leaves.")
    except (IndexError, KeyError, OSError, TypeError, msgspec.DecodeError) as error:
        raise ValueError("spatial checkpoint could not be decoded safely.") from error
    expected_digest = metadata.get("parameter_sha256")
    if not isinstance(expected_digest, str) or spatial_parameter_digest(model) != expected_digest:
        raise ValueError("spatial checkpoint digest does not match its contents.")
    return model, metadata
