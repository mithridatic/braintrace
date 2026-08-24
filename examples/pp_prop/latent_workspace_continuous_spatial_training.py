"""Compiled PP-prop training for the V41 continuous spatial ARC model."""

from __future__ import annotations

from collections import Counter
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

import braintrace

from examples.pp_prop.latent_workspace_continuous_spatial_model import (
    ANSWER_HEAD_VERSION,
    PROPOSAL_SOURCE,
    ContinuousSpatialARC,
    ContinuousSpatialConfig,
)
from examples.pp_prop.latent_workspace_direct_generation import (
    first_prediction_bytes,
    strict_task_pass_at_1,
)
from examples.pp_prop.latent_workspace_online_training import (
    OnlineEpisode,
    OnlineTrainingChunk,
    decode_hierarchical_online_outputs,
    fourth_root_balanced_hierarchical_mass,
    hierarchical_whole_grid_step_loss,
    stack_online_episodes,
)


def _parameter_group(path: tuple[object, ...]) -> str:
    groups = {
        "input_conv": "input",
        "recurrent_conv": "recurrent",
        "color_head": "color",
        "height_head": "height",
        "width_head": "width",
    }
    try:
        return groups[str(path[0])]
    except KeyError as error:
        raise ValueError(f"V41 parameter path {path!r} has no group.") from error


def _parameter_tree(value: object):
    return jax.tree.flatten(value)


def _array(value: object) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value))


def continuous_spatial_parameter_leaf_arrays(
    model: ContinuousSpatialARC,
) -> dict[str, np.ndarray]:
    """Return every ordered V41 parameter leaf as a copied array.

    Parameters
    ----------
    model : ContinuousSpatialARC
        Model whose checkpoint-owned leaves are returned.

    Returns
    -------
    dict
        Ordered ``state-path#leaf-index`` to contiguous parameter arrays.
    """

    if not isinstance(model, ContinuousSpatialARC):
        raise TypeError("model must be a ContinuousSpatialARC instance.")
    arrays = {}
    for path, state in model.states(brainstate.ParamState).items():
        state_name = ".".join(map(str, path))
        for index, leaf in enumerate(_parameter_tree(state.value)[0]):
            arrays[f"{state_name}#{index}"] = _array(leaf).copy()
    return arrays


def continuous_spatial_parameter_arrays(
    model: ContinuousSpatialARC,
) -> dict[str, np.ndarray]:
    """Return deterministic V41 parameter bytes grouped by model role.

    Parameters
    ----------
    model : ContinuousSpatialARC
        Model whose trainable parameters are grouped.

    Returns
    -------
    dict
        Contiguous byte arrays for input, recurrent, colour, and shape groups.
    """

    if not isinstance(model, ContinuousSpatialARC):
        raise TypeError("model must be a ContinuousSpatialARC instance.")
    chunks = {name: [] for name in ("input", "recurrent", "color", "height", "width")}
    for path, state in model.states(brainstate.ParamState).items():
        chunks[_parameter_group(path)].extend(
            _array(leaf).tobytes() for leaf in _parameter_tree(state.value)[0]
        )
    return {
        name: np.frombuffer(b"".join(values), dtype=np.uint8).copy()
        for name, values in chunks.items()
    }


def continuous_spatial_parameter_digest(model: ContinuousSpatialARC) -> str:
    """Hash the ordered V41 parameter schema and bytes.

    Parameters
    ----------
    model : ContinuousSpatialARC
        Model whose exact checkpoint state is hashed.

    Returns
    -------
    str
        Lowercase SHA-256 digest.
    """

    if not isinstance(model, ContinuousSpatialARC):
        raise TypeError("model must be a ContinuousSpatialARC instance.")
    digest = hashlib.sha256()
    for path, state in model.states(brainstate.ParamState).items():
        digest.update(".".join(map(str, path)).encode("utf-8"))
        for leaf in _parameter_tree(state.value)[0]:
            array = _array(leaf)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


class ContinuousSpatialPPPropTrainer:
    """Train V41 with compiled single-step PP-prop.

    Parameters
    ----------
    model : ContinuousSpatialARC
        Model updated in place.
    batch_size : int
        Positive static batch size.
    learning_rate : float, default=0.001
        Positive Adam learning rate.
    trace_decay : float, default=2 ** (-1 / 40)
        PP-prop exponential eligibility decay in ``(0, 1]``.
    """

    algorithm = "pp_prop"
    vjp_method = "single-step"
    loss_version = "fourth_root_hierarchical_v41"

    def __init__(
        self,
        model: ContinuousSpatialARC,
        *,
        batch_size: int,
        learning_rate: float = 0.001,
        trace_decay: float = 2.0 ** (-1.0 / 40.0),
    ):
        if not isinstance(model, ContinuousSpatialARC):
            raise TypeError("model must be a ContinuousSpatialARC instance.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise TypeError("batch_size must be a positive integer.")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        for name, raw in (
            ("learning_rate", learning_rate),
            ("trace_decay", trace_decay),
        ):
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
            grouped[_parameter_group(path)].append(path)
        self.groups = {name: tuple(paths) for name, paths in grouped.items()}
        if any(not paths for paths in self.groups.values()):
            raise ValueError("Every V41 parameter group must be nonempty.")
        self._train_many = self._make_train_many()

    def _reset(self) -> None:
        brainstate.nn.reset_all_states(self.model, batch_size=self.batch_size)
        self.learner.reset_state(batch_size=self.batch_size)

    def _make_train_many(self):
        learner = self.learner
        optimizer = self.optimizer
        groups = self.groups

        def gradient_norm(gradients, paths):
            leaves = [
                leaf
                for path in paths
                for leaf in jax.tree.leaves(gradients[path])
            ]
            return jnp.sqrt(
                sum(jnp.sum(jnp.square(jnp.asarray(leaf))) for leaf in leaves)
            )

        def train_one(
            events,
            rows,
            masks,
            _class_weights,
            heights,
            widths,
            decode_mask,
        ):
            self._reset()
            gate_mass, color_mass = fourth_root_balanced_hierarchical_mass(
                rows, masks
            )

            def step_loss(
                event,
                row,
                mask,
                step_gate_mass,
                step_color_mass,
                height,
                width,
            ):
                return hierarchical_whole_grid_step_loss(
                    learner(event),
                    row,
                    mask,
                    height,
                    width,
                    step_gate_mass,
                    step_color_mass,
                )

            gradients, objective = learner.etrace_grad(
                events,
                rows,
                masks,
                gate_mass,
                color_mass,
                heights,
                widths,
                step_fn=step_loss,
                mask=decode_mask,
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            norms = jnp.stack(
                [gradient_norm(gradients, groups[name]) for name in groups]
            )
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
        """Run one compiled update chunk.

        Parameters
        ----------
        chunk : OnlineTrainingChunk
            Update-major target-isolated training arrays.

        Returns
        -------
        losses : jax.Array
            One scalar objective per optimizer update.
        gradient_norms : dict
            Maximum observed norm for every V41 parameter group.
        """

        if not isinstance(chunk, OnlineTrainingChunk):
            raise TypeError("chunk must be an OnlineTrainingChunk instance.")
        if chunk.events.shape[2] != self.batch_size:
            raise ValueError(
                f"chunk batch axis must equal trainer batch_size {self.batch_size}."
            )
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


def evaluate_continuous_spatial_model(
    model: ContinuousSpatialARC,
    episodes: tuple[OnlineEpisode, ...],
    *,
    trace_decay: float = 2.0 ** (-1.0 / 40.0),
    batch_size: int = 10,
) -> dict[str, object]:
    """Execute target-free V41 recurrence and strict greedy scoring.

    Parameters
    ----------
    model : ContinuousSpatialARC
        Frozen checkpoint-owned recurrent model.
    episodes : tuple of OnlineEpisode
        Ordered target-free inputs with scorer-side labels.
    trace_decay : float, default=2 ** (-1 / 40)
        Trace setting used only by the forward compiler wrapper.
    batch_size : int, default=10
        Positive static evaluation batch size.

    Returns
    -------
    dict
        Exact task memberships, candidates, digest, and diagnostics.
    """

    if not isinstance(model, ContinuousSpatialARC):
        raise TypeError("model must be a ContinuousSpatialARC instance.")
    if not isinstance(episodes, tuple) or not episodes:
        raise ValueError("episodes must be a nonempty tuple.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be a positive integer.")
    static_batch = min(int(batch_size), len(episodes))
    if static_batch < 1:
        raise ValueError("batch_size must be a positive integer.")
    padding = (-len(episodes)) % static_batch
    padded = episodes + (episodes[-1],) * padding
    batches = tuple(
        stack_online_episodes(padded[index : index + static_batch])
        for index in range(0, len(padded), static_batch)
    )
    brainstate.nn.init_all_states(model, batch_size=static_batch)
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
        candidate = decode_hierarchical_online_outputs(
            outputs, episode.decode_mask, dependencies
        )
        candidate["answer_head_version"] = ANSWER_HEAD_VERSION
        candidate["proposal_source"] = PROPOSAL_SOURCE
        candidates.append(candidate)
        predictions.append(candidate["grid"])
        rows = episode.target_rows[episode.decode_mask.astype(np.bool_)]
        height = int(episode.target_heights[-1]) + 1
        width = int(episode.target_widths[-1]) + 1
        targets.append(rows[:height, :width].tolist())
        task_ids.append(episode.task_id)
    score = strict_task_pass_at_1(predictions, targets, task_ids)
    candidate_bytes = first_prediction_bytes(candidates)
    shape_pairs = [
        (np.asarray(prediction), np.asarray(target))
        for prediction, target in zip(predictions, targets, strict=True)
        if np.asarray(prediction).shape == np.asarray(target).shape
    ]
    foreground_total = sum(
        int(np.count_nonzero(target != 0)) for _, target in shape_pairs
    )
    background_total = sum(
        int(np.count_nonzero(target == 0)) for _, target in shape_pairs
    )
    foreground_correct = sum(
        int(np.count_nonzero((prediction == target) & (target != 0)))
        for prediction, target in shape_pairs
    )
    background_correct = sum(
        int(np.count_nonzero((prediction == target) & (target == 0)))
        for prediction, target in shape_pairs
    )
    predicted_colors = Counter(
        int(color)
        for prediction in predictions
        for color in np.asarray(prediction).reshape(-1)
    )
    return {
        **score,
        "query_count": len(episodes),
        "task_count": len(set(task_ids)),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_bytes_size": len(candidate_bytes),
        "diagnostics": {
            "shape_exact_count": len(shape_pairs),
            "foreground_correct": foreground_correct,
            "foreground_total": foreground_total,
            "foreground_accuracy": foreground_correct / max(foreground_total, 1),
            "background_correct": background_correct,
            "background_total": background_total,
            "background_accuracy": background_correct / max(background_total, 1),
            "predicted_color_counts": {
                str(color): count
                for color, count in sorted(predicted_colors.items())
            },
        },
        "candidates": candidates,
    }


def save_continuous_spatial_checkpoint(
    model: ContinuousSpatialARC, path: pathlib.Path
) -> str:
    """Save an exact-schema V41 checkpoint.

    Parameters
    ----------
    model : ContinuousSpatialARC
        Model whose ordered parameters are saved.
    path : pathlib.Path
        Destination ``.npz`` path.

    Returns
    -------
    str
        Ordered parameter digest.
    """

    if not isinstance(model, ContinuousSpatialARC):
        raise TypeError("model must be a ContinuousSpatialARC instance.")
    path = pathlib.Path(path)
    arrays = {}
    leaves_metadata = []
    cursor = 0
    for state_path, state in model.states(brainstate.ParamState).items():
        state_name = ".".join(map(str, state_path))
        for index, leaf in enumerate(_parameter_tree(state.value)[0]):
            key = f"leaf_{cursor:04d}"
            array = _array(leaf)
            arrays[key] = array
            leaves_metadata.append(
                {
                    "key": key,
                    "state_path": state_name,
                    "tree_leaf_index": index,
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                }
            )
            cursor += 1
    digest = continuous_spatial_parameter_digest(model)
    metadata = {
        "schema_version": 1,
        "architecture": asdict(model.config),
        "parameter_sha256": digest,
        "leaves": leaves_metadata,
    }
    arrays["__metadata__"] = np.frombuffer(
        msgspec.json.encode(metadata, order="sorted"), dtype=np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez(stream, **arrays)
    return digest


def load_continuous_spatial_checkpoint(
    path: pathlib.Path,
) -> tuple[ContinuousSpatialARC, dict[str, object]]:
    """Load and validate an exact-schema V41 checkpoint.

    Parameters
    ----------
    path : pathlib.Path
        Existing V41 checkpoint path.

    Returns
    -------
    model : ContinuousSpatialARC
        Restored model.
    metadata : dict
        Validated checkpoint metadata.
    """

    path = pathlib.Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            if "__metadata__" not in names:
                raise ValueError("V41 checkpoint metadata is missing.")
            metadata = msgspec.json.decode(bytes(archive["__metadata__"]))
            if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
                raise ValueError("V41 checkpoint schema is unsupported.")
            architecture = metadata.get("architecture")
            leaves_metadata = metadata.get("leaves")
            if not isinstance(architecture, dict) or not isinstance(
                leaves_metadata, list
            ):
                raise ValueError("V41 checkpoint metadata is invalid.")
            model = ContinuousSpatialARC(ContinuousSpatialConfig(**architecture))
            expected = {
                item.get("key")
                for item in leaves_metadata
                if isinstance(item, dict)
            }
            if names != expected | {"__metadata__"}:
                raise ValueError("V41 checkpoint leaf set is invalid.")
            cursor = 0
            for state_path, state in model.states(brainstate.ParamState).items():
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
                        or list(array.shape) != item.get("shape")
                        or array.dtype.str != item.get("dtype")
                        or not np.all(np.isfinite(array))
                    ):
                        raise ValueError("V41 checkpoint leaf schema is invalid.")
                    restored.append(jnp.asarray(array))
                    cursor += 1
                state.value = jax.tree.unflatten(structure, restored)
            if cursor != len(leaves_metadata):
                raise ValueError("V41 checkpoint has unexpected leaves.")
    except (IndexError, KeyError, OSError, TypeError, msgspec.DecodeError) as error:
        raise ValueError("V41 checkpoint could not be decoded safely.") from error
    expected_digest = metadata.get("parameter_sha256")
    if (
        not isinstance(expected_digest, str)
        or continuous_spatial_parameter_digest(model) != expected_digest
    ):
        raise ValueError("V41 checkpoint digest does not match its contents.")
    return model, metadata

