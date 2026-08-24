"""Single-step PP-prop training for the V42 task-gated operator bank."""

from __future__ import annotations

import hashlib

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from examples.pp_prop.latent_workspace_direct_generation import first_prediction_bytes
from examples.pp_prop.latent_workspace_expert_model import (
    ANSWER_HEAD_VERSION,
    PROPOSAL_SOURCE,
    TaskGatedOnlineRNN,
)
from examples.pp_prop.latent_workspace_online_training import (
    OnlineEpisode,
    OnlinePPPropTrainer,
    fourth_root_balanced_hierarchical_mass,
    hierarchical_whole_grid_step_loss,
    evaluate_online_model,
)


def parameter_leaf_arrays(model: TaskGatedOnlineRNN) -> dict[str, np.ndarray]:
    """Return every ordered V42 trainable leaf as a copied array.

    Parameters
    ----------
    model : TaskGatedOnlineRNN
        Model whose exact trainable leaves are returned.

    Returns
    -------
    dict
        Ordered ``state-path#leaf-index`` to contiguous parameter arrays.
    """

    if not isinstance(model, TaskGatedOnlineRNN):
        raise TypeError("model must be a TaskGatedOnlineRNN instance.")
    arrays = {}
    for path, state in model.states(brainstate.ParamState).items():
        name = ".".join(map(str, path))
        for index, leaf in enumerate(jax.tree.leaves(state.value)):
            arrays[f"{name}#{index}"] = np.ascontiguousarray(
                np.asarray(leaf)
            ).copy()
    return arrays


class TaskGatedPPPropTrainer(OnlinePPPropTrainer):
    """Train V42 with an unweighted hierarchical PP-prop objective.

    Parameters
    ----------
    model : TaskGatedOnlineRNN
        Task-gated model updated in place.
    batch_size : int
        Positive static batch size.
    learning_rate : float, default=0.001
        Positive Adam learning rate.
    trace_decay : float, default=2 ** (-1 / 40)
        PP-prop eligibility decay in ``(0, 1]``.
    """

    loss_version = "task_gated_fourth_root_v42"

    def __init__(
        self,
        model: TaskGatedOnlineRNN,
        *,
        batch_size: int,
        learning_rate: float = 0.001,
        trace_decay: float = 2.0 ** (-1.0 / 40.0),
    ):
        if not isinstance(model, TaskGatedOnlineRNN):
            raise TypeError("model must be a TaskGatedOnlineRNN instance.")
        super().__init__(
            model,
            batch_size=batch_size,
            learning_rate=learning_rate,
            trace_decay=trace_decay,
        )

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


def evaluate_task_gated_model(
    model: TaskGatedOnlineRNN,
    episodes: tuple[OnlineEpisode, ...],
    *,
    trace_decay: float = 2.0 ** (-1.0 / 40.0),
    batch_size: int = 10,
) -> dict[str, object]:
    """Execute and score the single mixed V42 neural candidate.

    Parameters
    ----------
    model : TaskGatedOnlineRNN
        Frozen checkpoint-owned task-gated recurrent model.
    episodes : tuple of OnlineEpisode
        Target-free model inputs with scorer-side labels.
    trace_decay : float, default=2 ** (-1 / 40)
        Forward compiler trace setting; traces do not determine outputs.
    batch_size : int, default=10
        Positive static evaluation batch size.

    Returns
    -------
    dict
        Exact score, memberships, candidate bytes, and diagnostics.
    """

    if not isinstance(model, TaskGatedOnlineRNN):
        raise TypeError("model must be a TaskGatedOnlineRNN instance.")
    result = evaluate_online_model(
        model,
        episodes,
        trace_decay=trace_decay,
        batch_size=batch_size,
    )
    for candidate in result["candidates"]:
        candidate["answer_head_version"] = ANSWER_HEAD_VERSION
        candidate["proposal_source"] = PROPOSAL_SOURCE
    candidate_bytes = first_prediction_bytes(result["candidates"])
    result["candidate_sha256"] = hashlib.sha256(candidate_bytes).hexdigest()
    result["candidate_bytes_size"] = len(candidate_bytes)
    return result
