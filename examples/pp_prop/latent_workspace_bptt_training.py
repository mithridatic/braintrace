"""Matched reverse-mode (BPTT) trainer for the credit-assignment experiment.

This trainer reproduces the exact V42/V44 training contract — same model,
same episodes, same ``task_gated_fourth_root_v42`` objective, same masked
mean reduction, same Adam and global-norm clipping — but computes gradients
by full reverse-mode backpropagation through the 360-step unroll instead of
single-step PP-prop eligibility traces. It exists to decide whether credit
assignment or representation limits the direct model's spatial-operator
learning; see ``docs/specs/2026-08-24-example21-causal-phase-map.md``.
"""

from __future__ import annotations

from numbers import Integral, Real

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np

from examples.pp_prop.latent_workspace_online_model import OnlineARCVanillaRNN
from examples.pp_prop.latent_workspace_online_training import (
    OnlineTrainingChunk,
    _parameter_group,
    fourth_root_balanced_hierarchical_mass,
    hierarchical_whole_grid_step_loss,
)


class TaskGatedBPTTTrainer:
    """Train the task-gated model with exact reverse-mode gradients.

    Parameters
    ----------
    model : OnlineARCVanillaRNN
        Task-gated model updated in place.
    batch_size : int
        Positive static batch size.
    learning_rate : float, default=0.001
        Positive Adam learning rate, matched to the PP-prop arm.
    """

    algorithm = "bptt"
    vjp_method = "reverse-mode"
    loss_version = "task_gated_fourth_root_v42"

    def __init__(
        self,
        model: OnlineARCVanillaRNN,
        *,
        batch_size: int,
        learning_rate: float = 0.001,
    ):
        if not isinstance(model, OnlineARCVanillaRNN):
            raise TypeError("model must be an OnlineARCVanillaRNN instance.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise TypeError("batch_size must be a positive integer.")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, Real
        ):
            raise TypeError("learning_rate must be a positive finite real.")
        self.learning_rate = float(learning_rate)
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive finite real.")
        self.model = model
        self.weights = model.states(brainstate.ParamState)
        brainstate.nn.init_all_states(model, batch_size=self.batch_size)
        self.optimizer = braintools.optim.Adam(lr=self.learning_rate)
        self.optimizer.register_trainable_weights(self.weights)
        grouped: dict[str, list[tuple[object, ...]]] = {
            "recurrent": [],
            "row_color": [],
            "height": [],
            "width": [],
        }
        for path in self.weights:
            grouped[_parameter_group(path)].append(path)
        self.groups = {name: tuple(paths) for name, paths in grouped.items()}
        if any(not paths for paths in self.groups.values()):
            raise ValueError("Every online parameter group must be nonempty.")
        self._train_many = self._make_train_many()

    def _make_train_many(self):
        groups = self.groups
        model = self.model
        optimizer = self.optimizer
        weights = self.weights
        batch_size = self.batch_size

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
            brainstate.nn.reset_all_states(model, batch_size=batch_size)
            gate_mass, color_mass = fourth_root_balanced_hierarchical_mass(
                rows, masks
            )

            def objective():
                outputs = brainstate.transform.for_loop(model, events)
                step_losses = jax.vmap(hierarchical_whole_grid_step_loss)(
                    outputs,
                    rows,
                    masks,
                    heights,
                    widths,
                    gate_mass,
                    color_mass,
                )
                mask = decode_mask.astype(step_losses.dtype)
                return jnp.sum(step_losses * mask) / jnp.maximum(
                    jnp.sum(mask), 1.0
                )

            gradients, loss = brainstate.transform.grad(
                objective, weights, return_value=True
            )()
            norms = jnp.stack(
                [gradient_norm(gradients, groups[name]) for name in groups]
            )
            optimizer.update(brainstate.nn.clip_grad_norm(gradients, 1.0))
            return loss, norms

        @brainstate.transform.jit
        def train_many(events, rows, masks, weights_, heights, widths, decode_mask):
            def body(values):
                return train_one(*values, decode_mask)

            return brainstate.transform.for_loop(
                body, (events, rows, masks, weights_, heights, widths)
            )

        return train_many

    def train_chunk(
        self, chunk: OnlineTrainingChunk
    ) -> tuple[jax.Array, dict[str, float]]:
        """Run a compiled update chunk and report group gradient maxima.

        Parameters
        ----------
        chunk : OnlineTrainingChunk
            Update-major target-isolated training arrays.

        Returns
        -------
        losses : jax.Array
            One scalar objective per optimizer update.
        gradient_norms : dict
            Maximum observed norm for each parameter group.
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
