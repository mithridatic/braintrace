"""Scene-summary routing model (V50) for direct ARC generation.

Adds fixed, task-neutral, input-only scene-summary channels to the frozen
V48 query-routing architecture: valid-cell count, per-colour counts, and
foreground bounding-box extents, computed exactly from the lossless
query-input block and consumed by the unchanged checkpoint-owned cell and
shape heads. The full lossless encoding is retained; the summaries are
redundant fixed arithmetic channels of the same class as the existing
validity-mask and coordinate planes. See
``docs/specs/2026-08-24-example21-scene-summary-v50.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import brainstate
import jax.numpy as jnp
import numpy as np

import braintrace

from examples.pp_prop.latent_workspace_online_model import (
    COLOR_COUNT,
    MAX_GRID_SIZE,
    OUTPUT_WIDTH,
)
from examples.pp_prop.latent_workspace_query_routing_model import (
    BASE_INPUT_WIDTH,
    QueryRoutingConfig,
    QueryRoutingGatedMemoryRNN,
    SOURCE_COUNT,
)

ARCHITECTURE_VERSION = "scene_summary_routing_v50"
ANSWER_HEAD_VERSION = "scene_summary_routing_decoder_v50"
PROPOSAL_SOURCE = "scene_summary_model_logits"
COUNT_BINS = MAX_GRID_SIZE + 1
SUMMARY_WIDTH = COUNT_BINS * (1 + COLOR_COUNT) + 2 * MAX_GRID_SIZE
V48_BLOCK_WIDTH = SOURCE_COUNT * (COLOR_COUNT + 1)
MODEL_INPUT_WIDTH = BASE_INPUT_WIDTH + V48_BLOCK_WIDTH + SUMMARY_WIDTH
SUMMARY_SLICE = slice(
    BASE_INPUT_WIDTH + V48_BLOCK_WIDTH,
    MODEL_INPUT_WIDTH,
)
BBOX_SLICE = slice(
    MODEL_INPUT_WIDTH - 2 * MAX_GRID_SIZE, MODEL_INPUT_WIDTH
)


def _count_onehot(counts: np.ndarray) -> np.ndarray:
    clipped = np.clip(counts, 0, COUNT_BINS - 1).astype(np.int64)
    return np.eye(COUNT_BINS, dtype=np.float32)[clipped]


def extend_events_scene_summary(events: np.ndarray) -> np.ndarray:
    """Append the fixed scene-summary block to every decode step.

    Parameters
    ----------
    events : numpy.ndarray
        V48-extended events shaped ``(..., time, 13731)`` whose decode
        steps carry the query colour one-hots and validity mask.

    Returns
    -------
    numpy.ndarray
        Events shaped ``(..., time, 14132)``; the appended 401 features
        repeat the exact valid-cell count, per-colour counts, and
        foreground bounding-box extents at every decode step and are
        exactly zero elsewhere.
    """

    from examples.pp_prop.latent_workspace_query_routing_model import (
        BLOCK_COLOR_SLICE,
        BLOCK_VALID_SLICE,
    )

    array = np.asarray(events)
    if array.shape[-1] != BASE_INPUT_WIDTH + V48_BLOCK_WIDTH:
        raise ValueError(
            f"events last dimension must be {BASE_INPUT_WIDTH + V48_BLOCK_WIDTH}."
        )
    colors = array[..., BLOCK_COLOR_SLICE].reshape(
        *array.shape[:-1], SOURCE_COUNT, COLOR_COUNT
    )
    validity = array[..., BLOCK_VALID_SLICE].reshape(
        *array.shape[:-1], SOURCE_COUNT
    )
    valid = validity > 0.5
    total = valid.sum(axis=-1)
    per_color = (valid[..., None] & (colors > 0.5)).sum(axis=-2)
    summary = [
        _count_onehot(total),
        _count_onehot(per_color).reshape(*array.shape[:-1], -1),
    ]
    valid.any(axis=-1)
    row_index = np.arange(SOURCE_COUNT)
    bbox = np.zeros((*array.shape[:-1], 2 * MAX_GRID_SIZE), dtype=np.float32)
    has_cell = total > 0
    if np.any(has_cell):
        row_positions = row_index // MAX_GRID_SIZE
        column_positions = row_index % MAX_GRID_SIZE
        masked_rows = np.where(valid, row_positions, -1)
        masked_columns = np.where(valid, column_positions, -1)
        row_min = np.where(
            has_cell,
            np.where(valid, row_positions, MAX_GRID_SIZE).min(axis=-1),
            0,
        )
        row_max = masked_rows.max(axis=-1)
        column_min = np.where(
            has_cell,
            np.where(valid, column_positions, MAX_GRID_SIZE).min(axis=-1),
            0,
        )
        column_max = masked_columns.max(axis=-1)
        bbox_height = np.clip(row_max - row_min + 1, 0, MAX_GRID_SIZE)
        bbox_width = np.clip(column_max - column_min + 1, 0, MAX_GRID_SIZE)
        height_onehot = np.zeros(
            (*array.shape[:-1], MAX_GRID_SIZE), dtype=np.float32
        )
        width_onehot = np.zeros_like(height_onehot)
        valid_height = (bbox_height >= 1) & has_cell
        valid_width = (bbox_width >= 1) & has_cell
        height_index = np.clip(bbox_height - 1, 0, MAX_GRID_SIZE - 1)
        width_index = np.clip(bbox_width - 1, 0, MAX_GRID_SIZE - 1)
        np.put_along_axis(
            height_onehot, height_index[..., None].astype(np.int64), 1.0, axis=-1
        )
        np.put_along_axis(
            width_onehot, width_index[..., None].astype(np.int64), 1.0, axis=-1
        )
        height_onehot *= valid_height[..., None]
        width_onehot *= valid_width[..., None]
        bbox = np.concatenate([height_onehot, width_onehot], axis=-1)
    summary.append(bbox.astype(np.float32))
    block = np.concatenate(summary, axis=-1).astype(array.dtype)
    blocks = np.zeros((*array.shape[:-1], SUMMARY_WIDTH), dtype=array.dtype)
    blocks[..., -MAX_GRID_SIZE:, :] = np.broadcast_to(
        block[..., -MAX_GRID_SIZE:, :][..., -1:, :],
        (*array.shape[:-1], MAX_GRID_SIZE, SUMMARY_WIDTH),
    )
    return np.concatenate([array, blocks], axis=-1)


@dataclass(frozen=True)
class SceneSummaryConfig(QueryRoutingConfig):
    """Configure the V50 scene-summary routing model.

    All V48 fields are inherited; only the event width and architecture
    identifier differ.
    """

    input_width: int = MODEL_INPUT_WIDTH
    architecture_version: str = ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.input_width != MODEL_INPUT_WIDTH:
            object.__setattr__(self, "input_width", MODEL_INPUT_WIDTH)


class SceneSummaryRoutingRNN(QueryRoutingGatedMemoryRNN):
    """Read exact scene summaries beside the V48 query-routing head.

    Parameters
    ----------
    config : SceneSummaryConfig
        Bound V48 dimensions plus the scene-summary event width.
    """

    answer_head_version = ANSWER_HEAD_VERSION
    proposal_source = PROPOSAL_SOURCE

    def __init__(self, config: SceneSummaryConfig):
        if not isinstance(config, SceneSummaryConfig):
            raise TypeError("config must be a SceneSummaryConfig instance.")
        super().__init__(config)
        self.config = config
        cell_feature_width = (
            config.hidden_width
            + COLOR_COUNT
            + MAX_GRID_SIZE
            + config.hidden_width * COLOR_COUNT
            + SUMMARY_WIDTH
        )
        shape_feature_width = (
            config.hidden_width
            + 2 * MAX_GRID_SIZE
            + config.hidden_width * 2 * MAX_GRID_SIZE
            + 2 * MAX_GRID_SIZE
        )
        with brainstate.random.seed_context(config.seed):
            self.cell_color_head = braintrace.nn.Linear(
                cell_feature_width, config.expert_count * COLOR_COUNT
            )
            self.height_head = braintrace.nn.Linear(
                shape_feature_width, MAX_GRID_SIZE
            )
            self.width_head = braintrace.nn.Linear(
                shape_feature_width, MAX_GRID_SIZE
            )

    def _cell_logits(self, hidden: jnp.ndarray, event: jnp.ndarray) -> jnp.ndarray:
        base = super()._cell_logits(hidden, event)
        summary = event[..., SUMMARY_SLICE]
        context = jnp.broadcast_to(
            hidden[..., None, :],
            (*hidden.shape[:-1], MAX_GRID_SIZE, hidden.shape[-1]),
        )
        summary_cells = jnp.broadcast_to(
            summary[..., None, :],
            (*hidden.shape[:-1], MAX_GRID_SIZE, SUMMARY_WIDTH),
        )
        query_colors = event[..., self.config.query_color_slice].reshape(
            *event.shape[:-1], MAX_GRID_SIZE, COLOR_COUNT
        )
        columns = jnp.broadcast_to(
            self.column_features,
            (*hidden.shape[:-1], MAX_GRID_SIZE, MAX_GRID_SIZE),
        )
        interaction = (context[..., :, None] * query_colors[..., None, :]).reshape(
            *hidden.shape[:-1], MAX_GRID_SIZE, hidden.shape[-1] * COLOR_COUNT
        )
        features = jnp.concatenate(
            (context, query_colors, columns, interaction, summary_cells), axis=-1
        )
        experts = self.cell_color_head(features).reshape(
            *hidden.shape[:-1],
            MAX_GRID_SIZE,
            self.config.expert_count,
            COLOR_COUNT,
        )
        weights = self._expert_weights(hidden)[..., None, :, None]
        return jnp.sum(experts * weights, axis=-2) + (
            base - super()._cell_logits(hidden, event)
        )

    def _shape_logits(
        self, hidden: jnp.ndarray, event: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        query_dimensions = jnp.concatenate(
            (
                event[..., self.config.query_height_slice],
                event[..., self.config.query_width_slice],
            ),
            axis=-1,
        )
        interaction = (hidden[..., :, None] * query_dimensions[..., None, :]).reshape(
            *hidden.shape[:-1], hidden.shape[-1] * 2 * MAX_GRID_SIZE
        )
        bbox = event[..., BBOX_SLICE]
        features = jnp.concatenate(
            (hidden, query_dimensions, interaction, bbox), axis=-1
        )
        return self.height_head(features), self.width_head(features)


assert SUMMARY_WIDTH == 401
assert MODEL_INPUT_WIDTH == 14132
assert OUTPUT_WIDTH == MAX_GRID_SIZE * COLOR_COUNT + 2 * MAX_GRID_SIZE
