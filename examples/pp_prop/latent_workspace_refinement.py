"""Pure state operations for learned row-wise ARC answer refinement."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp


MAX_GRID_SIZE = 30
COLOR_COUNT = 10


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class RowRefinementLayout:
    """Describe the row-event features used by learned answer feedback.

    Parameters
    ----------
    input_width
        Complete width of one row event.
    event_valid_index, demonstration_phase_index, query_phase_index
        Scalar validity and phase-channel indices.
    input_side_valid_index, output_side_valid_index
        Scalar row-side validity indices.
    normalized_start
        Start of the five normalized row and dimension scalars.
    row_index_start
        Start of the row-position one-hot.
    input_height_start, input_width_start
        Starts of the input dimension one-hots.
    output_height_start, output_width_start
        Starts of the learned-answer dimension distributions.
    input_mask_start, output_mask_start
        Starts of the input and learned-answer column masks.
    input_color_start, output_color_start
        Starts of the flattened ``30 x 10`` row color features.
    max_grid_size
        Static ARC grid side. Must be 30.
    color_count
        ARC color count. Must be 10.

    Examples
    --------
    .. code-block:: python

        >>> from latent_workspace_refinement import RowRefinementLayout
        >>> layout = RowRefinementLayout(
        ...     input_width=830,
        ...     event_valid_index=0,
        ...     demonstration_phase_index=1,
        ...     query_phase_index=2,
        ...     input_side_valid_index=13,
        ...     output_side_valid_index=14,
        ...     normalized_start=15,
        ...     row_index_start=20,
        ...     input_height_start=50,
        ...     input_width_start=80,
        ...     output_height_start=110,
        ...     output_width_start=140,
        ...     input_mask_start=170,
        ...     output_mask_start=200,
        ...     input_color_start=230,
        ...     output_color_start=530,
        ... )
        >>> layout.output_width
        9060
    """

    input_width: int
    event_valid_index: int
    demonstration_phase_index: int
    query_phase_index: int
    input_side_valid_index: int
    output_side_valid_index: int
    normalized_start: int
    row_index_start: int
    input_height_start: int
    input_width_start: int
    output_height_start: int
    output_width_start: int
    input_mask_start: int
    output_mask_start: int
    input_color_start: int
    output_color_start: int
    max_grid_size: int = MAX_GRID_SIZE
    color_count: int = COLOR_COUNT

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.input_width <= 0:
            raise ValueError("input_width must be positive")
        if self.max_grid_size != MAX_GRID_SIZE:
            raise ValueError("max_grid_size must be 30 for ARC refinement")
        if self.color_count != COLOR_COUNT:
            raise ValueError("color_count must be 10 for ARC refinement")

        scalar_indices = (
            self.event_valid_index,
            self.demonstration_phase_index,
            self.query_phase_index,
            self.input_side_valid_index,
            self.output_side_valid_index,
        )
        if len(set(scalar_indices)) != len(scalar_indices):
            raise ValueError("scalar feature indices must be unique")
        if any(index < 0 or index >= self.input_width for index in scalar_indices):
            raise ValueError("scalar feature indices must be inside input_width")

        expected_starts = (
            (self.row_index_start, self.normalized_start + 5),
            (self.input_height_start, self.row_index_start + self.max_grid_size),
            (self.input_width_start, self.input_height_start + self.max_grid_size),
            (self.output_height_start, self.input_width_start + self.max_grid_size),
            (self.output_width_start, self.output_height_start + self.max_grid_size),
            (self.input_mask_start, self.output_width_start + self.max_grid_size),
            (self.output_mask_start, self.input_mask_start + self.max_grid_size),
            (self.input_color_start, self.output_mask_start + self.max_grid_size),
            (
                self.output_color_start,
                self.input_color_start + self.row_width,
            ),
        )
        if any(actual != expected for actual, expected in expected_starts):
            raise ValueError("row refinement feature slices must be contiguous")
        if self.normalized_start < 0 or self.output_color_slice.stop > self.input_width:
            raise ValueError("row refinement feature slices must be inside input_width")
        occupied = set(range(self.normalized_start, self.output_color_slice.stop))
        if any(index in occupied for index in scalar_indices):
            raise ValueError(
                "scalar feature indices must not overlap row feature slices"
            )

    @property
    def row_width(self) -> int:
        """Return the flattened logits width of one 30-by-10 answer row."""

        return self.max_grid_size * self.color_count

    @property
    def shape_width(self) -> int:
        """Return the concatenated height and width logit count."""

        return 2 * self.max_grid_size

    @property
    def output_width(self) -> int:
        """Return explicit shape-plus-grid output width."""

        return self.shape_width + self.max_grid_size**2 * self.color_count

    @property
    def normalized_slice(self) -> slice:
        """Return normalized row/input/output dimension features."""

        return slice(self.normalized_start, self.normalized_start + 5)

    @property
    def row_index_slice(self) -> slice:
        """Return the row-position one-hot slice."""

        return slice(self.row_index_start, self.row_index_start + self.max_grid_size)

    @property
    def input_height_slice(self) -> slice:
        """Return the input-height one-hot slice."""

        return slice(
            self.input_height_start, self.input_height_start + self.max_grid_size
        )

    @property
    def input_width_slice(self) -> slice:
        """Return the input-width one-hot slice."""

        return slice(
            self.input_width_start, self.input_width_start + self.max_grid_size
        )

    @property
    def output_height_slice(self) -> slice:
        """Return the learned output-height distribution slice."""

        return slice(
            self.output_height_start, self.output_height_start + self.max_grid_size
        )

    @property
    def output_width_slice(self) -> slice:
        """Return the learned output-width distribution slice."""

        return slice(
            self.output_width_start, self.output_width_start + self.max_grid_size
        )

    @property
    def input_mask_slice(self) -> slice:
        """Return the query-row column-validity slice."""

        return slice(self.input_mask_start, self.input_mask_start + self.max_grid_size)

    @property
    def output_mask_slice(self) -> slice:
        """Return the soft answer-row column-validity slice."""

        return slice(
            self.output_mask_start, self.output_mask_start + self.max_grid_size
        )

    @property
    def input_color_slice(self) -> slice:
        """Return the flattened query-row color slice."""

        return slice(self.input_color_start, self.input_color_start + self.row_width)

    @property
    def output_color_slice(self) -> slice:
        """Return the flattened soft answer-row color slice."""

        return slice(self.output_color_start, self.output_color_start + self.row_width)


def _check_state_shapes(
    query_grid: jax.Array,
    query_shape: jax.Array,
    answer_grid: jax.Array | None,
    answer_shape: jax.Array | None,
    layout: RowRefinementLayout,
) -> int:
    expected_grid_tail = (
        layout.max_grid_size,
        layout.max_grid_size,
        layout.color_count,
    )
    if query_grid.ndim != 4 or query_grid.shape[1:] != expected_grid_tail:
        raise ValueError(f"query_grid must have shape (batch, {expected_grid_tail})")
    batch_size = query_grid.shape[0]
    if query_shape.shape != (batch_size, layout.shape_width):
        raise ValueError(
            f"query_shape must have shape ({batch_size}, {layout.shape_width})"
        )
    if answer_grid is not None and answer_grid.shape != query_grid.shape:
        raise ValueError("answer_grid must have the same shape as query_grid")
    if answer_shape is not None and answer_shape.shape != query_shape.shape:
        raise ValueError("answer_shape must have the same shape as query_shape")
    return batch_size


def capture_query_rows(
    query_grid: jax.Array,
    query_shape: jax.Array,
    event: jax.Array,
    advance: jax.Array,
    layout: RowRefinementLayout,
) -> tuple[jax.Array, jax.Array]:
    """Capture valid query rows while ignoring demonstrations and targets.

    Parameters
    ----------
    query_grid, query_shape
        Previous captured query state.
    event
        Batched row events shaped ``(batch, input_width)``.
    advance
        Boolean per-example physical advance gate.
    layout
        Validated row-event feature layout.

    Returns
    -------
    tuple of jax.Array
        Updated query grid and query shape.
    """

    query_grid = jnp.asarray(query_grid, dtype=jnp.float32)
    query_shape = jnp.asarray(query_shape, dtype=jnp.float32)
    event = jnp.asarray(event, dtype=jnp.float32)
    advance = jnp.asarray(advance, dtype=jnp.bool_)
    batch_size = _check_state_shapes(query_grid, query_shape, None, None, layout)
    if event.shape != (batch_size, layout.input_width):
        raise ValueError(f"event must have shape ({batch_size}, {layout.input_width})")
    if advance.shape != (batch_size,):
        raise ValueError(f"advance must have shape ({batch_size},)")

    row_one_hot = event[:, layout.row_index_slice]
    capture_gate = (
        advance
        & (event[:, layout.event_valid_index] > 0.5)
        & (event[:, layout.demonstration_phase_index] <= 0.5)
        & (event[:, layout.query_phase_index] > 0.5)
        & (event[:, layout.input_side_valid_index] > 0.5)
        & (jnp.sum(row_one_hot, axis=-1) > 0.5)
    )
    event_colors = event[:, layout.input_color_slice].reshape(
        batch_size, layout.max_grid_size, layout.color_count
    )
    row_gate = capture_gate[:, None, None, None] & (row_one_hot[:, :, None, None] > 0.5)
    next_grid = jnp.where(row_gate, event_colors[:, None, :, :], query_grid)
    event_shape = jnp.concatenate(
        (
            event[:, layout.input_height_slice],
            event[:, layout.input_width_slice],
        ),
        axis=-1,
    )
    next_shape = jnp.where(capture_gate[:, None], event_shape, query_shape)
    return next_grid, next_shape


def _soft_column_mask(width_probabilities: jax.Array) -> jax.Array:
    return jnp.flip(
        jnp.cumsum(jnp.flip(width_probabilities, axis=-1), axis=-1), axis=-1
    )


def _validated_row_indices(row_indices: jax.Array) -> jax.Array:
    raw = jnp.asarray(row_indices)
    if raw.ndim != 1 or not jnp.issubdtype(raw.dtype, jnp.integer):
        raise ValueError("row_indices must be a one-dimensional integer array")
    return raw.astype(jnp.int32)


def build_refinement_feedback_event(
    query_grid: jax.Array,
    query_shape: jax.Array,
    answer_grid: jax.Array,
    answer_shape: jax.Array,
    row_indices: jax.Array,
    layout: RowRefinementLayout,
) -> jax.Array:
    """Build a differentiable latent feedback event for selected rows.

    The result deliberately keeps the event-valid channel at zero. It can drive
    the physical feed-forward path while remaining a latent tick for memory
    write and query-capture semantics.

    Parameters
    ----------
    query_grid, query_shape
        Captured target-free query state.
    answer_grid, answer_shape
        Current learned answer logits.
    row_indices
        Per-example row indices shaped ``(batch,)``.
    layout
        Validated row-event feature layout.

    Returns
    -------
    jax.Array
        Feedback events shaped ``(batch, input_width)``.
    """

    query_grid = jnp.asarray(query_grid, dtype=jnp.float32)
    query_shape = jnp.asarray(query_shape, dtype=jnp.float32)
    answer_grid = jnp.asarray(answer_grid, dtype=jnp.float32)
    answer_shape = jnp.asarray(answer_shape, dtype=jnp.float32)
    row_indices = _validated_row_indices(row_indices)
    batch_size = _check_state_shapes(
        query_grid, query_shape, answer_grid, answer_shape, layout
    )
    if row_indices.shape != (batch_size,):
        raise ValueError(f"row_indices must have shape ({batch_size},)")

    safe_rows = jnp.mod(row_indices, layout.max_grid_size)
    gather_index = safe_rows[:, None, None, None]
    query_rows = jnp.take_along_axis(query_grid, gather_index, axis=1)[:, 0]
    answer_rows = jnp.take_along_axis(answer_grid, gather_index, axis=1)[:, 0]
    input_height = query_shape[:, : layout.max_grid_size]
    input_width = query_shape[:, layout.max_grid_size :]
    output_height = jax.nn.softmax(answer_shape[:, : layout.max_grid_size], axis=-1)
    output_width = jax.nn.softmax(answer_shape[:, layout.max_grid_size :], axis=-1)
    input_row_valid = jnp.take_along_axis(
        _soft_column_mask(input_height), safe_rows[:, None], axis=-1
    )[:, 0]
    output_row_valid = jnp.take_along_axis(
        _soft_column_mask(output_height), safe_rows[:, None], axis=-1
    )[:, 0]
    dimension_values = jnp.arange(1, layout.max_grid_size + 1, dtype=jnp.float32)
    normalized = jnp.stack(
        (
            (safe_rows.astype(jnp.float32) + 1.0) / layout.max_grid_size,
            (input_height * dimension_values).sum(axis=-1) / layout.max_grid_size,
            (input_width * dimension_values).sum(axis=-1) / layout.max_grid_size,
            (output_height * dimension_values).sum(axis=-1) / layout.max_grid_size,
            (output_width * dimension_values).sum(axis=-1) / layout.max_grid_size,
        ),
        axis=-1,
    )

    event = jnp.zeros((batch_size, layout.input_width), dtype=jnp.float32)
    event = event.at[:, layout.query_phase_index].set(1.0)
    event = event.at[:, layout.input_side_valid_index].set(input_row_valid)
    event = event.at[:, layout.output_side_valid_index].set(output_row_valid)
    event = event.at[:, layout.normalized_slice].set(normalized)
    event = event.at[:, layout.row_index_slice].set(
        jax.nn.one_hot(safe_rows, layout.max_grid_size, dtype=jnp.float32)
    )
    event = event.at[:, layout.input_height_slice].set(input_height)
    event = event.at[:, layout.input_width_slice].set(input_width)
    event = event.at[:, layout.output_height_slice].set(output_height)
    event = event.at[:, layout.output_width_slice].set(output_width)
    event = event.at[:, layout.input_mask_slice].set(
        _soft_column_mask(input_width) * input_row_valid[:, None]
    )
    event = event.at[:, layout.output_mask_slice].set(
        _soft_column_mask(output_width) * output_row_valid[:, None]
    )
    event = event.at[:, layout.input_color_slice].set(
        (query_rows * input_row_valid[:, None, None]).reshape(
            batch_size, layout.row_width
        )
    )
    event = event.at[:, layout.output_color_slice].set(
        (
            jax.nn.softmax(answer_rows, axis=-1) * output_row_valid[:, None, None]
        ).reshape(batch_size, layout.row_width)
    )
    return event


def build_latent_row_decode_event(
    query_grid: jax.Array,
    query_shape: jax.Array,
    row_indices: jax.Array,
    layout: RowRefinementLayout,
) -> jax.Array:
    """Build a decoder-only row event without predicted-answer feedback.

    Parameters
    ----------
    query_grid, query_shape
        Frozen target-free query state.
    row_indices
        Explicit decoder row indices shaped ``(batch,)``.
    layout
        Validated row-event layout.

    Returns
    -------
    jax.Array
        Query and row-index features with every output/feedback feature zero.
    """

    zero_grid = jnp.zeros_like(query_grid)
    zero_shape = jnp.zeros_like(query_shape)
    event = build_refinement_feedback_event(
        query_grid,
        query_shape,
        zero_grid,
        zero_shape,
        row_indices,
        layout,
    )
    event = event.at[:, layout.output_side_valid_index].set(0.0)
    event = event.at[:, layout.normalized_start + 3 : layout.normalized_start + 5].set(
        0.0
    )
    event = event.at[:, layout.output_height_slice].set(0.0)
    event = event.at[:, layout.output_width_slice].set(0.0)
    event = event.at[:, layout.output_mask_slice].set(0.0)
    event = event.at[:, layout.output_color_slice].set(0.0)
    return event


def scatter_answer_rows(
    answer_grid: jax.Array, row_logits: jax.Array, row_indices: jax.Array
) -> jax.Array:
    """Replace one selected answer row per batch slot.

    Parameters
    ----------
    answer_grid
        Existing logits shaped ``(batch, 30, 30, 10)``.
    row_logits
        Replacement rows shaped ``(batch, 300)``.
    row_indices
        Selected row per batch slot.

    Returns
    -------
    jax.Array
        Updated answer grid.
    """

    answer_grid = jnp.asarray(answer_grid, dtype=jnp.float32)
    row_logits = jnp.asarray(row_logits, dtype=jnp.float32)
    row_indices = _validated_row_indices(row_indices)
    if answer_grid.ndim != 4 or answer_grid.shape[1:] != (30, 30, 10):
        raise ValueError("answer_grid must have shape (batch, 30, 30, 10)")
    batch_size = answer_grid.shape[0]
    if row_logits.shape != (batch_size, 300):
        raise ValueError(f"row_logits must have shape ({batch_size}, 300)")
    if row_indices.shape != (batch_size,):
        raise ValueError(f"row_indices must have shape ({batch_size},)")
    safe_rows = jnp.mod(row_indices, MAX_GRID_SIZE)
    row_gate = jax.nn.one_hot(safe_rows, MAX_GRID_SIZE, dtype=jnp.bool_)
    return jnp.where(
        row_gate[:, :, None, None],
        row_logits.reshape(batch_size, 1, MAX_GRID_SIZE, COLOR_COUNT),
        answer_grid,
    )


def next_reasoning_index(row_indices: jax.Array) -> jax.Array:
    """Advance per-example row indices with a 30-row wrap.

    Parameters
    ----------
    row_indices
        Integer row indices.

    Returns
    -------
    jax.Array
        Indices incremented modulo 30.
    """

    return jnp.mod(_validated_row_indices(row_indices) + 1, MAX_GRID_SIZE)


def refinement_training_logits(
    answer_shape: jax.Array, answer_row: jax.Array
) -> jax.Array:
    """Concatenate per-tick shape and current-row training logits.

    This compact output keeps the history-dependent full answer grid out of a
    compiled per-tick training carry. Full-grid logits are materialized only at
    selected sweep checkpoints by :func:`refinement_output_logits`.

    Parameters
    ----------
    answer_shape
        Height and width logits shaped ``(..., 60)``.
    answer_row
        Current row logits shaped ``(..., 300)`` with matching leading axes.

    Returns
    -------
    jax.Array
        Per-tick training logits shaped ``(..., 360)``.
    """

    answer_shape = jnp.asarray(answer_shape, dtype=jnp.float32)
    answer_row = jnp.asarray(answer_row, dtype=jnp.float32)
    if answer_shape.shape[-1:] != (60,):
        raise ValueError("answer_shape must have trailing shape (60,)")
    if answer_row.shape[-1:] != (300,):
        raise ValueError("answer_row must have trailing shape (300,)")
    if answer_shape.shape[:-1] != answer_row.shape[:-1]:
        raise ValueError("answer_shape and answer_row leading axes must match")
    return jnp.concatenate((answer_shape, answer_row), axis=-1)


def _floating_logits(logits: jax.Array, name: str, width: int) -> jax.Array:
    values = jnp.asarray(logits)
    if values.ndim < 1 or values.shape[-1] != width:
        raise ValueError(f"{name} must have trailing shape ({width},)")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise ValueError(f"{name} must have a floating-point dtype")
    return values


def split_refinement_training_logits(
    training_logits: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Split compact per-tick logits into shape and current-row logits.

    Parameters
    ----------
    training_logits
        Floating-point logits with any leading axes and trailing width 360.

    Returns
    -------
    tuple of jax.Array
        Shape logits with trailing width 60 and row logits with trailing width
        300. The shape logits contain height first and width second.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from latent_workspace_refinement import split_refinement_training_logits
        >>> shape, row = split_refinement_training_logits(jnp.zeros((2, 360)))
        >>> shape.shape, row.shape
        ((2, 60), (2, 300))
    """

    values = _floating_logits(training_logits, "training_logits", 360)
    return values[..., :60], values[..., 60:]


def split_refinement_output_logits(
    output_logits: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split explicit checkpoint logits into height, width, and colors.

    Parameters
    ----------
    output_logits
        Floating-point logits with any leading axes and trailing width 9060.

    Returns
    -------
    tuple of jax.Array
        Height and width logits with trailing width 30, followed by color
        logits with trailing shape ``(30, 30, 10)``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from latent_workspace_refinement import split_refinement_output_logits
        >>> height, width, colors = split_refinement_output_logits(
        ...     jnp.zeros((2, 9060))
        ... )
        >>> height.shape, width.shape, colors.shape
        ((2, 30), (2, 30), (2, 30, 30, 10))
    """

    values = _floating_logits(output_logits, "output_logits", 9060)
    colors = values[..., 60:].reshape(*values.shape[:-1], 30, 30, 10)
    return values[..., :30], values[..., 30:60], colors


def _integer_target(value: jax.Array, name: str, shape: tuple[int, ...]) -> jax.Array:
    values = jnp.asarray(value)
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not jnp.issubdtype(values.dtype, jnp.integer):
        raise ValueError(f"{name} must have an integer dtype")
    return values


def row_refinement_loss_per_example(
    logits: jax.Array,
    target_height: jax.Array,
    target_width: jax.Array,
    target_colors: jax.Array,
    row_indices: jax.Array,
) -> jax.Array:
    """Return deep-supervision loss for each selected answer-row tick.

    Both 30-way shape axes are supervised once at the completed-sweep row 29.
    Color cross entropy is averaged over valid columns of every selected valid
    row. A nonterminal row beyond the target height contributes no loss. This
    prevents 30 repeated shape terms from overwhelming the one color term seen
    by each row during a sweep. Height and width targets are zero-based class
    indices: class zero represents a size of one and class 29 represents a
    size of 30.

    The color term is then scaled by ``30 / (target_height + 1)`` so that every
    example contributes the same total color weight over a completed sweep. The
    zeroing above is per row, while the caller's per-tick reduction divides by
    the number of *supervised ticks*, which does not depend on row validity --
    so without this scale a 30-row target would supply ten times the color
    gradient of a 3-row target from inside the same batch mean. That is a
    relative reweighting between examples rather than a global scale, so Adam's
    scale invariance does not remove it, and it falls hardest on the small
    outputs that are the only combinatorially reachable exact targets. Width
    needs no such treatment: the column average already normalises it.

    Parameters
    ----------
    logits
        Floating-point compact logits shaped ``(batch, 360)``.
    target_height, target_width
        Zero-based integer shape classes shaped ``(batch,)``.
    target_colors
        Integer padded color grids shaped ``(batch, 30, 30)``.
    row_indices
        Integer current answer-row indices shaped ``(batch,)``.

    Returns
    -------
    jax.Array
        Per-example loss vector shaped ``(batch,)``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from latent_workspace_refinement import row_refinement_loss_per_example
        >>> loss = row_refinement_loss_per_example(
        ...     jnp.zeros((1, 360)),
        ...     jnp.asarray([1]),
        ...     jnp.asarray([2]),
        ...     jnp.zeros((1, 30, 30), dtype=jnp.int32),
        ...     jnp.asarray([0]),
        ... )
        >>> loss.shape
        (1,)
    """

    logits = _floating_logits(logits, "logits", 360)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, 360)")
    batch_size = logits.shape[0]
    vector_shape = (batch_size,)
    target_height = _integer_target(target_height, "target_height", vector_shape)
    target_width = _integer_target(target_width, "target_width", vector_shape)
    target_colors = _integer_target(
        target_colors,
        "target_colors",
        (batch_size, MAX_GRID_SIZE, MAX_GRID_SIZE),
    )
    row_indices = _integer_target(row_indices, "row_indices", vector_shape)

    shape_logits, row_logits = split_refinement_training_logits(logits)
    height_logits = shape_logits[:, :MAX_GRID_SIZE]
    width_logits = shape_logits[:, MAX_GRID_SIZE:]
    height_loss = -jnp.take_along_axis(
        jax.nn.log_softmax(height_logits, axis=-1),
        target_height[:, None],
        axis=-1,
    )[:, 0]
    width_loss = -jnp.take_along_axis(
        jax.nn.log_softmax(width_logits, axis=-1),
        target_width[:, None],
        axis=-1,
    )[:, 0]

    row_selector = jax.nn.one_hot(row_indices, MAX_GRID_SIZE, dtype=target_colors.dtype)
    selected_targets = jnp.sum(target_colors * row_selector[:, :, None], axis=1)
    color_logits = row_logits.reshape(batch_size, MAX_GRID_SIZE, COLOR_COUNT)
    color_nll = -jnp.take_along_axis(
        jax.nn.log_softmax(color_logits, axis=-1),
        selected_targets[..., None],
        axis=-1,
    )[..., 0]
    valid_row = row_indices <= target_height
    valid_columns = jnp.arange(MAX_GRID_SIZE)[None, :] <= target_width[:, None]
    valid_colors = valid_row[:, None] & valid_columns
    color_loss = jnp.sum(jnp.where(valid_colors, color_nll, 0.0), axis=-1)
    color_loss /= jnp.maximum(jnp.sum(valid_colors, axis=-1), 1)
    sweep_rows = target_height.astype(jnp.float32) + 1.0
    color_loss = color_loss * (MAX_GRID_SIZE / sweep_rows)
    completed_sweep = row_indices == (MAX_GRID_SIZE - 1)
    shape_loss = jnp.where(completed_sweep, height_loss + width_loss, 0.0)
    return shape_loss + color_loss


def refinement_output_logits(
    answer_shape: jax.Array, answer_grid: jax.Array
) -> jax.Array:
    """Concatenate explicit learned shape and full-grid color logits.

    Parameters
    ----------
    answer_shape
        Height and width logits shaped ``(..., 60)``.
    answer_grid
        Color logits shaped ``(..., 30, 30, 10)`` with matching leading axes.

    Returns
    -------
    jax.Array
        Explicit logits shaped ``(..., 9060)``.
    """

    answer_shape = jnp.asarray(answer_shape, dtype=jnp.float32)
    answer_grid = jnp.asarray(answer_grid, dtype=jnp.float32)
    if answer_shape.shape[-1:] != (60,):
        raise ValueError("answer_shape must have trailing shape (60,)")
    if answer_grid.shape[-3:] != (30, 30, 10):
        raise ValueError("answer_grid must have trailing shape (30, 30, 10)")
    if answer_shape.shape[:-1] != answer_grid.shape[:-3]:
        raise ValueError("answer_shape and answer_grid leading axes must match")
    return jnp.concatenate(
        (answer_shape, answer_grid.reshape(*answer_grid.shape[:-3], 9000)), axis=-1
    )
