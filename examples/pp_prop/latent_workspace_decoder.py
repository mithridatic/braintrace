"""Edit-rule ARC decoder: shape as a rule, colour as an edit on the query.

The legacy decoder emits 900 cell colours from a rank-16 CP tensor off a 128-d
hidden with no reference to the query input. Measured against the repo's own
trivial predictors it loses on both axes exact match needs -- shape 0.348 against
0.869, pixel 0.383 against 0.634 -- and no query in 419 clears pixel 0.95.

This module supplies the two replacements specified in
``docs/specs/2026-08-18-example21-decoder-redesign.md``:

**Shape as a rule.** Twelve deterministic size maps (a rational scale factor
crossed with which input axis it reads) plus one learned absolute fallback. The
head only *selects*; it never has to induce the map from the recurrent state,
which is what demonstration pairing has never represented above chance. Because
the height and width gates select over bases keyed to *different* input axes,
identical gate vectors still produce different outputs whenever the input is not
square -- which is what breaks the observed ``h == w`` collapse.

**Colour as an edit.** Per output cell, a mixture over ``{copy, copy_transpose,
palette, explicit}``, where the copy components read the model's own captured
query grid through a nearest-neighbour alignment. Training and decoding use the
same alignment, so there is no train/test gap.

Everything here is a pure function of head outputs and captured query features,
so the heads stay inside ``model.update`` and therefore inside the ETP
compiler's parameter set.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

MAX_GRID_SIZE = 30
COLOR_COUNT = 10
COLOR_COMPONENT_COUNT = 4
NEGATIVE_INFINITY = -1e9
COPY_SHARPNESS = 8.0

#: ``(numerator, denominator)`` scale factors applied to an input side length.
SHAPE_SCALE_FACTORS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (1, 2),
    (1, 3),
)
#: Each scale factor is offered reading its own axis and reading the other one,
#: so transposing shape maps are expressible without a separate rule family.
SHAPE_SOURCE_COUNT = 2
SHAPE_RULE_COUNT = len(SHAPE_SCALE_FACTORS) * SHAPE_SOURCE_COUNT
#: Twelve deterministic rules plus the learned absolute fallback.
SHAPE_SLOT_COUNT = SHAPE_RULE_COUNT + 1


def _shape_rule_tables() -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Build the deterministic size maps and the axis each one reads.

    Returns
    -------
    tuple of numpy.ndarray
        ``(matrix, reads_own_axis)`` where ``matrix`` is
        ``(SHAPE_RULE_COUNT, MAX_GRID_SIZE, MAX_GRID_SIZE)`` holding ``0.0``
        where rule *r* maps input side ``i + 1`` to output side ``s + 1`` and a
        large negative constant elsewhere, and ``reads_own_axis`` marks the
        rules keyed to the axis being predicted.
    """
    matrix = np.full(
        (SHAPE_RULE_COUNT, MAX_GRID_SIZE, MAX_GRID_SIZE),
        NEGATIVE_INFINITY,
        dtype=np.float32,
    )
    reads_own_axis = np.zeros((SHAPE_RULE_COUNT,), dtype=np.bool_)
    rule = 0
    for source in range(SHAPE_SOURCE_COUNT):
        for numerator, denominator in SHAPE_SCALE_FACTORS:
            reads_own_axis[rule] = source == 0
            for index in range(MAX_GRID_SIZE):
                side = index + 1
                scaled = side * numerator
                if scaled % denominator:
                    continue
                output = scaled // denominator
                if 1 <= output <= MAX_GRID_SIZE:
                    matrix[rule, index, output - 1] = 0.0
            rule += 1
    return matrix, reads_own_axis


SHAPE_RULE_MATRIX, SHAPE_RULE_READS_OWN_AXIS = _shape_rule_tables()


def decoder_output_width() -> int:
    """Return the compact width of an already-final edit-rule output.

    Returns
    -------
    int
        ``height(30) | width(30) | colors(30 * 30 * 10)``.

    Examples
    --------
    .. code-block:: python

        >>> from latent_workspace_decoder import decoder_output_width
        >>> decoder_output_width()
        9060
    """
    return 2 * MAX_GRID_SIZE + MAX_GRID_SIZE * MAX_GRID_SIZE * COLOR_COUNT


def split_decoder_logits(
    compact: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split a compact edit-rule vector into height, width and colour logits.

    Parameters
    ----------
    compact : jax.Array
        Array whose trailing axis is :func:`decoder_output_width`.

    Returns
    -------
    tuple of jax.Array
        ``(height, width, colors)`` shaped ``(..., 30)``, ``(..., 30)`` and
        ``(..., 30, 30, 10)``.
    """
    compact = jnp.asarray(compact)
    expected = decoder_output_width()
    if compact.shape[-1] != expected:
        raise ValueError(
            f"compact trailing axis must be {expected}, got {compact.shape}"
        )
    height = compact[..., :MAX_GRID_SIZE]
    width = compact[..., MAX_GRID_SIZE : 2 * MAX_GRID_SIZE]
    colors = compact[..., 2 * MAX_GRID_SIZE :].reshape(
        *compact.shape[:-1], MAX_GRID_SIZE, MAX_GRID_SIZE, COLOR_COUNT
    )
    return height, width, colors


def shape_axis_logits(
    gate_logits: jax.Array,
    absolute_logits: jax.Array,
    input_height: jax.Array,
    input_width: jax.Array,
    *,
    predict_height: bool,
) -> jax.Array:
    """Select one output side length over deterministic rules plus a fallback.

    Parameters
    ----------
    gate_logits : jax.Array
        Slot preferences shaped ``(..., SHAPE_SLOT_COUNT)``.
    absolute_logits : jax.Array
        Learned fallback distribution shaped ``(..., MAX_GRID_SIZE)``.
    input_height, input_width : jax.Array
        Query input side one-hots shaped ``(..., MAX_GRID_SIZE)``.
    predict_height : bool
        Which output axis is being predicted.  A rule marked as reading its own
        axis reads ``input_height`` here when this is ``True``, and
        ``input_width`` when it is ``False``.

    Returns
    -------
    jax.Array
        Normalised log-probabilities shaped ``(..., MAX_GRID_SIZE)``.
    """
    matrix = jnp.asarray(SHAPE_RULE_MATRIX)
    own = jnp.asarray(SHAPE_RULE_READS_OWN_AXIS)
    own_axis = input_height if predict_height else input_width
    other_axis = input_width if predict_height else input_height
    from_own = jnp.einsum("...i,ris->...rs", own_axis, matrix)
    from_other = jnp.einsum("...i,ris->...rs", other_axis, matrix)
    rule_bases = jnp.where(own[:, None], from_own, from_other)
    fallback = jax.nn.log_softmax(absolute_logits, axis=-1)
    bases = jnp.concatenate((rule_bases, fallback[..., None, :]), axis=-2)
    slots = jax.nn.log_softmax(gate_logits, axis=-1)
    return jax.nn.log_softmax(
        jax.scipy.special.logsumexp(slots[..., None] + bases, axis=-2), axis=-1
    )


def _nearest_source_index(output_side: jax.Array, input_side: jax.Array) -> jax.Array:
    """Map every output index to its nearest-neighbour source index.

    ``src(i) = clip(floor(i * input_side / output_side), 0, input_side - 1)``,
    evaluated for all ``MAX_GRID_SIZE`` output positions at once.
    """
    positions = jnp.arange(MAX_GRID_SIZE, dtype=jnp.float32)
    safe_output = jnp.maximum(output_side, 1).astype(jnp.float32)
    safe_input = jnp.maximum(input_side, 1).astype(jnp.float32)
    scaled = jnp.floor(
        positions * safe_input[..., None] / safe_output[..., None]
    ).astype(jnp.int32)
    return jnp.clip(scaled, 0, (jnp.maximum(input_side, 1) - 1)[..., None])


def _gather_cells(
    query_grid: jax.Array, rows: jax.Array, columns: jax.Array
) -> jax.Array:
    """Gather ``query_grid[b, rows[b, i], columns[b, j]]`` for every ``(i, j)``."""
    gathered_rows = jnp.take_along_axis(query_grid, rows[..., None, None], axis=-3)
    return jnp.take_along_axis(gathered_rows, columns[..., None, :, None], axis=-2)


def color_cell_logits(
    gate_logits: jax.Array,
    palette_logits: jax.Array,
    explicit_logits: jax.Array,
    query_grid: jax.Array,
    output_shape: tuple[jax.Array, jax.Array],
    input_shape: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """Mix copy, transposed copy, palette and explicit colour per output cell.

    Parameters
    ----------
    gate_logits : jax.Array
        Per-cell component preferences shaped ``(..., 30, 30, 4)``.
    palette_logits : jax.Array
        One grid-wide colour distribution shaped ``(..., COLOR_COUNT)``.
    explicit_logits : jax.Array
        Free per-cell colour logits shaped ``(..., 30, 30, COLOR_COUNT)``.
    query_grid : jax.Array
        Captured query input as colour one-hots shaped ``(..., 30, 30, 10)``.
    output_shape, input_shape : tuple of jax.Array
        ``(height, width)`` integer side lengths shaped ``(...,)``.

    Returns
    -------
    jax.Array
        Normalised per-cell log-probabilities shaped ``(..., 30, 30, 10)``.
    """
    output_height, output_width = output_shape
    input_height, input_width = input_shape
    rows = _nearest_source_index(output_height, input_height)
    columns = _nearest_source_index(output_width, input_width)
    # The transposed component reads the input's other axis on each side, so a
    # transposing shape rule and a transposing colour copy stay consistent.
    rows_transposed = _nearest_source_index(output_height, input_width)
    columns_transposed = _nearest_source_index(output_width, input_height)

    aligned = _gather_cells(query_grid, rows, columns)
    transposed = jnp.swapaxes(
        _gather_cells(query_grid, columns_transposed, rows_transposed), -3, -2
    )
    copy_component = jax.nn.log_softmax(aligned * COPY_SHARPNESS, axis=-1)
    transpose_component = jax.nn.log_softmax(transposed * COPY_SHARPNESS, axis=-1)
    palette_component = jnp.broadcast_to(
        jax.nn.log_softmax(palette_logits, axis=-1)[..., None, None, :],
        copy_component.shape,
    )
    explicit_component = jax.nn.log_softmax(explicit_logits, axis=-1)

    components = jnp.stack(
        (
            copy_component,
            transpose_component,
            palette_component,
            explicit_component,
        ),
        axis=-2,
    )
    weights = jax.nn.log_softmax(gate_logits, axis=-1)
    return jax.scipy.special.logsumexp(weights[..., None] + components, axis=-2)


def decode_side_length(axis_logits: jax.Array) -> jax.Array:
    """Return the predicted side length implied by axis log-probabilities.

    The colour alignment needs a concrete side length, so the argmax is taken
    with its gradient stopped: the shape head is trained by its own objective,
    not through the colour path.
    """
    return jax.lax.stop_gradient(jnp.argmax(axis_logits, axis=-1) + 1)
