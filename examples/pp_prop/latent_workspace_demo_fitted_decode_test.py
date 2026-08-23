"""Tests for the demonstration-fitted decoder."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from latent_workspace_demo_fitted_decode import (
    CELLS,
    DECODER_WIDTH,
    OUTSIDE_CLASS,
    _extent_log_likelihood,
    _pack_window,
    build_demonstration_batch,
    demonstration_fitted_windows,
)
from latent_workspace_demonstration_forest import DemonstrationForestConfig


def _identity_task(seed=0, size=4, count=3):
    generator = np.random.default_rng(seed)
    grids = [
        generator.integers(0, 6, size=(size, size)).astype(np.int32)
        for _ in range(count + 1)
    ]
    return [(grid, grid.copy()) for grid in grids[:count]], grids[count]


def test_a_batch_pads_onto_the_thirty_by_thirty_canvas():
    pairs, query = _identity_task()
    batch = build_demonstration_batch(pairs, query)
    assert batch.demonstration_grids.shape == (3, 30, 30)
    assert batch.demonstration_targets.shape == (3, CELLS)
    assert batch.query_height == 4 and batch.query_width == 4


def test_cells_beyond_a_demonstration_output_are_labelled_outside():
    pair = (np.array([[1, 2]]), np.array([[3]]))
    batch = build_demonstration_batch([pair], np.array([[1, 2]]))
    targets = batch.demonstration_targets[0].reshape(30, 30)
    assert int(targets[0, 0]) == 3
    assert int(targets[0, 1]) == OUTSIDE_CLASS
    assert int(targets[10, 10]) == OUTSIDE_CLASS


def test_an_empty_demonstration_set_is_refused():
    with pytest.raises(ValueError):
        build_demonstration_batch([], np.array([[1]]))


def test_no_queries_is_refused():
    with pytest.raises(ValueError):
        demonstration_fitted_windows([], DemonstrationForestConfig())


@pytest.mark.parametrize("extent", [1, 5, 17, 30])
def test_extent_likelihood_peaks_at_the_occupied_prefix(extent):
    inside = np.full(30, 0.02, np.float32)
    inside[:extent] = 0.98
    scores = np.asarray(_extent_log_likelihood(jnp.asarray(inside)))
    assert int(scores.argmax()) == extent - 1


def test_extent_likelihood_prefers_the_longer_prefix_when_evidence_extends():
    short = np.full(30, 0.02, np.float32)
    short[:3] = 0.98
    long = short.copy()
    long[3] = 0.98
    assert int(np.asarray(_extent_log_likelihood(jnp.asarray(long))).argmax()) == 3
    assert int(np.asarray(_extent_log_likelihood(jnp.asarray(short))).argmax()) == 2


def test_the_packed_window_uses_the_row_refinement_layout():
    probabilities = np.full((CELLS, 11), 1.0 / 11.0, np.float32)
    window = np.asarray(_pack_window(jnp.asarray(probabilities)))
    assert window.shape == (DECODER_WIDTH,)
    assert np.isfinite(window).all()


def test_a_confident_two_by_two_prediction_decodes_its_own_extent():
    probabilities = np.zeros((30, 30, 11), np.float32)
    probabilities[..., OUTSIDE_CLASS] = 1.0
    probabilities[:2, :2] = 0.0
    probabilities[:2, :2, 4] = 1.0
    window = np.asarray(_pack_window(jnp.asarray(probabilities.reshape(CELLS, 11))))
    assert int(window[:30].argmax()) == 1
    assert int(window[30:60].argmax()) == 1
    colours = window[60:].reshape(30, 30, 10)
    assert int(colours[0, 0].argmax()) == 4


def test_windows_have_the_decoder_width_and_dtype():
    pairs, query = _identity_task()
    batch = build_demonstration_batch(pairs, query)
    config = DemonstrationForestConfig(depth=4, tree_count=1)
    windows = demonstration_fitted_windows([batch, batch], config)
    assert windows.shape == (2, DECODER_WIDTH)
    assert windows.dtype == np.float32


def test_an_identity_task_is_answered_exactly():
    """The head must reproduce a rule its demonstrations fully determine.

    Output equals input is the one transformation the shipped copy machine
    already gets right, so a head that could not do it would be a regression
    rather than a replacement.
    """

    pairs, query = _identity_task(seed=5)
    batch = build_demonstration_batch(pairs, query)
    config = DemonstrationForestConfig(depth=10, tree_count=1, feature_fraction=1.0)
    window = demonstration_fitted_windows([batch], config)[0]
    height = int(window[:30].argmax()) + 1
    width = int(window[30:60].argmax()) + 1
    assert (height, width) == query.shape
    colours = window[60:].reshape(30, 30, 10).argmax(-1)
    assert np.array_equal(colours[:height, :width], query)


def test_a_shrinking_task_decodes_a_smaller_extent_than_its_input():
    """Extent comes from the fitted head, never from the query's own shape."""

    generator = np.random.default_rng(1)
    pairs = []
    for _ in range(4):
        grid = generator.integers(1, 6, size=(6, 6)).astype(np.int32)
        grid[3:] = 0
        pairs.append((grid, grid[:3].copy()))
    query = generator.integers(1, 6, size=(6, 6)).astype(np.int32)
    query[3:] = 0
    batch = build_demonstration_batch(pairs, query)
    config = DemonstrationForestConfig(depth=10, tree_count=1, feature_fraction=1.0)
    window = demonstration_fitted_windows([batch], config)[0]
    assert int(window[:30].argmax()) + 1 == 3
    assert int(window[30:60].argmax()) + 1 == 6
