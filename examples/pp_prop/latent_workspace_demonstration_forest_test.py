"""Tests for the demonstration-fitted decision forest."""

from __future__ import annotations

from pathlib import Path

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

from latent_workspace_demonstration_forest import (
    CLASS_COUNT,
    DemonstrationForestConfig,
    demonstration_forest_probabilities,
    fit_demonstration_forest,
)


def _key(seed: int):
    return brainstate.random.RandomState(seed).value


def test_forest_randomness_uses_brainstate_only():
    source = Path(fit_demonstration_forest.__code__.co_filename).read_text(
        encoding="utf-8"
    )

    assert "jax.random" not in source


def _conjunction_problem(rows, seed=0):
    """Labels that are an exact depth-two conjunction of two binary features."""

    generator = np.random.default_rng(seed)
    features = (generator.random((rows, 24)) < 0.4).astype(np.float32)
    labels = np.where(
        features[:, 3] > 0.5,
        np.where(features[:, 7] > 0.5, 1, 2),
        np.where(features[:, 11] > 0.5, 3, 4),
    ).astype(np.int32)
    return jnp.asarray(features), jnp.asarray(labels)


@pytest.mark.parametrize(
    "field,value",
    [
        ("depth", 0),
        ("tree_count", 0),
        ("feature_fraction", 0.0),
        ("feature_fraction", 1.5),
        ("backoff", 0.0),
        ("class_count", -1),
    ],
)
def test_configuration_rejects_impossible_shapes(field, value):
    with pytest.raises(ValueError):
        DemonstrationForestConfig(**{field: value})


def test_configuration_rejects_booleans_dressed_as_counts():
    with pytest.raises(ValueError):
        DemonstrationForestConfig(depth=True)


def test_a_tree_recovers_an_exact_conjunction():
    features, labels = _conjunction_problem(600)
    config = DemonstrationForestConfig(depth=6, tree_count=1, feature_fraction=1.0)
    forest = fit_demonstration_forest(_key(0), features, labels, config)
    predicted = np.asarray(
        demonstration_forest_probabilities(forest, features, config)
    ).argmax(-1)
    assert float((predicted == np.asarray(labels)).mean()) == 1.0


def test_the_recovered_rule_transfers_to_unseen_cells():
    features, labels = _conjunction_problem(600)
    held_out, held_out_labels = _conjunction_problem(300, seed=1)
    config = DemonstrationForestConfig(depth=6, tree_count=1, feature_fraction=1.0)
    forest = fit_demonstration_forest(_key(0), features, labels, config)
    predicted = np.asarray(
        demonstration_forest_probabilities(forest, held_out, config)
    ).argmax(-1)
    assert float((predicted == np.asarray(held_out_labels)).mean()) == 1.0


def test_shapes_follow_the_configuration():
    features, labels = _conjunction_problem(80)
    config = DemonstrationForestConfig(depth=4, tree_count=3)
    splits, counts = fit_demonstration_forest(
        _key(0), features, labels, config
    )
    assert splits.shape == (3, 4, 16)
    assert counts.shape == (3, 5, 16, CLASS_COUNT)


def test_probabilities_are_normalised():
    features, labels = _conjunction_problem(200)
    config = DemonstrationForestConfig(depth=5, tree_count=4)
    forest = fit_demonstration_forest(_key(0), features, labels, config)
    probabilities = np.asarray(
        demonstration_forest_probabilities(forest, features, config)
    )
    assert probabilities.shape == (200, CLASS_COUNT)
    assert np.allclose(probabilities.sum(-1), 1.0, atol=1e-5)
    assert float(probabilities.min()) >= 0.0


def test_an_unreached_leaf_backs_off_instead_of_guessing_uniformly():
    """A cell no demonstration resembles must not read as eleven-way uniform.

    Extent is decoded from the probability that a cell lies outside the output
    grid, so a uniform leaf would claim a 10/11 chance of being inside and
    inflate every decoded extent. The path read-back exists to prevent that.
    """

    features = jnp.asarray(np.eye(8, dtype=np.float32))
    labels = jnp.asarray(np.full(8, 10, np.int32))
    config = DemonstrationForestConfig(depth=6, tree_count=1, feature_fraction=1.0)
    forest = fit_demonstration_forest(_key(0), features, labels, config)
    stranger = jnp.zeros((1, 8), jnp.float32)
    probabilities = np.asarray(
        demonstration_forest_probabilities(forest, stranger, config)
    )
    assert int(probabilities.argmax(-1)[0]) == 10
    assert float(probabilities[0, 10]) > 0.9


def test_feature_subsampling_still_fits_the_demonstrations():
    features, labels = _conjunction_problem(400)
    config = DemonstrationForestConfig(depth=7, tree_count=8, feature_fraction=0.5)
    forest = fit_demonstration_forest(_key(0), features, labels, config)
    predicted = np.asarray(
        demonstration_forest_probabilities(forest, features, config)
    ).argmax(-1)
    assert float((predicted == np.asarray(labels)).mean()) > 0.95


def test_a_constant_label_needs_no_split():
    features, _ = _conjunction_problem(50)
    labels = jnp.full((50,), 7, jnp.int32)
    config = DemonstrationForestConfig(depth=3, tree_count=1, feature_fraction=1.0)
    forest = fit_demonstration_forest(_key(0), features, labels, config)
    probabilities = np.asarray(
        demonstration_forest_probabilities(forest, features, config)
    )
    assert set(probabilities.argmax(-1).tolist()) == {7}


def test_fitting_is_deterministic_under_a_fixed_key():
    features, labels = _conjunction_problem(120)
    config = DemonstrationForestConfig(depth=5, tree_count=2, feature_fraction=0.7)
    first = fit_demonstration_forest(_key(3), features, labels, config)
    second = fit_demonstration_forest(_key(3), features, labels, config)
    assert np.array_equal(np.asarray(first[0]), np.asarray(second[0]))
    assert np.allclose(np.asarray(first[1]), np.asarray(second[1]))


def test_different_keys_give_different_feature_subsets():
    features, labels = _conjunction_problem(120)
    config = DemonstrationForestConfig(depth=5, tree_count=2, feature_fraction=0.3)
    first = fit_demonstration_forest(_key(0), features, labels, config)
    second = fit_demonstration_forest(_key(11), features, labels, config)
    assert not np.array_equal(np.asarray(first[0]), np.asarray(second[0]))
