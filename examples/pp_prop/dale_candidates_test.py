"""Tests for measured Dale candidate selection and sparse constraints."""

import jax.numpy as jnp
import numpy as np
import pytest
import brainevent

from examples.pp_prop.dale_candidates import (
    DaleMeasurements,
    encode_dale_weights,
    make_dale_weight_fn,
    measure_dale_candidates,
    sparse_dale_matmul,
    validate_deferred_biology,
)


def _measurements() -> DaleMeasurements:
    return DaleMeasurements(
        "accepted-parent",
        np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        np.array([1, 1, -1, -1, 1, -1, -1, -1], dtype=np.float32),
        np.array([4, 3, 2, 1], dtype=np.float32),
        np.array([1, 2, 3, 4], dtype=np.float32),
        np.array([4, 3, 2, 1], dtype=np.float32),
        np.array([1, 1, 2, 2], dtype=np.float32),
    )


def test_selection_is_measured_stable_and_shares_parent() -> None:
    first = measure_dale_candidates(_measurements(), fraction=0.5)
    second = measure_dale_candidates(_measurements(), fraction=0.5)
    assert first.parent_id == second.parent_id == "accepted-parent"
    np.testing.assert_array_equal(first.excitatory, second.excitatory)
    np.testing.assert_array_equal(first.inhibitory, second.inhibitory)
    assert first.excitatory.size == first.inhibitory.size == 2


def test_ties_use_neuron_index() -> None:
    measurement = _measurements()
    tied = DaleMeasurements(
        measurement.parent_id,
        measurement.rows,
        np.ones_like(measurement.weights),
        np.ones(4), np.ones(4), np.ones(4), np.ones(4),
    )
    selected = measure_dale_candidates(tied, fraction=0.5)
    np.testing.assert_array_equal(selected.excitatory, [0, 1])
    np.testing.assert_array_equal(selected.inhibitory, [0, 1])


def test_dale_weight_fn_preserves_effective_signs_and_floor() -> None:
    transform = make_dale_weight_fn(jnp.array([1, -1, 0]))
    effective = np.asarray(transform(jnp.array([-3.0, 2.0, -4.0])))
    assert effective[0] >= 1e-6
    assert effective[1] <= -1e-6
    assert effective[2] == -4.0


def test_inverse_encoding_round_trips_typed_and_raw_values() -> None:
    signs = jnp.array([1, -1, 0])
    effective = jnp.array([0.25, -0.75, -2.0])
    raw = encode_dale_weights(effective, signs)
    np.testing.assert_allclose(np.asarray(make_dale_weight_fn(signs)(raw)), effective, atol=1e-6)


def test_sparse_dale_matmul_uses_local_weight_fn() -> None:
    sparse = brainevent.CSR(
        jnp.ones(2),
        jnp.array([0, 1]),
        jnp.array([0, 1, 2]),
        shape=(2, 2),
        backend="jax_raw",
    )
    output = sparse_dale_matmul(
        jnp.array([1.0, 1.0]),
        encode_dale_weights(jnp.array([0.5, -0.5]), jnp.array([1, -1])),
        sparse,
        jnp.array([1, -1]),
    )
    np.testing.assert_allclose(np.asarray(output), [0.5, -0.5], atol=1e-6)


def test_invalid_typed_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        encode_dale_weights(jnp.array([0.0]), jnp.array([1]))


def test_deferred_biology_defaults_to_disabled() -> None:
    validate_deferred_biology()
    validate_deferred_biology(ampa=False, gabaa=False, nmda=False)
    with pytest.raises(ValueError, match="deferred biology"):
        validate_deferred_biology(ampa=True)


def test_unknown_biology_option_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown biology"):
        validate_deferred_biology(calcium=True)
