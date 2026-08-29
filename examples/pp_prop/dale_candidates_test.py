"""Tests for measured Dale candidate selection and sparse constraints."""

from types import SimpleNamespace

import brainevent
import jax.numpy as jnp
import numpy as np
import pytest

from examples.pp_prop.dale_candidates import (
    DaleMeasurements,
    deferred_biology_defaults,
    effective_dale_weights,
    encode_dale_weights,
    make_dale_weight_fn,
    measure_dale_candidates,
    project_dale_raw_weights,
    run_dale_arm,
    run_dale_candidates,
    sparse_dale_matmul,
    strict_dale_gate,
    validate_deferred_biology,
    validate_effective_signs,
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


def test_typed_value_below_floor_is_clamped_by_inverse_encoding() -> None:
    raw = encode_dale_weights(jnp.array([0.0]), jnp.array([1]))
    effective = make_dale_weight_fn(jnp.array([1]))(raw)
    assert float(effective[0]) == pytest.approx(1e-6, abs=1e-10)


def test_deferred_biology_defaults_to_disabled() -> None:
    validate_deferred_biology()
    validate_deferred_biology(ampa=False, gabaa=False, nmda=False)
    with pytest.raises(ValueError, match="deferred biology"):
        validate_deferred_biology(ampa=True)


def test_unknown_biology_option_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown biology"):
        validate_deferred_biology(calcium=True)


def test_selection_uses_only_untyped_parent_neurons() -> None:
    measurement = _measurements()
    typed = DaleMeasurements(
        measurement.parent_id, measurement.rows, measurement.weights,
        measurement.activity, measurement.gradient_mass,
        measurement.task_ownership, measurement.lesion_evidence,
        np.array([1, 0, 0, -1]),
    )
    selected = measure_dale_candidates(typed, fraction=1.0)
    assert set(selected.excitatory.tolist()) == {1, 2}
    assert set(selected.inhibitory.tolist()) == {1, 2}
    with pytest.raises(ValueError, match="no untyped"):
        measure_dale_candidates(DaleMeasurements(
            typed.parent_id, typed.rows, typed.weights, typed.activity,
            typed.gradient_mass, typed.task_ownership, typed.lesion_evidence,
            np.ones(4, dtype=np.int8),
        ))


def test_effective_transform_uses_exact_softplus_and_signed_raw_fallback() -> None:
    signs = jnp.array([1, -1, 0])
    raw = jnp.array([0.0, 0.0, -2.0])
    effective = effective_dale_weights(raw, signs)
    expected = jnp.array([jnp.log(2.0), -jnp.log(2.0), -2.0])
    np.testing.assert_allclose(effective, expected)
    assert validate_effective_signs(raw, signs)
    with pytest.raises(ValueError, match="equal shape"):
        validate_effective_signs(jnp.ones(2), signs)
    projected = project_dale_raw_weights(jnp.array([-100.0, -2.0]), signs[:2])
    assert float(effective_dale_weights(projected, signs[:2])[0]) == pytest.approx(1e-6, abs=1e-10)


def test_dale_validation_rejects_invalid_signs_and_floor() -> None:
    with pytest.raises(ValueError, match="type signs"):
        make_dale_weight_fn(jnp.array([2]))
    with pytest.raises(ValueError, match="positive"):
        make_dale_weight_fn(jnp.array([1]), floor=0)
    with pytest.raises(ValueError, match="equal shape"):
        encode_dale_weights(jnp.ones(2), jnp.ones(1))
    with pytest.raises(ValueError, match="positive"):
        encode_dale_weights(jnp.ones(1), jnp.ones(1), floor=0)
    with pytest.raises(ValueError, match="equal shape"):
        project_dale_raw_weights(jnp.ones(2), jnp.ones(1))


def test_measurement_validation_fails_closed() -> None:
    measurement = _measurements()
    invalid = (
        DaleMeasurements("p", np.array([[0]]), measurement.weights,
                         measurement.activity, measurement.gradient_mass,
                         measurement.task_ownership, measurement.lesion_evidence),
        DaleMeasurements("p", measurement.rows, measurement.weights,
                         np.ones((4, 1)), measurement.gradient_mass,
                         measurement.task_ownership, measurement.lesion_evidence),
        DaleMeasurements("p", measurement.rows, measurement.weights,
                         np.ones(3), measurement.gradient_mass,
                         measurement.task_ownership, measurement.lesion_evidence),
        DaleMeasurements("p", np.array([9] * 8), measurement.weights,
                         measurement.activity, measurement.gradient_mass,
                         measurement.task_ownership, measurement.lesion_evidence),
        DaleMeasurements("p", measurement.rows, measurement.weights,
                         measurement.activity, measurement.gradient_mass,
                         measurement.task_ownership, measurement.lesion_evidence,
                         np.array([2, 0, 0, 0])),
    )
    messages = ("equal length", "one-dimensional", "shared neuron", "existing", "type signs")
    for value, message in zip(invalid, messages):
        with pytest.raises(ValueError, match=message):
            measure_dale_candidates(value)
    with pytest.raises(ValueError, match="interval"):
        measure_dale_candidates(measurement, fraction=0)


def test_deferred_defaults_cover_every_optional_feature() -> None:
    defaults = deferred_biology_defaults()
    assert defaults and not any(defaults.values())
    for name in defaults:
        with pytest.raises(ValueError, match="deferred biology"):
            validate_deferred_biology(**{name: True})


class _EagerTransform:
    @staticmethod
    def jit(function):
        return function

    @staticmethod
    def for_loop(function, values):
        return np.stack([function(value) for value in values])


def test_dale_runner_isolates_both_arms_and_requires_exact_gate() -> None:
    parent = SimpleNamespace(parent_id="accepted-parent", updates=0)
    measurements = _measurements()
    built = []

    def build(parent_value, indices, sign):
        candidate = SimpleNamespace(parent_id=parent_value.parent_id, updates=0, sign=sign)
        built.append((parent_value, tuple(indices), sign))
        return candidate

    def update(candidate, _index):
        candidate.updates += 1
        return candidate.updates

    def strict(value):
        return (value.updates == 64,)

    result = run_dale_candidates(
        parent, measurements, build, update, strict,
        transform=_EagerTransform, clock=iter((0.0, 1.0, 1.0, 2.0)).__next__,
    )
    assert result["candidate_count"] == 1
    assert [arm["sign"] for arm in result["arms"]] == [1, -1]
    assert all(arm["updates"] == 64 and arm["promoted"] for arm in result["arms"])
    assert all(item[0] is parent for item in built)
    assert strict_dale_gate((False, True), (True, True), updates=64)
    assert not strict_dale_gate((True,), (False,), updates=64)
    with pytest.raises(ValueError, match="64"):
        strict_dale_gate((False,), (True,), updates=8)
    with pytest.raises(ValueError, match="equal length"):
        strict_dale_gate((False,), (False, True), updates=64)
    assert not strict_dale_gate((False,), (True,), updates=64, elapsed_seconds=301.0)


def test_dale_runner_rejects_invalid_arm_and_parent() -> None:
    parent = SimpleNamespace(parent_id="accepted-parent", updates=0)
    with pytest.raises(ValueError, match="Dale sign"):
        run_dale_arm(parent, [0], 0, lambda *_: parent, lambda *_: 0,
                     lambda _: (False,), transform=_EagerTransform)
    with pytest.raises(ValueError, match="measurements"):
        run_dale_candidates(SimpleNamespace(parent_id="other"), _measurements(),
                            lambda *_: parent, lambda *_: 0,
                            lambda _: (False,), transform=_EagerTransform)
