"""Focused BrainCell compatibility tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import braincell
import brainunit as u
import jax.numpy as jnp
import pytest


_PATH = Path(__file__).with_name("21-braincell-arc.py")
_SPEC = importlib.util.spec_from_file_location("braincell_arc", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixture)


def test_braincell_pin_and_imports():
    assert braincell.__version__ == "0.1.0"
    assert fixture.brainstate.__name__ == "brainstate"
    assert fixture.braintrace.__name__ == "braintrace"


def test_hodgkin_huxley_constructor_values_and_reset_state():
    cell = fixture.CompatibilityHodgkinHuxley()
    cell.init_state()
    assert cell.length.to_decimal(u.um) == pytest.approx(10.0)
    assert cell.radius.to_decimal(u.um) == pytest.approx(5.0)
    assert cell.C.to_decimal(u.uF / u.cm**2) == pytest.approx(1.0)
    assert cell.V_th.to_decimal(u.mV) == pytest.approx(0.0)
    assert jnp.allclose(cell.V.value.to_decimal(u.mV), -65.0)
    assert jnp.all(cell.spike.value == 0)


def test_csr_and_current_density_contract():
    csr = fixture.input_csr()
    assert csr.shape == (1, 4)
    assert jnp.array_equal(csr.indices, jnp.arange(4))
    assert jnp.array_equal(csr.indptr, jnp.asarray([0, 4]))
    assert jnp.allclose(csr.data, jnp.asarray([0.1, 0.0, 0.0, 0.0]))
    current = fixture.bounded_current_density(1.0)
    assert current.unit == u.mA / u.cm**2
    cell = fixture.CompatibilityHodgkinHuxley()
    cell.init_state()
    with pytest.raises(Exception, match="units do not match"):
        fixture.advance_one_step(cell, 1.0 * u.mA)


def test_finite_difference_fixture_has_declared_tolerance():
    result = fixture.finite_difference_fixture()
    assert result["absolute_error"] <= result["tolerance"]
    assert jnp.isfinite(result["pp_prop"])
    assert result["pp_prop"] != 0.0


def test_pp_prop_compiler_sees_input_and_recurrent_relations():
    relations = fixture.pp_prop_relation_fixture()
    assert len(relations) == 2
    assert all(relation.connected_hidden_paths for relation in relations)
    assert all(relation.trainable_vars for relation in relations)


def test_spike_path_fixture_is_finite_and_crosses_threshold():
    result = fixture.spike_path_fixture()
    assert result == {
        "threshold_crossed": True,
        "finite_voltage": True,
        "finite_spikes": True,
        "finite_gradient": True,
        "nonzero_gradient": True,
    }
