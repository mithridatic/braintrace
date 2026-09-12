"""Declared glial electricity, compiled evolution and fresh-source replay."""

from dataclasses import replace
import json

import braincell
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

from .h01_glia import H01GlialSelection, load_glial_fragment
from .h01_glia_test import source
from .h01_glial_cable import build_glial_cable
from .h01_network_step import H01NetworkStep


def electrical():
    return dict(initial_mv=-100.,inside_k_mm=140.,outside_k_mm=10.,temperature_c=34.,
                gkir_ms_cm2=.4,cm_uf_cm2=1.,ra_ohm_cm=100.,max_cv_length_um=.5,
                basis='Diagnostic fixed pools and Kir-only membrane; not fitted physiology')


def test_compiled_tapered_fragment_and_fresh_reconstruction(tmp_path):
    path, document, _=source(tmp_path)
    with brainstate.environ.context(precision=64):
        def construct():
            fragment=load_glial_fragment(path,H01GlialSelection(document))
            cell, record=build_glial_cable(fragment,electrical())
            net=braincell.Network()
            net.add_population('glia',cell)
            return H01NetworkStep(net),cell,record
        step,cell,record=construct()
        def rollout(driver, cable):
            def advance(_):
                driver.update(sample_probes=False)
                return cable.V.value.to_decimal(u.mV)
            return brainstate.transform.for_loop(advance,jnp.arange(20))
        actual=rollout(step,cell)
        assert np.isfinite(actual).all() and np.any(np.asarray(actual[-1]) != -100.)
        assert cell.n_cv > 4 and step.tick.value==20
        fresh,fresh_cell,fresh_record=construct()
        assert record==fresh_record
        np.testing.assert_array_equal(rollout(fresh,fresh_cell),actual)
        step.reset_state()
        np.testing.assert_array_equal(rollout(step,cell),actual)
        changed=dict(electrical(),gkir_ms_cm2=.2)
        _,other=build_glial_cable(load_glial_fragment(path,H01GlialSelection(document)),changed)
        assert record['assembly_sha256'] != other['assembly_sha256']


@pytest.mark.parametrize('change',[dict(basis=''),dict(extra=0),dict(initial_mv=True),dict(inside_k_mm=0),
    dict(gkir_ms_cm2=-1),dict(temperature_c=-274),dict(ra_ohm_cm=float('nan'))])
def test_invalid_electrical_settings(tmp_path,change):
    path, document, _=source(tmp_path)
    fragment=load_glial_fragment(path,H01GlialSelection(document))
    with pytest.raises(ValueError):
        build_glial_cable(fragment,dict(electrical(),**change))


def test_changed_reconstruction_identity_is_rejected(tmp_path):
    path, document, _=source(tmp_path)
    fragment=load_glial_fragment(path,H01GlialSelection(document))
    record=fragment.evidence
    record['geometry_sha256']='0'*64
    with pytest.raises(ValueError,match='geometry changed'):
        build_glial_cable(replace(fragment,evidence_json=json.dumps(record)),electrical())


def test_tapered_cable_spreads_local_voltage_by_axial_current(tmp_path):
    path, document, _=source(tmp_path)
    with brainstate.environ.context(precision=64):
        fragment=load_glial_fragment(path,H01GlialSelection(document))
        cell,_=build_glial_cable(fragment,dict(electrical(),gkir_ms_cm2=0.))
        net=braincell.Network()
        net.add_population('glia',cell)
        step=H01NetworkStep(net)
        cell.V.value=cell.V.value.to_decimal(u.mV).at[0,0].set(-90.)*u.mV
        brainstate.transform.for_loop(lambda _: step.update(sample_probes=False),jnp.arange(20))
        voltage=np.asarray(cell.V.value.to_decimal(u.mV))
        assert np.isfinite(voltage).all()
        assert voltage[0,0] < -90. and np.any(voltage[0,1:] > -100.)
        assert voltage.min() >= -100.-1e-8 and voltage.max() <= -90.+1e-8
