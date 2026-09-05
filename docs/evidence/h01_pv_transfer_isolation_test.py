"""Pure decision, mapping, and held-equal logic of the transfer isolation audit."""
import copy
import numpy as np
import pytest
from docs.evidence import h01_pv_transfer_isolation as iso


def test_neuron_names_map_to_branch_names():
    geometry = [{"name": "NeuronTemplate[0].soma[0]", "nseg": 9}, {"name": "NeuronTemplate[0].dend[12]", "nseg": 171}]
    assert iso.neuron_counts_by_branch(geometry) == {"soma_0": 9, "dend_12": 171}
    with pytest.raises(ValueError, match="Duplicate"):
        iso.neuron_counts_by_branch(geometry+[geometry[0]])


@pytest.mark.parametrize("passed,expected", [
    ({"maxcv_cvode": False, "maxcv_fixed": False, "matched_cvode": True, "matched_fixed": True}, "mesh"),
    ({"maxcv_cvode": False, "maxcv_fixed": True, "matched_cvode": False, "matched_fixed": True}, "integration"),
    ({"maxcv_cvode": False, "maxcv_fixed": False, "matched_cvode": False, "matched_fixed": False}, "time_level_open"),
    ({"maxcv_cvode": False, "maxcv_fixed": False, "matched_cvode": False, "matched_fixed": True}, "interaction"),
    ({"maxcv_cvode": True, "maxcv_fixed": False, "matched_cvode": False, "matched_fixed": True}, "no_prediction_matched"),
    ({"maxcv_cvode": True, "maxcv_fixed": True, "matched_cvode": True, "matched_fixed": True}, "no_prediction_matched"),
])
def test_decision_names_only_the_predicted_patterns(passed, expected):
    assert iso.decide(passed) == expected


def _metas():
    geometry = [{"name": "NeuronTemplate[0].soma[0]", "nseg": 9}, {"name": "NeuronTemplate[0].axon[0]", "nseg": 27}]
    neuron = {k: 1 for k in iso.NEURON_HELD}
    neuron.update(geometry=geometry, integration={"method": "CVode"}, dt_ms=.025, current_na=.27,
                  temperature_c=34., initial_voltage_mv=-80., stimulus_on_ms=270., duration_ms=330.)
    fixed = copy.deepcopy(neuron)
    fixed["integration"] = {"method": "fixed step"}
    fixed["dt_ms"] = .000625
    half = copy.deepcopy(fixed)
    half["dt_ms"] = .0003125
    cell = {k: 1 for k in iso.BRAINCELL_HELD}
    cell.update(solver="staggered", dt_ms=.000625, current_na=.27, temperature_c=34., initial_voltage_mv=-80.,
                stimulus_on_ms=270., duration_ms=330., mesh={"policy": "MaxCVLen"})
    matched = copy.deepcopy(cell)
    matched["mesh"] = {"policy": "CVPerBranchList", "source_sha256": "abc",
                       "branches": ["soma_0", "axon_0"], "cv_per_branch": [9, 27]}
    matched_half = copy.deepcopy(matched)
    matched_half["dt_ms"] = .0003125
    return {"neuron_cvode": neuron, "neuron_fixed": fixed, "neuron_fixed_halfdt": half,
            "braincell_maxcv": cell, "braincell_matched": matched, "braincell_matched_halfdt": matched_half}


def test_held_equal_design_passes_and_reports_shared_dt():
    metas = _metas()
    held = iso.assert_held_equal(metas, "abc")
    assert held["dt_ms"] == .000625 and held["neuron_nseg"] == [9, 27]
    assert iso.assert_matched_mesh(metas["neuron_cvode"], metas["braincell_matched"]) == {"soma_0": 9, "axon_0": 27}


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m["neuron_fixed"].update(sodium_h_slope_mv=6.), "NEURON field"),
    (lambda m: m["neuron_fixed"]["geometry"][0].update(nseg=27), "NEURON mesh"),
    (lambda m: m["braincell_matched"].update(current_na=.19), "BrainCell field"),
    (lambda m: m["braincell_matched"]["mesh"].update(source_sha256="other"), "copied from"),
    (lambda m: [m[k].update(current_na=.19) for k in ("braincell_maxcv", "braincell_matched", "braincell_matched_halfdt")], "cross-simulator"),
    (lambda m: m["braincell_matched"].update(dt_ms=.00125), "fixed-step dt"),
    (lambda m: m["neuron_fixed_halfdt"].update(dt_ms=.000625), "half the shared dt"),
    (lambda m: m["braincell_matched_halfdt"]["mesh"].update(cv_per_branch=[9, 9]), "matched meshes differ"),
])
def test_any_unheld_element_is_rejected(mutate, match):
    metas = _metas()
    mutate(metas)
    with pytest.raises(ValueError, match=match):
        iso.assert_held_equal(metas, "abc")


def test_matched_mesh_mismatch_is_rejected():
    metas = _metas()
    metas["braincell_matched"]["mesh"]["cv_per_branch"] = [9, 9]
    with pytest.raises(ValueError, match="axon_0"):
        iso.assert_matched_mesh(metas["neuron_cvode"], metas["braincell_matched"])


def test_cell_result_reports_late_event_errors_and_gates():
    t = np.arange(0., 340.001, .01)

    def trace(shift):
        v = np.full_like(t, -80.)
        for k in range(8):
            centre = 275.+6.*k+(shift if k >= 6 else 0.)
            v += 120.*np.exp(-((t-centre)/.3)**2)
        return {"time_ms": t, "voltage_mv": v}
    same = iso.cell_result(trace(0.), trace(0.))
    drift = iso.cell_result(trace(0.), trace(-1.7))
    assert same["passed"] and same["rise_error_event_8_ms"] == 0.
    assert not drift["passed"] and drift["rise_error_event_8_ms"] == pytest.approx(-1.7, abs=1e-6)
    assert drift["max_abs_rise_error_ms"] == pytest.approx(1.7, abs=1e-6)
