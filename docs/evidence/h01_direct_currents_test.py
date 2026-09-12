"""Tests for spatial boundaries, current signs, and capacitor charge closure."""

import copy

import numpy as np
import pytest

from docs.evidence.h01_direct_currents import extract_l2, local_currents


def capacitor():
    t = np.array([0., .01, .2, .7, 1.])
    v = -70+10*t
    data = {"time_ms": t, "voltage_mv": v, "applied_current_na": np.full_like(t, .01),
                "soma_icap_ma_cm2": np.full_like(t, .01), "soma_pas_ma_cm2": np.zeros_like(t),
                "n0": v+1, "n1": v-1}
    report = {"charge_balance_geometry": {"location": "soma(0.5)", "area_um2": 100., "cm_uf_cm2": 1.,
        "neighbours": [{"location": f"dend({i})", "voltage_key": f"n{i}", "resistance_mohm": 10.} for i in (0, 1)]},
        "soma_observation_units": {"soma_icap_ma_cm2": "mA/cm2; local", "soma_pas_ma_cm2": "mA/cm2; outward positive"}}
    return data, report


def test_opposite_axial_paths_survive_and_local_charge_closes():
    data, report = capacitor()
    currents, boundary = local_currents(data, report)
    assert boundary["status"] == "available"
    np.testing.assert_allclose(currents["axial_0"], .1)
    np.testing.assert_allclose(currents["axial_1"], -.1)
    np.testing.assert_allclose(currents["balance_residual"], 0., atol=1e-16)
    record, arrays = extract_l2(data, report, (0., 1.))
    w = record["windows"][0]
    assert w["max_abs_capacitor_charge_error_pc"] < 1e-15
    np.testing.assert_allclose(arrays["w0_charge_storage_pc"][-1], .01)
    np.testing.assert_allclose(arrays["w0_dissipation_axial_0_pw"], .1)
    np.testing.assert_allclose(arrays["w0_dissipation_axial_1_pw"], .1)


def test_area_is_per_segment_and_density_sign_is_explicit():
    data, report = capacitor()
    data["soma_pas_ma_cm2"][:] = .2
    small, _ = local_currents(data, report)
    report["charge_balance_geometry"]["area_um2"] = 300.
    large, _ = local_currents(data, report)
    np.testing.assert_allclose(small["pas"], -.2)
    np.testing.assert_allclose(large["pas"], -.6)


@pytest.mark.parametrize("mutation", ["area", "cm", "location", "units", "sign", "resistance", "neighbour_location", "shape", "nan"])
def test_invalid_current_boundary_is_rejected(mutation):
    data, report = capacitor()
    geo = report["charge_balance_geometry"]
    if mutation == "area": geo["area_um2"] = -1
    if mutation == "cm": geo.pop("cm_uf_cm2")
    if mutation == "location": geo.pop("location")
    if mutation == "units": report["soma_observation_units"]["soma_pas_ma_cm2"] = "unknown"
    if mutation == "sign": report["soma_observation_units"]["soma_pas_ma_cm2"] = "mA/cm2"
    if mutation == "resistance": geo["neighbours"][0]["resistance_mohm"] = 0
    if mutation == "neighbour_location": geo["neighbours"][0].pop("location")
    if mutation == "shape": data["n0"] = np.array([1.])
    if mutation == "nan": data["soma_pas_ma_cm2"][0] = np.nan
    with pytest.raises(ValueError):
        local_currents(data, report)


@pytest.mark.parametrize("missing", ["soma_icap_ma_cm2", "soma_pas_ma_cm2", "n0", "applied_current_na"])
def test_missing_probe_preserves_partial_observations_without_balance_pass(missing):
    data, report = capacitor()
    data.pop(missing)
    record, _arrays = extract_l2(data, report, (0., 1.))
    assert record["current_evidence"] == "partial"
    assert "max_abs_current_balance_residual_na" not in record["windows"][0]


def test_missing_geometry_produces_voltage_only_not_whole_soma_guess():
    data, report = capacitor()
    record, arrays = extract_l2(data, {}, (0., 1.))
    assert record["current_evidence"] == "unavailable"
    assert not any("current_" in k for k in arrays)
    for key in ("neighbours",):
        broken = copy.deepcopy(report)
        broken["charge_balance_geometry"].pop(key)
        assert local_currents(data, broken)[1]["status"] == "partial"
    empty = {k: v for k, v in data.items() if not k.startswith("soma_")}
    report["soma_observation_units"] = {}
    assert local_currents(empty, report)[1]["status"] == "partial"


def test_uncovered_window_is_not_integrated_and_invalid_state_is_rejected():
    data, report = capacitor()
    record, _ = extract_l2(data, report, (0., 2.))
    assert record["windows"][0]["status"] == "unavailable"
    report["soma_observation_units"]["soma_cai_mm"] = "mM"
    data["soma_cai_mm"] = np.array([np.nan]*5)
    with pytest.raises(ValueError):
        extract_l2(data, report, (0., 1.))
