"""Tests for the SP12 KsAHP dose calculator and stage scorer."""

import numpy as np
import pytest

import h01_e_sahp as sahp


def _spike_trace(spike_ms=(1147.,), hold_mv=-65., stop_ms=2300.):
    time = np.arange(0., stop_ms, .02)
    voltage = np.full_like(time, hold_mv)
    for s in spike_ms:
        voltage[(time >= s) & (time < s+1.)] = 30.
    return time, voltage


def test_gate_is_silent_below_vhalf_and_opens_for_one_ms_above():
    time, voltage = _spike_trace()
    z = sahp.gate_trace(time, voltage)
    before = z[time < 1147.]
    assert before.max() < 1e-9
    peak = z[(time >= 1147.) & (time < 1148.5)].max()
    assert peak == pytest.approx(1.-np.exp(-1.), abs=.02)
    late = z[(time >= 2147.) & (time < 2147.1)][0]
    assert late == pytest.approx(peak*np.exp(-1.), rel=.03)


def test_single_spike_voltage_keeps_only_the_first_spike():
    time, voltage = _spike_trace(spike_ms=(1147., 1216.))
    single, spike_ms, above_ms = sahp.single_spike_voltage(time, voltage, hold_mv=-67.)
    assert spike_ms == pytest.approx(1147., abs=.03)
    assert above_ms == pytest.approx(1., abs=.03)
    assert single[(time >= 1216.) & (time < 1217.)].max() == -67.
    assert single[(time >= 1147.) & (time < 1148.)].max() == 30.
    with pytest.raises(ValueError):
        sahp.single_spike_voltage(time, np.full_like(time, -65.), hold_mv=-67.)


def test_dose_scales_the_target_offset_by_gate_driving_force_and_area():
    gbar = sahp.dose_for_offset(z_mean=.5, offset_na=.02, plateau_mv=-67.)
    assert gbar*.5*40.*sahp.SOMA_AREA_UM2*sahp.DENSITY_TO_NA == pytest.approx(.02)
    with pytest.raises(ValueError):
        sahp.dose_for_offset(0., .02, -67.)


def test_register_reports_the_gate_and_one_dose_per_target(tmp_path):
    time, voltage = _spike_trace(spike_ms=(1147., 1216.))
    npz = tmp_path/"m.npz"
    np.savez(npz, time_ms=time, voltage_mv=voltage)
    out = sahp.register(npz, [.02, .05], hold_mv=-67.)
    assert out["first_spike_ms"] == pytest.approx(1147., abs=.03)
    assert 0 < out["gate_late_mean"] < out["gate_after_first_spike"] < 1
    assert out["doses"]["0.050 nA"]["gbar_s_cm2"] == pytest.approx(2.5*out["doses"]["0.020 nA"]["gbar_s_cm2"])


def _levels(late, mid, post=-84.):
    return {"late_pulse": {"p50_mv": late}, "mid_pulse": {"p50_mv": mid}, "post_pulse": {"p50_mv": post}}


def test_dose_bands_hold_for_a_single_spike_at_the_human_level():
    summary = {"count": 1, "axon_first": True}
    pre = {"prespike_max_dev_mv": .01, "first_spike_shift_ms": .0}
    bands = sahp.dose_bands(summary, _levels(-67.5, -67.), _levels(-67.06, -67.47), pre)
    assert all(b["held"] for b in bands)
    bands = sahp.dose_bands({"count": 3, "axon_first": True}, _levels(-64.8, -64.), _levels(-67.06, -67.47),
                            {"prespike_max_dev_mv": .5, "first_spike_shift_ms": None})
    assert [b["row"] for b in bands if not b["held"]] == ["count", "late_p50_minus_human_mv", "mid_p50_minus_human_mv",
                                                          "prespike_max_dev_mv", "first_spike_shift_ms"]


def test_control_bands_expect_nap_removal_to_keep_spikes_and_a_small_shift():
    bands = sahp.control_bands({"count": 3}, _levels(-65.5, -63.), _levels(-65.1, -64.))
    assert all(b["held"] for b in bands)
    bands = sahp.control_bands({"count": 1}, _levels(-67.1, -67.), _levels(-65.1, -64.))
    assert not any(b["held"] for b in bands)


def test_prespike_deviation_compares_only_before_the_reference_spike():
    time, reference = _spike_trace(spike_ms=(1147.,))
    _, run = _spike_trace(spike_ms=(1147.,))
    run = run+np.where(time > 1160., -3., 0.)
    pre = sahp.prespike_deviation({"time_ms": time, "voltage_mv": run}, {"time_ms": time, "voltage_mv": reference})
    assert pre["prespike_max_dev_mv"] == 0.
    assert pre["first_spike_shift_ms"] == 0.
