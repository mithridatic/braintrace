"""Tests for the stage-12 scorer (the recording's measurement chain)."""

import numpy as np
import pytest

import h01_topographic_chain as ch


def _synthetic_edges(tau_p_us, fc_khz, t0_ms, delta_r, n=4, noise=.01, seed=0, tau_cmd_us=0.):
    rng = np.random.default_rng(seed)
    t = np.arange(-.3, .5+1e-9, .02)
    edges = []
    for k in range(n):
        i0 = .9+.1*k
        v = ch.chain_response(t, i0, 7.94, 9.5*i0, tau_p_us, fc_khz, t0_ms, delta_r, tau_cmd_us)+rng.normal(0., noise, t.size)
        edges.append({"sweep": k, "t_ms": t, "v_mv": v, "i0_na": i0, "rb_mohm": 7.94})
    return edges


def test_single_pole_and_bessel_preserve_a_constant():
    x = np.full(400, 3.)
    assert ch.single_pole(x, .005, .03)[-1] == pytest.approx(3., abs=1e-6)
    assert ch.bessel4(x, .005, 10.)[-1] == pytest.approx(3., abs=1e-3)
    assert np.array_equal(ch.single_pole(x, .005, 0.), x)


def test_chain_response_shows_the_pipette_transient_only_when_tau_is_live():
    t = np.arange(-.1, .3, .02)
    dead = ch.chain_response(t, 1.2, 7.94, 9.5, 0., 10., 0., .3)
    live = ch.chain_response(t, 1.2, 7.94, 9.5, 50., 10., 0., .3)
    assert dead.min() > -.2
    assert live.min() < -3.


def test_fit_chain_recovers_tau_p_through_a_slow_command():
    edges = _synthetic_edges(30., 20., .06, .4, tau_cmd_us=80.)
    fit = ch.fit_chain(edges, 20.)
    assert fit["tau_p_us"] == pytest.approx(30., abs=8.)
    assert fit["tau_cmd_us"] == pytest.approx(80., abs=30.)
    assert fit["delta_r_mohm"] == pytest.approx(.4, abs=.15)
    assert fit["rms_mv"] < .03


def test_profile_bounds_tau_near_the_truth():
    edges = _synthetic_edges(0., 20., .05, .3, noise=.02, tau_cmd_us=60.)
    pr = ch.profile_tau(edges, 20., grid=(0., 5., 20., 50., 100.))
    assert pr["tau_p_best_us"] <= 5.
    assert pr["tau_p_bound_us"] < 50.


def test_noise_corner_reads_a_flat_and_a_filtered_floor(monkeypatch):
    rng = np.random.default_rng(1)
    t = np.arange(0., 2000., .02)
    white = rng.normal(0., .08, t.size)
    monkeypatch.setattr(ch, "nwb_sweep", lambda path, s: (t, white))
    flat = ch.noise_corner("x", sweeps=[1, 2])
    assert flat["flat_to_nyquist"] and flat["fc_khz"] == ch.FC_FLOOR_KHZ
    monkeypatch.setattr(ch, "nwb_sweep", lambda path, s: (t, ch.bessel4(white, .02, 5.)))
    filtered = ch.noise_corner("x", sweeps=[1, 2])
    assert not filtered["flat_to_nyquist"] and filtered["slope_10_to_20_khz_db_per_octave"] < -12.


def test_apply_chain_lowers_a_fast_rise_and_keeps_a_slow_one():
    t = np.arange(0., 20., .02)
    fast = -60.+100./(1.+np.exp(-(t-10.)/.05))
    slow = -60.+10./(1.+np.exp(-(t-10.)/2.))
    assert np.max(np.gradient(ch.apply_chain(t, fast, 50., 10.), t)) < .9*np.max(np.gradient(fast, t))
    assert np.max(np.gradient(ch.apply_chain(t, slow, 50., 10.), t)) == pytest.approx(np.max(np.gradient(slow, t)), rel=.01)


def _spike_trace(peak=40., rise_scale=.04):
    t = np.arange(0., 2100., .02)
    v = np.full_like(t, -70.)
    on = (t >= 1020.) & (t < 2020.)
    v[on] = -60.
    for s in (1100., 1250.):
        seg = (t >= s) & (t < s+1.)
        v[seg] = -60.+(peak+60.)*np.exp(-((t[seg]-s-.4)/(rise_scale*4))**2)
    return t, v


def test_read_spike_reports_the_landmarks_and_decide_reads_the_cells():
    t, v = _spike_trace()
    r = ch.read_spike(t, v, (1020., 2020.))
    assert r["count"] == 2 and r["peak_mv"] == pytest.approx(40., abs=.5) and r["rise_v_s"] > 100.
    chain = {"tau_p_us": 5., "fc_khz": 10., "rms_mv": .02, "delta_r_mohm": .3, "t0_ms": .05, "tau_cmd_us": 40.}
    profile = {"tau_p_bound_us": 10., "tau_p_best_us": 5., "rows": []}
    noise = {"n_sweeps": 2, "rms_mv": .08, "db_vs_1_2_khz": {1: 0., 10: -4., 20: -4.5}, "slope_10_to_20_khz_db_per_octave": -.5, "flat_to_nyquist": True, "fc_khz": 20.}
    raw = ch.read_spike(t, v, (1020., 2020.)); filt = ch.read_spike(t, ch.apply_chain(t, v, 5., 10.), (1020., 2020.))
    human = dict(raw, rise_v_s=raw["rise_v_s"]*.5, peak_mv=raw["peak_mv"]-4., span_mv=filt["span_mv"]+.2, rapidness_per_ms=30.)
    arm = {"raw": {"soma_slide": {"take_off_mv": [-57., -58.], "slide_mv": -1.}, "approach": [.4, 2.5], "axon1_lead_ms": [.3, .3], "soma_span_mv": [4., 4.]},
           "chain": {"soma_slide": {"take_off_mv": [-57.02, -58.03], "slide_mv": -1.01}, "approach": [.41, 2.5], "axon1_lead_ms": [.3, .31], "soma_span_mv": [4.1, 4.1]}}
    result = {"E": {"chain": chain, "profile": profile, "noise": noise, "human": human, "fit_step": {"status": "read", "raw": raw, "chain": filt, "chain_floor": filt}, "control_step": {"status": "not run"},
                    "peak_bound": {"tau_p_us": 40., **dict(filt, rise_v_s=raw["rise_v_s"]*.8)}, "arm": arm},
              "I": {"chain": chain, "profile": profile, "noise": noise, "human": dict(raw, span_mv=2.7, rapidness_per_ms=52.), "fit_step": {"status": "read", "raw": dict(raw, span_mv=14.), "chain": dict(filt, span_mv=13.)}},
              "reading": ["r"]}
    d = ch.decide(result)
    assert d["verdict"] == "PASS"
    assert any(c["reading"] == "b" and "less than half" in c["prediction"] and c["pass"] for c in d["checks"])
    result["E"]["arm"]["chain"]["soma_slide"]["take_off_mv"] = [-57.5, -58.]
    assert ch.decide(result)["verdict"] == "FAIL"
    result["decision"] = ch.decide(result)
    md = ch.markdown(result)
    for piece in ("## E chain", "| recording sweep 53 |", "## I onset through the I chain", "Verdict: FAIL"):
        assert piece in md
