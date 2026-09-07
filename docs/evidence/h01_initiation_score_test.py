import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h01_initiation_score as score  # noqa: E402


def trace(spike_ms, shoulder=False, stop_ms=1200., rest=-70.):
    time = np.arange(1000., stop_ms, .01)
    v = np.full_like(time, rest)
    t0 = spike_ms
    if shoulder:
        ramp = (time >= t0-.06) & (time < t0)
        v[ramp] = -58.+(-48.+58.)*(time[ramp]-(t0-.06))/.06
        slow = (time >= t0) & (time < t0+.3)
        v[slow] = -48.+4.*(time[slow]-t0)/.3
        t0 = t0+.3
        rest_after = -44.
    else:
        rest_after = rest
    rise = (time >= t0) & (time < t0+.1)
    fall = (time >= t0+.1) & (time < t0+.9)
    recover = (time >= t0+.9) & (time < t0+6.)
    v[rise] = rest_after+(30.-rest_after)*(time[rise]-t0)/.1
    v[fall] = 30.-(30.+75.)*(time[fall]-t0-.1)/.8
    v[recover] = -75.+(rest+75.)*(time[recover]-t0-.9)/5.1
    return time, v


def test_axon_first_and_lead():
    time, soma = trace(1050.)
    _, axon = trace(1049.8)
    report = score.score_run(time, soma, axon, (1020., 1200.))
    assert report["spikes"] == 1 and report["axon_first"]
    assert report["axon_leads_ms"] == pytest.approx(.2, abs=.03)
    assert report["shoulder"] is None
    assert report["usable_view"]["count"] == 1


def test_shoulder_detected_inside_bands():
    time, soma = trace(1050., shoulder=True)
    report = score.score_run(time, soma, soma, (1020., 1200.))
    assert report["shoulder"] is not None
    assert -55. <= report["shoulder"]["voltage_mv"] <= -45.
    assert 100. <= report["shoulder"]["slope_v_s"] <= 300.
    assert not report["axon_first"]
    profile = report["slope_profile_v_s"]
    assert profile["-45"] == pytest.approx(13.3, abs=1.)
    assert profile["-35"] == pytest.approx(740., abs=5.)


def test_no_spike_returns_note():
    time = np.arange(1000., 1100., .01)
    flat = np.full_like(time, -70.)
    assert score.score_run(time, flat, flat, (1020., 1100.))["spikes"] == 0


def test_shoulder_needs_three_samples():
    assert score.shoulder(np.array([0., 1.]), np.array([-50., -49.]), 0., 1.) is None


def test_main_writes_report(tmp_path, monkeypatch):
    time, soma = trace(1050.)
    np.savez(tmp_path/"run.npz", time_ms=time, voltage_mv=soma, axon_voltage_mv=soma)
    monkeypatch.setattr(score, "EVIDENCE", tmp_path)
    monkeypatch.setattr(score, "PULSE_MS", (1020., 1200.))
    score.main(["--run", "run"])
    assert (tmp_path/"run-initiation.json").exists()
