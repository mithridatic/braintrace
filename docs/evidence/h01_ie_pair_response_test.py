import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h01_ie_pair_response as pair  # noqa: E402


def traces(amplitude=-1., tau=4., onset=10., n=4000, dt=.01):
    time = np.arange(n)*dt
    events_i = np.zeros(n, bool)
    events_i[int(onset/dt)] = True
    base = np.full(n, -60.)
    ipsp = np.where(time >= onset, amplitude*(1-np.exp(-(time-onset)/.5))*np.exp(-np.maximum(time-onset-.5, 0.)/tau), 0.)
    disconnected = {"time_ms": time, "E_voltage": base, "E_incoming_voltage": base, "I_events": events_i,
                    "E_events": np.zeros(n, bool)}
    connected = {**disconnected, "E_voltage": base+ipsp, "E_incoming_voltage": base+3*ipsp}
    return connected, disconnected


def test_condition_args_encode_hold_and_synapse():
    args = pair.condition_args("ei", .5, pair.LITERATURE, 40.)
    assert "--control" in args and args[args.index("--control")+1] == "ei"
    assert args[args.index("--e-current-na")+1] == "0.5" and args[args.index("--e-pulse-ms")+1] == "40.0"
    assert args[args.index("--inhibitory-reversal-mv")+1] == "-75.0"


def test_difference_rows_measure_amplitude_latency_and_decay():
    connected, disconnected = traces()
    row = pair.difference_rows(connected, disconnected, "E_voltage")
    expected = float((connected["E_voltage"]-disconnected["E_voltage"]).min())
    assert row["amplitude_mv"] == pytest.approx(expected) and -1. < expected < -.5
    assert 0. < row["latency_to_10pct_ms"] < .2
    assert 3. < row["decay_to_37pct_ms"] < 5.5
    assert row["pre_spike_max_abs_difference_mv"] == 0.


def test_difference_rows_without_i_spike():
    connected, disconnected = traces()
    connected["I_events"] = np.zeros_like(connected["I_events"])
    assert "note" in pair.difference_rows(connected, disconnected, "E_voltage")


def test_score_pair_and_render():
    connected, disconnected = traces()
    report = {"hold_na": .5, "conditions": {"literature": pair.score_pair(connected, disconnected)}}
    soma, site = (report["conditions"]["literature"][k]["amplitude_mv"] for k in ("soma", "receptor_site"))
    assert site == pytest.approx(3*soma)
    assert report["conditions"]["literature"]["e_hold_mv_before_i_spike"] == pytest.approx(-60.)
    text = pair.render(report)
    assert "| literature | -60.0 | soma |" in text


def test_run_condition_invokes_the_wrapper(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pair.subprocess, "run", lambda command, **kw: calls.append(command))
    stem = pair.run_condition("x", ["-m", "mod"], cache=tmp_path, python="py")
    assert stem == tmp_path/"pair-x" and calls[0][:3] == ["py", "-m", "mod"] and calls[0][-1] == str(stem)


def test_hold_tag_has_no_dot():
    assert pair.hold_tag(.4) == "hold400pa" and "." not in pair.hold_tag(.25)
