"""Multivari data: rows per cycle from a synthetic trace, rise rate, missing traces recorded."""

import numpy as np

import h01_multivari_data as data


def _trace(spikes_ms, stop_ms=1500.):
    time = np.arange(0., stop_ms, .02)
    voltage = np.full_like(time, -70.)
    for spike in spikes_ms:
        voltage += 95.*np.exp(-((time-spike)/.25)**2)-8.*np.exp(-((time-spike-3.)/2.)**2)
    return time, voltage


def test_rows_from_trace_one_row_per_cycle_with_positive_rise():
    time, voltage = _trace([300., 340., 380., 420.])
    rows = data.rows_from_trace("I", "model", "0.19 nA", None, time, voltage, (270., 1270.))
    assert len(rows) >= 3 and all(r["rise_v_per_s"] > 0 for r in rows)
    assert [r["cycle"] for r in rows] == list(range(1, len(rows)+1))


def test_rise_rate_none_when_window_is_empty():
    assert data.rise_rate(np.array([0., 1.]), np.array([0., 1.]), -5.) is None


def test_model_rows_report_missing_traces_by_runner(monkeypatch):
    monkeypatch.setattr(data, "MODEL_TRACES", {("I", "9 nA"): "no-such-folder/no-such-trace"})
    rows, missing = data.model_rows("I", data.CELLS["I"])
    assert rows == [] and missing[0]["input"] == "9 nA" and "Docker" in missing[0]["runner"]
