"""Exporter and simulation driver judged together on every registered and sealed sweep.

The continuation spec's prevention rule: a holdout release is checked at both entry
points before the source waveform is opened. A registered (calibration) sweep must be
accepted by the exporter and by the driver's input guard; a sealed sweep must be
rejected by both until its prediction file exists.
"""

import json
import runpy
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

FOLDER = Path(__file__).resolve().parent


@pytest.fixture
def exporter(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(FOLDER))
    import h01_l2_sweep_export as module

    monkeypatch.setattr(module, "EVIDENCE", tmp_path/"evidence")
    (tmp_path/"evidence").mkdir()
    return module


def _cache(tmp_path, sweeps):
    cache = tmp_path/"cache"
    cache.mkdir()
    (cache/"long-square-index.json").write_text(json.dumps(
        [{"sweep": f"Sweep_{s}", "bias_a": -3.7e-12} for s in sweeps]))
    with h5py.File(cache/"recording.nwb", "w") as nwb:
        for sweep in sweeps:
            group = nwb.create_group(f"acquisition/timeseries/Sweep_{sweep}")
            group.create_dataset("data", data=np.full(40, -.07, dtype=np.float32))
            group.create_dataset("starting_time", data=0.).attrs["rate"] = 50000.
            nwb.create_dataset(f"stimulus/presentation/Sweep_{sweep}/data", data=np.full(40, 2e-10))
    return cache


def _driver_exit(monkeypatch, tmp_path, sweep):
    """The driver's outcome for one sweep with no source files: SystemExit or FileNotFoundError."""
    fake_neuron = types.ModuleType("neuron")
    fake_neuron.h = object()
    monkeypatch.setitem(sys.modules, "neuron", fake_neuron)
    monkeypatch.setattr(sys, "argv", ["reference", "--cache", str(tmp_path/"absent"),
                                      "--output", str(tmp_path/"out"), "--sweep", str(sweep)])
    try:
        runpy.run_path(str(FOLDER/"h01_l2_neuron_reference.py"), run_name="__main__")
    except SystemExit as error:
        return "rejected", error.code
    except FileNotFoundError as error:
        return "accepted", str(error)
    raise AssertionError("driver reached neither the input guard nor source loading")


def test_registered_sweep_accepted_by_exporter_and_driver(exporter, monkeypatch, tmp_path):
    cache = _cache(tmp_path, exporter.CALIBRATION)
    for sweep in exporter.CALIBRATION:
        assert exporter.sealed_reason(sweep) is None
        exporter.main(["--cache", str(cache), "--sweep", str(sweep)])
        assert (cache/f"sweep-{sweep}.npz").exists()
        state, detail = _driver_exit(monkeypatch, tmp_path, sweep)
        assert state == "accepted" and "541563728_fit.json" in detail


@pytest.mark.parametrize("sweep", [54, 55])
def test_sealed_sweep_rejected_by_both_until_prediction_then_accepted_by_both(
        exporter, monkeypatch, tmp_path, capsys, sweep):
    cache = _cache(tmp_path, [sweep])
    prediction = exporter.SEALED[sweep]
    with pytest.raises(SystemExit, match=f"sealed until {prediction}"):
        exporter.main(["--cache", str(cache), "--sweep", str(sweep)])
    assert not (cache/f"sweep-{sweep}.npz").exists()
    state, code = _driver_exit(monkeypatch, tmp_path, sweep)
    assert (state, code) == ("rejected", 2)
    assert f"sealed until {prediction}" in capsys.readouterr().err
    (tmp_path/"evidence"/prediction).write_text("{}")
    exporter.main(["--cache", str(cache), "--sweep", str(sweep)])
    assert (cache/f"sweep-{sweep}.npz").exists()
    assert _driver_exit(monkeypatch, tmp_path, sweep)[0] == "accepted"


def test_unknown_sweep_rejected_by_driver_and_absent_from_exporter_registry(exporter, monkeypatch, tmp_path):
    assert 52 not in exporter.DRIVER_SWEEPS
    assert _driver_exit(monkeypatch, tmp_path, 52) == ("rejected", 2)
