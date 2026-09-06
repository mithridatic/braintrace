"""Tests for the L2 sweep exporter."""

import json

import h5py
import numpy as np
import pytest

from docs.evidence.h01_l2_sweep_export import export_sweep, main, matches


@pytest.fixture
def cache(tmp_path):
    (tmp_path/"long-square-index.json").write_text(json.dumps([{"sweep": "Sweep_7", "bias_a": -2e-12}]))
    with h5py.File(tmp_path/"recording.nwb", "w") as nwb:
        group = nwb.create_group("acquisition/timeseries/Sweep_7")
        group.create_dataset("data", data=np.linspace(-.07, -.06, 50, dtype=np.float32))
        group.create_dataset("starting_time", data=0.).attrs["rate"] = 50000.
        nwb.create_dataset("stimulus/presentation/Sweep_7/data", data=np.full(50, 1e-10))
    return tmp_path


def test_export_builds_the_driver_waveform_format(cache):
    arrays = export_sweep(cache, 7)
    assert arrays["time_ms"][1] == pytest.approx(.02)
    assert arrays["corrected_voltage_mv"][0] == pytest.approx(-84.)
    assert arrays["command_current_na"][0] == pytest.approx(.1)
    assert float(arrays["bias_current_na"]) == pytest.approx(-.002)
    assert arrays["total_current_na"][0] == pytest.approx(.098)


def test_main_writes_once_and_verifies(cache, capsys):
    main(["--cache", str(cache), "--sweep", "7"])
    assert (cache/"sweep-7.npz").exists()
    with pytest.raises(SystemExit):
        main(["--cache", str(cache), "--sweep", "7"])
    main(["--cache", str(cache), "--sweep", "7", "--verify"])
    assert max(json.loads(capsys.readouterr().out.strip().splitlines()[-1]).values()) == 0.
    assert matches(export_sweep(cache, 7), cache/"sweep-7.npz")["time_ms"] == 0.
