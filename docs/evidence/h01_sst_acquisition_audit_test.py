"""Audit logic on synthetic inputs; the cached NWB is exercised only when present."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h01_sst_acquisition_audit as audit  # noqa: E402

HL5MN1 = "proc biophys_HL5MN1(){\n\tforsec $o1.all {\n\t\tcm = 1\n\t}\n}\n"
HL23SST = HL5MN1.replace("biophys_HL5MN1", "biophys_HL23SST")


def test_hoc_identity_accepts_name_only_difference():
    result = audit.hoc_identity(HL5MN1, HL23SST)
    assert result["identical_parameters"] is True
    assert result["differing_lines"] == [("proc biophys_HL5MN1(){", "proc biophys_HL23SST(){")]


def test_hoc_identity_rejects_changed_value_or_length():
    changed = audit.hoc_identity(HL5MN1, HL23SST.replace("cm = 1", "cm = 2"))
    assert changed["identical_parameters"] is False and len(changed["differing_lines"]) == 2
    longer = audit.hoc_identity(HL5MN1, HL23SST+"\n// extra\n")
    assert longer["identical_parameters"] is False
    assert longer["differing_lines"][-1] == ("<5 lines>", "<7 lines>")


def test_step_window_finds_the_nominal_amplitude_samples():
    waveform = np.zeros(100)
    waveform[10:20] = 50.
    waveform[40:60] = 110.
    assert audit.step_window_ms(waveform, 110., 1000.) == (40., 60.)


def test_pinned_files_cover_every_mechanism_and_the_recording():
    names = set(audit.FILES)
    assert {f"mod/{m}.mod" for m in audit.MECHANISMS} <= names
    assert {"biophys_HL5MN1.hoc", "biophys_HL23SST.hoc", "HL5MN1.swc", "NeuronTemplate.hoc", "LICENSE",
            "571700399_ephys.nwb"} <= names
    assert audit.FILES["571700399_ephys.nwb"] == audit.NWB_URL
    assert all(url.startswith("http") for url in audit.FILES.values())


def test_export_sweep_crops_and_corrects(tmp_path):
    h5py = pytest.importorskip("h5py")
    rate = 50000
    samples = 3*rate
    waveform = np.zeros(samples)
    waveform[int(1.02*rate):int(2.02*rate)] = 100e-12
    voltage = np.linspace(-.07, -.06, samples)
    path = tmp_path/"synthetic.nwb"
    with h5py.File(path, "w") as nwb:
        group = nwb.create_group("acquisition/timeseries/Sweep_44")
        group.create_dataset("data", data=voltage)
        group.create_dataset("starting_time", data=0.).attrs["rate"] = float(rate)
        group.create_dataset("bias_current", data=43e-12)
        stimulus = nwb.create_group("stimulus/presentation/Sweep_44")
        stimulus.create_dataset("data", data=waveform)
        stimulus.create_dataset("aibs_stimulus_name", data=np.bytes_(b"Long Square"))
        stimulus.create_dataset("aibs_stimulus_amplitude_pa", data=100.)
    with h5py.File(path, "r") as nwb:
        record = audit.export_sweep(nwb, 44, tmp_path/"active-0.npz")
        table = audit.sweep_table(nwb)
    exported = np.load(tmp_path/"active-0.npz")
    assert record["step_ms_exported"] == [270., 1270.] and record["bias_current_pa"] == pytest.approx(43.)
    assert len(exported["voltage"]) == samples-audit.CROP_START_SAMPLES
    assert exported["voltage"][0] == pytest.approx(voltage[audit.CROP_START_SAMPLES]*1000.-14.)
    assert exported["time"][1] == pytest.approx(.02)
    assert table == [{"sweep": 44, "stimulus": "Long Square", "amplitude_pa": 100., "sampling_rate_hz": 50000.,
                      "samples": samples, "bias_current_pa": pytest.approx(43.)}]


@pytest.mark.skipif(not (audit.CACHE/"571700399_ephys.nwb").exists(), reason="donor cache not fetched")
def test_cached_acquisition_matches_the_versioned_report():
    report = audit.audit()
    versioned = json.loads((audit.ROOT/"docs/evidence/h01-sst-acquisition.json").read_text())
    assert report["files"] == versioned["files"]
    assert report["verification_1_hoc_identity"]["identical_parameters"] is True
    assert [e["sweep"] for e in report["exports"]] == [44, 35, 23]
    assert report["exports"] == versioned["exports"]
