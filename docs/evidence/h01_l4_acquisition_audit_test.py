"""Audit logic on synthetic inputs; the cached bundle and NWB are exercised only when present."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h01_l4_acquisition_audit as audit  # noqa: E402

FIT = {"genome": [{"section": "soma", "name": "gbar_NaTs", "mechanism": "NaTs", "value": 1.},
                  {"section": "soma", "name": "decay_CaDynamics", "mechanism": "CaDynamics", "value": 2.},
                  {"section": "soma", "name": "g_pas", "mechanism": "", "value": 3.}]}


def test_genome_mechanisms_skip_the_leak_rows():
    assert audit.genome_mechanisms(FIT) == ["CaDynamics", "NaTs"]


def test_mechanism_identity_compares_bundle_against_the_l2_import(tmp_path):
    bundle, reference = tmp_path/"bundle", tmp_path/"reference"
    bundle.mkdir(), reference.mkdir()
    for name in audit.MECHANISMS:
        (bundle/f"{name}.mod").write_text(name)
        (reference/f"{name}.mod").write_text(name)
    assert audit.mechanism_identity(bundle, reference)["identical"] is True
    (reference/"Ih.mod").write_text("changed")
    result = audit.mechanism_identity(bundle, reference)
    assert result["identical"] is False and result["files"]["Ih"]["identical"] is False
    (reference/"Ih.mod").unlink()
    result = audit.mechanism_identity(bundle, reference)
    assert result["files"]["Ih"]["l2_import_sha256"] is None and result["identical"] is False


def test_step_window_and_crossings():
    waveform = np.zeros(100)
    waveform[40:60] = 110.
    assert audit.step_window_ms(waveform, 110., 1000.) == (40., 60.)
    time = np.arange(100.)
    voltage = np.full(100, -70.)
    voltage[[45, 50, 51, 70]] = 0.  # two events inside the step (45; 50-51 is one crossing), one after
    crossings = audit.upward_crossings(voltage, time, 40., 60.)
    assert crossings.tolist() == [45., 50.]


def test_pinned_files_name_the_bundle_recording_and_metadata():
    assert set(audit.FILES) == {"626170709.zip", "527952752_ephys.nwb", "527952884_fit.json",
                                "allen-cell-detail.json", "neuronal-models.json"}
    assert audit.FILES["626170709.zip"].endswith("/neuronal_model/download/626170709")
    assert audit.FILES["527952752_ephys.nwb"].endswith("well_known_file_download/618205555")
    assert all(url.startswith("http") for url in audit.FILES.values())
    assert audit.FIT_SWEEPS == (69, 70, 71, 72) and audit.EXPORTS[0] == 69
    assert len(audit.MECHANISMS) == 11 and "NaTs" in audit.MECHANISMS and "NaTg" not in audit.MECHANISMS


def _synthetic_nwb(path, sweep, name, amplitude_pa, stop_s, bias_pa, rate=50000):
    import h5py

    samples = int((stop_s+1.)*rate)
    waveform = np.zeros(samples)
    waveform[int(1.02*rate):int(stop_s*rate)] = amplitude_pa*1e-12
    voltage = np.full(samples, -.066)
    voltage[int(1.05*rate):int(1.05*rate)+10] = 0.01
    voltage[int(1.5*rate):int(1.5*rate)+10] = 0.01
    with h5py.File(path, "w") as nwb:
        group = nwb.create_group(f"acquisition/timeseries/Sweep_{sweep}")
        group.create_dataset("data", data=voltage)
        group.create_dataset("starting_time", data=0.).attrs["rate"] = float(rate)
        group.create_dataset("bias_current", data=bias_pa*1e-12)
        stimulus = nwb.create_group(f"stimulus/presentation/Sweep_{sweep}")
        stimulus.create_dataset("data", data=waveform)
        stimulus.create_dataset("aibs_stimulus_name", data=np.bytes_(name.encode()))
        stimulus.create_dataset("aibs_stimulus_amplitude_pa", data=amplitude_pa)
    return samples


def test_export_sweep_writes_the_l2_driver_format_and_reads_bias(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path/"synthetic.nwb"
    samples = _synthetic_nwb(path, 69, "Square - 2s Suprathreshold", 100., 3.02, -16.7)
    listed = {"sweep_number": 69, "leak_pa": -16.7, "bridge_balance_mohm": 9.2, "num_spikes": 2}
    with h5py.File(path, "r") as nwb:
        record = audit.export_sweep(nwb, 69, tmp_path/"sweep-69.npz", listed)
        table = audit.sweep_table(nwb, [listed])
    exported = np.load(tmp_path/"sweep-69.npz")
    assert set(exported) == {"time_ms", "reported_voltage_mv", "corrected_voltage_mv", "command_current_na",
                             "bias_current_na", "total_current_na"}
    assert exported["time_ms"][1] == pytest.approx(.02) and len(exported["time_ms"]) == samples
    assert exported["corrected_voltage_mv"][0] == pytest.approx(-66.-14.)
    assert exported["command_current_na"].max() == pytest.approx(.1)
    assert float(exported["bias_current_na"]) == pytest.approx(-.0167)
    assert record["step_ms"] == [1020., 3020.] and record["crossing_count_minus20mv"] == 2
    assert record["first_crossing_after_onset_ms"] == pytest.approx(30.)
    assert record["bias_current_pa"] == pytest.approx(-16.7) and record["allen_num_spikes"] == 2
    assert table[0]["leak_pa"] == -16.7 and table[0]["bridge_balance_mohm"] == 9.2 and table[0]["samples"] == samples


def test_export_sweep_rejects_a_bias_that_disagrees_with_ephys_sweeps(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path/"synthetic.nwb"
    _synthetic_nwb(path, 39, "Long Square", 90., 2.02, 2.3)
    with h5py.File(path, "r") as nwb:
        with pytest.raises(AssertionError, match="leak_pa"):
            audit.export_sweep(nwb, 39, tmp_path/"sweep-39.npz", {"leak_pa": -5., "bridge_balance_mohm": 9., "num_spikes": 1})
        record = audit.export_sweep(nwb, 39, tmp_path/"sweep-39.npz", {"leak_pa": 2.3, "bridge_balance_mohm": 9., "num_spikes": 1})
    assert record["step_ms"] == [1020., 2020.]


@pytest.mark.skipif(not (audit.CACHE/"527952752_ephys.nwb").exists(), reason="donor cache not fetched")
def test_cached_acquisition_matches_the_versioned_report():
    report = audit.audit()
    versioned = json.loads((audit.ROOT/"docs/evidence/h01-l4-acquisition.json").read_text())
    assert report["files"] == versioned["files"] and report["bundle_files"] == versioned["bundle_files"]
    assert report["verification_1_mechanism_set"]["identical_to_l2_import"] is True
    assert report["verification_1_mechanism_set"]["genome_mechanisms"] == sorted(audit.MECHANISMS)
    assert [e["sweep"] for e in report["exports"]] == [69, 39, 66]
    assert report["exports"] == versioned["exports"]
    assert all(e["crossing_count_minus20mv"] == e["allen_num_spikes"] for e in report["exports"])
    assert [r["allen_num_spikes"] for r in report["verification_2_fit_sweeps"]["rows"]] == [20, 17, 19, 20]
    assert report["neuron_reference"] == versioned["neuron_reference"]
    reference = audit.CACHE/"neuron-reference"
    assert json.loads((reference/"donor.json").read_text()) == versioned["neuron_reference"]
    assert audit.sha256(reference/"527952884_fit.json") == versioned["neuron_reference"]["fit_sha256"]
    assert sorted(p.stem for p in reference.glob("*.mod")) == sorted(audit.MECHANISMS)
