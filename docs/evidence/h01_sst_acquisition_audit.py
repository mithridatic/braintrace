"""Acquisition provenance of the HL5MN1 (Yao 2022 HL23SST) donor and its Allen recordings.

Pins every fetched file by URL and sha256, records the three verifications the donor
survey demanded (hoc identity with ModelDB 267595, the NWB sweep table, the fit-source
statement in the Yao 2022 methods), and exports the pre-registered long-square sweeps of
Allen specimen 571700636 in the released PV export convention (750 ms crop, -14 mV
junction correction, bias current read separately from the stimulus waveform).
Specification: docs/specs/2026-09-07-h01-donor-hl5mn1-import.md.
"""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/".cache/human-sst-l3"
SPECIMEN_ID = 571700636
SPECIMEN_NAME = "H17.06.006.11.09.05"
NWB_URL = "http://api.brain-map.org/api/v2/well_known_file_download/618228061"
L5_COMMIT = "dd472f19a0d1bfbbba59677cfd82c6e9f8a80590"
L23_COMMIT = "4b970fb5881929d192691e68a2a146c2e97766f6"
L5_RAW = f"https://raw.githubusercontent.com/agmccrei/HumanL5Circuit_AGM2022/{L5_COMMIT}/"
L23_RAW = f"https://raw.githubusercontent.com/KantYao/Human-L2-3-Cortical-Microcircuit/{L23_COMMIT}/"
MECHANISMS = ("CaDynamics", "Ca_HVA", "Ca_LVA", "Ih", "Im", "K_P", "K_T", "Kv3_1", "NaTg", "Nap", "SK")
FILES = {
    "biophys_HL5MN1.hoc": L5_RAW+"L5Circuit/default_circuit/models/biophys_HL5MN1.hoc",
    "NeuronTemplate.hoc": L5_RAW+"L5Circuit/default_circuit/models/NeuronTemplate.hoc",
    "HL5MN1.swc": L5_RAW+"L5Circuit/default_circuit/morphologies/HL5MN1.swc",
    "LICENSE": L5_RAW+"LICENSE",
    "biophys_HL23SST.hoc": L23_RAW+"L23Net/models/biophys_HL23SST.hoc",
    "571700636_fit.json": "https://api.brain-map.org/api/v2/well_known_file_download/626185239",
    "571700399_ephys.nwb": NWB_URL,
    **{f"mod/{m}.mod": L5_RAW+f"L5Circuit/default_circuit/mod/{m}.mod" for m in MECHANISMS},
}
RATE_HZ = 50000.
CROP_START_SAMPLES = 37500
JUNCTION_MV = -14.
EXPORTS = ((44, "active", 0), (35, "active", 1), (23, "passive", 0))
YAO_METHODS = ("bioRxiv 10.1101/2021.02.17.431698 v5 (2021-11-22) Methods, Experimental Data: "
               "'putative SST (Neuron ID: 571700636), PV (Neuron ID: 529807751) and VIP (Neuron ID: 525018757) "
               "interneurons available from the Allen Brain Atlas'; 'five hyperpolarizing and depolarizing "
               "current steps ... Three depolarizing supra-threshold current steps ... low, medium and high firing rates'")


def sha256(path):
    """Hex digest of one file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hoc_identity(hl5mn1_text, hl23sst_text):
    """Compare the two biophysics files line by line; only the procedure name may differ."""
    left, right = hl5mn1_text.splitlines(), hl23sst_text.splitlines()
    differing = [(a, b) for a, b in zip(left, right) if a.strip() != b.strip()]
    if len(left) != len(right):
        differing.append((f"<{len(left)} lines>", f"<{len(right)} lines>"))
    name_only = differing == [("proc biophys_HL5MN1(){", "proc biophys_HL23SST(){")]
    return {"identical_parameters": name_only, "differing_lines": differing}


def sweep_table(nwb):
    """One row per acquisition sweep: name, amplitude, rate, samples, bias."""
    rows = []
    for name in nwb["acquisition/timeseries"]:
        group, stimulus = nwb[f"acquisition/timeseries/{name}"], nwb[f"stimulus/presentation/{name}"]
        rows.append({"sweep": int(name.split("_")[1]),
                     "stimulus": stimulus["aibs_stimulus_name"][()].decode(),
                     "amplitude_pa": float(stimulus["aibs_stimulus_amplitude_pa"][()]),
                     "sampling_rate_hz": float(group["starting_time"].attrs["rate"]),
                     "samples": int(group["data"].shape[0]),
                     "bias_current_pa": float(group["bias_current"][()])*1e12 if "bias_current" in group else None})
    return sorted(rows, key=lambda r: r["sweep"])


def step_window_ms(stimulus_pa, amplitude_pa, rate_hz):
    """Start and end of the samples at the sweep's nominal amplitude."""
    at = np.nonzero(np.abs(stimulus_pa-amplitude_pa) < .5)[0]
    return at[0]*1000./rate_hz, (at[-1]+1)*1000./rate_hz


def export_sweep(nwb, sweep, path):
    """Crop, correct and save one sweep; return its provenance record."""
    group, stimulus = nwb[f"acquisition/timeseries/Sweep_{sweep}"], nwb[f"stimulus/presentation/Sweep_{sweep}"]
    rate = float(group["starting_time"].attrs["rate"])
    assert rate == RATE_HZ, f"Sweep {sweep} is sampled at {rate} Hz, not {RATE_HZ}."
    assert stimulus["aibs_stimulus_name"][()].decode() == "Long Square"
    amplitude = float(stimulus["aibs_stimulus_amplitude_pa"][()])
    waveform = stimulus["data"][()].astype(np.float64)*1e12
    start_ms, stop_ms = step_window_ms(waveform, amplitude, rate)
    assert (start_ms, stop_ms) == (1020., 2020.), f"Sweep {sweep} steps over {start_ms}-{stop_ms} ms."
    raw_mv = group["data"][()].astype(np.float64)*1000.
    voltage = raw_mv[CROP_START_SAMPLES:]+JUNCTION_MV
    time = np.arange(len(voltage))*1000./rate
    np.savez_compressed(path, time=time, voltage=voltage)
    return {"sweep": sweep, "export": path.name, "export_sha256": sha256(path),
            "stimulus": "Long Square", "amplitude_pa": amplitude,
            "step_ms_recorded": [start_ms, stop_ms], "step_ms_exported": [start_ms-750., stop_ms-750.],
            "sample_count": len(voltage), "crop_start_samples": CROP_START_SAMPLES, "crop_start_ms": 750.,
            "junction_correction_mv": JUNCTION_MV,
            "bias_current_pa": float(group["bias_current"][()])*1e12,
            "stimulus_levels_pa": np.unique(np.round(waveform, 3)).tolist(),
            "pre_step_mean_mv": float(voltage[(time > 200.) & (time < 270.)].mean()),
            "sampling_rate_hz": rate}


def audit(cache=CACHE):
    """Assemble the provenance report; every assertion is a pinned acquisition fact."""
    files = {name: {"url": url, "sha256": sha256(cache/name), "bytes": (cache/name).stat().st_size}
             for name, url in FILES.items()}
    identity = hoc_identity((cache/"biophys_HL5MN1.hoc").read_text(), (cache/"biophys_HL23SST.hoc").read_text())
    swc_header = (cache/"HL5MN1.swc").read_text().splitlines()[1]
    assert SPECIMEN_NAME in swc_header, "The morphology header does not name the fitted specimen."
    fit = json.loads((cache/"571700636_fit.json").read_text())
    with h5py.File(cache/"571700399_ephys.nwb", "r") as nwb:
        generated = [v.decode() for v in nwb["general/generated_by"][()]]
        assert generated == ["pipeline", "IVSCC", "version", "1.0"]
        assert int(nwb["general/aibs_specimen_id"][()]) == SPECIMEN_ID
        assert nwb["general/aibs_specimen_name"][()].decode() == SPECIMEN_NAME
        table = sweep_table(nwb)
        exports = [export_sweep(nwb, sweep, cache/f"{kind}-{index}.npz") for sweep, kind, index in EXPORTS]
    return {
        "donor": "HL5MN1 (= Yao 2022 HL23SST); Allen specimen 571700636, MTG layer 3, aspiny; subtype unknown (putative SST)",
        "licence": "GPL-3.0 (agmccrei/HumanL5Circuit_AGM2022 LICENSE)",
        "repository_commits": {"agmccrei/HumanL5Circuit_AGM2022": L5_COMMIT,
                               "KantYao/Human-L2-3-Cortical-Microcircuit": L23_COMMIT},
        "files": files,
        "verification_1_hoc_identity": identity,
        "verification_2_nwb": {"http": "200, Content-Disposition 571700399_ephys.nwb", "pipeline": generated,
                               "specimen_id": SPECIMEN_ID, "specimen_name": SPECIMEN_NAME, "sweeps": table},
        "verification_3_fit_source": {"statement": YAO_METHODS, "swc_header": swc_header,
                                      "allen_perisomatic_fit_sweeps": fit["fitting"][0]["sweeps"],
                                      "allen_perisomatic_junction_mv": fit["fitting"][0]["junction_potential"]},
        "mechanism_set_identical_to_hl5bn1": "byte-identical .mod files and NeuronTemplate.hoc; kinetic parameters in the hoc differ (see spec)",
        "exports": exports,
        "interpretation": "Allen reports voltages uncorrected for the -14 mV liquid junction potential; the export applies it. Bias current is the holding current, separate from the stimulus levels.",
        "qualification": "Acquisition provenance only; model behaviour is not validated.",
    }


def main():
    report = audit()
    output = ROOT/"docs/evidence/h01-sst-acquisition.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"files": {k: v["sha256"] for k, v in report["files"].items()},
                      "identical_parameters": report["verification_1_hoc_identity"]["identical_parameters"],
                      "exports": report["exports"]}, indent=2))


if __name__ == "__main__":
    main()
