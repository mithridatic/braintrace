"""Acquisition provenance of the Allen L4 pyramidal donor (specimen 527952884, model 626170709).

Pins every fetched file by URL, size and sha256, records the verifications the donor survey
demanded (the genome's mechanism set equals the imported L2 donor's eleven mechanisms and the
mod files are byte-identical; the fit sweeps 69-72 protocol; per-sweep bias and bridge balance
from ``ephys_sweeps.json`` against the NWB), exports the registered sweeps in the L2 driver's
waveform format, and lays out the ``neuron-reference/`` directory the driver reads.
Specification: docs/specs/2026-09-07-h01-donor-allen-l4-import.md.
"""

import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/".cache/human-pyramidal-l4"
L2_CACHE = ROOT.parent/"h01-braincell/.cache/human-pyramidal-l2/source-model"
DONOR_KEY = "l4-pyramidal-allen-527952884"
SPECIMEN_ID = 527952884
SPECIMEN_NAME = "H16.06.008.01.31.06"
MODEL_ID = 626170709
API = "http://api.brain-map.org/api/v2/"
FILES = {
    "626170709.zip": f"http://api.brain-map.org/neuronal_model/download/{MODEL_ID}",
    "527952752_ephys.nwb": API+"well_known_file_download/618205555",
    "527952884_fit.json": "https://api.brain-map.org/api/v2/well_known_file_download/626185209",
    "allen-cell-detail.json": API+"data/query.json?criteria=model::ApiCellTypesSpecimenDetail,rma::criteria,[specimen__id$eq527952884]",
    "neuronal-models.json": API+"data/query.json?criteria=model::NeuronalModel,rma::criteria,[specimen_id$eq527952884],rma::include,well_known_files,neuronal_model_template",
}
BUNDLE = ("fit_parameters.json", "reconstruction.swc", "ephys_sweeps.json", "manifest.json", "model_metadata.json")
MECHANISMS = ("CaDynamics", "Ca_HVA", "Ca_LVA", "Ih", "Im", "K_P", "K_T", "Kv3_1", "NaTs", "Nap", "SK")
LICENCE = ("Allen Institute Terms of Use (https://alleninstitute.org/terms-of-use/): research and non-commercial "
           "use, citation required (Allen Cell Types Database, Allen Institute for Brain Science, "
           "celltypes.brain-map.org), no commercial redistribution without written permission")
RATE_HZ = 50000.
JUNCTION_MV = -14.
FIT_SWEEPS = (69, 70, 71, 72)
EXPORTS = (69, 39, 66)
STEP_MS = {"Long Square": (1020., 2020.), "Square - 2s Suprathreshold": (1020., 3020.)}
THRESHOLD_MV = -20.


def sha256(path):
    """Hex digest of one file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def genome_mechanisms(fit):
    """Mechanism names the genome references (the empty string marks ``g_pas`` rows)."""
    return sorted({row["mechanism"] for row in fit["genome"] if row["mechanism"]})


def mechanism_identity(bundle_mods, reference_mods):
    """Per-mechanism hashes of the bundle's mod files against the L2 import's copies."""
    rows = {}
    for name in MECHANISMS:
        bundle, reference = bundle_mods/f"{name}.mod", reference_mods/f"{name}.mod"
        rows[name] = {"bundle_sha256": sha256(bundle),
                      "l2_import_sha256": sha256(reference) if reference.exists() else None}
        rows[name]["identical"] = rows[name]["bundle_sha256"] == rows[name]["l2_import_sha256"]
    return {"files": rows, "identical": all(r["identical"] for r in rows.values())}


def sweep_table(nwb, ephys_rows):
    """One row per acquisition sweep joined with ``ephys_sweeps.json`` bias and bridge values."""
    by_number = {row["sweep_number"]: row for row in ephys_rows}
    rows = []
    for name in nwb["acquisition/timeseries"]:
        group, stimulus = nwb[f"acquisition/timeseries/{name}"], nwb[f"stimulus/presentation/{name}"]
        number = int(name.split("_")[1])
        listed = by_number.get(number, {})
        rows.append({"sweep": number,
                     "stimulus": stimulus["aibs_stimulus_name"][()].decode(),
                     "amplitude_pa": float(stimulus["aibs_stimulus_amplitude_pa"][()]),
                     "sampling_rate_hz": float(group["starting_time"].attrs["rate"]),
                     "samples": int(group["data"].shape[0]),
                     "bias_current_pa": float(group["bias_current"][()])*1e12 if "bias_current" in group else None,
                     "leak_pa": listed.get("leak_pa"), "bridge_balance_mohm": listed.get("bridge_balance_mohm"),
                     "allen_num_spikes": listed.get("num_spikes")})
    return sorted(rows, key=lambda r: r["sweep"])


def step_window_ms(stimulus_pa, amplitude_pa, rate_hz):
    """Start and end of the samples at the sweep's nominal amplitude."""
    at = np.nonzero(np.abs(stimulus_pa-amplitude_pa) < .5)[0]
    return at[0]*1000./rate_hz, (at[-1]+1)*1000./rate_hz


def upward_crossings(voltage_mv, time_ms, start_ms, stop_ms):
    """Times of upward crossings of the -20 mV threshold inside the step."""
    up = np.nonzero((voltage_mv[1:] >= THRESHOLD_MV) & (voltage_mv[:-1] < THRESHOLD_MV))[0]+1
    crossing_ms = time_ms[up]
    return crossing_ms[(crossing_ms >= start_ms) & (crossing_ms <= stop_ms)]


def export_sweep(nwb, sweep, path, listed):
    """Save one sweep in the L2 driver's waveform format; return its provenance record."""
    group, stimulus = nwb[f"acquisition/timeseries/Sweep_{sweep}"], nwb[f"stimulus/presentation/Sweep_{sweep}"]
    rate = float(group["starting_time"].attrs["rate"])
    assert rate == RATE_HZ, f"Sweep {sweep} is sampled at {rate} Hz, not {RATE_HZ}."
    name = stimulus["aibs_stimulus_name"][()].decode()
    amplitude = float(stimulus["aibs_stimulus_amplitude_pa"][()])
    command_na = stimulus["data"][()].astype(np.float64)*1e9
    start_ms, stop_ms = step_window_ms(command_na*1e3, amplitude, rate)
    assert (start_ms, stop_ms) == STEP_MS[name], f"Sweep {sweep} steps over {start_ms}-{stop_ms} ms."
    reported = group["data"][()].astype(np.float64)*1000.
    corrected = reported+JUNCTION_MV
    time = np.arange(len(reported))*1000./rate
    bias_na = float(group["bias_current"][()])*1e9
    assert abs(bias_na*1e3-listed["leak_pa"]) < .01, f"Sweep {sweep}: NWB bias differs from ephys_sweeps leak_pa."
    np.savez(path, time_ms=time, reported_voltage_mv=reported, corrected_voltage_mv=corrected,
             command_current_na=command_na, bias_current_na=np.float64(bias_na), total_current_na=command_na+bias_na)
    crossings = upward_crossings(corrected, time, start_ms, stop_ms)
    return {"sweep": sweep, "export": path.name, "export_sha256": sha256(path), "stimulus": name,
            "amplitude_pa": amplitude, "step_ms": [start_ms, stop_ms], "sample_count": len(time),
            "sampling_rate_hz": rate, "junction_correction_mv": JUNCTION_MV,
            "bias_current_pa": bias_na*1e3, "leak_pa": listed["leak_pa"],
            "bridge_balance_mohm": listed["bridge_balance_mohm"],
            "stimulus_levels_pa": np.unique(np.round(command_na*1e3, 2)).tolist(),
            "pre_step_mean_corrected_mv": float(corrected[(time > start_ms-70.) & (time < start_ms)].mean()),
            "crossing_count_minus20mv": int(len(crossings)),
            "allen_num_spikes": listed["num_spikes"],
            "first_crossing_after_onset_ms": float(crossings[0]-start_ms) if len(crossings) else None}


def lay_out_reference(cache, fit_path):
    """Copy the driver inputs into ``neuron-reference/`` and write ``donor.json``."""
    reference = cache/"neuron-reference"
    reference.mkdir(exist_ok=True)
    for name in MECHANISMS:
        shutil.copyfile(cache/"source-model/modfiles"/f"{name}.mod", reference/f"{name}.mod")
    shutil.copyfile(fit_path, reference/"527952884_fit.json")
    shutil.copyfile(cache/"source-model/reconstruction.swc", reference/"morphology.swc")
    donor = {"donor_key": DONOR_KEY, "model_id": MODEL_ID, "specimen_id": SPECIMEN_ID,
             "fit": "527952884_fit.json", "fit_sha256": sha256(reference/"527952884_fit.json"),
             "morphology": "morphology.swc", "morphology_sha256": sha256(reference/"morphology.swc"),
             "sweeps": list(EXPORTS), "mechanisms": list(MECHANISMS),
             "specification": "docs/specs/2026-09-07-h01-donor-allen-l4-import.md"}
    (reference/"donor.json").write_text(json.dumps(donor, indent=2)+"\n")
    return donor


def audit(cache=CACHE, l2_cache=L2_CACHE):
    """Assemble the provenance report; every assertion is a pinned acquisition fact."""
    files = {name: {"url": url, "sha256": sha256(cache/name), "bytes": (cache/name).stat().st_size}
             for name, url in FILES.items()}
    bundle = {name: {"sha256": sha256(cache/"source-model"/name), "bytes": (cache/"source-model"/name).stat().st_size}
              for name in BUNDLE}
    fit_path = cache/"source-model/fit_parameters.json"
    fit = json.loads(fit_path.read_text())
    assert sha256(fit_path) == files["527952884_fit.json"]["sha256"], "Bundle fit differs from well-known file 626185209."
    assert fit["fitting"][0]["sweeps"] == list(FIT_SWEEPS)
    mechanisms = genome_mechanisms(fit)
    assert mechanisms == sorted(MECHANISMS), f"Genome mechanisms {mechanisms} differ from the L2 import."
    identity = mechanism_identity(cache/"source-model/modfiles", l2_cache)
    unused = sorted(p.stem for p in (cache/"source-model/modfiles").glob("*.mod") if p.stem not in MECHANISMS)
    detail = json.loads((cache/"allen-cell-detail.json").read_text())["msg"][0]
    assert (detail["specimen__id"], detail["structure__layer"], detail["tag__dendrite_type"]) == (SPECIMEN_ID, "4", "spiny")
    models = json.loads((cache/"neuronal-models.json").read_text())["msg"]
    assert [m["id"] for m in models] == [MODEL_ID] and models[0]["neuronal_model_template_id"] == 329230710
    ephys_rows = json.loads((cache/"source-model/ephys_sweeps.json").read_text())
    by_number = {row["sweep_number"]: row for row in ephys_rows}
    swc_header = (cache/"source-model/reconstruction.swc").read_text().splitlines()[1]
    assert SPECIMEN_NAME in swc_header, "The morphology header does not name the fitted specimen."
    reference_dir = cache/"neuron-reference"
    reference_dir.mkdir(exist_ok=True)
    with h5py.File(cache/"527952752_ephys.nwb", "r") as nwb:
        generated = [v.decode() for v in nwb["general/generated_by"][()]]
        assert generated == ["pipeline", "IVSCC", "version", "1.0"]
        assert int(nwb["general/aibs_specimen_id"][()]) == SPECIMEN_ID
        assert nwb["general/aibs_specimen_name"][()].decode() == SPECIMEN_NAME
        table = sweep_table(nwb, ephys_rows)
        exports = [export_sweep(nwb, sweep, reference_dir/f"sweep-{sweep}.npz", by_number[sweep]) for sweep in EXPORTS]
    fit_rows = [r for r in table if r["sweep"] in FIT_SWEEPS]
    assert all(r["stimulus"] == "Square - 2s Suprathreshold" and abs(r["amplitude_pa"]-100.) < .01 for r in fit_rows)
    donor = lay_out_reference(cache, fit_path)
    return {
        "donor": f"Allen human specimen {SPECIMEN_ID} ({SPECIMEN_NAME}), MTG layer 4, spiny, apical truncated; "
                 f"perisomatic model {MODEL_ID} (template 329230710); donor key {DONOR_KEY}",
        "licence": LICENCE,
        "files": files, "bundle_files": bundle,
        "verification_1_mechanism_set": {"genome_mechanisms": mechanisms, "identical_to_l2_import": identity["identical"],
                                         "mod_files": identity["files"], "unused_bundle_mod_files": unused},
        "verification_2_fit_sweeps": {"sweeps": list(FIT_SWEEPS), "rows": fit_rows,
                                      "protocol": "Square - 2s Suprathreshold, 100 pA, 1020-3020 ms, 50 kHz"},
        "verification_3_nwb": {"http": "200, Content-Disposition 527952752_ephys.nwb", "pipeline": generated,
                               "specimen_id": SPECIMEN_ID, "specimen_name": SPECIMEN_NAME, "sweeps": table},
        "fit": {"passive": fit["passive"][0], "conditions": fit["conditions"][0], "fitting": fit["fitting"][0],
                "swc_header": swc_header, "explained_variance_ratio":
                json.loads((cache/"source-model/model_metadata.json").read_text())["neuronal_model_runs"][0]["explained_variance_ratio"]},
        "neuron_reference": donor,
        "exports": exports,
        "interpretation": ("Allen reports voltages uncorrected for the -14 mV liquid junction potential; the export "
                           "applies it once in corrected_voltage_mv. bias_current is the holding current, separate "
                           "from the command levels; the driver plays the command only unless --include-recorded-bias."),
        "qualification": "Acquisition provenance only; model behaviour is not validated.",
    }


def main():
    report = audit()
    output = ROOT/"docs/evidence/h01-l4-acquisition.json"
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"files": {k: v["sha256"] for k, v in report["files"].items()},
                      "mechanism_set_identical": report["verification_1_mechanism_set"]["identical_to_l2_import"],
                      "exports": [(e["sweep"], e["amplitude_pa"], e["crossing_count_minus20mv"], e["allen_num_spikes"])
                                  for e in report["exports"]]}, indent=2))


if __name__ == "__main__":
    main()
