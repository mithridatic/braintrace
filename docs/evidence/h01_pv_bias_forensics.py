"""Isolation split on the I input datum: held state or membrane current?

The recorded ``bias_current`` of the PV calibration sweeps is either a current the
model must carry in addition to the command, or a holding current whose effect the
published fit already absorbed by resting at the held potential. The two readings
predict different resting voltages for the model, so the pre-pulse baseline decides
between them (Hartshorne 2020, Isolation split: inputs versus function).
"""

import json
from pathlib import Path

import h5py
import numpy as np

JUNCTION_MV = -14.
BASELINE_SAMPLES = (1000, 50000)
REPEAT_SWEEPS = (40, 41, 42)
PASSIVE_SWEEPS = (20, 21, 22, 23)
CALIBRATION_SWEEPS = (35, 39)
DLF = {2: 2.95, 3: 1.81, 4: 1.47}
MODEL_TRACES = {"bias_0": "h01-pv-neuron-bias-control.npz",
                "bias_measured": "h01-pv-neuron-bias-measured.npz"}


def sweep_record(nwb, sweep):
    """Bias, command, pre-pulse baseline and late-pulse voltage of one sweep."""
    group = nwb[f"acquisition/timeseries/Sweep_{sweep}"]
    stimulus = nwb[f"stimulus/presentation/Sweep_{sweep}/data"][()].astype(np.float64)*1e12
    voltage = group["data"][()].astype(np.float64)*1000.+JUNCTION_MV
    levels = np.unique(np.round(stimulus, 1))
    command = float(levels[np.argmax(np.abs(levels))])
    pulse = np.flatnonzero(np.abs(stimulus-command) < .5)
    start, stop = BASELINE_SAMPLES
    record = {"sweep": sweep, "name": group["aibs_stimulus_name"][()].decode(),
              "bias_pa": float(group["bias_current"][()])*1e12, "command_pa": command,
              "baseline_mv": float(voltage[start:stop].mean()),
              "baseline_sd_mv": float(voltage[start:stop].std())}
    if record["name"] == "Long Square" and len(pulse) >= 5000:
        record["late_pulse_mv"] = float(voltage[pulse[-1]-5000:pulse[-1]].mean())
    return record


def held_sweeps(nwb):
    """Every sweep that records a holding current and a long or repeated square."""
    names = ("Long Square", "Square - 0.5ms Subthreshold")
    records = []
    for name in sorted(nwb["acquisition/timeseries"], key=lambda s: int(s.split("_")[1])):
        group = nwb["acquisition/timeseries"][name]
        if "bias_current" in group and group["aibs_stimulus_name"][()].decode() in names:
            records.append(sweep_record(nwb, int(name.split("_")[1])))
    return records


def input_resistance_mohm(records):
    """Median steady deflection per command over the hyperpolarising passive sweeps."""
    values = [(r["late_pulse_mv"]-r["baseline_mv"])/r["command_pa"]*1000.
              for r in records if r["sweep"] in PASSIVE_SWEEPS and r["command_pa"] < 0]
    return float(np.median(values))


def baseline_slope_mohm(records):
    """Least-squares slope of the held baseline against the holding current."""
    bias = np.array([r["bias_pa"] for r in records])
    baseline = np.array([r["baseline_mv"] for r in records])
    slope = np.polyfit(bias, baseline, 1)[0]
    return float(slope*1000.)


def repeat_limit_mv(records, sweeps=REPEAT_SWEEPS):
    """Dixon decision limit of the baseline from repeated sweeps at one holding current."""
    values = [r["baseline_mv"] for r in records if r["sweep"] in sweeps]
    return (max(values)-min(values))*DLF[len(values)]


def model_rest_mv(path, window=(100., 265.)):
    """Mean model voltage over the pre-pulse window of a saved trace."""
    data = np.load(path)
    mask = (data["time_ms"] >= window[0]) & (data["time_ms"] <= window[1])
    return float(data["voltage_mv"][mask].mean())


def verdict(human_mv, limit_mv, rests):
    """Name the model input whose resting state lies within the human baseline limit."""
    inside = {k: abs(v-human_mv) <= limit_mv for k, v in rests.items()}
    if inside["bias_0"] and not inside["bias_measured"]:
        return "held state: the fit absorbed the holding current; command current only"
    if inside["bias_measured"] and not inside["bias_0"]:
        return "membrane current: the recorded bias belongs to the input"
    return "undecided: both or neither input rests inside the human baseline limit"


def assemble(records, rests):
    """Build the forensic report from sweep records and model rest voltages."""
    calibration = [r["baseline_mv"] for r in records if r["sweep"] in CALIBRATION_SWEEPS]
    human = float(np.mean(calibration))
    limit = repeat_limit_mv(records)
    resistance = input_resistance_mohm(records)
    bias = next(r["bias_pa"] for r in records if r["sweep"] == CALIBRATION_SWEEPS[0])
    return {"human_held_baseline_mv": human, "calibration_baselines_mv": calibration,
            "repeat_sweeps": list(REPEAT_SWEEPS), "baseline_decision_limit_mv": limit,
            "input_resistance_mohm": resistance,
            "baseline_vs_bias_slope_mohm": baseline_slope_mohm(records),
            "predicted_shift_if_current_mv": resistance*bias/1000.,
            "model_rest_mv": rests,
            "model_rest_error_mv": {k: v-human for k, v in rests.items()},
            "model_input_resistance_mohm": (rests["bias_measured"]-rests["bias_0"])/bias*1000.,
            "verdict": verdict(human, limit, rests), "sweeps": records}


def render(report):
    """Markdown summary of the forensic split."""
    lines = ["# I input datum: held state or membrane current", "",
             "| Quantity | Value |", "| --- | --- |"]
    for key in ("human_held_baseline_mv", "baseline_decision_limit_mv", "input_resistance_mohm",
                "baseline_vs_bias_slope_mohm", "predicted_shift_if_current_mv",
                "model_input_resistance_mohm"):
        lines.append(f"| {key} | {report[key]:.3f} |")
    for key, value in report["model_rest_error_mv"].items():
        lines.append(f"| model rest error, {key} | {value:+.3f} mV |")
    lines += ["", f"**Verdict:** {report['verdict']}", ""]
    return "\n".join(lines)


def main():
    """Write the forensic report beside the other PV evidence."""
    folder = Path(__file__).parent
    with h5py.File(folder.parents[1]/".cache/human-pv/528687507_ephys.nwb", "r") as nwb:
        records = held_sweeps(nwb)
    rests = {k: model_rest_mv(folder/v) for k, v in MODEL_TRACES.items()}
    report = assemble(records, rests)
    (folder/"h01-pv-bias-forensics.json").write_text(json.dumps(report, indent=2))
    (folder/"h01-pv-bias-forensics.md").write_text(render(report))
    print(render(report))


if __name__ == "__main__":
    main()
