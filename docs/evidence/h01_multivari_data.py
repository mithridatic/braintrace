"""Per-cycle spike features for the H01 multivari charts, from every trace still on disk.

Stored form: one row per (cell, machine, input, repeat, cycle) with width, AHP, cycle
length, peak and the maximum rise rate. Human rows come from the Allen recordings in
``.cache`` (every suprathreshold long-square sweep, repeats included); model rows come
from whichever model traces survive under ``docs/evidence``. Traces that a full chart
would need but that are absent are listed under ``missing`` with the runner that
produces them, so the gap is recorded rather than hidden.
"""

import json
import sys
from pathlib import Path

import numpy as np

from h01_isolation_report import human_trace, model_trace, nwb_sweep
from h01_usable_tier import CELLS, ROOT, cycle_table

EVIDENCE = Path(__file__).resolve().parent
MIN_SUPRATHRESHOLD_PA = {"E": 190., "I": 100.}
NWB = {"E": (".cache/human-pyramidal-l2/recording.nwb", ".cache/human-pyramidal-l2/long-square-index.json"),
       "I": (".cache/human-pv/528687507_ephys.nwb", None)}
MODEL_TRACES = {("I", "0.19 nA"): "h01-multivari-runs/i-kv3-close2-019",
                ("I", "0.23 nA"): "h01-multivari-runs/i-kv3-close2-023",
                ("I", "0.27 nA"): "h01-i-energetic/e-kv3-close2-027",
                ("E", "200 pA"): "h01-multivari-runs/e-b3-sweep56",
                ("E", "250 pA"): "h01-multivari-runs/e-b3-sweep50",
                ("E", "310 pA"): "h01-multivari-runs/e-b3-sweep53",
                ("E", "350 pA"): "h01-multivari-runs/e-b3-sweep55"}
MODEL_RUNNERS = {"E": "h01_l2_braincell_reference.py (Docker image; see h01_usable_tier CELLS['E'])",
                 "I": "h01_pv_braincell_reference.py (Docker image; see h01_usable_tier CELLS['I'])"}


def rise_rate(time, voltage, peak_ms, window_ms=2.):
    """Maximum dV/dt (V/s) in the window before one spike peak."""
    mask = (time >= peak_ms-window_ms) & (time <= peak_ms)
    if mask.sum() < 3:
        return None
    return float(np.max(np.gradient(voltage[mask], time[mask])))


def rows_from_trace(cell, machine, label, repeat, time, voltage, pulse):
    time, voltage = np.asarray(time, float), np.asarray(voltage, float)
    return [{"cell": cell, "machine": machine, "input": label, "repeat": repeat, "cycle": r["cycle"],
             "width_ms": r["width_ms"], "ahp_mv": r["ahp_mv"], "cycle_ms": r["cycle_ms"],
             "peak_mv": r["peak_mv"], "rise_v_per_s": rise_rate(time, voltage, r["peak_ms"])}
            for r in cycle_table(time, voltage, pulse, label)]


def e_sweep_amplitudes(root):
    """Sweep number -> long-square amplitude (pA) for the L2 recording."""
    index = json.loads((root/NWB["E"][1]).read_text())
    return {int(r["sweep"].split("_")[1]): float(r["amplitude_pa"]) for r in index}


def human_rows(cell, spec, root=ROOT):
    """Every human long-square sweep at or above the suprathreshold floor, repeats kept apart."""
    rows = []
    if cell == "E":
        pulse = spec["pulse_ms"]
        for sweep, amp in sorted(e_sweep_amplitudes(root).items()):
            if amp < MIN_SUPRATHRESHOLD_PA["E"]:
                continue
            time, voltage = nwb_sweep(root/NWB["E"][0], sweep)
            rows += rows_from_trace(cell, "human", f"{amp:.0f} pA", sweep, time, voltage, pulse)
    else:
        for label, (human_path, _) in spec["inputs"].items():
            rows += rows_from_trace(cell, "human", label, None, *human_trace(root/human_path), spec["pulse_ms"])
        path, sweeps, window = spec["repeats"]
        for sweep in sweeps:
            rows += rows_from_trace(cell, "human", "repeat", sweep, *nwb_sweep(root/path, sweep), window)
    return rows


def model_rows(cell, spec):
    """Rows from surviving model traces; names of the missing ones."""
    rows, missing = [], []
    for (trace_cell, label), model_name in MODEL_TRACES.items():
        if trace_cell != cell:
            continue
        try:
            time, voltage = model_trace(EVIDENCE, model_name)
        except (FileNotFoundError, KeyError, OSError):
            missing.append({"cell": cell, "input": label, "trace": model_name, "runner": MODEL_RUNNERS[cell]})
            continue
        rows += rows_from_trace(cell, "model", label, None, time, voltage, spec["pulse_ms"])
    return rows, missing


def main(out=EVIDENCE/"h01-multivari-data.json"):
    rows, missing = [], []
    for cell in ("E", "I"):
        spec = CELLS[cell]
        rows += human_rows(cell, spec)
        model, gaps = model_rows(cell, spec)
        rows += model
        missing += gaps
    summary = {cell: {m: sum(1 for r in rows if r["cell"] == cell and r["machine"] == m) for m in ("human", "model")}
               for cell in ("E", "I")}
    Path(out).write_text(json.dumps({"rows": rows, "missing": missing, "row_counts": summary}, indent=1)+"\n")
    return summary, missing


if __name__ == "__main__":
    print(json.dumps(main(*[Path(a) for a in sys.argv[1:2]]), indent=1))
