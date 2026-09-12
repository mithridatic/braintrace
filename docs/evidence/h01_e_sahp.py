"""SP12 E cell: a spike-triggered slow outward conductance (KsAHP) on the B3 soma.

Two jobs, both pre-registration first (``docs/specs/2026-09-11-h01-e-spike-triggered-outward.md``):

* ``register``: integrate the KsAHP gate along the retained ``m0-b3-currents-sweep56``
  trace with only its first spike kept, and convert a target whole-cell offset (nA) at
  the late-pulse plateau into a soma density ``gbar`` (S/cm2). The dose is fixed from
  the recorded trace before any run.
* ``score``: apply the stage bands to the runs of ``h01-e-sahp/``: count, late- and
  mid-pulse p50 against the human sweep, an unchanged pre-spike trace and first-spike
  time against ``m0-b3-currents`` (the gate is silent before a spike), and the comparison
  arm's own prediction (Nap removed; a lever of the same size that is not spike-triggered).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from h01_e_current_paths import PLATEAU_WINDOWS, cycle_summary, load_run, _window_stats
from docs.evidence.h01_direct_observations import first_crossing_ms

EVIDENCE = Path(__file__).resolve().parent
GATE = {"vhalf_mv": -20., "slope_mv": 2., "tau_on_ms": 1., "tau_off_ms": 1000.}
EK_MV = -107.
SOMA_AREA_UM2 = 592.0593838294642
DENSITY_TO_NA = 1e-2
LEVEL_MV = 1.
PRESPIKE_MV = .1
FIRST_SPIKE_MS = 1.
HUMAN_SWEEP56 = ".cache/human-pyramidal-l2/sweep-56.npz"
REFERENCE = ("h01-e-currents", "m0-b3-currents-sweep56")


def gate_trace(time, voltage, gate=GATE):
    """Exponential-Euler integration of the KsAHP gate ``z`` along a voltage trace."""
    time, voltage = np.asarray(time, float), np.asarray(voltage, float)
    zinf = 1./(1.+np.exp(-(voltage-gate["vhalf_mv"])/gate["slope_mv"]))
    ztau = gate["tau_off_ms"]+(gate["tau_on_ms"]-gate["tau_off_ms"])*zinf
    z = np.empty_like(voltage)
    z[0] = zinf[0]
    for i in range(1, time.size):
        decay = np.exp(-(time[i]-time[i-1])/ztau[i])
        z[i] = zinf[i]+(z[i-1]-zinf[i])*decay
    return z


def single_spike_voltage(time, voltage, hold_mv, vhalf_mv=GATE["vhalf_mv"], margin_ms=1.):
    """The trace with only its first crossing of ``vhalf_mv`` kept; later samples are held at ``hold_mv``."""
    time, voltage = np.asarray(time, float), np.asarray(voltage, float)
    above = voltage > vhalf_mv
    ups = np.flatnonzero(~above[:-1] & above[1:])
    if ups.size == 0:
        raise ValueError("The trace has no spike above the gate half-point.")
    downs = np.flatnonzero(above[:-1] & ~above[1:])
    down = downs[downs > ups[0]][0]
    return np.where(time > time[down]+margin_ms, hold_mv, voltage), float(time[ups[0]+1]), float(time[down+1]-time[ups[0]+1])


def dose_for_offset(z_mean, offset_na, plateau_mv, ek_mv=EK_MV, area_um2=SOMA_AREA_UM2):
    """Soma density (S/cm2) whose mean gated current over a window equals ``offset_na`` (outward)."""
    if z_mean <= 0 or plateau_mv <= ek_mv:
        raise ValueError("The gate must be open and the plateau above the potassium reversal.")
    return offset_na/(z_mean*(plateau_mv-ek_mv)*area_um2*DENSITY_TO_NA)


def register(model_npz, targets_na, hold_mv, gate=GATE, window=PLATEAU_WINDOWS["late_pulse"]):
    """Fix the doses from the retained trace: gate per spike, late-window mean gate, density per target."""
    data = np.load(model_npz)
    time, voltage = np.asarray(data["time_ms"], float), np.asarray(data["voltage_mv"], float)
    single, spike_ms, above_ms = single_spike_voltage(time, voltage, hold_mv, gate["vhalf_mv"])
    z = gate_trace(time, single, gate)
    mask = (time >= window[0]) & (time < window[1])
    z_mean = float(z[mask].mean())
    return {"gate": gate, "reference_trace": str(model_npz), "first_spike_ms": spike_ms,
            "ms_above_vhalf": above_ms, "gate_after_first_spike": float(z.max()),
            "hold_mv": hold_mv, "late_window_ms": list(window), "gate_late_mean": z_mean,
            "ek_mv": EK_MV, "soma_area_um2": SOMA_AREA_UM2,
            "doses": {f"{t:.3f} nA": {"target_offset_na": t, "gbar_s_cm2": dose_for_offset(z_mean, t, hold_mv)}
                      for t in targets_na}}


def human_levels(human_npz):
    human = np.load(human_npz)
    ht, hv = human["time_ms"], human["corrected_voltage_mv"]
    return {name: _window_stats(ht, hv, w) for name, w in PLATEAU_WINDOWS.items()}


def prespike_deviation(data, reference, vhalf_mv=GATE["vhalf_mv"]):
    """Largest voltage difference to the reference before the reference's first spike, and the first-spike shift."""
    t, v = np.asarray(data["time_ms"], float), np.asarray(data["voltage_mv"], float)
    rt, rv = np.asarray(reference["time_ms"], float), np.asarray(reference["voltage_mv"], float)
    ref_spike = first_crossing_ms(rt, rv, 0., level=vhalf_mv)
    spike = first_crossing_ms(t, v, 0., level=vhalf_mv)
    stop = ref_spike-1.
    grid = rt[(rt >= 1000.) & (rt <= stop)]
    deviation = float(np.abs(np.interp(grid, t, v)-np.interp(grid, rt, rv)).max())
    return {"prespike_max_dev_mv": deviation, "first_spike_ms": spike, "reference_first_spike_ms": ref_spike,
            "first_spike_shift_ms": None if spike is None else spike-ref_spike}


def _band(row, predicted, observed, held):
    return {"row": row, "predicted": predicted, "observed": observed, "held": bool(held)}


def dose_bands(summary, model, human, pre):
    """Stage-0 bands of a KsAHP dose at 200 pA."""
    late = _difference(model["late_pulse"]["p50_mv"], human["late_pulse"]["p50_mv"])
    mid = _difference(model["mid_pulse"]["p50_mv"], human["mid_pulse"]["p50_mv"])
    return [_band("count", "1-2", summary["count"], 1 <= summary["count"] <= 2),
            _band("late_p50_minus_human_mv", f"|x| <= {LEVEL_MV}", late, late is not None and abs(late) <= LEVEL_MV),
            _band("mid_p50_minus_human_mv", f"|x| <= {LEVEL_MV}", mid, mid is not None and abs(mid) <= LEVEL_MV),
            _band("prespike_max_dev_mv", f"<= {PRESPIKE_MV}", pre["prespike_max_dev_mv"], pre["prespike_max_dev_mv"] <= PRESPIKE_MV),
            _band("first_spike_shift_ms", f"|x| <= {FIRST_SPIKE_MS}", pre["first_spike_shift_ms"],
                  pre["first_spike_shift_ms"] is not None and abs(pre["first_spike_shift_ms"]) <= FIRST_SPIKE_MS),
            _band("axon_first", True, summary["axon_first"], summary["axon_first"])]


def control_bands(summary, model, reference_model):
    """The comparison arm's registered prediction: Nap removal does not land the human's 200 pA response."""
    shift = _difference(model["late_pulse"]["p50_mv"], reference_model["late_pulse"]["p50_mv"])
    return [_band("count", ">= 3", summary["count"], summary["count"] >= 3),
            _band("late_p50_minus_b3_mv", "-1.5 to 0", shift, shift is not None and -1.5 <= shift <= 0.)]


def _difference(a, b):
    return None if a is None or b is None else a-b


def score_candidate(folder, name, arm, human, reference_data):
    data, report = load_run(folder, f"{name}-sweep56")
    summary = cycle_summary(data)
    model = {n: _window_stats(np.asarray(data["time_ms"], float), np.asarray(data["voltage_mv"], float), w)
             for n, w in PLATEAU_WINDOWS.items()}
    pre = prespike_deviation(data, reference_data)
    if arm == "dose":
        bands = dose_bands(summary, model, human, pre)
    else:
        reference_model = {n: _window_stats(np.asarray(reference_data["time_ms"], float),
                                            np.asarray(reference_data["voltage_mv"], float), w)
                           for n, w in PLATEAU_WINDOWS.items()}
        bands = control_bands(summary, model, reference_model)
    gate = {k: float(np.asarray(data[k]).max()) for k in data if k.startswith("soma_KsAHP")}
    return {"candidate": name, "arm": arm, "summary": summary, "levels": model, "prespike": pre, "gate_max": gate,
            "bands": bands, "bands_held": sum(b["held"] for b in bands), "bands_total": len(bands),
            "wall_seconds": report.get("integration_seconds"), "candidate_sha256": report.get("candidate_json", {}).get("sha256")}


def decide(folder, candidates, human_npz, reference=REFERENCE):
    """Stage-0 decision: PASS when a dose holds every band; the survivor is the dose with count 1 closest to the human late level."""
    human = human_levels(human_npz)
    reference_data, _ = load_run(EVIDENCE/reference[0], reference[1])
    scored = [score_candidate(folder, c["name"], c["arm"], human, reference_data) for c in candidates]
    passing = [s for s in scored if s["arm"] == "dose" and s["bands_held"] == s["bands_total"]]
    survivor = min(passing, key=lambda s: (s["summary"]["count"] != 1, abs(s["bands"][1]["observed"])), default=None)
    control = next((s for s in scored if s["arm"] == "control"), None)
    control_note = None if control is None else (
        "Nap removal holds its registered prediction: it does not land the human's response."
        if control["bands_held"] == control["bands_total"] else
        "Nap removal broke its registered prediction; the spike-triggered reading is not isolated.")
    return {"stage": "0", "human": human, "reference": "/".join(reference), "candidates": scored,
            "decision": "PASS" if passing else "FAIL",
            "survivor": None if survivor is None else survivor["candidate"], "control_note": control_note}


def render(decision):
    lines = [f"# SP12 stage 0: {decision['decision']}", "",
             f"Human 200 pA late p50 {decision['human']['late_pulse']['p50_mv']:.2f} mV, mid p50 {decision['human']['mid_pulse']['p50_mv']:.2f} mV.",
             f"Reference: {decision['reference']}. Survivor: {decision['survivor']}.", ""]
    if decision["control_note"]:
        lines += [decision["control_note"], ""]
    for c in decision["candidates"]:
        lines += [f"## {c['candidate']} ({c['arm']}): {c['bands_held']} of {c['bands_total']} bands", "",
                  "| Row | Predicted | Observed | Held |", "| --- | --- | ---: | --- |"]
        for b in c["bands"]:
            obs = b["observed"] if not isinstance(b["observed"], float) else f"{b['observed']:.3f}"
            lines.append(f"| {b['row']} | {b['predicted']} | {obs} | {b['held']} |")
        lines += ["", f"Count {c['summary']['count']}; peaks {[round(p, 2) for p in c['summary']['peak_ms']]} ms; "
                  f"levels mid {c['levels']['mid_pulse']['p50_mv']:.2f}, late {c['levels']['late_pulse']['p50_mv']:.2f}, "
                  f"post {c['levels']['post_pulse']['p50_mv']} mV (sample p50; None = unavailable); gate max {c['gate_max']}.", ""]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--model-npz", type=Path, default=EVIDENCE/REFERENCE[0]/(REFERENCE[1]+".npz"))
    reg.add_argument("--target-na", type=float, action="append", required=True)
    reg.add_argument("--hold-mv", type=float, required=True, help="post-spike plateau the dose is sized at (the human late p50)")
    reg.add_argument("--output", type=Path)
    sc = sub.add_parser("score")
    sc.add_argument("--folder", type=Path, required=True)
    sc.add_argument("--manifest", type=Path, required=True)
    sc.add_argument("--human-root", type=Path, required=True)
    sc.add_argument("--output-stem", default="stage-0-reanalysis")
    args = parser.parse_args(argv)
    if args.command == "register":
        result = register(args.model_npz, args.target_na, args.hold_mv)
        text = json.dumps(result, indent=2)+"\n"
        (args.output.write_text(text, encoding="utf-8") if args.output else print(text))
        return
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidates = [c for c in manifest["candidates"] if c["stage"] == "0"]
    from docs.evidence.h01_direct_report import write_bundle
    direct = write_bundle([args.folder/f"{c['name']}-sweep56" for c in candidates],
                          args.folder/f"{args.output_stem}-direct", [args.human_root/HUMAN_SWEEP56])
    decision = decide(args.folder, candidates, args.human_root/HUMAN_SWEEP56)
    decision.update(direct_observation=direct, causal_verdict="not_established", visual_review="pending",
                    qc_basis="Original registered sample-percentile bands, with explicit coverage checks; not a causal verdict.")
    decision["manifest"] = str(args.manifest.name)
    (args.folder/f"{args.output_stem}.json").write_text(json.dumps(decision, indent=2)+"\n", encoding="utf-8")
    (args.folder/f"{args.output_stem}.md").write_text(render(decision), encoding="utf-8")
    print(render(decision))


if __name__ == "__main__":
    main()
