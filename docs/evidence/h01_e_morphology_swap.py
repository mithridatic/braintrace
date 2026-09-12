"""SP15 stage 1 (Q2 for the E upstroke): hold the function, change the input.

The B3 genome and every B3 flag are kept; the only change is the morphology: the Allen
541563728 reconstruction is replaced by an H01 proofread skeleton. Two runs at the same
mesh (nseg factor 1): the donor anatomy (mesh control against the nseg-9 B3 value) and
the H01 anatomy. The scorer reads the maximum rise of spike 1 from both.

``prepare`` converts one H01 component to an SWC the L2 driver accepts: positions from
32 x 32 x 33 nm voxels to micrometres, radii from nanometres; the soma-labelled nodes
are collapsed to one soma point (centroid, largest soma-node radius); the dendritic
subtree with the greatest cable length is labelled apical (the fit carries apical rows)
and the rest basal; axon-labelled nodes keep their label (the driver replaces the axon
with its 60 um stub). A donor.json beside it points the driver at the B3 fit and the
human sweep-56 waveform.
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import deque
from pathlib import Path

import numpy as np

from h01_spike_cycle_energetics import landmarks
from h01_topographic_stage0 import resample

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[1]/".cache/worktree-recovery-2026-09-09"
ARCHIVE = ROOT/"h01/.cache/h01/proofread104.zip"
L2_CACHE = ROOT/"h01/.cache/human-pyramidal-l2"
POSITION_UM = np.array([.032, .032, .033])
RADIUS_UM = .001
PULSE_MS = (1020., 2020.)
B3_NSEG9_RISE_V_S = 597.7
B3_SOMA_RADIUS_UM = 13.72802734375/2  # the donor soma (L = diam = 13.73 um) is held; only the cable is swapped
MESH_TOLERANCE = .10


def parse_swc(text):
    rows = [[float(x) for x in line.split()] for line in text.splitlines() if line.strip() and not line.startswith("#")]
    arr = np.asarray(rows, float)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError("SWC rows must have seven columns.")
    return arr


def bfs_order(ids, parents):
    index = {int(i): k for k, i in enumerate(ids)}
    children = {int(i): [] for i in ids}
    roots = []
    for k, p in enumerate(parents):
        (roots if p == -1 else children[int(p)]).append(k)
    if len(roots) != 1:
        raise ValueError("The component must have exactly one root.")
    order, pending = [], deque(roots)
    while pending:
        k = pending.popleft()
        order.append(k)
        pending.extend(children[int(ids[k])])
    if len(order) != len(ids):
        raise ValueError("The component is not a single tree.")
    return order, children, index


def convert(text, soma_radius_um=None):
    """H01 component text to a NEURON-ready SWC (micrometres, one soma point, apic/dend/axon labels).

    ``soma_radius_um`` overrides the soma sphere radius (default: the largest soma-node radius,
    which for H01 skeletons is a trunk radius of about 1 um, not a soma).
    """
    rows = parse_swc(text)
    ids, types, parents = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 6].astype(int)
    pos, rad = rows[:, 2:5]*POSITION_UM, rows[:, 5]*RADIUS_UM
    order, children, index = bfs_order(ids, parents)
    soma = np.flatnonzero(types == 1)
    if soma.size == 0:
        raise ValueError("The component has no soma-labelled node.")
    centre = pos[soma].mean(axis=0)
    soma_r = float(rad[soma].max()) if soma_radius_um is None else float(soma_radius_um)
    soma_set = set(int(ids[k]) for k in soma)
    # nodes whose parent chain reaches a soma node without leaving the soma set attach to the new soma
    new_parent = {}
    for k in order:
        p = int(parents[k])
        if int(ids[k]) in soma_set:
            continue
        new_parent[k] = 0 if (p == -1 or p in soma_set) else index[p]
    # label by subtree: the longest-cable subtree from the soma is apical
    subtree_root = {}
    length = {}
    for k in order:
        if int(ids[k]) in soma_set or k not in new_parent:
            continue
        r = k if new_parent[k] == 0 else subtree_root[new_parent[k]]
        subtree_root[k] = r
        seg = 0. if new_parent[k] == 0 else float(np.linalg.norm(pos[k]-pos[new_parent[k]]))
        length[r] = length.get(r, 0.)+seg
    apical_root = max(length, key=length.get) if length else None
    out_index = {}
    lines = [f"1 1 {centre[0]:.6f} {centre[1]:.6f} {centre[2]:.6f} {soma_r:.6f} -1"]
    for k in order:
        if int(ids[k]) in soma_set:
            continue
        out_index[k] = len(lines)+1
        label = 2 if types[k] == 2 else (4 if subtree_root[k] == apical_root else 3)
        parent = 1 if new_parent[k] == 0 else out_index[new_parent[k]]
        lines.append(f"{out_index[k]} {label} {pos[k,0]:.6f} {pos[k,1]:.6f} {pos[k,2]:.6f} {max(rad[k], .05):.6f} {parent}")
    header = ("# H01 proofread_104 component converted for the L2 NEURON driver: positions 32x32x33 nm voxels to um,\n"
              "# radii nm to um (floor 0.05 um), soma nodes collapsed to one point, longest subtree labelled apical (4).\n")
    summary = {"nodes_in": int(len(rows)), "nodes_out": len(lines), "soma_nodes_collapsed": int(soma.size),
               "soma_radius_um": soma_r, "largest_soma_node_radius_um": float(rad[soma].max()), "apical_cable_um": float(length.get(apical_root, 0.)),
               "basal_cable_um": float(sum(v for r, v in length.items() if r != apical_root)),
               "axon_nodes": int((types == 2).sum())}
    return header+"\n".join(lines)+"\n", summary


def prepare(cell_id, component, out_dir, archive=ARCHIVE, cache=L2_CACHE, sweeps=(56, 50, 53, 43)):
    """Write the converted SWC, copy the B3 fit and the human sweep exports, and write donor.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    member = f"{cell_id}.{component}.swc"
    with zipfile.ZipFile(archive) as z:
        raw = z.read(member)
    text, summary = convert(raw.decode(), soma_radius_um=B3_SOMA_RADIUS_UM)
    swc = out_dir/f"h01-{cell_id}.swc"
    swc.write_text(text, encoding="utf-8")
    fit = out_dir/"541563728_fit.json"
    shutil.copyfile(cache/"541563728_fit.json", fit)
    for s in sweeps:
        shutil.copyfile(cache/f"sweep-{s}.npz", out_dir/f"sweep-{s}.npz")
    donor = {"donor_key": f"h01-{cell_id}-b3-genome", "model_id": 626170538, "specimen_id": 541563728,
             "fit": fit.name, "fit_sha256": hashlib.sha256(fit.read_bytes()).hexdigest(),
             "morphology": swc.name, "morphology_source": {"archive": archive.name, "member": member,
                                                             "member_sha256": hashlib.sha256(raw).hexdigest()},
             "sweeps": list(sweeps), "conversion": summary,
             "note": "SP15 stage 1: the B3 genome on an H01 L2 pyramidal skeleton with the B3 soma sphere held (13.73 um); the input swapped is the cable. The waveform files are the human L2 exports."}
    (out_dir/"donor.json").write_text(json.dumps(donor, indent=2)+"\n", encoding="utf-8")
    return donor


def upstroke(npz):
    """Spike-1 landmarks on a uniform 0.02 ms grid.

    The solver writes an adaptive grid that is dense through the upstroke, which inflates the
    sampled maximum rise. Every rise quoted in this campaign is measured after resampling, so
    the control, the doses and the retained fine-mesh reference are read the same way.
    """
    data = np.load(npz)
    t, v, _ = resample(np.asarray(data["time_ms"], float), np.asarray(data["voltage_mv"], float),
                      np.asarray(data["voltage_mv"], float))
    rows = landmarks(t, v, PULSE_MS)
    return {"count": len(rows), "max_rise_v_s": rows[0]["max_rise_v_s"] if rows else None,
            "first_spike_ms": rows[0]["threshold_ms"]-PULSE_MS[0] if rows else None,
            "threshold_mv": rows[0]["threshold_mv"] if rows else None}


def decide(control, test, reference=B3_NSEG9_RISE_V_S):
    """Mesh check on the control; then does the upstroke follow the anatomy (inputs) or not (function)?"""
    mesh_dev = None if control["max_rise_v_s"] is None else abs(control["max_rise_v_s"]-reference)/reference
    mesh_ok = mesh_dev is not None and mesh_dev <= MESH_TOLERANCE
    verdict = "mesh control failed; no reading"
    delta = None
    if mesh_ok and test["max_rise_v_s"] is not None:
        delta = test["max_rise_v_s"]-control["max_rise_v_s"]
        limit = max(3.*abs(control["max_rise_v_s"]-reference), MESH_TOLERANCE*reference)
        verdict = ("inputs: the upstroke follows the anatomy" if abs(delta) > limit
                   else "function: the upstroke is insensitive to the anatomy")
    elif mesh_ok:
        verdict = "H01 anatomy did not spike; no upstroke to compare"
    return {"control": control, "test": test, "reference_nseg9_v_s": reference, "mesh_deviation": mesh_dev,
            "mesh_ok": mesh_ok, "delta_v_s": delta, "verdict": verdict}


def graded_decide(series, control_reference=B3_NSEG9_RISE_V_S, mesh_tolerance=MESH_TOLERANCE, reference_count=None):
    """Read the graded cable series: does the upstroke leave the control band, monotonically, while still spiking?

    ``series`` is an ordered list of ``{"dose", "upstroke"}`` starting with the x1 control. When
    ``reference_count`` is given, the control must also reproduce the reference response (the same
    spike count): a control that matches one summary number but not the response is not a control.
    """
    if not series or series[0]["upstroke"]["max_rise_v_s"] is None:
        return {"verdict": "no control reading", "series": series}
    control = series[0]["upstroke"]["max_rise_v_s"]
    control_count = series[0]["upstroke"].get("count")
    if reference_count is not None and control_count != reference_count:
        return {"verdict": f"control invalid: it gives {control_count} spikes against the reference {reference_count}; "
                           "the series reads a different response, not the dose",
                "control_v_s": control, "control_count": control_count, "reference_count": reference_count,
                "series": series}
    band = max(3.*abs(control-control_reference), mesh_tolerance*control)
    rises = [(s["dose"], s["upstroke"]["max_rise_v_s"]) for s in series if s["upstroke"]["max_rise_v_s"] is not None]
    spiking = [d for d, _ in rises]
    deltas = [{"dose": d, "max_rise_v_s": r, "delta_v_s": r-control, "outside_band": abs(r-control) > band} for d, r in rises]
    values = [r for _, r in rises]
    monotone = all(b <= a+1e-9 for a, b in zip(values, values[1:])) or all(b >= a-1e-9 for a, b in zip(values, values[1:]))
    outside = [x for x in deltas if x["outside_band"]]
    silent = [s["dose"] for s in series if s["upstroke"]["max_rise_v_s"] is None]
    if len(rises) < 2:
        verdict = "no reading: every dose silenced the cell at this input"
    elif outside and monotone:
        verdict = "inputs: the upstroke follows the cable load"
    elif outside:
        verdict = "outside the band but not monotone: the dose is not acting through the cable load"
    elif silent:
        verdict = "function: the upstroke holds its band at every dose that spikes; larger doses only silence the cell"
    else:
        verdict = "function: the upstroke holds its band across the whole series"
    return {"control_v_s": control, "control_count": control_count, "reference_count": reference_count,
            "reference_nseg9_v_s": control_reference, "band_v_s": band,
            "monotone": monotone, "doses_spiking": spiking, "doses_silent": silent,
            "rows": deltas, "series": series, "verdict": verdict}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cell-id", default="955432427")
    p.add_argument("--component", type=int, default=0)
    p.add_argument("--out", type=Path, default=EVIDENCE/"h01-e-morphology")
    s = sub.add_parser("score")
    s.add_argument("--folder", type=Path, required=True)
    s.add_argument("--control", required=True, help="stem of the donor-anatomy run")
    s.add_argument("--test", required=True, help="stem of the H01-anatomy run")
    g = sub.add_parser("grade")
    g.add_argument("--folder", type=Path, required=True, help="folder of the graded runs")
    g.add_argument("--control-npz", type=Path, required=True, help="the x1 control trace")
    g.add_argument("--dose", action="append", required=True, metavar="LABEL=STEM",
                   help="ordered doses after the control, e.g. x1.5=c15-area-sweep56")
    g.add_argument("--reference-v-s", type=float, default=B3_NSEG9_RISE_V_S,
                   help="the fine-mesh maximum rise this series' control is checked against (default: the 200 pA value)")
    g.add_argument("--reference-count", type=int,
                   help="spike count of the fine-mesh reference response; the control must reproduce it")
    g.add_argument("--output-stem", default="stage-2-decision")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.cell_id, args.component, args.out), indent=2))
        return
    if args.command == "grade":
        series = [{"dose": "x1", "upstroke": upstroke(args.control_npz)}]
        for text in args.dose:
            label, stem = text.split("=", 1)
            series.append({"dose": label, "upstroke": upstroke(args.folder/f"{stem}.npz")})
        decision = graded_decide(series, control_reference=args.reference_v_s, reference_count=args.reference_count)
        (args.folder/f"{args.output_stem}.json").write_text(json.dumps(decision, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(decision, indent=2))
        return
    decision = decide(upstroke(args.folder/f"{args.control}.npz"), upstroke(args.folder/f"{args.test}.npz"))
    (args.folder/"stage-1-decision.json").write_text(json.dumps(decision, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
