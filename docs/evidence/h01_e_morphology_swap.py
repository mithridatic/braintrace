"""Source-preserving H01 export and retained SP15 diagnostic scorers.

The historical stage-1 converter confused H01 dendrite/astrocyte annotations
with standard SWC soma/axon codes. Preparation now preserves every source node
and edge with neutral SWC types and original annotations in a sidecar. It does
not generate a donor.json for the L2 runner, whose soma and axon substitutions
are incompatible with this anatomy contract. Historical records stay intact.
"""

import argparse
import hashlib
import json
import zipfile
from collections import deque
from pathlib import Path

import numpy as np

from h01_anatomy_area_audit import cable_geometry, read_swc

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[1]/".cache"   # restored 2026-09-14 from the 2026-09-09 worktree stash
ARCHIVE = ROOT/"h01/proofread104.zip"
L2_CACHE = ROOT/"human-pyramidal-l2"
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
    """Export exact source geometry with neutral types and reversible annotations.

    Parameters
    ----------
    text : str
        H01 SWC component in source position/radius units.
    soma_radius_um : float or None, optional
        Retained only to reject the old soma-substitution API explicitly.

    Returns
    -------
    tuple of str and dict
        Micrometre SWC and source annotation/identifier maps. No soma collapse,
        radius floor, inferred apical branch or axon replacement is performed.
    """
    if soma_radius_um is not None:
        raise ValueError('Donor soma substitution is incompatible with source-preserving export.')
    rows = read_swc(text)
    geometry = cable_geometry(rows, position_um=POSITION_UM, radius_um=RADIUS_UM)
    ids, types, parents = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 6].astype(int)
    order, _, _ = bfs_order(ids, parents)
    new_ids = {int(ids[k]): j + 1 for j, k in enumerate(order)}
    pos, rad = rows[:, 2:5] * POSITION_UM, rows[:, 5] * RADIUS_UM
    lines = []
    for k in order:
        parent = -1 if parents[k] == -1 else new_ids[int(parents[k])]
        lines.append(f'{new_ids[int(ids[k])]} 0 {pos[k,0]:.12g} {pos[k,1]:.12g} '
                     f'{pos[k,2]:.12g} {rad[k]:.12g} {parent}')
    header = ('# H01 source-preserved geometry in micrometres; SWC type 0 is neutral.\n'
              '# Original H01 annotation codes are retained in the anatomy sidecar.\n')
    return header + '\n'.join(lines) + '\n', {
        'schema': 'h01-source-preserved-v1', 'nodes_in': len(rows), 'nodes_out': len(rows),
        'soma_nodes_collapsed': 0, 'radius_floor_applied': False,
        'source_soma_samples': int((types == 3).sum()),
        'source_annotation_by_node_id': {str(int(i)): int(t) for i, t in zip(ids, types)},
        'source_node_id_by_export_id': {str(new_ids[int(i)]): int(i) for i in ids},
        'source_geometry': geometry,
        'runtime': 'neutral-label BrainCell import with source annotations; not the L2 donor runner',
    }


def prepare(cell_id, component, out_dir, archive=ARCHIVE):
    """Write source-preserving anatomy to a new directory without a donor model.

    Parameters
    ----------
    cell_id : str
        Released H01 cell ID.
    component : int
        Exact archived component ID.
    out_dir : pathlib.Path
        New artifact directory; existing directories are never overwritten.
    archive : pathlib.Path, optional
        Original H01 SWC archive.

    Returns
    -------
    dict
        Hashes, annotation maps and geometry, with physiology unqualified.
    """
    member = f"{cell_id}.{component}.swc"
    with zipfile.ZipFile(archive) as z:
        raw = z.read(member)
    text, summary = convert(raw.decode())
    out_dir.mkdir(parents=True, exist_ok=False)
    swc = out_dir/f"h01-{cell_id}.swc"
    swc.write_text(text, encoding="utf-8")
    donor = {"cell_id": str(cell_id), "physiology": "unqualified",
             "morphology": swc.name, "morphology_source": {"archive": archive.name, "member": member,
                                                             "member_sha256": hashlib.sha256(raw).hexdigest()},
             "morphology_sha256": hashlib.sha256(swc.read_bytes()).hexdigest(),
             "conversion": summary,
             "note": "Corrected source anatomy; no donor soma, axon stub or apical inference."}
    (out_dir/"anatomy.json").write_text(json.dumps(donor, indent=2)+"\n", encoding="utf-8")
    return donor


def upstroke(npz):
    """Spike-1 landmarks on a uniform 0.02 ms grid.

    The solver writes an adaptive grid that is dense through the upstroke, which inflates the
    sampled maximum rise. Every rise quoted in this campaign is measured after resampling, so
    the control, the doses and the retained fine-mesh reference are read the same way.
    """
    from h01_spike_cycle_energetics import landmarks
    from h01_topographic_stage0 import resample
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
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
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
        result = prepare(args.cell_id, args.component, args.out, archive=args.archive)
        print(json.dumps({"cell_id": result['cell_id'], "physiology": result['physiology'],
                          "output": str(args.out)}, indent=2))
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
