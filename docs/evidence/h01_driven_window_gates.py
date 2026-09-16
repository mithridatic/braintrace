"""Apply the existing runtime, control and refinement gates to the Vast driven-window runs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_population_control_gate import CONTROLS, audit_controls
from docs.evidence.h01_population_refinement_gate import audit_refinement
from docs.evidence.h01_population_runtime_gate import audit_runtime


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(folder, plan):
    """Return (build, plan, arrays) and the hashes of the three inputs for one run folder."""
    folder = Path(folder)
    paths = dict(build=folder/"run-build.json", plan=Path(plan), traces=folder/"run-traces.npz")
    build = json.loads(paths["build"].read_text())
    arrays = dict(np.load(paths["traces"]))
    return (build, json.loads(paths["plan"].read_text()), arrays), {k: sha256(v) for k, v in paths.items()}


def decide(reference_path, runs, refinement):
    """Gate verdicts for four controls and one refinement pair; no physiology promoted."""
    reference = json.loads(Path(reference_path).read_text())
    hashes = {"reference": sha256(reference_path)}
    loaded = {}
    for control, (folder, plan) in runs.items():
        loaded[control], hashes[control] = load_run(folder, plan)
    runtime = {control: audit_runtime(run[0], reference, run[1], run[2]) for control, run in loaded.items()}
    controls = audit_controls(reference, loaded) if set(loaded) == set(CONTROLS) else None
    coarse, hashes["refinement_coarse"] = load_run(*refinement["coarse"])
    fine, hashes["refinement_fine"] = load_run(*refinement["fine"])
    refined = audit_refinement(reference, coarse, fine)
    passed = (all(v["status"] == "passed" for v in runtime.values())
              and controls is not None and controls["status"] == "passed" and refined["status"] == "passed")
    return dict(
        driven_window=1. if passed else 0.,
        verdict="PASS" if passed else "FAIL",
        scope=f"{len(reference.get('simulated_cell_ids', []))}-cell driven window on the Vast executor: runtime, "
              "matched-control and timestep gates only; physiology, functional inhibition and human provenance "
              "are not qualified by this decision",
        runtime={c: dict(status=v["status"], failures=v["failures"]) for c, v in runtime.items()},
        controls=None if controls is None else dict(status=controls["status"], failures=controls["failures"]),
        refinement=dict(status=refined["status"], failures=refined["failures"],
                        worst_voltage_mV=max((max(o["voltage_errors_mV"].values()) for o in refined["observations"]), default=None),
                        worst_event_ms=max((o["max_event_timing_error_ms"] or 0. for o in refined["observations"]), default=None),
                        cells_compared=len(refined["observations"])),
        input_hashes=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--runs", required=True, help="JSON: {control: [folder, plan]} for the four controls")
    parser.add_argument("--refinement", required=True, help="JSON: {coarse: [folder, plan], fine: [folder, plan]}")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decision = decide(args.reference, json.loads(args.runs), json.loads(args.refinement))
    Path(args.output).write_text(json.dumps(decision, indent=2)+"\n")
    print(decision["verdict"], json.dumps({k: decision[k] for k in ("runtime", "controls", "refinement")}))


if __name__ == "__main__":
    main()
