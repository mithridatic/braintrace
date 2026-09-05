"""Compare the first 2 ms on donor geometry to independent pinned NEURON traces.

This is an initialized-current and short-response transfer gate. It does not
qualify spike transfer, human physiology, or an H01 connected circuit.
"""
import hashlib
import json
from pathlib import Path
import brainstate
import brainunit as u
import numpy as np
from braintrace.datasets.h01_l2_cell import make_l2_cell
from braintrace.datasets.h01_pv_cell import make_pv_cell


def main():
    """Write direct error traces for source and candidate donor models.

    Returns
    -------
    None
        Write a report and samples beside this driver.
    """
    root = Path(__file__).parent
    rows, arrays = [], {}
    choices = (("E", "candidate", "h01-l2-kv3-ninety-ca133"),
               ("E", "source", "h01-l2-source-cvode-endpoint"),
               ("I", "candidate", "h01-pv-regional-mesh-axon2187"),
               ("I", "source", "h01-pv-neuron-019-cvode10"))
    with brainstate.environ.context(precision=64):
        for role, mode, name in choices:
            geometry = json.loads((root/("h01-l2-geometry-reference.json" if role == "E" else "h01-pv-geometry-reference.json")).read_text())
            metadata = json.loads((root/(name+".json")).read_text())
            reference = np.load(root/(name+".npz"))
            if role == "E":
                cell = make_l2_cell(geometry, mode=mode, delay_ms=0., duration_ms=3.,
                                   current_na=metadata["added_bias_na"])
            else:
                cell = make_pv_cell(geometry, mode=mode)
            result = cell.run(dt=.001*u.ms, duration=2.*u.ms)
            v = np.asarray(result.traces["voltage"].to_decimal(u.mV)).ravel()
            t = .001*np.arange(1, len(v)+1)
            expected = np.interp(t, reference["time_ms"], reference["voltage_mv"])
            error = v-expected
            limit = .1
            row = dict(role=role, mode=mode, reference=name,
                       reference_sha256=hashlib.sha256((root/(name+".npz")).read_bytes()).hexdigest(),
                       max_absolute_soma_error_mv=float(np.max(np.abs(error))), limit_mv=limit,
                       passed=bool(np.isfinite(error).all() and np.max(np.abs(error)) <= limit))
            rows.append(row)
            arrays.update({role+"_"+mode+"_time_ms": t, role+"_"+mode+"_voltage_mv": v,
                           role+"_"+mode+"_error_mv": error})
            print(row, flush=True)
    report = dict(cases=rows, all_passed=all(r["passed"] for r in rows), duration_ms=2., dt_ms=.001,
                  qualification="Initial response on donor geometry only; spike and human-response transfer unqualified.")
    (root/"h01-ei-transfer-check.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(root/"h01-ei-transfer-check.npz", **arrays)
    if not report["all_passed"]:
        raise RuntimeError("Donor response transfer gate failed.")


if __name__ == "__main__":
    main()
