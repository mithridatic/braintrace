"""Close the timestep ladder on cell 7196644737 from the recovered traces.

The population ledger scored the `timestep` term 0.000 with the cause "three finer rungs were run
but their traces were deleted with the raw-trace sweep, so the ladder cannot be closed without
re-running them". That cause was wrong. All five rungs survive in the 2026-09-09 worktree-recovery
stash, which was restored to `.cache/h01/readiness` on 2026-09-14. This module reads them and
closes the ladder as arithmetic: no simulation is run.

The ladder is five runs of the same isolated cell under the implicit calcium solver, 10 ms, 1 nA,
at dt 0.005, 0.0025, 0.00125, 0.000625 and 0.0003125 ms. The campaign's contract is 1 mV: a step
is qualified when halving it again moves the trace by less than that.

Each rung samples on its own grid. Adjacent rungs are compared only at the coarse rung's own
sample times, which every finer grid contains exactly (each is a bisection of the one above), so
no interpolation enters the comparison and no error is invented by resampling. That matters here:
the trace reaches +384 mV on a 0.165 um^2 output compartment, so an interpolated comparison across
a spike edge would manufacture tens of millivolts of difference that the solver never produced.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent.parent / ".cache" / "h01" / "readiness"

CELL = "7196644737"
RUNGS = [("005", 0.005), ("0025", 0.0025), ("00125", 0.00125),
         ("000625", 0.000625), ("0003125", 0.0003125)]
GATE_MV = 1.0
DECISION = "h01-ready-cell7196644737-implicit-decision.json"


def trace(tag, root=None):
    """One rung: its time grid and the recorded output voltage."""
    path = (Path(root) if root else CACHE) / f"cell{CELL}-implicit{tag}.npz"
    d = np.load(path, allow_pickle=True)
    t = np.asarray(d["time_ms"], float).ravel()
    v = np.asarray(d["output_voltage"], float).ravel()
    return t, v


def compare(coarse, fine):
    """Max |dV| between two rungs at the coarse rung's own sample times.

    Returns None when the fine grid does not contain the coarse grid, which would mean the two
    rungs are not a bisection pair and the comparison would need interpolation.
    """
    (tc, vc), (tf, vf) = coarse, fine
    if len(tf) != 2 * len(tc):
        return None
    # each grid starts at its own dt, so coarse sample j is fine sample 2j+1 exactly;
    # index arithmetic avoids the float equality that searchsorted would need
    idx = 2 * np.arange(len(tc)) + 1
    if np.max(np.abs(tf[idx] - tc)) > 1e-9 * max(1., float(tc[-1])):
        return None
    diff = np.abs(vf[idx] - vc)
    k = int(np.argmax(diff))
    return {"max_error_mv": float(diff[k]), "at_ms": float(tc[k]),
            "coarse_mv": float(vc[k]), "fine_mv": float(vf[idx][k]),
            "n_compared": int(len(tc))}


def ladder(root=None):
    """Every adjacent pair, coarse to fine, with the gate verdict on each."""
    traces = {tag: trace(tag, root) for tag, _ in RUNGS}
    out = []
    for (tag_c, dt_c), (tag_f, dt_f) in zip(RUNGS, RUNGS[1:]):
        c = compare(traces[tag_c], traces[tag_f])
        row = {"dt_coarse_ms": dt_c, "dt_fine_ms": dt_f}
        if c is None:
            row.update({"status": "not a bisection pair; not compared", "passes_gate": False})
        else:
            row.update(c)
            row["passes_gate"] = c["max_error_mv"] < GATE_MV
        out.append(row)
    return out


def qualified_step(rows):
    """The coarsest dt whose refinement moves the trace by less than the gate.

    A step is qualified only if it and every finer pair below it also pass: a single passing pair
    in the middle of a diverging ladder is not convergence.
    """
    for i, r in enumerate(rows):
        if r.get("passes_gate") and all(x.get("passes_gate") for x in rows[i:]):
            return r["dt_coarse_ms"]
    return None


def report(root=None):
    rows = ladder(root)
    dt = qualified_step(rows)
    finest = rows[-1] if rows else None
    return {
        "cell": CELL,
        "scope": ("Timestep qualification of the isolated cell against the campaign's 1 mV "
                  "contract, read from the five recovered implicit-solver traces. Numerical "
                  "convergence only; no physiology claim."),
        "gate_mv": GATE_MV,
        "solver": "h01_staggered_calcium_implicit",
        "duration_ms": 10.0,
        "pairs": rows,
        "qualified_dt_ms": dt,
        "verdict": ("QUALIFIED at dt %g ms" % dt) if dt else
                   "NOT QUALIFIED at any step on this ladder",
        "finest_pair_error_mv": finest["max_error_mv"] if finest and "max_error_mv" in finest else None,
        "supersedes": (f"{DECISION}, which compared only the two coarsest rungs and reported the "
                       f"1 mV gate as FAIL. The three finer rungs had already run; their traces "
                       f"were recovered on 2026-09-14."),
    }


def main():
    r = report()
    out = HERE / "h01-timestep-ladder.json"
    out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    for p in r["pairs"]:
        e = p.get("max_error_mv")
        print(f"dt {p['dt_coarse_ms']:<10g} -> {p['dt_fine_ms']:<10g} "
              f"max |dV| = {e:9.3f} mV at t = {p.get('at_ms', float('nan')):7.4f} ms   "
              f"{'PASS' if p['passes_gate'] else 'FAIL'}")
    print(r["verdict"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
