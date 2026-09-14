"""The population accuracy ledger: what "accuracy for the 104 H01 cells" can mean, term by term.

The single-cell campaign produced one number, 71.8 percent, and it is easy to multiply that by
whatever coverage fraction is at hand and call the product a population accuracy. Doing so hides
the terms that are zero. This module assembles the product from committed evidence and refuses to
drop a term it cannot fill, for the same reason the stage-15/16 scorer refuses to drop an element
a short train cannot exhibit: dropping a requirement quietly raises the score of exactly the thing
that fails it.

Five terms, each read from a committed decision JSON:

  donor      the donor model's accuracy against a real human recording, ten elements at 310 pA
             through the measured chain. Measured for ONE donor (B3), which supplies 28 of the
             104 cells. The other three donors had their published fits reproduced and REJECTED
             against their own recordings, and no ten-element score exists for them.
  coverage   how many H01 cells have a donor matched in both layer and class (55 of 104).
  construction  whether the cell can be imported, constructed, initialised and stepped. Split into
             the gates that pass (import, construction, initialisation, a forward pass on a
             synthetic probe) and the gate that does not (a driven window at a physiological
             duration with a qualified timestep).
  timestep   whether the integration step is qualified against the campaign's own 1 mV contract.
  anatomy    whether a donor's fitted conductances still behave on H01 anatomy. Measured once,
             and the reading is that they do not: the H01 skeleton is silent at the donor's own
             200 pA input.

A term with no supporting evidence is reported as 0.0 with its cause, never omitted.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

ACCURACY = "h01-topographic/human-vs-rodent-accuracy.json"
TYPES = "h01-population-types.json"
IMPORT104 = "h01-population-import-104.json"
BUILD104 = "h01-ready-104-implicit-build-decision.json"
INIT104 = "h01-ready-104-implicit-init-decision.json"
PROBE104 = "h01-arc-probe-104.json"
DRIVEN104 = "h01-ready-104-ei-10ms-decision.json"
DRIVEN_R2 = "h01-ready-104-implicit-ei-10ms-r2-launch.json"
DTLADDER = "h01-ready-cell7196644737-implicit-decision.json"
LADDER = "h01-timestep-ladder.json"
ANATOMY = "h01-e-morphology/stage-1-decision.json"

CELLS = 104
B3_KEY = "l2-pyramidal-allen-541563728"


def _read(name):
    with open(HERE / name, encoding="utf-8") as fh:
        return json.load(fh)


def _term(name, value, measured, statement, cause, files):
    """One factor of the product. `value` is a fraction in [0, 1]; `measured` says whether the
    value came from a reading or is an unearned assumption standing in for a missing one."""
    return {"term": name, "value": float(value), "measured": bool(measured),
            "statement": statement, "cause": cause, "evidence": list(files)}


def donor_term():
    """Donor accuracy. Scoped to the cells whose donor actually carries a ten-element score."""
    acc = _read(ACCURACY)
    types = _read(TYPES)
    scored = sum(1 for r in types["rows"] if r.get("donor_key") == B3_KEY and r.get("match") == "matched")
    return _term(
        "donor_accuracy", acc["b3"]["mean_pct"] / 100., True,
        f"B3 scores {acc['b3']['mean_pct']:.1f} percent on {acc['b3']['n_elements']} elements against "
        f"Allen 541563728's own recording at 310 pA. That donor is the type match for {scored} of "
        f"{CELLS} cells.",
        f"The other three donors (Allen L4 527952884, HL5MN1, HL5BN1) had their published fits "
        f"reproduced and rejected against their own recordings; no ten-element score exists for them, "
        f"so this term is measured for {scored} cells only and assumed for the rest.",
        [ACCURACY, TYPES])


def coverage_term():
    """Type-match coverage: layer AND class, from the released tags."""
    types = _read(TYPES)
    matched = types["summary"]["matched"]
    return _term(
        "type_coverage", matched / CELLS, True,
        f"{matched} of {CELLS} cells have a donor matched in both layer and class "
        f"({100. * matched / CELLS:.1f} percent).",
        f"The remaining {CELLS - matched} are matched on polarity or on part of the label only; they "
        f"would inherit a model fitted to a different cell class.",
        [TYPES])


def build_term():
    """Construction: every gate that has a committed reading, in order."""
    imp, build, init = _read(IMPORT104), _read(BUILD104), _read(INIT104)
    probe = _read(PROBE104)
    ok = (imp["passed"] and build["status"] == "passed" and len(build["failures"]) == 0
          and init["status"] == "passed" and probe["status"] == "forward_pass")
    n_imported = sum(1 for c in imp["cells"] if c["passed"])
    return _term(
        "construction", (n_imported / CELLS) if ok else 0., True,
        f"All {n_imported} of {CELLS} largest soma-bearing components import with zero missing "
        f"segments; construction passes with {build['n_compartments']:,} compartments and no "
        f"failures; initialisation passes; a forward pass completes all six phases at dt "
        f"{probe['dt_ms']} ms, peak RSS {probe['peak_rss_mb'] / 1024.:.1f} GiB.",
        "This supersedes the earlier 12-of-104 figure: the one-point SWC branch that stopped 7 "
        "components from loading is repaired. The forward pass drives an all-ones synthetic "
        f"441-feature probe, not an encoded episode, and records physiology as "
        f"'{probe['physiology']}'.",
        [IMPORT104, BUILD104, INIT104, PROBE104])


def driven_term():
    """A driven window at a physiological duration. No reading exists, so the term is zero."""
    dec, r2 = _read(DRIVEN104), _read(DRIVEN_R2)
    return _term(
        "driven_window", 0., False,
        "No qualified driven window at 104 cells. The 10 ms explicit run failed on "
        f"{dec['failure']} after {dec['wall_seconds']:.0f} s; the implicit retry aborted at the "
        f"wall cap after {r2['seconds']:.0f} s with exit code {r2['exit_code']} and no result.",
        "Construction is not simulation. Until a driven window of physiological length completes, "
        "no population number describes anything the cells do.",
        [DRIVEN104, DRIVEN_R2])


def timestep_term():
    """The integration step against the campaign's own 1 mV contract."""
    lad = _read(LADDER)
    errs = [p["max_error_mv"] for p in lad["pairs"]]
    dt = lad["qualified_dt_ms"]
    return _term(
        "timestep", 1. if dt else 0., True,
        f"Qualified at dt {dt} ms: refining to {lad['pairs'][-1]['dt_fine_ms']} ms moves the trace "
        f"by {errs[-1]:.3f} mV, inside the {lad['gate_mv']:.0f} mV contract. The four adjacent "
        f"pairs give {', '.join(f'{e:.2f}' for e in errs)} mV, halving with the step (first order)."
        if dt else "Not qualified at any step on this ladder.",
        f"This supersedes {DTLADDER}, which compared only the two coarsest rungs and reported the "
        f"gate as FAIL at 6.36 mV. The three finer rungs had already run; their traces were "
        f"recovered from the 2026-09-09 worktree stash and the ladder closes as "
        f"arithmetic, with no new simulation. Two limits on the reading: the gate is met by the "
        f"FINEST pair on the ladder, so no rung below 0.0003125 ms confirms that convergence "
        f"continues (the first-order ratios say it should, which is an extrapolation, not a "
        f"measurement); and it is convergence on one isolated cell, not a claim about the other "
        f"103.",
        [LADDER, DTLADDER])


def anatomy_term():
    """Whether a donor's fitted conductances survive the move onto H01 anatomy."""
    ana = _read(ANATOMY)
    return _term(
        "anatomy_transfer", 0., True,
        f"Measured once and it fails: {ana['verdict']}. On the donor's own reconstruction the same "
        f"parameters fire {ana['control']['count']} spikes at the same input.",
        "The H01 skeleton is a far larger electrical load than the Allen reconstruction the fit was "
        "made on (onset capacitance 785 pF against 125 pF; input resistance about 38 MOhm against "
        "about 98), so at 200 pA it sits on a plateau instead of spiking. Every other term in this "
        "ledger is conditional on this one, and this one has a negative reading.",
        [ANATOMY])


def ledger():
    """All terms, in the order a reader needs them. Terms are never dropped."""
    return [donor_term(), coverage_term(), build_term(), driven_term(), timestep_term(),
            anatomy_term()]


def product(terms, assume=()):
    """The product of the terms, with the terms named in `assume` taken as 1.0.

    `assume` exists so that an optimistic figure can be quoted only alongside the exact list of
    unearned assumptions that produced it. With `assume` empty the product is zero, which is the
    honest reading: three terms have no positive evidence.
    """
    assumed, value, used = list(assume), 1., []
    for t in terms:
        v = 1. if t["term"] in assumed else t["value"]
        used.append({"term": t["term"], "used": v, "assumed": t["term"] in assumed})
        value *= v
    return {"value_pct": 100. * value, "assumptions": assumed, "factors": used}


def report():
    terms = ledger()
    return {
        "scope": ("What a single accuracy percentage for the 104 H01 cells would have to be a "
                  "product of. No H01 cell has electrophysiology, so every term is a transfer "
                  "claim and none of them is a measurement of an H01 cell."),
        "cells": CELLS,
        "terms": terms,
        "product_as_measured": product(terms),
        "product_if_construction_counts_as_simulation": product(
            terms, assume=["driven_window", "anatomy_transfer"]),
        "note": ("The second product is the figure obtained by counting construction as a "
                 "working simulation and setting the two unqualified terms to one. It is quoted only with "
                 "its assumptions attached; it also assumes the three rejected donors score like "
                 "the one donor that was scored, which the anatomy term actively contradicts."),
    }


def main():
    out = HERE / "h01-population-accuracy-ledger.json"
    out.write_text(json.dumps(report(), indent=2) + "\n", encoding="utf-8")
    r = report()
    for t in r["terms"]:
        print(f"{t['term']:<18} {t['value']:.3f}  measured={t['measured']}")
    print(f"as measured: {r['product_as_measured']['value_pct']:.2f} percent")
    print(f"with two terms assumed: "
          f"{r['product_if_construction_counts_as_simulation']['value_pct']:.2f} percent")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
