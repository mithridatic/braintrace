"""Stage 1 of the I reserve: score the single evaluation against its registered bands.

The bands, the rejection clause and the loop limits are those pre-registered in
``docs/specs/2026-09-07-h01-i-reserve.md`` and ``h01-i-reserve-manifest.json``.
Observed values come from the usable-tier report of the run (retained per-cycle
tables), the initiation scores (axon-first must hold) and the campaign log (wall
clocks). Nothing here re-tunes a band after the run.
"""

import argparse
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
CYCLE_LIMIT_MS = 12.8
PEAK_AT_MOST_MV = 22.
WIDTH_TOL_MS = .006
BANDS = {
    "0.19 nA": {"cycle2_ms": {"from": 34.76, "at_most": 22.}, "count": {"from": 14, "range": (15, 17)},
                "width1_ms": .221, "troughs_mv": (-80.13, -80.54, -80.65), "loop_width_ms": (.218, .328)},
    "0.27 nA": {"cycle2_ms": {"from": 16.37, "at_most": 12.}, "count": {"from": 37, "range": (40, 43)},
                "width1_ms": .221, "troughs_mv": (-78.99, -79.30, -79.61), "loop_width_ms": (.214, .320)},
}


def _pick(table, cycle, key):
    rows = [r for r in table if r["cycle"] == cycle]
    return rows[0][key] if rows else None


def band_rows(label, table):
    """Predicted, observed and held/missed for every registered band of one input."""
    band = BANDS[label]
    cycle2 = _pick(table, 2, "cycle_ms")
    width1 = _pick(table, 1, "width_ms")
    peak = max((r["peak_mv"] for r in table), default=None)
    rows = [{"row": "cycle2_ms", "predicted": f"<= {band['cycle2_ms']['at_most']} (from {band['cycle2_ms']['from']})",
             "observed": cycle2, "held": cycle2 is not None and cycle2 <= band["cycle2_ms"]["at_most"]},
            {"row": "count", "predicted": f"{band['count']['range'][0]} to {band['count']['range'][1]} (from {band['count']['from']})",
             "observed": len(table), "held": band["count"]["range"][0] <= len(table) <= band["count"]["range"][1]},
            {"row": "width1_ms", "predicted": f"{band['width1_ms']} +/- {WIDTH_TOL_MS}", "observed": width1,
             "held": width1 is not None and abs(width1-band["width1_ms"]) <= WIDTH_TOL_MS},
            {"row": "peak_mv", "predicted": f"<= {PEAK_AT_MOST_MV}", "observed": peak,
             "held": peak is not None and peak <= PEAK_AT_MOST_MV}]
    for cycle, reference in enumerate(band["troughs_mv"], start=1):
        trough = _pick(table, cycle, "ahp_mv")
        rows.append({"row": f"trough{cycle}_mv", "predicted": f"within 1 of {reference}", "observed": trough,
                     "held": trough is not None and abs(trough-reference) <= 1.})
    return [dict(r, input=label) for r in rows]


def rejection(tables):
    """The pre-registered rejection clause: cycle 2 at 0.27 nA moves less than the limit, or the loop breaks."""
    cycle2 = _pick(tables["0.27 nA"], 2, "cycle_ms")
    change = None if cycle2 is None else BANDS["0.27 nA"]["cycle2_ms"]["from"]-cycle2
    breaks = []
    for label, table in tables.items():
        width1, low, high = _pick(table, 1, "width_ms"), *BANDS[label]["loop_width_ms"]
        peak = max((r["peak_mv"] for r in table), default=None)
        if width1 is None or not low <= width1 <= high:
            breaks.append(f"{label}: cycle-1 width {width1} outside {low}-{high} ms")
        if peak is None or peak > PEAK_AT_MOST_MV:
            breaks.append(f"{label}: peak {peak} above {PEAK_AT_MOST_MV} mV")
    below = change is None or change < CYCLE_LIMIT_MS
    return {"cycle2_027_change_ms": change, "cycle_limit_ms": CYCLE_LIMIT_MS, "change_below_limit": below,
            "loop_breaks": breaks, "met": below or bool(breaks)}


def tier_verdicts(report):
    """Usable and contract verdicts per input from the usable-tier rows and counts."""
    out = {}
    for label, counts in report["counts"].items():
        rows = [r for r in report["rows"] if r["input"] == label]
        usable = {v: sorted({f"{r['row']}{'' if r['cycle'] is None else r['cycle']}" for r in rows if r["verdict"] == v})
                  for v in ("pass", "fail", "unresolvable")}
        contract_fail = sorted({f"{r['row']}{'' if r['cycle'] is None else r['cycle']}" for r in rows if r["contract"] == "fail"})
        count_ok = counts["human"] == counts["model"]
        out[label] = {"counts": counts, "usable": usable,
                      "usable_verdict": "pass" if not usable["fail"] else "fail",
                      "contract": {"count": "pass" if count_ok else "FAIL", "failing_rows": contract_fail,
                                   "verdict": "pass" if count_ok and not contract_fail else "FAIL"}}
    return out


def decide(report, initiation, log, candidate_sha256):
    """The stage-1 decision: bands, rejection, tiers, initiation, wall clocks."""
    tables = {label: sides["model"] for label, sides in report["tables"].items()}
    bands = [row for label in BANDS for row in band_rows(label, tables[label])]
    reject = rejection(tables)
    axon_first = {k: bool(v.get("axon_first")) for k, v in initiation.items()}
    passed = not reject["met"] and all(r["held"] for r in bands) and all(axon_first.values())
    return {"stage": "1", "candidate": "g-natg-axon15", "candidate_sha256": candidate_sha256,
            "decision": ("PASS: every band held, rejection not met, axon-first holds" if passed else
                         "FAIL at cap: " + ("rejection clause met" if reject["met"] else "a band missed or axon-first lost")),
            "verdict": "PASS" if passed else "FAIL", "bands": bands,
            "bands_held": sum(r["held"] for r in bands), "bands_total": len(bands), "rejection": reject,
            "initiation_axon_first": axon_first,
            "initiation": {k: {f: v.get(f) for f in ("soma_crossing_ms", "axon_crossing_ms", "axon_leads_ms", "spikes")}
                           for k, v in initiation.items()},
            "tiers": tier_verdicts(report),
            "runs": [{k: r[k] for k in ("input", "seconds", "returncode", "aborted")} for r in log
                     if r["name"] == "g-natg-axon15"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usable", type=Path, default=EVIDENCE/"h01-i-reserve"/"g-natg-axon15-usable.json")
    parser.add_argument("--folder", type=Path, default=EVIDENCE/"h01-i-reserve")
    parser.add_argument("--out", type=Path, default=EVIDENCE/"h01-i-reserve"/"stage-1-decision.json")
    args = parser.parse_args(argv)
    report = json.loads(args.usable.read_text(encoding="utf-8"))
    initiation = {name: json.loads((args.folder/f"g-natg-axon15-{name}-initiation.json").read_text(encoding="utf-8"))
                  for name in ("019", "027")}
    log = json.loads((args.folder/"campaign-log.json").read_text(encoding="utf-8"))
    sha = json.loads((args.folder/"g-natg-axon15-027.json").read_text(encoding="utf-8"))["candidate_json"]["sha256"]
    decision = decide(report, initiation, log, sha)
    args.out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({k: decision[k] for k in ("decision", "bands_held", "bands_total", "rejection")}, indent=1))


if __name__ == "__main__":
    main()
