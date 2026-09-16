"""Write the Vast chain scripts that drive every H01 cell under its donor's protocol.

Spec: docs/specs/2026-09-16-h01-keep-drop.md. One command per cell through
``var/h01-driven/run_transfer.sh``; cells are sorted by id and dealt round-robin into
``n`` chains that run concurrently, each pinned to its own CPU set and a 0.2 GPU memory share.
"""

import argparse
import json
from pathlib import Path

SPEC = "docs/specs/2026-09-16-h01-keep-drop.md"
RUNNER = "var/h01-driven/run_transfer.sh"
CPU_SETS = ["0-31", "32-63", "64-95", "96-127"]
CAP_S = {"primary": 1800, "dthalf": 3600}

PROTOCOL = {
    "l2-pyramidal-allen-541563728": dict(polarity="E", pulse_on_ms=1020, pulse_ms=1000, duration_ms=2300,
                                         current_na=.31, registered_count=10, repeat_counts=None, donor_model_count=10),
    "l4-pyramidal-allen-527952884": dict(polarity="E", pulse_on_ms=1020, pulse_ms=1000, duration_ms=2300,
                                         current_na=.09, registered_count=12, repeat_counts=None, donor_model_count=8),
    "l5-pv-basket-hl5bn1": dict(polarity="I", pulse_on_ms=270, pulse_ms=1000, duration_ms=1500,
                                current_na=.19, registered_count=12, repeat_counts=None, donor_model_count=14),
    "l3-sst-interneuron-hl5mn1": dict(polarity="I", pulse_on_ms=270, pulse_ms=1000, duration_ms=1500,
                                      current_na=.10, registered_count=14, repeat_counts=(14, 14, 13, 12),
                                      donor_model_count=16),
}


def command(cell, donor_key, dt_half=False):
    """One launcher line for a cell: primary run, or the dt-half repeat when ``dt_half``."""
    p = PROTOCOL[donor_key]
    label = f"transfer-all-{cell}" + ("-dthalf" if dt_half else "")
    parts = [f"$R {label} {CAP_S['dthalf' if dt_half else 'primary']} --",
             f"--cell {cell} --donor {donor_key} --polarity {p['polarity']}",
             f"--pulse-on-ms {p['pulse_on_ms']} --pulse-ms {p['pulse_ms']} --duration-ms {p['duration_ms']}",
             f"--current-na {p['current_na']:g} --registered-count {p['registered_count']}"]
    if p["repeat_counts"]:
        parts.append("--repeat-counts " + " ".join(str(c) for c in p["repeat_counts"]))
    parts.append(f"--donor-model-count {p['donor_model_count']}")
    if dt_half:
        parts.append("--dt-ms 0.0025")
    return " ".join(parts)


def chains(cells_with_donors, n=4):
    """Sort (cell, donor) pairs by cell id and deal them round-robin into ``n`` lists."""
    ordered = sorted(cells_with_donors, key=lambda pair: pair[0])
    return [ordered[k::n] for k in range(n)]


def write_chains(rows, out_dir, phase, n=4):
    """Write keep-chain-<k>.sh for k = 1..n; returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt_half = phase == "dthalf"
    suffix = "-dthalf" if dt_half else ""
    paths = []
    for k, chain in enumerate(chains(rows, n), start=1):
        lines = ["#!/usr/bin/env bash", f"# Keep/drop {phase} chain {k} of {n}; rule and protocol in {SPEC}.",
                 "cd /workspace/braintrace", f"R={RUNNER}", f"export CPUSET={CPU_SETS[k-1]} MEMFRAC=.2"]
        lines += [command(cell, donor, dt_half) for cell, donor in chain]
        lines.append(f"echo CHAIN-DONE > var/h01-driven/keep-chain-{k}{suffix}.done")
        path = out_dir/f"keep-chain-{k}{suffix}.sh"
        path.write_text("\n".join(lines)+"\n", newline="\n")
        paths.append(path)
    return paths


def currents_for(rows):
    """Per-cell soma pulse amplitude (nA) for the kept-network probe: each cell's donor primary input."""
    return {cell: PROTOCOL[donor]["current_na"] for cell, donor in rows}


def rows_from_types(types_path, candidates=None):
    rows = json.loads(Path(types_path).read_text())["rows"]
    pairs = [(row["cell_id"], row["donor_key"]) for row in rows]
    if candidates is not None:
        wanted = set(candidates)
        pairs = [pair for pair in pairs if pair[0] in wanted]
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--types", type=Path, default=Path("docs/evidence/h01-population-types.json"))
    parser.add_argument("--phase", choices=["primary", "dthalf", "currents"], required=True)
    parser.add_argument("--candidates", type=Path,
                        help="decision.json whose 'candidates' (dthalf) or 'kept' (currents) list selects cells")
    parser.add_argument("--out-dir", type=Path, help="chain scripts directory (primary, dthalf)")
    parser.add_argument("--currents-out", type=Path, help="JSON {cell_id: nA} for --currents-json (currents phase)")
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()
    if args.phase != "primary" and args.candidates is None:
        parser.error("--candidates is required for the dthalf and currents phases")
    key = "kept" if args.phase == "currents" else "candidates"
    candidates = None if args.candidates is None else json.loads(args.candidates.read_text())[key]
    rows = rows_from_types(args.types, candidates)
    if args.phase == "currents":
        if args.currents_out is None:
            parser.error("--currents-out is required for the currents phase")
        args.currents_out.write_text(json.dumps(currents_for(rows), indent=2, sort_keys=True)+"\n", newline="\n")
        print(args.currents_out, len(rows), "cells")
        return
    if args.out_dir is None:
        parser.error("--out-dir is required for chain phases")
    for path in write_chains(rows, args.out_dir, args.phase, args.chains):
        print(path)


if __name__ == "__main__":
    main()
