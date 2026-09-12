"""Required direct-observation artifacts for H01 E-cell diagnostic analyses."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_direct_currents import extract_l2
from docs.evidence.h01_direct_observations import observe


def _source(path):
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def write_bundle(stems, output, human_paths=(), plots=True):
    """Save direct samples, multivari charts, and explicit evidence limitations.

    Parameters
    ----------
    stems : iterable of path-like
        Model run prefixes, each with matching NPZ and JSON files.
    output : path-like
        Artifact prefix, distinct from historical registered decisions.
    human_paths : iterable of path-like
        Human NPZ exports on the same 1020-2020 ms pulse clock.
    plots : bool
        Render charts. False is for a compute-only export awaiting rendering.

    Returns
    -------
    dict
        Artifact paths and observation status. Never a causal or human pass.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records, arrays = [], {}
    for stem in map(Path, stems):
        npz, meta = stem.with_suffix(".npz"), stem.with_suffix(".json")
        with np.load(npz) as data:
            report = json.loads(meta.read_text(encoding="utf-8"))
            record, selected = extract_l2(data, report)
        key = f"r{len(records)}"
        records.append({"label": f"{stem.name} | sweep {report.get('sweep', '?')} | model",
                            "array_key": key, "observation": record, "sources": [_source(npz), _source(meta)]})
        arrays.update({f"{key}_{name}": values for name, values in selected.items()})
    for path in map(Path, human_paths):
        with np.load(path) as data:
            # Command current is an observed input, not a measured ionic-current budget.
            currents = {"command": data["total_current_na"]} if "total_current_na" in data else {}
            record, selected = observe(data["time_ms"], data["corrected_voltage_mv"], (1020., 2020.),
                currents, metadata={"source": "human", "location": "recorded electrode",
                                        "units": "ms, corrected mV, nA", "limitation": "Human ionic currents unavailable."})
            record["current_evidence"] = "command_only" if currents else "unavailable"
        key = f"r{len(records)}"
        records.append({"label": f"{path.stem} | human", "array_key": key, "observation": record, "sources": [_source(path)]})
        arrays.update({f"{key}_{name}": values for name, values in selected.items()})
    if not records:
        raise ValueError("At least one model or human trace is required.")
    archive = Path(f"{output}.npz")
    np.savez_compressed(archive, **arrays)
    bundle = {"schema": "h01-direct-bundle-v1", "records": records, "selected_samples": _source(archive),
                  "causal_verdict": "not_established", "human_qualification": "not_assessed",
                  "visual_review": "pending", "registered_qc": "Separate historical record; not replaced by this analysis.",
                  "software_sources": [_source(Path(__file__).with_name(name)) for name in
                    ("h01_direct_report.py", "h01_direct_observations.py", "h01_direct_currents.py", "h01_direct_plots.py")]}
    bundle["plots"] = []
    if plots:
        from docs.evidence.h01_direct_plots import render_bundle
        bundle["plots"] = render_bundle(bundle, arrays, output)
    Path(f"{output}.json").write_text(json.dumps(bundle, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# H01 direct observations", "", "Inspect the multivari and conjugate-pair charts before interpreting a cause.",
             "A small cycle sample retains all original samples within each selected window.",
             "Missing phases and mismatched events are not causal equivalence. Human ionic currents are unavailable.", ""]
    for item in records:
        obs = item["observation"]
        lines += [f"## {item['label']}", "", f"Current observations: {obs['current_evidence']}. Causal verdict: not established.", "",
                  "| Window | Cycle | Phase | Absolute ms | Voltage mV | Status |",
                  "| --- | ---: | --- | ---: | ---: | --- |"]
        for w in obs["windows"]:
            if not w["phase_samples"]:
                lines.append(f"| {w['kind']} | {w['cycle']} | unavailable | | | {w['status']} |")
            for p in w["phase_samples"]:
                value = "unavailable" if p["voltage_mv"] is None else f"{p['voltage_mv']:.6g}"
                lines.append(f"| {w['kind']} | {w['cycle']} | {p['phase']} | {p['time_ms']:.6f} | {value} | {p['status']} |")
        lines.append("")
    for plot in bundle["plots"]:
        lines += [f"![{Path(plot).stem}]({Path(plot).name})", ""]
    lines += ["Visual review: pending. Record which patterns occur within phases, between cycles, and between inputs.",
              "Do not infer a channel identity from the chart alone. Retain the named intervention and its contradicting outcome."]
    Path(f"{output}.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"json": f"{output}.json", "markdown": f"{output}.md", "arrays": str(archive), "plots": bundle["plots"],
                "visual_review": "pending", "causal_verdict": "not_established"}


def main(argv=None):
    """Run a retained-trace analysis without launching a simulation.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to the process arguments.

    Returns
    -------
    None
        Writes artifacts and prints their locations.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], type=Path)
    parser.add_argument("--human", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true", help="Compute-only export; visual review remains pending.")
    args = parser.parse_args(argv)
    print(json.dumps(write_bundle(args.run, args.output, args.human, not args.no_plots), indent=2))


if __name__ == "__main__":
    main()
