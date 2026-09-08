"""Youden plots of contract residuals: one calibration input against the other.

Both axes carry the same residual kind on identical square scales (Hartshorne 2020,
scan pages p148-p150). Distance from the diagonal is input-to-input reproducibility of
the candidate error; distance along the diagonal is the shared candidate error. The
allowance square marks the approved contract limit.
"""

import argparse
import json
from pathlib import Path

FOLDER = Path(__file__).parent


def paired_residuals(report, name, inputs, kind):
    """Residuals of one record name at two inputs, matched by event index.

    Parameters
    ----------
    report : dict
        Loaded scorecard JSON.
    name : str
        Record name shared by both inputs.
    inputs : tuple of str
        The two input keys, e.g. ``("I 0.19", "I 0.27")``.
    kind : str
        Row kind to pair, e.g. ``"rise_crossing_ms"``.

    Returns
    -------
    list of tuple
        ``(key, residual_a, residual_b)`` for rows present and finite at both inputs.
    """
    rows = {}
    for record in report["records"]:
        if record["name"] == name and record["input"] in inputs:
            rows[record["input"]] = {r["key"]: r for r in record["rows"] if r["kind"] == kind}
    if set(rows) != set(inputs):
        raise ValueError(f"Record {name!r} is not scored at both inputs {inputs}.")
    a, b = (rows[i] for i in inputs)
    pairs = []
    for key in a:
        if key in b and a[key]["residual"] is not None and b[key]["residual"] is not None:
            pairs.append((key, a[key]["residual"], b[key]["residual"]))
    return pairs


def square_limits(pairs, allowance):
    """Identical symmetric axis limits covering every point and the allowance square."""
    extent = max([allowance]+[abs(v) for _, x, y in pairs for v in (x, y)])*1.1
    return (-extent, extent)


def plot(pairs, allowance, title, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    limits = square_limits(pairs, allowance)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(limits, limits, color="grey", linewidth=.8)
    axis.add_patch(plt.Rectangle((-allowance, -allowance), 2*allowance, 2*allowance, fill=False, color="red", linewidth=.8))
    axis.scatter([x for _, x, _ in pairs], [y for _, _, y in pairs], s=12)
    for key, x, y in pairs:
        axis.annotate(key.split("_")[0], (x, y), fontsize=6)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=FOLDER/"h01-contract-scorecard.json")
    parser.add_argument("--name", required=True)
    parser.add_argument("--inputs", nargs=2, required=True)
    parser.add_argument("--kind", default="rise_crossing_ms")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.scorecard.read_text())
    allowance = report["allowances"].get(args.kind, report["derived_allowances"].get(args.kind))
    pairs = paired_residuals(report, args.name, tuple(args.inputs), args.kind)
    plot(pairs, allowance, f"{args.name}: {args.kind} at {args.inputs[0]} (x) vs {args.inputs[1]} (y)", args.output)
    print(json.dumps({"points": len(pairs), "allowance": allowance, "output": str(args.output)}))


if __name__ == "__main__":
    main()
