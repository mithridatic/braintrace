#!/usr/bin/env python3
"""Complete the Gate 2 acceptance measurements and attribute them by channel.

Two of the four Gate 2 criteria are not in the summary metrics, and both are
the direct measurement of a named defect:

* ``shape_square_fraction`` -- 79.2% of legacy candidate-1 shapes were square
  because two 30-way softmaxes collapsed onto one function (D2).
* the count of queries reaching pixel >= 0.95 -- legacy managed 0 of 419, the
  signature of a rank-16 colour tensor that cannot express a grid (D3).

The summary also cannot answer the question that decides what to do next. Most
ARC outputs are the same size as their input, so a decoder biased towards
"same shape" scores well on shape while having learned nothing about the
demonstrations. Splitting every metric by whether the target shape equals the
input shape separates the structural regularity from any actual rule
selection, and splitting by channel keeps the verified-rule solves from being
read as neural ones.

Per-query numbers are taken from each record's own ``score`` block, which is
what the frozen scorer actually submitted. The stored ``candidates`` grids are
*not* a substitute: on a query the rule channel wins, they still hold the
model's proposal, so re-deriving accuracy from them silently attributes the
rule channel's solves to the network and reports zero exact matches for a run
that had twenty-seven.

Usage:  example21_shape_attribution.py RESULT_JSON [--effort 32] [--data DIR]
"""

from __future__ import annotations

import argparse
import msgspec
import pathlib

LEGACY_SQUARE_FRACTION = 0.7920


def _shape_class(data_dir: pathlib.Path, task_id: str, query_index: int) -> str:
    """Return whether this query's output keeps its input's shape."""
    name = task_id.split(":")[1]
    task = msgspec.json.decode((data_dir / f"{name}.json").read_text(encoding="utf-8"))
    pair = task["test"][query_index]
    same = len(pair["output"]) == len(pair["input"]) and len(pair["output"][0]) == len(
        pair["input"][0]
    )
    return "same-shape" if same else "resize"


def _accumulate(rows: list[dict], data_dir: pathlib.Path) -> dict:
    buckets: dict[tuple[str, str], dict[str, float]] = {}
    square = 0
    for row in rows:
        score = row["score"]
        candidate = row["candidates"][0]
        square += int(candidate["height"] == candidate["width"])
        key = (
            "rule" if row.get("rule_solved") else "model",
            _shape_class(data_dir, row["task_id"], row["query_index"]),
        )
        bucket = buckets.setdefault(
            key, {"n": 0, "shape": 0, "pixel": 0.0, "solved": 0, "exact": 0}
        )
        pixel = float(score["valid_cell_pixel_accuracy_diagnostic"])
        bucket["n"] += 1
        bucket["shape"] += int(bool(score["shape_accuracy_diagnostic"]))
        bucket["pixel"] += pixel
        bucket["solved"] += int(pixel >= 0.95)
        bucket["exact"] += int(bool(score["pass_at_1"]))
    return {"buckets": buckets, "square": square, "total": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=pathlib.Path)
    parser.add_argument("--effort", default="32")
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=pathlib.Path("var/arc-agi-1/data/evaluation"),
    )
    arguments = parser.parse_args()
    result = msgspec.json.decode(arguments.result.read_text(encoding="utf-8"))
    rows = result["evaluation"]["checkpoint_queries"][arguments.effort]
    summary = _accumulate(rows, arguments.data)
    buckets = summary["buckets"]
    total = summary["total"]

    print(f"queries {total}   effort {arguments.effort}")
    print(
        f"shape square fraction {summary['square'] / total:.4f}"
        f"   (legacy {LEGACY_SQUARE_FRACTION})"
    )
    print()
    header = f"{'channel':7} {'class':11} {'n':>4} {'shape':>7} {'pixel':>7}"
    print(f"{header} {'>=.95':>6} {'exact':>6}")
    for key in sorted(buckets):
        bucket = buckets[key]
        count = bucket["n"]
        print(
            f"{key[0]:7} {key[1]:11} {count:4} {bucket['shape'] / count:7.4f} "
            f"{bucket['pixel'] / count:7.4f} {int(bucket['solved']):6} "
            f"{int(bucket['exact']):6}"
        )

    model = {key: value for key, value in buckets.items() if key[0] == "model"}
    count = sum(value["n"] for value in model.values())
    if not count:
        return 0
    print(
        f"\nmodel only: n={count} "
        f"shape={sum(v['shape'] for v in model.values()) / count:.4f} "
        f"pixel={sum(v['pixel'] for v in model.values()) / count:.4f} "
        f">=0.95={int(sum(v['solved'] for v in model.values()))} "
        f"exact={int(sum(v['exact'] for v in model.values()))}"
    )
    resize = model.get(("model", "resize"))
    if resize and resize["n"]:
        print(
            f"model shape accuracy where the output resizes: "
            f"{int(resize['shape'])}/{int(resize['n'])} = "
            f"{resize['shape'] / resize['n']:.4f}"
        )
        print(
            "  A same-shape bias alone scores 0 here, and a uniform 30-way "
            "guess scores ~0.033."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
