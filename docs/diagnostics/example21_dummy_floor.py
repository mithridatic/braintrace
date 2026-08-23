#!/Usr/bin/env python3
"""Phase-1 diagnostic: score trivial predictors through Example 21's own scorer.

Establishes the floor that the pp-prop model must beat before any claim about
capacity or training budget is meaningful. Read-only; touches no production code.
"""

from __future__ import annotations

import collections
import msgspec
import pathlib
import sys

import numpy as np

sys.path.insert(0, "examples/pp_prop")
from latent_workspace_analysis import (
    aggregate_arc_metrics,
    score_query_candidates,
)

ARC = pathlib.Path(sys.argv[1])
MAX_SIDE = 30


def load_split(name: str) -> list[tuple[str, dict]]:
    return [
        (path.stem, msgspec.json.decode(path.read_text()))
        for path in sorted((ARC / name).glob("*.json"))
    ]


def demo_shape_rule(task: dict, query_input: np.ndarray) -> tuple[int, int]:
    """Predict output shape from the demonstrations, falling back to the input."""
    ratios = set()
    shapes = set()
    for pair in task["train"]:
        source = np.asarray(pair["input"])
        target = np.asarray(pair["output"])
        shapes.add(target.shape)
        if (
            target.shape[0] % source.shape[0] == 0
            and target.shape[1] % source.shape[1] == 0
        ):
            ratios.add(
                (target.shape[0] // source.shape[0], target.shape[1] // source.shape[1])
            )
        else:
            ratios.add(None)
    if len(ratios) == 1 and (ratio := next(iter(ratios))) is not None:
        return (
            min(MAX_SIDE, query_input.shape[0] * ratio[0]),
            min(MAX_SIDE, query_input.shape[1] * ratio[1]),
        )
    if len(shapes) == 1:
        return next(iter(shapes))
    return query_input.shape


def predictors(task: dict, query_input: np.ndarray) -> dict[str, np.ndarray]:
    background = collections.Counter(query_input.ravel().tolist()).most_common(1)[0][0]
    rule_shape = demo_shape_rule(task, query_input)
    return {
        "copy_input": query_input.copy(),
        "zeros_input_shape": np.zeros_like(query_input),
        "mode_color_input_shape": np.full_like(query_input, background),
        "zeros_demo_shape": np.zeros(rule_shape, dtype=int),
        "copy_or_rule_shape": (
            query_input.copy()
            if rule_shape == query_input.shape
            else np.full(rule_shape, background, dtype=int)
        ),
        "uniform_random_input_shape": np.random.default_rng(0).integers(
            0, 10, size=query_input.shape
        ),
    }


def main() -> None:
    for split in ("evaluation",):
        tasks = load_split(split)
        scores: dict[str, list] = collections.defaultdict(list)
        for task_id, task in tasks:
            for index, pair in enumerate(task["test"]):
                query_input = np.asarray(pair["input"])
                target = np.asarray(pair["output"])
                for name, grid in predictors(task, query_input).items():
                    scores[name].append(
                        score_query_candidates(
                            [grid], target, task_id=task_id, query_index=index
                        )
                    )
        print(
            f"=== split={split}  tasks={len(tasks)}  queries={len(next(iter(scores.values())))}"
        )
        print(f"{'predictor':<28} {'pass@1':>8} {'shape':>8} {'pixel':>8}")
        for name, entries in scores.items():
            metrics = aggregate_arc_metrics(entries)
            print(
                f"{name:<28} {metrics['query_pass_at_1']:>8.4f}"
                f" {metrics['shape_accuracy_diagnostic']:>8.4f}"
                f" {metrics['valid_cell_pixel_accuracy_diagnostic']:>8.4f}"
            )
        print(
            f"{'pp-prop effort 32 (measured)':<28} {0.0:>8.4f} {0.0859:>8.4f} {0.0628:>8.4f}"
        )


if __name__ == "__main__":
    main()
