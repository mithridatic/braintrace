#!/usr/bin/env python3
"""Gate 1: score the demonstration-verified rule channel alone on ARC.

Measures the exact pass@1 / pass@2 floor the rule channel establishes before any
neural work. Read-only; touches no production state.

Usage
-----
``python docs/diagnostics/example21_rule_channel.py var/arc-agi-1/data [split]``
"""

from __future__ import annotations

import collections
import msgspec
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "examples/pp_prop")
from latent_workspace_rules import verified_rule_candidates  # noqa: E402

ARC = pathlib.Path(sys.argv[1])
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "evaluation"


def main() -> None:
    """Score the rule channel over one ARC split and print the summary."""

    paths = sorted((ARC / SPLIT).glob("*.json"))
    started = time.perf_counter()
    query_total = query_at_1 = query_at_2 = 0
    task_at_1 = task_at_2 = 0
    solved_by: collections.Counter[str] = collections.Counter()
    admitted_any = 0
    solved_tasks: list[str] = []

    for path in paths:
        task = msgspec.json.decode(path.read_text())
        demos = [
            (np.asarray(pair["input"], np.int32), np.asarray(pair["output"], np.int32))
            for pair in task["train"]
        ]
        task_hits_1: list[bool] = []
        task_hits_2: list[bool] = []
        for query in task["test"]:
            source = np.asarray(query["input"], np.int32)
            truth = np.asarray(query["output"], np.int32)
            candidates = verified_rule_candidates(demos, source)[:2]
            if candidates:
                admitted_any += 1
            hits = [
                grid.shape == truth.shape and np.array_equal(grid, truth)
                for _, grid in candidates
            ]
            query_total += 1
            first = bool(hits[:1] and hits[0])
            any_hit = any(hits)
            query_at_1 += first
            query_at_2 += any_hit
            task_hits_1.append(first)
            task_hits_2.append(any_hit)
            if any_hit:
                name = next(n for (n, _), hit in zip(candidates, hits) if hit)
                solved_by[name.split(":")[0]] += 1
        if task_hits_1 and all(task_hits_1):
            task_at_1 += 1
        if task_hits_2 and all(task_hits_2):
            task_at_2 += 1
            solved_tasks.append(path.stem)

    elapsed = time.perf_counter() - started
    print(f"=== rule channel  split={SPLIT}  tasks={len(paths)}  queries={query_total}")
    print(f"query   pass@1={query_at_1 / query_total:.4f} ({query_at_1})")
    print(f"query   pass@2={query_at_2 / query_total:.4f} ({query_at_2})")
    print(f"strict  pass@1={task_at_1 / len(paths):.4f} ({task_at_1}/{len(paths)})")
    print(f"strict  pass@2={task_at_2 / len(paths):.4f} ({task_at_2}/{len(paths)})")
    print(f"queries with any admitted rule: {admitted_any}/{query_total}")
    print(f"elapsed: {elapsed:.1f} s")
    print("solves by family:", dict(solved_by.most_common()))
    print("solved tasks:", " ".join(solved_tasks))


if __name__ == "__main__":
    main()
