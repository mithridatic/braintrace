"""Event budget of one Example 21 evolve run, derived before launch.

Every ARC query is encoded by ``arc_contracts.encode_episode`` into 705 events
of which only the non-zero rows advance the model: one episode header, 64
events per demonstration pair (two grids of 32 events each), 32 for the query
grid, two request markers and 30 answer rows.  A query with ``D``
demonstration pairs therefore advances ``65 + 64 * D`` events, independent of
grid size.  This module counts those events for every stage of the evolve
lifecycle under one :class:`~examples.pp_prop.example21_evolve.PipelineConfig`,
multiplies them by supplied per-event costs, and reports whether each stage
fits a wall-clock cap.  It can also launch a scoped H01 run through the
unchanged coordinator so the pinned entry point stays byte-identical.

The derivation, with the numbers of the pinned H01 profile, is recorded in
``docs/evidence/h01-event-budget.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.pp_prop.example21_evolve import (
    DEFAULT_ROUNDS,
    DEFAULT_SCORE_TASKS,
    DEFAULT_SCREEN_TASKS,
    DEFAULT_UPDATES,
    CorpusManifest,
    PipelineConfig,
    build_update_schedule,
    score_scope_ids,
    screen_task_ids,
)

EPISODE_HEADER_EVENTS = 1
GRID_EVENTS = 32
DEMONSTRATION_EVENTS = 2 * GRID_EVENTS
REQUEST_EVENTS = 2
ANSWER_ROW_EVENTS = 30
QUERY_FIXED_EVENTS = (
    EPISODE_HEADER_EVENTS + GRID_EVENTS + REQUEST_EVENTS + ANSWER_ROW_EVENTS
)
OPERATION_ARMS_PER_ROUND = {"edge": 2, "neuron": 2}

# Measured on the 17 kept cells at the pinned dt 0.000625 ms / 160 substeps
# (docs/evidence/h01-evolve-event-profile.json, arm ``pinned``): forward
# seconds per advancing event, learner seconds per advancing event, and the
# fixed build + init + compile seconds paid once per rebuilt runtime.
PROFILE_PINNED = {
    "forward_seconds_per_event": 1.3567346297204494,
    "learner_seconds_per_event": 13.18038301775232,
    "fixed_seconds": 497.20272340718657,
}
PROFILE_DT0005 = {
    "forward_seconds_per_event": 0.17966430680826306,
    "learner_seconds_per_event": 1.337522776176532,
    "fixed_seconds": 418.0818205466494,
}
PROFILES = {"pinned": PROFILE_PINNED, "dt0005": PROFILE_DT0005}
DEFAULT_CAP_SECONDS = 3600.0


@dataclass(frozen=True)
class QueryEvents:
    """Advancing-event count of one supervised ARC query.

    Parameters
    ----------
    task_id : str
        ARC task identifier.
    query_index : int
        Query within the task.
    demonstrations : int
        Demonstration pairs the episode presents before the query.
    advancing : int
        Events with any set bit, the events the model steps.
    """

    task_id: str
    query_index: int
    demonstrations: int
    advancing: int


@dataclass(frozen=True)
class StageEvents:
    """Advancing events one lifecycle stage consumes.

    Parameters
    ----------
    stage : str
        Lifecycle stage or arm label.
    scoring_queries, scoring_events : int
        Queries and advancing events of the stage's scoring pass.
    updates, update_events : int
        Scheduled updates and their advancing events (zero for rescores).
    """

    stage: str
    scoring_queries: int
    scoring_events: int
    updates: int = 0
    update_events: int = 0

    @property
    def events(self) -> int:
        """Return all advancing events of the stage.

        Returns
        -------
        int
            Update events plus scoring events.
        """

        return self.update_events + self.scoring_events


def advancing_events(demonstrations: int) -> int:
    """Return the advancing events of a query with ``demonstrations`` pairs.

    Parameters
    ----------
    demonstrations : int
        Demonstration pairs presented before the query grid.

    Returns
    -------
    int
        ``65 + 64 * demonstrations``.

    Examples
    --------
    .. code-block:: python

        >>> advancing_events(2), advancing_events(10)
        (193, 705)
    """

    if demonstrations < 0:
        raise ValueError("Demonstration count must be nonnegative.")
    return QUERY_FIXED_EVENTS + DEMONSTRATION_EVENTS * demonstrations


def corpus_query_events(
    arc_root: str | os.PathLike[str],
) -> tuple[CorpusManifest, tuple[QueryEvents, ...]]:
    """Count advancing events of every training query in manifest order.

    Parameters
    ----------
    arc_root : path-like
        Raw ARC root containing ``data/training``.

    Returns
    -------
    tuple
        The training manifest and one :class:`QueryEvents` per query, in
        ``manifest.query_order``, counted from the real encoder output.
    """

    from examples.pp_prop.example21_arc_adapter import Example21ArcAdapter

    adapter = Example21ArcAdapter(arc_root)
    manifest = adapter.training_manifest()
    tasks = adapter._load_tasks("training")
    records = []
    for query in adapter._encoded_queries("training"):
        demonstrations = len(tasks[query.task_id].demonstrations)
        counted = int(query.advances.sum())
        if counted != advancing_events(demonstrations):
            raise ValueError(
                f"Encoder advanced {counted} events for {query.task_id} query "
                f"{query.query_index}; the 65 + 64 * D derivation no longer holds."
            )
        records.append(
            QueryEvents(query.task_id, query.query_index, demonstrations, counted)
        )
    return manifest, tuple(records)


def scoring_events(
    queries: Sequence[QueryEvents], task_ids: Sequence[str]
) -> tuple[int, int]:
    """Return the queries and advancing events one scoring pass steps.

    Parameters
    ----------
    queries : sequence of QueryEvents
        Corpus queries in manifest order.
    task_ids : sequence of str
        Scored task subset.

    Returns
    -------
    tuple of int
        Query count and advancing-event count.
    """

    selected = frozenset(task_ids)
    chosen = [query for query in queries if query.task_id in selected]
    return len(chosen), sum(query.advancing for query in chosen)


def block_events(
    manifest: CorpusManifest,
    queries: Sequence[QueryEvents],
    cursor: int,
    updates: int = DEFAULT_UPDATES,
) -> int:
    """Return the advancing events of one update block from ``cursor``.

    Parameters
    ----------
    manifest : CorpusManifest
        Training manifest whose query order the schedule walks.
    queries : sequence of QueryEvents
        Event counts of every manifest query.
    cursor : int
        Absolute update cursor at block start.
    updates : int, optional
        Block length.

    Returns
    -------
    int
        Sum of advancing events over the scheduled entries.
    """

    lookup = {(query.task_id, query.query_index): query.advancing for query in queries}
    schedule = build_update_schedule(manifest, cursor, updates)
    return sum(lookup[(entry.task_id, entry.query_index)] for entry in schedule.entries)


def first_round_stages(
    manifest: CorpusManifest,
    queries: Sequence[QueryEvents],
    config: PipelineConfig,
) -> tuple[StageEvents, ...]:
    """Return the stages of ``initialize`` and the first round, in order.

    The order follows ``_run_evolution``: initial scoring, the parent training
    block, then ``round-screen`` when the lineage screens, two arms each of
    the ``edge`` and ``neuron`` operations (the H01 backend blocks ``dale``
    before any event), and ``round-score`` to restore the complete scope.
    Every model stage advances the cursor by ``config.updates`` whether or not
    its arms ran, so each stage's block is a different schedule window.

    Parameters
    ----------
    manifest : CorpusManifest
        Training manifest.
    queries : sequence of QueryEvents
        Event counts of every manifest query.
    config : PipelineConfig
        Lineage configuration supplying the scope, the screen and the block.

    Returns
    -------
    tuple of StageEvents
        Stage event counts in execution order.
    """

    scope = score_scope_ids(manifest, config)
    screen = screen_task_ids(manifest, config)
    scope_pass = scoring_events(queries, scope)
    screen_pass = scoring_events(queries, screen) if screen else scope_pass
    stages = [StageEvents("initialize", *scope_pass)]
    cursor = 0
    block = block_events(manifest, queries, cursor, config.updates)
    stages.append(StageEvents("train", *scope_pass, config.updates, block))
    cursor += config.updates
    if screen:
        stages.append(StageEvents("round-screen", *screen_pass))
    for operation, arms in OPERATION_ARMS_PER_ROUND.items():
        block = block_events(manifest, queries, cursor, config.updates)
        for arm in range(arms):
            stages.append(
                StageEvents(
                    f"{operation}-arm{arm}", *screen_pass, config.updates, block
                )
            )
        cursor += config.updates
    if screen:
        stages.append(StageEvents("round-score", *scope_pass))
    return tuple(stages)


def stage_seconds(stage: StageEvents, costs: Mapping[str, float]) -> float:
    """Return the wall seconds one stage costs under per-event prices.

    Parameters
    ----------
    stage : StageEvents
        Stage event counts.
    costs : mapping
        ``forward_seconds_per_event``, ``learner_seconds_per_event`` and
        ``fixed_seconds`` (paid once per stage arm that rebuilds a runtime).

    Returns
    -------
    float
        Fixed cost plus update events at the learner price plus scoring
        events at the forward price.
    """

    return (
        costs["fixed_seconds"]
        + stage.update_events * costs["learner_seconds_per_event"]
        + stage.scoring_events * costs["forward_seconds_per_event"]
    )


def largest_scope_under_cap(
    queries: Sequence[QueryEvents],
    manifest: CorpusManifest,
    costs: Mapping[str, float],
    cap_seconds: float,
) -> int:
    """Return the largest ``score_tasks`` whose initial scoring pass fits a cap.

    Parameters
    ----------
    queries : sequence of QueryEvents
        Corpus queries in manifest order.
    manifest : CorpusManifest
        Training manifest.
    costs : mapping
        Per-event prices and fixed seconds.
    cap_seconds : float
        Wall-clock cap for the ``initialize`` stage.

    Returns
    -------
    int
        Leading task count, zero when not even one task fits.
    """

    fitting = 0
    for count in range(1, len(manifest.task_ids) + 1):
        _, events = scoring_events(queries, manifest.task_ids[:count])
        if (
            costs["fixed_seconds"] + events * costs["forward_seconds_per_event"]
            > cap_seconds
        ):
            break
        fitting = count
    return fitting


def updates_under_cap(
    queries: Sequence[QueryEvents], costs: Mapping[str, float], cap_seconds: float
) -> int:
    """Return the most updates of the shortest queries that fit a cap.

    The block floor is ``updates`` times the shortest query in the corpus, so
    this is the upper bound on any block that a cap admits, before scoring.

    Parameters
    ----------
    queries : sequence of QueryEvents
        Corpus queries.
    costs : mapping
        Per-event prices and fixed seconds.
    cap_seconds : float
        Wall-clock cap.

    Returns
    -------
    int
        Update count, zero when one shortest update does not fit.
    """

    shortest = min(query.advancing for query in queries)
    remaining = cap_seconds - costs["fixed_seconds"]
    if remaining <= 0:
        return 0
    return int(remaining // (shortest * costs["learner_seconds_per_event"]))


def event_budget(
    manifest: CorpusManifest,
    queries: Sequence[QueryEvents],
    config: PipelineConfig,
    *,
    costs: Mapping[str, float],
    cap_seconds: float = DEFAULT_CAP_SECONDS,
) -> dict[str, Any]:
    """Return the JSON-compatible event budget of one configuration.

    Parameters
    ----------
    manifest : CorpusManifest
        Training manifest.
    queries : sequence of QueryEvents
        Event counts of every manifest query.
    config : PipelineConfig
        Lineage configuration.
    costs : mapping
        Per-event prices and fixed seconds.
    cap_seconds : float, optional
        Wall-clock cap each stage is checked against.

    Returns
    -------
    dict
        Corpus totals, per-stage events and seconds, cap verdicts, and the
        largest scope and update count the cap admits.
    """

    stages = first_round_stages(manifest, queries, config)
    corpus_queries, corpus_events = scoring_events(queries, manifest.task_ids)
    scope = score_scope_ids(manifest, config)
    screen = screen_task_ids(manifest, config)
    shortest = min(query.advancing for query in queries)
    rows = []
    for stage in stages:
        seconds = stage_seconds(stage, costs)
        rows.append(
            {
                "stage": stage.stage,
                "updates": stage.updates,
                "update_events": stage.update_events,
                "scoring_queries": stage.scoring_queries,
                "scoring_events": stage.scoring_events,
                "events": stage.events,
                "seconds": seconds,
                "fits_cap": seconds <= cap_seconds,
            }
        )
    return {
        "schema": "example21-event-budget-v1",
        "config": config.to_dict(),
        "training_manifest_sha256": manifest.digest,
        "costs": dict(costs),
        "cap_seconds": cap_seconds,
        "corpus": {
            "tasks": len(manifest.task_ids),
            "queries": corpus_queries,
            "advancing_events": corpus_events,
            "mean_advancing_per_query": corpus_events / corpus_queries,
            "min_advancing_per_query": shortest,
            "max_advancing_per_query": max(query.advancing for query in queries),
        },
        "scope": {"score_tasks": len(scope), "screen_tasks": len(screen)},
        "block": {
            "updates": config.updates,
            "advancing_events_from_cursor_0": block_events(
                manifest, queries, 0, config.updates
            ),
            "floor_advancing_events": config.updates * shortest,
            "floor_seconds": costs["fixed_seconds"]
            + config.updates * shortest * costs["learner_seconds_per_event"],
        },
        "stages": rows,
        "first_round_seconds": sum(row["seconds"] for row in rows),
        "largest_score_tasks_under_cap": largest_scope_under_cap(
            queries, manifest, costs, cap_seconds
        ),
        "updates_under_cap": updates_under_cap(queries, costs, cap_seconds),
    }


def render_budget(budget: Mapping[str, Any]) -> str:
    """Return the budget as an operator-facing text table.

    Parameters
    ----------
    budget : mapping
        Output of :func:`event_budget`.

    Returns
    -------
    str
        Multi-line report.
    """

    corpus, block, costs = budget["corpus"], budget["block"], budget["costs"]
    lines = [
        (
            f"corpus: {corpus['tasks']} tasks, {corpus['queries']} queries, "
            f"{corpus['advancing_events']} advancing events "
            f"(min {corpus['min_advancing_per_query']}, mean "
            f"{corpus['mean_advancing_per_query']:.1f}, "
            f"max {corpus['max_advancing_per_query']} per query)"
        ),
        (
            f"scope: score_tasks={budget['scope']['score_tasks']} "
            f"screen_tasks={budget['scope']['screen_tasks']}"
        ),
        (
            f"costs: forward {costs['forward_seconds_per_event']:.3f} s/event, learner "
            f"{costs['learner_seconds_per_event']:.3f} s/event, "
            f"fixed {costs['fixed_seconds']:.0f} s; cap {budget['cap_seconds']:.0f} s"
        ),
        (
            f"block: {block['updates']} updates = "
            f"{block['advancing_events_from_cursor_0']} events from cursor 0; "
            f"floor {block['floor_advancing_events']} events = "
            f"{block['floor_seconds']:.0f} s"
        ),
        "",
        (
            f"{'stage':<14}{'updates':>8}{'upd.events':>12}{'queries':>9}"
            f"{'score.events':>14}{'events':>10}{'seconds':>12}  fits"
        ),
    ]
    for row in budget["stages"]:
        lines.append(
            f"{row['stage']:<14}{row['updates']:>8}{row['update_events']:>12}"
            f"{row['scoring_queries']:>9}{row['scoring_events']:>14}{row['events']:>10}"
            f"{row['seconds']:>12.0f}  {'yes' if row['fits_cap'] else 'no'}"
        )
    lines.append("")
    lines.append(f"first round total: {budget['first_round_seconds']:.0f} s")
    lines.append(
        f"largest score_tasks whose initialize fits the cap: "
        f"{budget['largest_score_tasks_under_cap']}"
    )
    lines.append(
        f"updates of the shortest query that fit the cap: {budget['updates_under_cap']} "
        f"(a block needs {block['updates']})"
    )
    return "\n".join(lines)


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        rounds=args.rounds,
        patience=args.patience,
        updates=args.updates,
        operations_per_stage=args.topology_operations_per_stage,
        screen_tasks=args.screen_tasks,
        score_tasks=args.score_tasks,
    )


def _costs_from_args(args: argparse.Namespace) -> dict[str, float]:
    costs = dict(PROFILES[args.profile])
    for name in costs:
        override = getattr(args, name)
        if override is not None:
            costs[name] = float(override)
    return costs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive the advancing-event budget of an Example 21 evolve run, or "
            "launch a scoped H01 run whose receipt carries that budget."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("budget", "evolve"),
        default="budget",
        help="print the dry-run budget (default) or launch a scoped H01 evolve run",
    )
    parser.add_argument("--arc-root", type=Path, required=True, help="raw ARC root")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--topology-operations-per-stage", type=int, default=1)
    parser.add_argument(
        "--screen-tasks",
        type=int,
        default=DEFAULT_SCREEN_TASKS,
        help="training tasks screened between structural operations (default: 64)",
    )
    parser.add_argument(
        "--score-tasks",
        type=int,
        default=DEFAULT_SCORE_TASKS,
        help=(
            "leading training tasks every complete-scope score covers; "
            "0 scores the complete corpus (default: 0)"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="pinned",
        help="measured per-event costs to start from (default: pinned dt)",
    )
    parser.add_argument("--forward-seconds-per-event", type=float, default=None)
    parser.add_argument("--learner-seconds-per-event", type=float, default=None)
    parser.add_argument("--fixed-seconds", type=float, default=None)
    parser.add_argument("--cap-seconds", type=float, default=DEFAULT_CAP_SECONDS)
    parser.add_argument("--json", action="store_true", help="print the budget as JSON")
    parser.add_argument("--output-dir", type=Path, help="evolve: durable run directory")
    parser.add_argument("--h01-manifest", type=Path, help="evolve: pinned H01 manifest")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    return parser


def _evolve_scoped(args: argparse.Namespace, config: PipelineConfig) -> dict[str, Any]:
    """Launch one scoped H01 evolve run through the unchanged coordinator."""

    import brainstate
    import jax

    from examples.pp_prop.example21_evolve import ConsoleProgressReporter, run_evolution
    from examples.pp_prop.h01_arc_adapter import H01ArcAdapter

    if args.output_dir is None or args.h01_manifest is None:
        raise ValueError(
            "A scoped evolve run requires --output-dir and --h01-manifest; pass both."
        )
    try:
        device = jax.devices(args.device)[0]
    except RuntimeError as error:
        raise ValueError(
            f"Requested device {args.device!r} is unavailable; select an available device."
        ) from error
    with jax.default_device(device), brainstate.environ.context(precision=64):
        state = run_evolution(
            H01ArcAdapter(args.arc_root, args.h01_manifest),
            args.output_dir,
            config=config,
            progress_reporter=ConsoleProgressReporter(),
        )
    report = {
        "mode": "evolve",
        "passed": bool(state.closed and state.evaluation_completed),
        "closed": bool(state.closed),
        "terminal_reason": state.terminal_reason,
        "round_index": int(state.round_index),
        "training_exact_tasks": int(state.accepted.score.exact_count),
        "training_task_count": len(state.accepted.score.task_ids),
        "score_tasks": config.score_tasks,
        "checkpoint": state.accepted.checkpoint_path,
        "checkpoint_sha256": state.accepted.checkpoint_sha256,
        "evaluation_digest": state.evaluation_digest,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "example21-evolve.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the event-budget command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse; ``None`` reads the process command line.

    Returns
    -------
    int
        Zero when the budget was printed or the launched run closed.
    """

    args = _parser().parse_args(argv)
    try:
        config = _config_from_args(args)
        if args.command == "evolve":
            report = _evolve_scoped(args, config)
            print(json.dumps(report, sort_keys=True))
            return 0 if report["passed"] else 1
        manifest, queries = corpus_query_events(args.arc_root)
        budget = event_budget(
            manifest,
            queries,
            config,
            costs=_costs_from_args(args),
            cap_seconds=args.cap_seconds,
        )
    except ValueError as error:
        _parser().error(str(error))
        return 2
    print(
        json.dumps(budget, sort_keys=True, indent=2)
        if args.json
        else render_budget(budget)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
