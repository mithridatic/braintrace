# Example 21 event budget: configured scope and dry-run receipt (2026-09-16)

## Problem

An Example 21 evolve run on the H01 backend (17 full-morphology human cells) is
bounded by ARC event volume, not by the cells. The per-event profile
(`campaign/h01-evolve-profile-20260916`, `docs/evidence/h01-evolve-event-profile.md`)
measured 13.2 s per learner event at the pinned dt (1.34 s at dt 0.005) and
1.357 s per forward event; `initialize()` scores all 400 training tasks
(116,128 advancing events) before the first block exists, and `train_parent()`
scores all 400 again after its 128 updates. Neither pass is reachable from the
only existing scope control, `--screen-tasks`, which `_stage_context` applies to
operation stages only.

## Requirements

1. The scoring scope of the whole lineage (initial score, trained parent,
   `round-score`, mastery) is a configured field, persisted in every receipt
   (`run-state.json` config and digest, pending and progress documents).
2. Default behaviour is byte-identical: the config digest of an existing run
   must not change, every existing test passes unchanged.
3. A dry run derives the advancing-event count per stage from the corpus and the
   configuration, multiplies by supplied per-event costs, and reports whether
   each stage fits a cap; a test pins the corpus numbers.
4. Pinned H01 runtime sources (`h01_session.py` `implementation_sha256`,
   which includes `examples/pp_prop/21-braincell-arc.py`) are not edited.

## Design

- `PipelineConfig.score_tasks: int = 0`. `0` is the complete corpus. A nonzero
  value names the leading `score_tasks` manifest tasks as the lineage's
  complete scope. `to_dict()` emits the key only when nonzero, so existing
  digests are unchanged and a scoped run's receipt always carries the key.
- `score_scope_ids(manifest, config)` returns that scope; every coordinator site
  that compared a score to `training_manifest.task_ids` compares to the scope
  instead (`RunState.__post_init__`, initial `_require_candidate`,
  `_is_full_score`, `_allowed_score_scopes`, `_rescore_scope_ids`, the progress
  `score_scope` label). `screen_task_ids` screens within the scope and is empty
  when the screen is not a proper subset of it.
- `_stage_context` passes the scope as `score_task_ids` to unscreened stages of
  a scoped lineage (empty, as before, for an unscoped lineage). The base and
  H01 adapters forward `context.score_task_ids` from `train_parent`, and
  `initialize` derives the scope from the config.
- The 128-update block is unchanged: `updates` stays pinned at 128 by
  `PipelineConfig` and by `H01ArcAdapter._train_scheduled`, and the schedule
  still walks the full manifest query order. Scoping shrinks the scoring
  passes; it does not shrink a block.
- `examples/pp_prop/example21_event_budget.py`: `budget` (default) prints the
  per-stage event counts and seconds for a configuration and per-event costs;
  `evolve` launches a scoped H01 run through the unchanged coordinator and
  adapter, writing the same `example21-evolve.json` report plus `score_tasks`.
  This keeps the pinned entry point untouched.

## Non-goals

- Reducing the cells, the dt, or the substeps.
- Making `updates` configurable below 128 (a release-gate change; the budget
  tool reports what update count a cap would need so that decision is explicit).
- Scoping the terminal evaluation corpus.
