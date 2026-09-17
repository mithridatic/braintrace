# Example 21 H01 backend: the event budget of an evolve run (2026-09-16)

Question: where the advancing ARC events of an evolve run come from, stage by stage, and
what a capped run on the 17 kept cells can complete once the scoring scope is a configured,
receipted quantity. Companion to the per-event profile
(`campaign/h01-evolve-profile-20260916`, `docs/evidence/h01-evolve-event-profile.md/.json`),
whose measured seconds per event are used here unchanged. Every event count below is
computed from the code path the run executes (`arc_contracts.encode_episode` over
`var/arc-agi-1`, `build_update_schedule` over the training manifest) by
`examples/pp_prop/example21_event_budget.py`; the four JSON receipts under
[h01-event-budget/](h01-event-budget/) are its `--json` output. Spec:
`docs/specs/2026-09-16-example21-event-budget.md`.

## Result in one line

An advancing event is one non-zero row of the 705-row episode encoding, and a query with `D`
demonstration pairs advances exactly `65 + 64 D` events (grid size plays no part); the
training corpus is 416 queries with `sum(D) = 1,392`, hence 116,128 events per complete
scoring pass, and a 128-update block is the next 128 queries of the manifest order (35,648
events from cursor 0, mean 278.5). The scope of every scoring pass is now
`PipelineConfig.score_tasks` (receipted in `run-state.json`); the block is not scoped, because
`updates` is pinned at 128 by two release gates, so no task subset brings one block under
3,600 s: the block floor is 128 x 193 = 24,704 events = 33,460 s at 1.34 s/event and
326,105 s at 13.2 s/event. What a 3,600 s cap admits is the initial scoring pass of at most 8
tasks (pinned) or 56 tasks (dt 0.005), and 1 or 12 updates of the shortest query.

## 1. Where one event comes from (`examples/pp_prop/arc_contracts.py`)

`encode_episode(task, query_index)` emits a fixed 705 x 441 Boolean event matrix
(`EVENTS = 705`, `EVENT_WIDTH = 441`) and the advance mask `np.any(encoded, axis=1)`: an
event advances the model when any bit is set, and the learner skips the others under
`cond` (the "padding events" of the profile). The 705 rows are:

| Rows | Source | Advancing? |
| ---: | --- | --- |
| 1 | episode header `_event(0, ...)` | yes |
| 10 x 64 | demonstration slots 0-9: a present pair is two `_grid_events` of 32 rows each (header, 30 row events, footer); an absent slot is `np.zeros((64, 441))` | present: all 64 (row events past the grid height still carry the kind/role/slot/height/width/row one-hots); absent: none |
| 32 | query grid `_grid_events(query, 2, 0)` | yes |
| 2 | request markers `_event(4)`, `_event(5)` | yes |
| 30 | answer rows `_event(6, ..., row)` | yes |

So advancing events per query `= 1 + 64 D + 32 + 2 + 30 = 65 + 64 D`, padding
`= 64 (10 - D)`, for `D` demonstration pairs (`MAX_DEMONSTRATIONS = 10`). The budget module
asserts this against every real encoder output it counts.

## 2. The corpus (`var/arc-agi-1`, 400 training tasks, 416 queries)

Demonstration pairs per task (400 tasks): 2 x 56, 3 x 237, 4 x 78, 5 x 18, 6 x 5, 7 x 3, 8 x 2,
10 x 1. Queries per task: 1 x 386, 2 x 12, 3 x 2 (= 416 queries). Per query:
2 x 57, 3 x 240, 4 x 80, 5 x 19, 6 x 7, 7 x 7, 8 x 4, 10 x 2, `sum(D) = 1,392`.

| Quantity | Derivation | Value | Profile JSON `corpus` |
| --- | --- | ---: | ---: |
| complete scoring pass | `416 x 65 + 64 x 1,392` | 116,128 | `corpus_advancing_events` 116,128 |
| padding per pass | `416 x 705 - 116,128` | 177,152 | `corpus_padding_events` 177,152 |
| min / mean / max per query | `D = 2 / 3.346 / 10` | 193 / 279.15 / 705 | 193 / 279.15 / 705 |
| block from cursor 0 | first 128 entries of `query_order` (122 distinct tasks, `sum(D) = 427`): `128 x 65 + 64 x 427` | 35,648 (278.5 per update) | `block_advancing_events` 35,648 |
| blocks at cursors 128 / 256 / 384 | next windows of the manifest order (384 wraps at 416) | 35,712 / 35,136 / 36,736 | not profiled |
| `--screen-tasks 2` pass | `007bbfb7` (D = 5), `00d62c1b` (D = 5): `2 x 65 + 64 x 10` | 770 | `screen_advancing_events` 770 |
| block floor | `128 x 193` (every update the shortest query) | 24,704 | not profiled |

The 278.5 per update is therefore the mean demonstration count of the first 128 scheduled
queries (3.336) and nothing else: not grid cells, not tokens, not the number of cells in
the network.

## 3. Where the events are spent (`examples/pp_prop/example21_evolve.py`, the adapters)

`build_update_schedule(manifest, cursor, updates)` takes entries
`manifest.query_order[(cursor + k) % 416]` for `k < updates`; every model stage advances
`cursor` by `config.updates` (`_expected_model_successor`), whether or not its arms ran, so
consecutive stages train on consecutive windows of the manifest order. Scoring scope, as the
code stood before this change:

| Stage (round 0 order) | Coordinator call | Scope before | Scope now |
| --- | --- | --- | --- |
| `initialize` | `adapter.initialize(config, output_dir)` -> `_write_runtime(task_ids=None)` | all 400 tasks | `score_tasks` prefix |
| `train` | `_run_parent_training` -> `train_parent(parent, schedule, context)`; `_stage_context` gave `score_task_ids=()`; both adapters called `_write_runtime` without `task_ids` | 128 updates + all 400 tasks | 128 updates + `score_tasks` prefix |
| `round-screen` | `_run_rescore_stage` -> `rescore`, `_rescore_scope_ids` = `screen_task_ids` | screen | screen (within the scope) |
| `edge`, `neuron`, `edge-revisit`, `dale` (2 arms each) | `_run_sibling_stage` -> `run_candidate(..., context)`, `context.score_task_ids` = screen | 128 updates + screen per arm (`dale` is blocked on H01 before any event) | unchanged |
| `round-score` | `rescore`, `_rescore_scope_ids` = `manifest.task_ids` | all 400 tasks | `score_tasks` prefix |
| `terminal-evaluation` | `evaluate_terminal(candidate, evaluation_manifest)` | 400 evaluation tasks | unchanged (not scoped) |

The 128-update pin is enforced twice and is not touched: `PipelineConfig.__post_init__`
("requires exactly 128 updates per block") and `H01ArcAdapter._train_scheduled`
("Production H01 requires the shared 128-update schedule"); `CandidateAttempt.completed`
requires `executed_updates == 128`. `--screen-tasks` never reached `initialize` or
`train_parent` because `_stage_context` screens `OPERATION_STAGES` only and both adapters'
`train_parent` scored the complete manifest; the 60-minute run
(`docs/evidence/h01-keep-drop/evolve/evolve-record.json`, `--screen-tasks 2 --rounds 1`)
therefore died inside the 116,128-event initial pass, as the profile found.

## 4. Seconds per stage (per-event costs from the profile JSON, arms `pinned` and `dt0005`)

Scoring passes are priced at the forward rate, update blocks at the learner rate, and every
stage pays the fixed build + init + compile cost once (`fixed_seconds`; each stage rebuilds
its runtime from a checkpoint). Configuration of the capped run
(`--screen-tasks 2 --rounds 1 --patience 1`), from
[capped-run-pinned.json](h01-event-budget/capped-run-pinned.json) and
[capped-run-dt0005.json](h01-event-budget/capped-run-dt0005.json):

| Stage | updates | update events | scored queries | scoring events | pinned s (1.357 fwd / 13.18 learner / 497 fixed) | dt 0.005 s (0.180 / 1.338 / 418) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `initialize` | 0 | 0 | 416 | 116,128 | 158,052 | 21,282 |
| `train` | 128 | 35,648 | 416 | 116,128 | 627,906 | 68,962 |
| `round-screen` | 0 | 0 | 2 | 770 | 1,542 | 556 |
| `edge` arm (x2) | 128 | 35,712 | 2 | 770 | 472,240 | 48,322 |
| `neuron` arm (x2) | 128 | 35,136 | 2 | 770 | 464,648 | 47,552 |
| `round-score` | 0 | 0 | 416 | 116,128 | 158,052 | 21,282 |
| first round | 640 | 177,344 | | | 2,819,328 (783 h) | 303,830 (84 h) |

`initialize` at the pinned rate is the profile's 157,555 s corpus pass plus 497 s fixed;
`train` is that plus the profile's 469,854 s block. The same configuration with
`--score-tasks 8` ([score8-pinned.json](h01-event-budget/score8-pinned.json),
[score8-dt0005.json](h01-event-budget/score8-dt0005.json)):

| Stage | scored queries | scoring events | total events | pinned s | dt 0.005 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `initialize` | 8 | 2,248 | 2,248 | 3,547 (fits 3,600) | 822 |
| `train` | 8 | 2,248 | 37,896 | 473,401 | 48,502 |
| `round-screen` | 2 | 770 | 770 | 1,542 | 556 |
| `edge` arm (x2) | 2 | 770 | 36,482 | 472,240 | 48,322 |
| `neuron` arm (x2) | 2 | 770 | 35,906 | 464,648 | 47,552 |
| `round-score` | 8 | 2,248 | 2,248 | 3,547 | 822 |

Initial scoring pass against the 3,600 s cap, by scope (fixed cost included):

| `score_tasks` | queries | events | pinned s | dt 0.005 s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 385 | 1,020 | 487 |
| 2 | 2 | 770 | 1,542 | 556 |
| 4 | 4 | 1,220 | 2,152 | 637 |
| 8 | 8 | 2,248 | 3,547 | 822 |
| 9 | 9 | 2,505 | 3,896 | 868 |
| 16 | 16 | 4,368 | 6,423 | 1,203 |
| 56 | 60 | 17,468 | 24,196 | 3,556 |
| 57 | 61 | 17,725 | 24,545 | 3,603 |
| 400 | 416 | 116,128 | 158,052 | 21,282 |

## 5. What a 3,600 s cap admits

- One full 128-update block: no task subset. The block does not depend on the scope (the
  schedule walks the manifest order, so it is 35,648 events from cursor 0 for every
  `score_tasks`), and even a hypothetical block of nothing but the shortest queries is
  24,704 events: 33,460 s at 1.34 s/event, 326,105 s at 13.2 s/event (fixed cost included).
  A block under the cap needs `updates <= 12` at 1.34 s/event (12 x 193 x 1.3375 + 418 =
  3,516 s; 8 updates at the 278.5-event mean) or `updates <= 1` at 13.2 s/event (193 x 13.18
  + 497 = 3,041 s; the mean query, 278.5 x 13.18 = 3,671 s, does not fit at all). That lever is
  the pinned `updates = 128`, a release-gate quantity this change leaves alone; the dry run
  prints the admitted update count so the decision is explicit.
- The initial scoring pass: `score_tasks <= 8` at the pinned dt (3,547 s), `<= 56` at dt 0.005
  (3,556 s), with `--screen-tasks` below `score_tasks` (`PipelineConfig` rejects a screen
  that is not a proper subset of the scope, so the default screen of 64 cannot silently
  turn a small scope into an unscreened lineage with a different stage order).
- A screened operation arm's scoring (770 events, 1,542 s / 556 s) fits; its block does not.

## 6. What changed (worktree `campaign/h01-event-budget-20260916`)

- `examples/pp_prop/example21_evolve.py`: `PipelineConfig.score_tasks` (default 0 = complete
  corpus; emitted by `to_dict()` only when nonzero, so every existing `config_sha256` is
  unchanged); `score_scope_ids(manifest, config)`; `screen_task_ids` screens within the
  scope; the complete-scope comparisons in `RunState.__post_init__`, the initial
  `_require_candidate`, `_is_full_score`, `_allowed_score_scopes`, `_rescore_scope_ids` and
  the progress `score_scope` label use the scope; `_stage_context` hands the scope to
  unscreened stages of a scoped lineage.
- `examples/pp_prop/example21_arc_adapter.py`: `initialize` scores the scope derived from
  the config; `train_parent` scores `context.score_task_ids`.
- `examples/pp_prop/h01_arc_adapter.py`: `train_parent` scores `context.score_task_ids`.
- `examples/pp_prop/example21_event_budget.py` (+ `_test.py`): the dry run
  (`budget`, default) and a launcher (`evolve`) for scoped H01 runs, so the pinned entry
  point `examples/pp_prop/21-braincell-arc.py` (in `h01_session.py`
  `implementation_sha256`) is untouched and the committed ARC manifest stays valid.
- No pinned H01 runtime source was edited.

Receipts: a scoped run's `run-state.json` carries `"score_tasks": N` in `config` and in
`config_sha256`; its `accepted.score.task_ids` has `N` entries; the launcher's
`example21-evolve.json` carries `score_tasks` and `training_task_count`. A run without the
key is a complete-corpus run.

Commands (laptop, `PYTHONPATH=.`):

```
python -m examples.pp_prop.example21_event_budget --arc-root var/arc-agi-1 \
    --screen-tasks 2 --rounds 1 --patience 1 --score-tasks 8 [--profile dt0005] [--json]
python -m examples.pp_prop.example21_event_budget evolve --arc-root var/arc-agi-1 \
    --h01-manifest docs/evidence/h01-keep-drop/arc-manifest/manifest.json \
    --output-dir var/h01-driven/scoped-evolve --device gpu \
    --rounds 1 --patience 1 --screen-tasks 2 --score-tasks 8
```

## Caveats

- Seconds are the profile's numbers multiplied out; the profile's own caveats apply
  (pinned arm +-25 percent under CPU contention; the dt 0.005 learner slope 1.34 is 16 percent
  below the real 705-event update's 1.595 s per advancing event, which would make the dt 0.005
  block 57,000 s rather than 48,000 s).
- The fixed cost is charged once per stage; a stage with two arms rebuilds twice, which the
  table shows as two rows.
- The terminal evaluation (400 evaluation tasks, not scoped) is not in the tables; it is
  only reached after the configured rounds close.
- `Example21ArcAdapter._score_runtime` chooses its scoring bucket lengths by whether the
  scored set is the whole manifest, so a scoped complete-scope pass compiles the
  screen-sized program rather than the 416-query one; the H01 adapter's `score_queries`
  stacks whatever queries it is given. The fixed cost charged per stage (497 s / 418 s) was
  measured for the unscoped programs and is carried unchanged.
- Byte-identity check: the one current-schema local receipt
  (`var/example21-evolve-ops8/run-state.json`) reproduces its recorded `config_sha256`
  through the new `PipelineConfig.from_dict`, exactly as under the previous coordinator.
