# H01 ARC evolution machinery run at 104 cells

Date: 2026-09-13
Status: approved design; implementation follows the companion plan.

## Goal

Prove that the complete H01 evolution lifecycle executes on the real 104-cell
source population: every executable structural arm mutates the topology,
rebuilds the cables, learns on real encoded ARC episodes, scores, and is
checkpointed with lineage; and the topology visual exports. ARC accuracy is
recorded but is not an acceptance criterion. Speed is a later objective; this
run establishes that the machinery works and measures what it costs.

## Population and environment

- Population: all 104 selected source cells from
  `docs/evidence/h01-arc-probe-104.json` (status `forward_pass`, no nonfinite
  states) with the two constructible anatomical contacts from
  `docs/evidence/h01-verified-network.json`.
- Execution host: the Vast.ai box `braintrace-gpu` (RTX 4090 24 GB, 64 GB host
  RAM share, H01 archive cache at `/h01`, evidence directory `/evidence`,
  interpreter `/workspace/venv314/bin/python`). The box runs a pinned commit
  pulled over git; nothing runs from a dirty tree. The ARC corpus is gitignored
  and is re-fetched on the box; the raw root must contain exactly 400 practice
  task files.
- Worktree: this worktree and branch are shared with the fit-to-human session.
  Only the paths named below are staged (`git add <path>`, never `-A`).

## Step 0: source manifest

No `h01-arc-manifest-v1` document exists yet. Generate it on the box with
`examples/h01_arc_manifest.py --archive /h01/<archive> --output
docs/evidence/h01-arc-manifest/manifest.json`. The content-addressed morphology
asset is written beside it under `assets/` and stays on the box (gitignored
size); the manifest JSON is committed. The adapter refuses to start unless the
manifest has exactly 104 active cells.

## Step 1: the missing measurement

Run `examples/h01_arc_episode_probe.py` with the manifest at 104 cells for task
`025d127b`, query 0 (193 encoded events): one real encoded episode, one
pp-prop update, direct scoring before and after, save, restore, repeated score.
The wall limit is raised above 900 s under the user's authorization of learning
updates longer than 15 minutes; the RSS limit is raised to 48 GiB. Recorded
outputs: warm per-event cost, first-update cost including compilation, warm
update cost, peak host RSS, peak live device bytes, loss before and after.

These are the first measured 104-cell ARC learning numbers. They cost Step 3;
no duration for Step 3 is quoted before they exist. Step 1 needs no code
change and starts on the box while Step 2 is written locally.

## Step 2: two relaxations

Both are behaviour-preserving for the existing BrainCell backend.

1. `H01ArcAdapter._train_scheduled` accepts a training block whose length is
   the pipeline's configured `updates`, not the hardcoded 128. The
   `executed_updates` value, the recovery cursor bound, and the "cursor below
   block" resume branch all derive from the schedule length. A block whose
   entry count disagrees with its cursor span is still rejected.
2. `PipelineConfig.score_tasks` (integer, default 0 meaning the complete
   corpus) bounds the task subset used by the initial direct score, by
   `round-score`, and by terminal evaluation, in manifest order, the same way
   `screen_tasks` already bounds operation screening. The evolve command
   exposes it as `--score-tasks`. The rescore evidence verifier and the
   lineage records use the bounded scope so that a bounded run still closes.
   The value is recorded in the run configuration so a resume with a
   different value is rejected like any other configuration change.

Each change has co-located tests: a small-fixture H01 block of length 2
completes and reports `executed_updates=2`; a mismatched block is rejected;
`score_tasks` narrows the three score scopes and is validated between 0 and
400; default 0 reproduces the current full-corpus order.

## Step 3: the machinery run

```
21-braincell-arc.py evolve --model-backend h01 --h01-manifest <manifest>
  --arc-root <root> --output-dir /evidence/h01-arc-evolve-104 --device gpu
  --rounds 1 --patience 1 --updates 2 --screen-tasks 2 --score-tasks 2
  --topology-operations-per-stage 1
```

Expected arm ledger for round 1:

| Stage | Arms | Expected outcome |
| --- | --- | --- |
| train | training | completed, 2 episodes |
| edge | add, prune | completed; add creates a synthetic contact, prune removes the lowest-scored contact |
| neuron | add, prune | completed; add clones the top-ranked cell (synthetic instance), prune retires the lowest-ranked |
| edge-revisit | add, prune | completed |
| dale | excitatory, inhibitory | blocked: "Inherited H01 E/I identities cannot change" (by design; polarity is a source property) |
| compression-edge | prune | completed |
| compression-neuron | prune | completed |
| round-screen, round-score | rescore | completed on the bounded scope |
| terminal-evaluation | — | completed on the bounded evaluation scope |

Every mutation releases the parent runtime and rebuilds the full 104-cell
network (measured construction and initialization together are on the order
of twenty minutes before compilation); this is the deferred speed work. The
run is launched detached with a log and resumes from durable checkpoints if
interrupted. `edge-prune` may legitimately report `blocked` if no active
contact remains after earlier prunes; `edge-add` may report `blocked` if no
absent directed pair exists. Both are recorded, not treated as failures.

## Evidence and acceptance

`docs/evidence/h01-arc-evolve-104.md` with raw JSON and the exported PNGs:

- per-arm outcome table (completed / blocked / failed, with the literal reason
  for blocked and failed);
- topology mutation ledger from the accepted checkpoint: operation, stage,
  lineage of synthetic instances and contacts, active counts before and after;
- `topology.png` (clones as squares, synthetic contacts dashed, E blue, I red)
  and `score-history.png`;
- per-stage wall time and peak memory; the Step 1 measurement table;
- restore verification of the accepted checkpoint.

Pass: every non-Dale arm reaches `completed` (or a documented `blocked`
reason listed above), the round closes, terminal evaluation completes, and the
PNGs export. Any `failed` arm is reported verbatim with its traceback and the
run is reported as not passing. ARC exact counts are reported as measured.

## Out of scope

Speeding up construction, initialization, compilation or per-event cost;
numerical qualification at 0.005/0.0025 ms; physiological qualification;
any ARC score threshold; identity-changing Dale mutations; morphological
(anatomical-shape) rendering of the population.
