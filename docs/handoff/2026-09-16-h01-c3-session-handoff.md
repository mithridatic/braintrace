# Handoff 2026-09-16 (late): H01 C3 partner campaign, stopped mid-run

Everything was stopped on J's instruction. This file is the resume point. Nothing here is a
result until the follower's final pass and the decision JSON are written and committed.

## Where things are

- **Git.** `main` = `9e84bd01` (laptop and box). It is the merge of every campaign branch:
  `campaign/h01-c3-c3-keep-20260916` (contains WS-A graph, WS-B skeletons, WS-D code, the
  partner spec) → `campaign/h01-event-budget-20260916` → `campaign/h01-fused-population-20260916`
  (contains `campaign/h01-evolve-profile-20260916`). No conflicts. All worktrees removed on both
  machines; local campaign branches deleted; the remote branches still exist on GitHub (merged).
- **Vast box 50616476** (`ssh braintrace-gpu`): **stopped** (`vastai start instance 50616476`
  to resume; disk kept). `/workspace/braintrace` is on `main` `9e84bd01`. The other running
  instance `50935717` (`comsol-tr2`) is not this project's and was not touched.
- **Receipts preserved on the box** under `/workspace/archive-20260916/` (7.6 GB, gitignored
  material moved out of the deleted worktrees):
  - `var-c3-keep/` — the whole keep campaign: `keep-primary-decision.json` (rolling status),
    `transfer-all-<cell>{,-ramp,-dthalf}/` (launch.json, run.json, run.npz, run.log/err,
    terminal.json, time.txt), chain/follower logs, `launched-utc.txt`.
  - `var-fused/` — fused-population profile scratch (`dtstate/`, `dtwin/`, `gate5/`, gate5.sh,
    the parity/index scripts).
  - `var-c3-graph/`, `var-c3-skel/`, `var-profile/` — WS-A/WS-B/evolve-profile scratch.
  - C3 downloads now live in main's cache: `/workspace/braintrace/.cache/h01/`
    `c3-candidates-20260916.zip` (sha 75d15798…), `c3-segment-properties.json` (sha 6178bfe1…),
    `c3-graph/`, `c3-kept-graph/`, `c3-control-1684504313.zip`, `c3-pilot-20260916.zip`,
    `proofread-base-membership.json`.

## Keep test on the 40 C3 candidates — state at stop

Gate: `failure` (hard edges `finite / no_spike_before_pulse / fires_at_highest /
no_block_in_recorded_range / recruitable_in_human_range / dt_half_reproduces`; plausibility
`count / rest / return _in_type_spread`, datum = donor value, tolerance = 2 sd of the Allen
human population of the donor's type). Spec: `docs/specs/2026-09-16-h01-c3-partner-expansion.md`;
evidence folder `docs/evidence/h01-c3-keep-drop/` (types, kept-types, human-datums,
kept-gate-preflight, chains, run_transfer.sh, README).

Runs: **40/40 primary terminal, 40/40 candidate ramps terminal, 22/22 dt-half repeats on every
holding candidate terminal.** Verdicts from the rolling `keep-primary-decision.json`
(preliminary — the follower's `--phase final` pass has NOT been run):

| | hold | dropped |
|---|---|---|
| I (20) | **13** | 7 |
| E (20) | **9** | 11 |

Held E: 2062261179, 36052250628, 693197378, 2468824745, 1669011958, 2950643092, 927076559,
2136098827, 2091491230.
Held I: 3504577656, 3518943222, 2440908874, 4524762575, 2193655685, 28949696072, 30406000355,
1539076840, 37218168369, 2207657768, 925441648, 28629556145, 2776001292.
Dropped: 14 × `recruitable_in_human_range` (model rheobase above the human's highest firing
amplitude: L2 355–565 pA vs 350; PV 216–294 vs 270), 2 × `fires_at_highest` (29938695074,
37421824682), 2702762854 `no_spike_before_pulse,count_in_type_spread`, 3126060193
`return_in_type_spread`.

Registered predictions: I 0–2 of 20 → **13 held** (no fit pilot needed); E 1–4 → 9.

## Kept-17 qualification — what is left

The 17 already-kept proofread cells get the same ramp (0 → 3× donor current) so both sets carry
one gate. Done: 9 of 17 ramps (1669770671, 1684504313, 2001418787, 2103991145, 2252715458,
2848552900, 3111823553, 3571083397, 3761379470). Readings so far: rheobase 0.1–0.2 nA, **no
block edge under 3× on any of them** (prediction "half the L2 cells show a block edge" is 0/7 so
far). Killed in flight (partial dirs exist, rerun them): 4010150634, 4197933517, 4437316933,
5013648003. Never started: 5173982155, 5439194879, 6833911543, 751294744.
Kept preflight (`kept-gate-preflight.json`): 16/17 hold before ramps; 5013648003 fails
`return_in_type_spread`.
dt-half repeats for the kept 17 were not queued under this gate (0/17); the original keep/drop
recorded 17/17 exact at the same driver dt — decide whether the follower needs them redone.

## To resume (all on the box, main, `/workspace/venv314`)

1. `vastai start instance 50616476`; `ssh braintrace-gpu`; `cd /workspace/braintrace`;
   `mkdir -p var && ln -s /workspace/archive-20260916/var-c3-keep var/c3-keep` (or `mv` it back).
2. Delete the four partial ramp dirs (`transfer-all-{4010150634,4197933517,4437316933,5013648003}-ramp`)
   so the chain re-queues them, then relaunch with the same commands the chains used (see
   `var/c3-keep/keep-chain-{1,2}.log` head and `docs/evidence/h01-c3-keep-drop/chains/`), two
   chains max, cpusets 0-31 / 32-63, `XLA_PYTHON_CLIENT_MEM_FRACTION=.2`. ~20–25 min per ramp
   under contention. 8 ramps left.
3. Follower final pass — the exact command that was running:
   `docs/evidence/h01_keep_drop_follower.py --folder var/c3-keep --types docs/evidence/h01-c3-keep-drop/types.json --gate failure --datums docs/evidence/h01-c3-keep-drop/human-datums.json --chains 2 --runner docs/evidence/h01-c3-keep-drop/run_transfer.sh --output var/c3-keep/keep-primary-decision.json --cpuset 64-95 --memfrac .12 --extra-args --archive .cache/h01/c3-candidates-20260916.zip --archive-sha256 75d15798e35cb1f111198f89c03e5210dec2cbd7f42f233783f46b1eb626faf7 --components docs/evidence/h01-c3-candidates-components.json --cell-table .cache/h01/c3-segment-properties.json --cell-table-sha256 6178bfe1c0a876001d02f641dda1d5af3b1d3512935b6114d4765e7988c44b3f`
   then `h01_anatomy_transfer_decision.py --phase final` → `docs/evidence/h01-c3-keep-drop/decision.json`
   with `rule_counts` for all four rule sets, receipts via `h01_keep_drop_receipts.py` into
   `docs/evidence/h01-c3-keep-drop/runs/`.
4. Same commit: `docs/h01-causal-model.md` entry + Reasoning Studio nodes `qc3_e`, `qc3_i`,
   `datum_fire`, `datum_rest` closed with evidence paths (`docs/evidence/h01-reasoning/`,
   `build_search_tree.py`, export and grep the SVG).
5. Then WS-D data half (plan in `.claude/plans/context-yes-and-fluffy-cosmos.md` and the spec):
   contacts among kept-17 ∪ 22 holds from the C3 graph (AXON-pre rows only —
   `h01-c3-neuron-graph-is-sparse`), delivery sweep per contact (I→E with the post cell held
   depolarised), probe evidence, two-archive manifest, capped evolve launch via
   `examples/pp_prop/example21_event_budget.py evolve`.

## Fused population (merged, usable)

`build_network(fused=True)` in `examples/pp_prop/h01_runtime.py`; `fused_population` default
False. Parity vs per-cell on the kept manifest: every conducting gate 2e-8, V 9e-7 mV, 33/33
spikes; launches 327 vs 5,104 per substep (17 cells); static slots → 0 recompiles on clone +
contact. Spec `docs/specs/2026-09-16-h01-fused-population.md`, evidence
`docs/evidence/h01-fused-population-profile/`. Not done: step 5 (uncontended 40 ms spiking-window
gate; the armed queue was killed, script kept at `archive-20260916/var-fused/gate5.sh`), step 6
(per-mechanism channel fusion; changes `implementation_sha256`), `H01Session.mutate` still
rebuilds from topology.

**Fused batch driver (planned, not started).** The per-cell driver
`docs/evidence/h01_anatomy_transfer_run.py` has no fused path, so fused does not speed a single
run; it batches N cells at ~one cell's launch cost. Plan: `docs/evidence/h01_fused_batch_run.py`
that builds one contact-free forest for a cell list, injects each cell's own step/ramp at the
soma, writes receipts in the per-cell layout (+ `method: fused`), parity-checked over the full
window against a finished per-cell `-ramp` and `-dthalf` receipt (spike count/times, every gate
reading, ΔV) before any batch counts. Use it for the dt-half tail and every later batch.

## Open item to settle before the manifest is built

`h01_anatomy_transfer_run.py` defaults `--dt-ms .005` and the dt-half repeats run at `.0025`
(seen in the live command lines). The Example 21 runtime dt is pinned at 0.000625 / 160
substeps and the fused evidence shows dt 0.005 errs 1.42–1.78× the dt-half error on every state
variable. The keep verdicts (counts, rheobase, block edges) were therefore measured at 0.005,
as the original 104-cell keep/drop was. Decide whether the kept/held sets need a re-read at the
pinned dt before they enter the manifest (a fused batch would make that cheap) — this is a
consistency question about the gate, not a proposal to change dt.

## Rules that bind the next session (unchanged)

Box only (laptop runs nothing); `PYTHONPATH=. JAX_PLATFORMS=cpu /workspace/venv314/bin/python -m pytest <module>_test.py -q -p no:cacheprovider`;
dt is settled; never dumb down a cell; causal model in the same commit as each decision;
Reasoning Studio is the register; never `git stash`; no attribution trailers; co-located tests;
kill by PID only; two GPU chains max; never quote unmeasured durations.
