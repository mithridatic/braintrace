# C3 test-to-failure gate: 40 candidates and the 17 kept cells (step C)

Spec: [2026-09-16-h01-c3-partner-expansion.md](../../specs/2026-09-16-h01-c3-partner-expansion.md),
section C. Register: `qc3_e`, `qc3_i` (search tree), `io_gate`, `datum_fire`, `datum_rest`
(campaign tree) under [h01-reasoning/](../h01-reasoning/). Executor: Vast 50616476, worktree
`/workspace/braintrace-c3-keep` on branch `campaign/h01-c3-c3-keep-20260916`; receipts under
`var/c3-keep/` there. Nothing below has run yet: this page is the preparation record.

## Gate

`h01_anatomy_transfer_decision.py --keep --gate failure --datums human-datums.json`. Rules
`finite / rheobase_in_step / fires_at_highest / count_in_repeat_range / rest_in_donor_spread /
no_spike_before_pulse / dt_half_reproduces`; datums and tolerances in
[human-datums.json](human-datums.json) (LJP-corrected long-square families of the four donor
recordings). The former 10 mV / 30 percent bands are written beside every verdict as
`legacy_verdict` and are not the gate. A cell whose ramp or dt-half repeat has not run is
`pending` (its unmeasured rules `None`), never kept and never counted as dropped; a measured
rule that fails drops the cell whether or not the ramp has run.

## Inputs

| File | Content |
| --- | --- |
| [types.json](types.json) | The 40 candidates in the `h01-population-types.json` shape: donor as assigned by `h01-c3-candidates.json` (19 PV, 1 SST, 18 L2-donor, 2 L4-donor; `donor_match` recorded, 20 layer-only), `source: c3 non-proofread`, ordered interneurons first then pyramidal in rank order. Written by `h01_c3_keep_types.py`. |
| [kept-types.json](kept-types.json) | The 17 kept cells (13 L2 donor, 4 L4 donor) from the population types and the keep/drop decision; proofread archive. |
| [chains/keep-chain-1.sh](chains/keep-chain-1.sh), [chains/keep-chain-2.sh](chains/keep-chain-2.sh) | The two chains from `h01_c3_keep_chain.py`: per candidate the donor step run then the ramp (0 to 3x the donor input, `--ramp-na`), per kept cell the ramp only; 49 + 48 = 97 runs before dt-half. Candidate lines carry `--archive .cache/h01/c3-candidates-20260916.zip --archive-sha256 75d15798... --components h01-c3-candidates-components.json --cell-table .cache/h01/c3-segment-properties.json --cell-table-sha256 6178bfe1...`; kept lines use the proofread defaults. |
| [chains/follower.sh](chains/follower.sh) | `h01_keep_drop_follower.py --gate failure` for the candidates: launches the dt-half repeat of every candidate whose step run and ramp hold the measured rules; same extra arguments as the chain lines. |
| [chains/dry-run.json](chains/dry-run.json) | Run counts and the first commands of each chain. Estimate: 97 x 190 s = 18,430 s (5.1 h) before dt-half, where 190 s is the aggregate per-run cost measured in the 2026-09-16 keep/drop campaign (launch-bound, independent of chain count); it is an estimate from that campaign, not a measurement of these runs. |
| [run_transfer.sh](run_transfer.sh) | Receipted launcher for the c3-keep worktree (`ROOT=/workspace/braintrace-c3-keep`, receipts `var/c3-keep/<label>/{launch,terminal}.json`, `run.{log,err,json,npz}`); refuses `/workspace/braintrace`. |
| [kept-gate-preflight.json](kept-gate-preflight.json) | The failure gate applied offline to the 17 kept cells' existing keep/drop receipts (primary and dt-half; ramp pending). Not a decision. See below. |

## Preflight reading on the 17 kept cells (no ramp yet)

`--folder ../h01-keep-drop/runs --ramp-folder runs` on [kept-types.json](kept-types.json):
0 kept, 17 dropped, 0 pending. The ramp rules report `None` (`pending: ramp_missing` on every
cell), but every kept cell already fails a measured rule of the new gate:

- `count_in_repeat_range` fails on 16 of 17: the L2 and L4 donors have one registered sweep at
  the primary input (`repeat_counts` `[10]` and `[12]`), so the repeat range is a single value
  and the rule is exact equality; kept counts are 7-11 (L2) and 9, 11, 13, 15 (L4). Only
  4437316933 (count 10) holds.
- `rest_in_donor_spread` fails on 15 of 17, on the return leg: the pre-pulse rest sits inside
  the band on most cells (L2 datum -84.01 +- 0.92 mV, L4 -80.60 +- 1.42 mV) but the level
  200 ms after the pulse is 3.2 to 9.5 mV below the datum on every L2 cell (-87.2 to -93.5 mV) and
  1.2 to 1.6 mV below on the L4 cells; 2001418787 and 4010150634 (L4) hold.

No tolerance is widened here (rule: none without a named human datum and its repeat spread).
The reading is reported to J with the question whether the 17 kept-cell ramps still earn box
time when no kept cell can pass the gate as registered.

## Launch recipe (only on "GO")

```text
ssh braintrace-gpu
cd /workspace/braintrace-c3-keep && git pull --ff-only
chmod +x docs/evidence/h01-c3-keep-drop/run_transfer.sh docs/evidence/h01-c3-keep-drop/chains/*.sh
mkdir -p var/c3-keep
nohup bash docs/evidence/h01-c3-keep-drop/chains/keep-chain-1.sh > var/c3-keep/keep-chain-1.log 2>&1 &
nohup bash docs/evidence/h01-c3-keep-drop/chains/keep-chain-2.sh > var/c3-keep/keep-chain-2.log 2>&1 &
nohup bash docs/evidence/h01-c3-keep-drop/chains/follower.sh > var/c3-keep/follower.log 2>&1 &
```

Two chains, nothing else on the box while they run; poll `var/c3-keep/*/terminal.json` with a
Monitor; kill by PID only. Record the first chain's timing before judging the estimate. After
the runs: `h01_keep_drop_receipts.py --folder var/c3-keep --out-dir docs/evidence/h01-c3-keep-drop/runs`
(JSON receipts committed, decimated `trace.npz` unversioned), then the decision:

```text
h01_anatomy_transfer_decision.py --keep --gate failure --folder var/c3-keep --types docs/evidence/h01-c3-keep-drop/types.json --output docs/evidence/h01-c3-keep-drop/decision.json
h01_anatomy_transfer_decision.py --keep --gate failure --folder docs/evidence/h01-keep-drop/runs --ramp-folder docs/evidence/h01-c3-keep-drop/runs --types docs/evidence/h01-c3-keep-drop/kept-types.json --output docs/evidence/h01-c3-keep-drop/kept-decision.json
```

Each decision is committed with the `docs/h01-causal-model.md` update (Y7) and the register
closure in one commit.
