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

## Datum correction before any run (2026-09-16)

The first preflight of the gate on the 17 kept cells' existing receipts dropped all 17 on
two rules. Both were datum-construction errors, not cell failures, and were corrected with
no tolerance widened (every tolerance is still read from the recording):

- `count_in_repeat_range` had compared the count with the single registered sweep at the
  primary amplitude (`[10]`, `[12]`), which carries no repeat spread. The datum is now
  `count_band`: min to max of the counts of every long-square sweep whose amplitude lies
  within one `sweep_step_pa` of `primary_pa` (inclusive, the primary's own repeats included),
  the same one-step-of-drive tolerance the rheobase rule uses. L2 290/310/330 pA -> [9, 12]
  (sweeps 52-54); L4 90/110 pA -> [12, 17] (39-40; the family has no 70 pA sweep, 60 pA is
  30 pA away); PV 170/190/210 pA -> [4, 23] (34-36); SST 90/100 x4/110 pA -> [12, 21]
  (32, 44-47, 33). `repeat_counts` and `measured_counts_at_primary` are unchanged.
- `rest_in_donor_spread` had scored the return leg (the 10 ms ending 200 ms after offset)
  against the pre-pulse datum. `h01_c3_human_datums.py` now reads each sweep's `after_mv`
  over exactly the runner's window (`h01_anatomy_transfer_run.windows`: `off+190 <= t <=
  off+200`, gated on the trace reaching `off+200`) and the family's `after_repeat_mean_mv` /
  `after_repeat_sd_mv`: L2 -84.94 (sd 0.96), L4 -81.03 (1.83), PV -87.18 (0.45), SST -79.18
  (2.27) mV; bands 2.87, 5.49, 1.34, 6.82 mV. Every sweep of all four families reaches the
  window (`after_sweeps_unreached` empty); a leg the recording does not reach would be
  recorded as not scored. The pre-pulse leg is unchanged.

`human-datums.json` was regenerated from the four NWB files (shas unchanged:
cc180b29..., e321fe93..., b6412208..., 218aa144...). The spec gate table, the register
nodes `datum_fire` / `datum_rest` and the causal model's qualification rules carry the
band construction.

## Preflight reading on the 17 kept cells with the corrected datums (no ramp yet)

`--folder ../h01-keep-drop/runs --ramp-folder runs` on [kept-types.json](kept-types.json)
([kept-gate-preflight.json](kept-gate-preflight.json), not a decision): 0 kept, 15 dropped,
2 pending on the ramp. Per rule: `finite` 17/17, `no_spike_before_pulse` 17/17,
`dt_half_reproduces` 17/17, `count_in_repeat_range` 7/17 (L2 5 of 13: counts 9, 10, 11 x3
inside [9, 12], eight cells at 7-8 outside; L4 2 of 4: 13 and 15 inside [12, 17], 9 and 11
outside), `rest_in_donor_spread` 5/17 (pre-pulse leg 15/17; return leg 6/17: every L4 cell
holds at -81.8 to -82.2 mV against -81.03 +- 5.49, two L2 cells hold at -87.2 and -87.7 mV
against -84.94 +- 2.87, the other eleven L2 cells return 3.6 to 8.6 mV below the datum at
-88.5 to -93.5 mV), `rheobase_in_step` and `fires_at_highest` unmeasured (17 pending on the
ramp). Cells holding every measured rule: 3761379470 and 5439194879 (both L4).

The count band is not touched. Ten kept cells still fail it (eight L2 at 7-8 spikes, two L4
at 9 and 11) and eleven L2 cells still return
below the family's post-pulse level; those are readings on the cells, reported as such.

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
