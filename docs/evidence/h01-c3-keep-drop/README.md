# C3 test-to-failure gate: 40 candidates and the 17 kept cells (step C)

Spec: [2026-09-16-h01-c3-partner-expansion.md](../../specs/2026-09-16-h01-c3-partner-expansion.md),
section C. Register: `qc3_e`, `qc3_i` (search tree), `io_gate`, `datum_fire`, `datum_rest`
(campaign tree) under [h01-reasoning/](../h01-reasoning/). Executor: Vast 50616476, worktree
`/workspace/braintrace-c3-keep` on branch `campaign/h01-c3-c3-keep-20260916`; receipts under
`var/c3-keep/` there. Nothing below has run yet: this page is the preparation record.

## Gate (split, J 2026-09-16; runs unchanged)

`h01_anatomy_transfer_decision.py --keep --gate failure --datums human-datums.json`.
(a) Hard failure edges: `finite`, `no_spike_before_pulse`, `recruitable_in_human_range` (ramp
rheobase at or below the human's highest recorded amplitude and the step run at the primary
drive firing at least once), `fires_at_highest`, `no_block_in_recorded_range`,
`dt_half_reproduces`. `rheobase_in_step` (1 s ramp rheobase within one sweep step of the human
long-square rheobase) is a recorded reading, not a rule (coordinator 2026-09-16, flagged to J:
different measurement chains; the donors' Allen rows document step-to-ramp offsets of +36 / +2 /
+72 / +27 pA, `human-datums.json` `allen_ramp_threshold`, pull
`.cache/h01/allen-donor-ephys-features.json` a701d082...); it sits in every receipt beside the
human step and slow-ramp thresholds. (b) Plausibility rules: `count_in_type_spread`,
`rest_in_type_spread`, `return_in_type_spread` — datum the donor's own value, tolerance 2 sd
across the human cells of the donor's type in the Allen Cell Types database
(`human-datums.json` `type_population`, from the sha-pinned pulls
`.cache/h01/allen-human-ephys-features.json` b19ed0b5... (413 human specimens) and
`allen-human-ephys-sweeps.json` b3bd3d51... (8,443 long-square sweeps);
`h01_allen_type_population.py`). Populations by `tag__dendrite_type` and `structure__layer`:
spiny L2/3 (199 cells, L2 donor), spiny L4 (37, L4 donor), aspiny all layers (79, both
interneuron donors — the API exposes no fast-spiking / non-fast-spiking label for human cells,
so the split J named is not available). Per cell the readings are on its 1 s long-square sweeps
within one step (20 pA) of the donor's primary drive: `num_spikes` (null = 0), ipfx
`pre_vm_mv` (500 ms before onset) and `post_vm_mv` (last 500 ms of the recording, 5-7 s after
offset — the table has no field for the 10 ms ending 200 ms after offset; `post_vm_mv` is the
closest documented post-stimulus level and is named as the fallback). Tolerances (2 sd): count
L2/3 14.2 (n 147), L4 20.1 (37), aspiny at 190 pA 62.2 (67), aspiny at 100 pA 43.1 (77); rest
7.7 / 8.3 / 9.4 / 9.8 mV; post-pulse 7.6 / 8.2 / 9.3 / 9.7 mV. Every receipt also carries
`donor_band_rules` (the single-donor bands: `count_band` across the sweeps within one step of
the primary, 3 across-sweep sd of each rest leg) and `legacy_verdict` (10 mV / 30 percent) as
comparison columns; `decision.json` `rule_counts` tallies all four sets. A cell whose ramp or
dt-half repeat has not run is `pending`; an unavailable datum is never a pass.

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

## Preflight reading on the 17 kept cells under the split gate (ramps pending)

`--folder ../h01-keep-drop/runs --ramp-folder runs` on [kept-types.json](kept-types.json)
([kept-gate-preflight.json](kept-gate-preflight.json), not a decision): 0 kept, 1 dropped,
16 pending on the ramp. Hard edges: `finite` 17/17, `no_spike_before_pulse` 17/17,
`dt_half_reproduces` 17/17, the three ramp edges unmeasured. Plausibility: `count_in_type_spread`
17/17 (counts 7-15 against donor datums 10 / 12 with tolerances 14.2 / 20.1),
`rest_in_type_spread` 17/17, `return_in_type_spread` 16/17 (5013648003 returns to -93.46 mV,
8.5 mV below the L2 family's -84.94 mV against a 7.6 mV tolerance). Single-donor comparison
columns on the same receipts: `count_in_repeat_range` 7/17, `rest_in_donor_spread` 5/17
(pre-pulse leg 15/17, post-pulse leg 6/17); legacy bands 17/17 on every rule.

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
