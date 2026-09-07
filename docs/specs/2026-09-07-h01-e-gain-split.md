# H01 E cell: gain split (SP3, 2026-09-07)

Status: registered before any run. Binds SP3 of the
[population programme](2026-09-07-h01-population-programme.md); inherits the SP0 rules
(Matryoshka before splitting, prediction/rejection/unchanged before every run, cap is a
stop, killed = untested, both tiers in every verdict, causal model updated in the decision
commit). Worktree `h01-e-split`, branch `feat/h01-e-split`. No simulation was run while
this spec and its code were prepared.

## 1. Matryoshka classification (from retained tables, no run)

Source: `docs/evidence/h01-e-usable/b3-all-inputs-usable.md` (frozen B3,
`h01-e-usable/b3-sk035-ca-decay.candidate.json`, sha256 `e4825c83...`).

**Elemental (input specified?)** Yes: recorded command plus recorded bias (-3.712 pA) at
every input; nseg factor 9; CVode 1e-10; stop 2100 ms.

**Cyclical (does the error repeat across inputs and cycles?)** Recomputed from the count
table:

| Segment | Human slope (spikes/pA) | Model slope (spikes/pA) | Ratio |
| --- | ---: | ---: | ---: |
| 250 -> 310 pA (5 -> 10 vs 8 -> 10) | 5/60 = 0.0833 | 2/60 = 0.0333 | 0.40 |
| 310 -> 350 pA (10 -> 13 vs 10 -> 12) | 3/40 = 0.0750 | 2/40 = 0.0500 | 0.67 |
| 250 -> 350 pA (5 -> 13 vs 8 -> 12) | 8/100 = 0.0800 | 4/100 = 0.0400 | 0.50 |

The programme figures (model 0.033 vs human 0.083 spikes/pA between 250 and 310 pA) are
confirmed. The same reading in mean-full-cycle rate: human (10.002-4.790)/60 = 0.087 Hz/pA,
model (10.045-7.728)/60 = 0.039 Hz/pA. The model curve is too flat on both sides of the
pinned 310 pA point: it overfires below and underfires above. This is a gain (f-I slope)
error, not an offset.

**Temporal (early vs late cycles?)** Late cycles are uniform within each input: model
250 pA cycles 4-8 run 184 -> 157 ms (monotone, 27 ms range) against human 220/307/273 ms;
model 310 pA cycles 4-10 run 121 -> 116 ms against human 110-124 ms. The error is set by
input level, not by position in the train, so the lever is a steady current that scales with
drive, not an adaptation constant. SK doses (Stage B) translate the curve; they cannot
rotate it, which is why B3 fixed 310 pA and overfired 250 pA.

**Structural.** Mesh and tolerance unchanged from the B3 qualification (Dixon limits
<= 0.0033 ms per `h01-matryoshka-e.md`); no numerical row is in play.

**Parked, not searched.** Rest -84 vs literature -72 mV: the donor's own pre-pulse
samples (rows 1019-1120 ms, `h01-matryoshka-e.md`) match the model within 1 mV, so the
model rests where this cell rested. Early widths (cycle 2 at 310, cycle 3 at 350) stay in
the width row and are scored, not targeted.

**Controlling property.** A current present between spikes at low input and small or absent
at high input: distributed Ih (`ih_density_factor 75` spread by `distribute_ih` to a uniform
9.99e-5 S/cm2 over soma, dend, apic) or the linear leak (`g_pas` in four regions,
reversal shifted -4 mV).

## 2. Inputs and holdouts

| Sweep | Command | Role | Status |
| --- | ---: | --- | --- |
| 43 | 110 pA | subthreshold calibration (F5 rows, 1 mV contract) | export on disk; not rescored on B3 |
| 50 | 250 pA | active calibration | spent |
| 53 | 310 pA | former holdout | spent |
| 55 | 350 pA | former holdout | spent |
| 56 | 200 pA | calibration: its repeats 56/59/60/61/62 already feed the AHP spread (`h01_usable_tier.py`), so the waveform is not blind; Allen count metadata visible (1 spike in each of 56/59/60/61/62) | exported for this spec |
| 54 | 330 pA | **new sealed holdout**: highest-amplitude long-square sweep never exported or scored | sealed until `h01-prediction-e3.json` exists |

Sweep 54 selection: candidates never opened with spikes are 48 (210), 49 (230), 52 (290),
54 (330); 51 (270) is the source-fit sweep. Every one of them is within 20 pA of an opened
sweep (43/50/53/55) or of the 200 pA repeat family, so the earlier "not adjacent" rule
cannot be met by any remaining sweep and is relaxed; the higher-amplitude rule of the
programme picks 54. Its Allen count metadata (12 spikes) was visible in
`sweep-metadata.json`, so count is NOT closed for this seal; waveform, timing, peaks,
widths and the per-cycle table are closed. Recorded in `h01-l2-reserved-input.json`.

Sweep 56 gives the E rate row its only repeat-based decision limit: five human repeats
(56/59/60/61/62) at the same command; limit = repeat range x DLF (1.47 for five). With one
spike per repeat the mean-full-cycle rate is undefined, so the resolvable 200 pA rows are
the count (range 0 -> exact) and the first-spike latency; `h01_usable_tier.py` scores both
from the repeats.

## 3. Stage 0 (evaluation 1): B3 at sweep 43 and sweep 56

Candidate `g0-b3` = the exact B3 flags (section 6). The new information is sweep 43 and
sweep 56; the runner applies the manifest inputs per candidate, so Stage 0 also re-runs
sweep 50 and sweep 53 under the `h01-e-gain/g0-b3-*` stems (about 2 x 650 s of the
evaluation). This is accepted: it puts every arm on the same four inputs and the same
stems, and the 250/310 rows double as a same-image repeat of B3.

- Prediction: sweep-43 late-return rows (1520-2120 ms) and onset rows (1019-1120 ms)
  within 1 mV of the human (B3's Stage B levers did not touch F5); sweep-56 count 1 (in the
  repeat band, which is exact), first-spike latency within the repeat limit; 250/310 rows
  reproduce `b3-all-inputs-usable.md` (8 and 10 spikes) within solver noise.
- Rejection: sweep 43 fails the 1 mV row -> B3's Stage B levers broke F5; every arm is
  scored on 43 (they are regardless) and the G arms are judged on the *change* of the 43
  rows, not their absolute pass.
- Decision file `h01-e-gain/stage-g0-decision.json`; it gates Stage G.

## 4. Stage G (evaluations 2-4), every arm at sweeps 43, 50, 53, 56

| Arm | Candidate | Flag (verified in `h01_l2_neuron_reference.py`) | Prediction | Rejection |
| --- | --- | --- | --- | --- |
| G1 | `g1-ih-half` | `ih_density_factor 37.5` (from 75; `--ih-density-factor`, scales somatic `gbar_Ih` before `distribute_ih` spreads it, so the uniform density halves) | 250 pA count 5-7, late cycles >= 200 ms; 310 pA count 9-10, rate within 1.5 Hz of 10.0 Hz; pre-pulse baseline -1 to -3 mV from B3; sweep-43 late return moves <= 1 mV; 200 pA count 1 | 310 count < 9, or 250 count still >= 8 |
| G2 | `g2-leak-150` | `leak_factor 1.5` (`--leak-factor`, scales `g_pas` in soma/axon/dend/apic; `leak_reversal_shift_mv -4` unchanged, so the reversal is unchanged) | both counts fall, 250 pA proportionally more (rheobase shifts right): 250 count <= 6, 310 count >= 8; 200 pA count 0-1 | 250 and 310 fall by the same spike count (pure shift) |
| G3 | `g3-reserve` (dose written into `stage-g-decision.json` before it runs) | dose of the better arm: the arm with the larger \|d(250 count)\| / \|d(310 rate)\| per unit factor; factor interpolated linearly to land 250 count 5 while the 310 rate change stays within 1.5 Hz | 250 count 5 +/- 1, 310 rate within 1.5 Hz, sweep 43 within 1 mV, 200 count 1 | as the chosen arm |

Both flags exist; no driver addition is needed for G1/G2. Stage G opens only when
`stage-g0-decision.json` records a decision; G3 opens only when `stage-g-decision.json`
does (the runner's gate rule). G1 and G2 run concurrently (two containers).

## 5. PASS/FAIL rule and reassessment

PASS: one profile whose rate and adaptation rows pass at 250/310/350 pA (350 from the
retained sweep-55 B3 trace for `g0-b3`; a G arm at 350 would need a fifth input and is
outside this cap, so a G-arm PASS is provisional on 250/310 plus 200 where resolvable) and
whose width rows show no cycle newly failing relative to `b3-all-inputs-usable.md`, with
sweep-43 rows within 1 mV and axon-first initiation at every active input.

FAIL at cap (4 evaluations) names the reassessment: the gain is set outside the fit's
passive family, i.e. human slow Na inactivation or Kv7/M kinetics absent from the Allen
genome, or a second human L2/3 donor with a recorded f-I curve (SP6b lead). No fifth
evaluation, no dose beyond G3.

## 6. Unchanged in every arm

All B3 flags from `h01-e-usable/b3-sk035-ca-decay.candidate.json`: `calcium_decay_factor 1.0`,
`distribute_ih true`, `insert_density ["NaTs:axon:3.814"]`, `kv3_closing_factor 0.9`,
`leak_reversal_shift_mv -4.0`, `regional_density ["SK:soma:0.35"]`,
`sodium_density_factor 0.9`, `sodium_opening_factor 2.0`, `sodium_recovery_factor 1.0`,
`record_soma_currents false`; settings `nseg_factor 9`, `cvode_atol 1e-10`, `stop_ms 2100.0`;
recorded command plus recorded bias (`--include-recorded-bias`); Allen 541563728 geometry
and mechanism library `kv3-closing-source`; image `braintrace-h01-neuron:9.0.2`; axon-first
initiation checked with `h01_initiation_score.py` on every active run.

## 7. Cap, cost, governance

- Cap 4 evaluations (`prior_evaluations 0`): Stage 0 (1) + G1 + G2 (2) + G3 (1).
- Manifest `docs/evidence/h01-e-gain-manifest.json`, runner `h01_campaign.py`, abort 1500 s
  per run, max 2 concurrent, output `h01-e-gain/`.
- Cost anchors (measured): 649-665 s per 250/310 pA run (B3, `h01-e-continuation-result.md`).
  Sweep 43 and sweep 56 are unmeasured. One evaluation is four serial runs, so even at the
  measured anchor an evaluation exceeds 15 min (>= 2 x 650 s plus two unmeasured runs);
  each evaluation therefore needs a measured estimate and per-job approval before the
  manifest opens, and the first Stage 0 launch reports the 43 and 56 durations before Stage G.
- Scoring: `h01_usable_tier.py --cell E-gain --candidate <name>` (four inputs, 200 pA
  repeat rows), `h01_contract_score.py` for the 1 mV rows, `h01_initiation_score.py` per
  active run; both tiers printed in every verdict.

## 8. Causal model

No observation exists yet, so no Y4 entry. The Stage 0 and Stage G predictions are entered in
`docs/h01-causal-model.md` Y4 as dated registered predictions; a miss is recorded against the
prediction when the decision JSON lands.
