# SP11. E cell: measure the B3 current paths, then one voltage-dependent lever

Status: spec written 2026-09-11 before any run. Branch
`campaign/h01-currents-2026-09-11`. Manifest
`docs/evidence/h01-e-currents-manifest.json`, output `docs/evidence/h01-e-currents/`.
Executor: the Vast.ai box, NEURON 9.0.2 (see the SP10 spec for the local executor).

## Question

Frozen B3 fires 4, 8, 10, 12 spikes at 200, 250, 310, 350 pA against the human's
1, 5, 10, 13 ([gain close](../evidence/h01-e-gain/stage-close-decision.json)).
The count error changes sign across inputs, so no constant offset and no passive
lever (Ih, leak: both rejected) describes it. The reassessment names a
voltage- or use-dependent current outside the passive family. The saved B3
traces carry no currents (`record_soma_currents` false), so the current that
carries the three excess low-drive spikes has never been measured.

## Stage 0: measurement (one evaluation, no parameter change)

Rerun unchanged B3 with `record_soma_currents` at sweeps 56, 50, 53 (200, 250,
310 pA). The recorded terms are the soma densities of NaTs, Nap, K_P, K_T, Kv3_1,
Im, SK, Ca_HVA, Ca_LVA, Ih and leak, the capacitive current, calcium, the Ih and
Im gates, and every axial neighbour voltage with its resistance.

Prediction: counts 4, 8, 10; every rise crossing within 0.1 ms and every trough
within 0.1 mV of `g0-b3`; axon-first initiation. Rejection: any count or
crossing difference (the probes or the executor changed the model; stop).

Analysis (`docs/evidence/h01_e_current_paths.py`): for each interspike window,
integrate every soma term (nA, inward positive) and report the net drift current
and each term's share at 200 pA against 310 pA. The lever selected for stage 1
must satisfy, from these numbers alone: (a) its interspike current at 200 pA is at
least the net drift current there (so a plausible dose can hold the plateau below
threshold), and (b) its share of the interspike balance at 310 pA is smaller than
at 200 pA (so the same dose leaves the 310 pA train inside its band). A term that
fails (b) is rejected before any run.

## Stage 1: one registered lever (up to two evaluations)

Candidates, in the order the stage-0 numbers rank them; each is a single
`--regional-density` change on B3 and is run at sweeps 56, 50, 53, 43:

- `n1-nap-*`: persistent sodium (Nap) density reduced. Nap is the inward current
  that sustains slow near-rheobase firing; removing part of it raises rheobase
  without moving the resting return (Nap is closed at -84 mV).
- `k1-im-*`: M current (Im) density raised. Im activates slowly near threshold and
  holds a low-drive plateau below threshold after the onset spike.

Prediction bands (registered here, dose fixed in the manifest after stage 0):
200 pA count 1 (band 1-2); 250 pA count 5-7; 310 pA count 9-10 with rate within
1.5 Hz of 10.0 Hz and late cycles 110-124 ms unchanged within 12.8 ms; sweep-43
onset and late-return rows move by less than 1 mV from `g0-b3`; axon-first
initiation retained. Rejection: 310 pA count below 9 or its late cycle moves more
than 12.8 ms (the lever rotates the curve through the high-drive train), or the
sweep-43 rows move by 1 mV or more (the lever is not silent at rest).

## Limits

Cap three evaluations. Abort 2400 s per run. Sweep 54 (330 pA) stays sealed;
sweep 55 is not rerun. Both tiers reported for every arm. A stage-1 pass is a
gain-split result for B3, not a contract pass and not a promotion; promotion
needs the sealed sweep-54 prediction under the population programme's rules.
The causal model Y4 section is updated in the same commit as each decision JSON.
