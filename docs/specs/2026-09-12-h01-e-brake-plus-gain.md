# SP14 (proposed). E cell: a saturating post-spike brake paired with one gain lever

Status: DRAFT 2026-09-12 from the SP13 close. Not approved, not run. It is a two-factor
test, so it needs the user's approval before code (a one-line mod change and a manifest).

## What SP12 and SP13 established

A spike-triggered outward gate on the B3 soma, silent without a spike, is sufficient for
the human's 200 pA response when its tail is long (5 s): one spike at B3's time, then a
level within 0.5 mV of the human for the rest of the pulse, with the 110 pA rows unchanged
to 0.000 mV. The same gate accumulates along a train (0.48 after one spike, 0.83 at
250 pA, 0.96 at 310 pA) and over-brakes: counts 3 and 7 against the human's 5 and 10,
while unchanged B3 already lands the 310 pA cycles (121 vs 120 ms).

## Hypothesis

Two conditions together: (1) the human's post-spike brake saturates after the first
spike (the second to tenth spikes add little), and (2) B3 lands the 310 pA rate only
because its high-drive gain is too low by about the amount the brake removes (the
cross-cell pattern: both humans have a steeper late gain than their donor models). A
saturating brake alone would still remove about three 310 pA spikes; the gain lever
must return them without adding spikes at 200 pA, where the brake holds the level.

## Implementation

`KsAHP.mod`: add `zmax` and a per-spike increment that saturates, e.g. the gate opens
towards `zmax` with `tau_on` only while above `vhalf` (already so) and `zmax = 0.5`
(one spike fills it). No new mechanism. Gain lever candidates, one per arm, each with a
registered dose derived from the retained 310 pA trace: somatic NaTs density up (the
peak/rise lever already established in Stage A), or axonal NaTs density up (the
initiation site). Both are existing flags.

## Registered predictions (to fix with doses before any run)

Stage 0 (sweeps 56, 53 together, cap two evaluations): 200 pA count 1 with the level
bands of SP13; 310 pA count 9-10 with late cycles within 12.8 ms of 120 ms. Stage 1
(sweeps 50, 43): 250 pA count 5-7 with cycles 4-5 of 250-350 ms; sweep-43 rows within
1 mV of the human and 0.1 mV of g0-b3. Rejection: no arm holds both stage-0 inputs; a
sweep-43 row moves; the 200 pA first spike moves by more than 1 ms unless the gain lever
is registered to move it. Both tiers reported; sweep 54 stays sealed.
