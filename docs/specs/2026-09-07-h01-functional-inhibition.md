# SP5. Functional inhibition at the measured I-to-E pair

Programme: [population programme](2026-09-07-h01-population-programme.md) SP5. Prior evidence:
[pair result](../evidence/h01-ie-pair-result.md) (W4), [literature pins](../evidence/h01-ie-synapse-literature.json).
Worktree `h01-pair` / branch `feat/h01-pair`. Causal-model section: Y5. Registered 2026-09-07;
no run has been made at registration.

## Question

W4 showed that one I spike through the literature receptor at the measured site moves the E
soma by 0.008 mV while E is charging without firing. That is a voltage response, not
inhibition. SP5 asks whether the same contact changes E's firing: does one I spike, landing
between E spikes, shift an E spike or lengthen an E cycle by more than the numerical
repeatability of the run? Stage 1 uses the current (unpromoted) profiles and is diagnostic;
stage 2 (promoted profiles from SP2/SP3) is listed at the end and is not opened here.

## Arms

| Arm | Name | Control | Receptor | Site | dt (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | `disconnected` | `disconnected` | none delivered | measured (probe only) | 0.005 |
| 2 | `measured` | `ei` | literature: 3.1 nS, -75 mV, 4.18 ms | contact 8105899, E `cable_location [2805, 0.93]` (measured) | 0.005 |
| 3 | `soma` | `ei` | the same receptor | E soma, an explicit perisomatic hypothesis, labelled `inferred` in the output JSON | 0.005 |
| 4 | `measured-halved` | `ei` | as arm 2 | as arm 2 | 0.0025 (dt-halving partner of arm 2) |

Everything else is identical across arms: measured connectivity, `h01_staggered_scan` solver,
`max_cv_um 10`, transmission delay 0.5 ms, the closing-restored I wrapper
(`docs/evidence/h01_i_source_closing_circuit.py`, which restores I NaTg closing kinetics and
labels the run a functional diagnostic), the E drive and the I pulse train below. Arm 3 is the
only arm whose receptor placement is not the measured endpoint; its JSON carries
`receptor_placement.status = "inferred"` and the result page repeats the label.

## E drive (a stimulus choice, registered before any run; not a fit)

**Rule.** E receives a constant somatic current from 0 ms to the end of the window. The value
is chosen so that the disconnected control fires at least 3 spikes in the 300 ms window with a
first spike early enough to leave at least two full cycles. It is not tuned against any
connected arm.

**Candidate value: 0.6 nA.** Derivation from the only direct evidence on this E model in this
circuit (the W4 0.4 nA hold, `h01-ie-pair-result.md`): the soma rests at -84 mV, reaches
-71 mV at 14.78 ms and -60 mV at 40 ms under 0.4 nA without firing. A single-exponential
charge through those two points gives a membrane time constant of 27.6 ms and an input
resistance of 78 MOhm (steady state -52.6 mV at 0.4 nA). With the model's threshold of
-57 mV (Y4, `h01-z-map-e.md`), 0.4 nA would reach threshold only at 54 ms and then sit
4 mV above it, near rheobase (the donor's reported rheobase is 200 pA on the Allen geometry;
the H01 geometry carries more dendritic load, `cm 2.303`). At 0.6 nA the steady state is
-37 mV and threshold is reached at 24 ms; on the donor's f-I curve (10 spikes/s at 310 pA,
slope 0.033 to 0.05 spikes/pA/s, `h01-e-usable-result.md`) 600 pA extrapolates to roughly 20
spikes/s, i.e. cycles near 50 ms, giving about 5 spikes in 300 ms and 1 to 2 in the 100 ms
benchmark. 0.6 nA is therefore the smallest round value that clears the ">= 3 spikes with two
full cycles" rule with margin and stays 0.4 nA below the 1 nA pulse this model is known to
fire to. Registered second value (the one permitted drive change, see stop rule): 0.8 nA
(steady state -21 mV, threshold at 15.5 ms).

## I drive (pulse train through the closing-restored wrapper)

I receives 1 nA, 3 ms somatic pulses (the W4 stimulus; one I spike per pulse at 14.78 ms for
an 8 ms onset, i.e. a pulse-to-spike latency of 6.78 ms). Because the measured connectivity
has no E-to-I projection, I's firing is identical in every arm and E's firing in the
disconnected arm does not depend on the I schedule. The schedule is therefore built in two
steps:

1. The disconnected control runs with the **provisional train** (onsets 20, 45, 70, ... ms,
   every 25 ms inside the window). Its E spike times are the control cycles.
2. `schedule` derives the **between-spike train**: for each pair of consecutive control E
   spikes, one pulse whose onset is the cycle midpoint minus the 6.78 ms latency, dropped if it
   would start before 0 ms or overlap the previous pulse. Arms 2, 3 and 4 run this schedule;
   the scorer records which I spikes landed in which control cycle rather than assuming it.

## Retained per E cycle (control cycle k = control spike k to control spike k+1)

| Quantity | Definition |
| --- | --- |
| `cycle_ms` | control cycle length |
| `connected_cycle_ms` | the connected arm's cycle k (spike k+1 minus spike k), `null` if either spike is missing |
| `shift_ms` | connected spike k minus control spike k (positive = delayed); `null` if missing |
| `i_spikes_ms` | I spikes falling inside the control cycle |
| `soma_ipsp_mv`, `site_ipsp_mv` | signed extremum of (connected minus control) E voltage at the soma and at the receptor-site probe, from the first I spike in the cycle to 20 ms later or to 2 ms before the earlier of the two arms' next spikes, whichever comes first (so a shifted spike does not enter the difference) |

An E spike count that differs between arm and control is retained as such (`e_count_*`) and
counts as a shift beyond any limit (a removed or added spike is not a small displacement).

## Decision limit (dt-halving pair, arms 2 and 4)

Per control cycle, range = |arm 2 minus arm 4| for `shift_ms` and for `connected_cycle_ms`.
Limit per quantity = mean range x 2.95 (DLF for two repeats, programme method), floored at one
step (0.005 ms) so a limit cannot fall below the sampling resolution. If arms 2 and 4 differ
in E spike count, the halving pair breaks and the decision is "numerically unresolved" (no
functional verdict is issued; this is not a pass and not a fail).

## Gate

**Functional inhibition** at an arm = at least one E spike shifted beyond the shift limit, or
at least one connected cycle lengthened beyond the cycle limit, or an E spike count change.

## Registered predictions

| Arm | Prediction | Rejection |
| --- | --- | --- |
| 1 `disconnected` | >= 3 E spikes in 300 ms; first spike between 20 and 60 ms; I fires once per pulse | fewer than 3 E spikes: one drive change to 0.8 nA, then stop |
| 2 `measured` | soma IPSP magnitude < 0.05 mV (W4 gave 0.008 mV at a 110-fold site-to-soma attenuation); no E spike shifted and no cycle lengthened beyond the halving limit; site response 0.5 to 2 mV, depolarising while the site sits below -75 mV | soma IPSP magnitude >= 0.05 mV, or any spike or cycle beyond the limit |
| 3 `soma` | soma IPSP magnitude >= 0.3 mV (derivation below) and at least one E spike delayed beyond the shift limit | soma IPSP magnitude < 0.05 mV: the perisomatic placement hypothesis is rejected as the explanation of the W4 deficit (the conductance is then too small at any site) |
| 4 `measured-halved` | E spike count equal to arm 2; per-cycle ranges below 0.1 ms | count differs: numerically unresolved |

**Arm 3 derivation (from `h01-ie-synapse-literature.json`).** Pinned unitary IPSC 27.6 pA at
-55 mV (szegedi2017) times the pinned pyramidal input resistance 102 MOhm (molnar2008) is
2.82 mV, the slow-synapse steady-state bound already written in the pin file. The receptor is
fast against this membrane: a 27.6 pA current decaying with 4.18 ms delivers Q = 115 fC, and
the membrane estimated above (tau_m 27.6 ms, R_in 78 MOhm) has C = tau_m / R_in = 0.27 nF,
so the somatic peak is about Q / C = 0.43 mV (equivalently 2.82 mV x 4.18 / 27.6). Between
spikes the E soma sits near -60 mV rather than -55 mV, so the driving force to -75 mV is 15 mV
instead of 20 mV: 0.43 x 15/20 = 0.32 mV. The registered threshold is 0.3 mV. The 3.1 nS
median conductance gives the same figure (3.1 nS x 15 mV x 78 MOhm x 4.18/27.6 = 0.55 mV, an
upper figure because it ignores the soma's own loading). A response of 0.3 mV at the soma is
about 40 times the W4 measured-site soma response, which is the whole placement contrast.

## Human tiers

Both tiers are **not applicable** and the decision JSON says so: the 1 mV contract tier
compares a modelled cell to a human recording of the same cell, and the usable tier scores a
single-cell spike train; there is no human recording of this pair (a modelled H01 contact with
borrowed synapse dynamics), so neither tier has a datum to score against. SP5 stage 1 is a
diagnostic on unpromoted profiles; its verdict is about the model, not the human.

## Cost, caps, run governance

- Cost anchor: 170 s per 40 ms run (W4, measured, three runs), about 4.25 s per simulated
  ms. Predicted: 100 ms benchmark about 7 min; 300 ms about 21 min; the halved partner about
  42 min. Anything predicted over 15 min needs a measured estimate and approval (SP0), so the
  order is: 100 ms benchmark (disconnected, 0.6 nA, provisional train) -> measured estimate for
  300 ms -> approval -> the four arms. The benchmark doubles as the E-drive check (>= 1 E spike
  expected by 100 ms).
- Cap: 6 runs + 2 halving runs. Planned: benchmark, disconnected, measured, soma (4 of 6),
  measured-halved (1 of 2). One drive change would use a fifth of 6.
- Detached jobs via `Start-Process`; progress line at least every 60 s; watchdog kill at 10 min
  without progress; killed = untested.
- No run in this task: the spec, tooling and manifest are registered; other measured runs are
  in progress on the machine.

## Stop rule

No E spike in the disconnected control (or fewer than 3 in 300 ms): one drive change to the
registered 0.8 nA and re-run the control; if that still fails, stop and report. Placement
(whether the PV contact belongs perisomatically on this pyramid) stays the user's decision; SP5
measures both placements and does not choose.

## Deliverables

- `braintrace/datasets/h01_ei_circuit.py`: `receptor_site` ("measured" or "soma", labelled)
  and `i_pulse_onsets_ms` (multi-segment I current clamp); tests in the sibling test file.
- `examples/h01_ei_circuit.py`: `--receptor-site`, `--i-pulse-onsets-ms`; test.
- `docs/evidence/h01_ie_functional_inhibition.py` (+ `_test.py`): modes `benchmark`,
  `schedule`, `run --arm`, `score`; reuses `h01_ie_pair_response.py`'s wrapper arguments.
- `docs/evidence/h01-ie-inhibition-manifest.json`: exact commands, predictions, cap, cost.
- `docs/evidence/h01-ie-inhibition/decision.json` (written by `score`), result page
  `h01-ie-inhibition-result.md`, and the Y5 causal-model entry in the same commit.

## Stage 2 (listed, not opened)

The same four arms with the profiles SP2/SP3 promote (E gain split survivor, I finalist),
opened only if a profile is promoted; its own manifest and predictions are written then.
