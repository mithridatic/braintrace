# H01 direct observation contract

Status: APPROVED 2026-09-11 (user: "yes that way future sessions will get what they need").
Requested after the causal-document audit; multivari charts and visual inspection
were explicitly added by the user during implementation.

Visual refinement: inspection of the first B3 multivari chart showed faster human
recovery in the first 6 ms even though the human later stayed lower at 200 pA.
Keep that early view and add a separate common-scale view at 20, 60, and 200 ms
after the same trough. These are observed phase samples, not a fitted mean or a
retrospectively registered intervention prediction. Offsets beyond a next threshold
remain visibly missing. This follows Figures 101/104 and the discussion on printed
p. 119: inspect the initial pattern before reducing or refining the observations.

## Problem and book basis

The retained H01 traces contain voltage, current, and state observations, but SP11
uses mean currents and current shares to exclude interventions, and SP12 summarizes
recovery with voltage percentiles. Those summaries discard the timing needed to
explain a response. SP11 additionally accepts a one-sample post-pulse window as a
level and averages a nonuniform sample grid without time weighting.

Hartshorne's Matryoshka procedure starts with a few tightly connected observations
within a cycle and then compares consecutive cycles and structures; its starting
sample of three is a search heuristic, not a statistical confidence guarantee
(chapter 3, local book `04-chapter-3.md`, sampling discussion around Figure 99).
Chapter 2's energetics discussion uses paired force/displacement and torque/angle
to expose behavior. Its electrical pairs are voltage/current for power and
voltage/charge for storage (printed pp. 83-85, scans p074-p076). Apply those actual
comparisons to the neuron; a voltage mean or a V-versus-dV/dt plot alone does not
supply measured current paths.

## Required behavior

Implement one shared direct-observation extractor and renderer, initially wired
into the B3 current-path and SP12 analyses. Existing registered campaign decisions
remain historical records with their original rules. Produce a separate diagnostic
record; do not silently change a registered band or rename an old pass.

1. Select the first, second, and last complete recovery cycles deterministically,
   deduplicating short trains. Retain the onset before the first spike and the
   post-last-spike tail as distinct windows. With one spike, record the onset and
   tail; with no spikes, retain the commanded pulse's voltage/current response.
   Do not invent cycles or substitute an earlier cycle when a required one is absent.
2. Keep all original samples within these few selected windows. Small sample means
   a few informative cycles or systems, not undersampling a spike waveform. Attach
   explicit landmark samples at the trough and registered recovery offsets, refusing
   an offset that crosses the next threshold or runs beyond recorded time.
3. Record the cell/candidate hash, input, pulse clock, cycle identity, compartment,
   area, capacitance, sign convention, initial-state/source metadata, and absolute
   sample times. Preserve an explicit location for every current. A local segment
   cannot silently stand for the soma, and axial flow cannot stand for a downstream
   ionic current.
4. Pair local membrane voltage with local signed applied, ionic, axial, and
   capacitive currents at the same times. Preserve each axial neighbor and its
   voltage difference. For an axial branch use that branch's voltage difference
   and current; do not multiply axial current by soma voltage and label it branch
   dissipation. Where a channel's reversal/conductance is not recorded, mark its
   dissipative-power accounting unavailable rather than infer it from voltage alone.
5. Retain cumulative charge by path through each selected window alongside voltage
   and direct currents. Compare capacitive charge change with C times voltage change,
   and record the time-resolved current-balance residual. Cumulative integration is
   an additional physical check, never a replacement for the original samples.
   Do not claim a whole-cell energy budget without the ion reservoirs and boundaries.
6. Plot human and model on their original pulse clock, and separately plot explicitly
   aligned recovery windows with their original event identities. Missing or extra
   spikes are visible; no ordinal pairing is silently treated as correspondence.
   Human recordings supply voltage and commanded current, not measured ionic currents.
7. Present the observed response and its time/location contrasts first. Means,
   percentiles, rates, and counts remain secondary QC summaries. They cannot by
   themselves set `causal_explanation_supported`, exclude a current family, or
   promote a candidate. A causal verdict requires a named registered intervention
   and its predicted direct response, with matched background settings.

## Fail-closed checks

Reject malformed observation records: nonfinite samples, nonincreasing time,
missing units or spatial geometry for conversions, and current/voltage length or
clock mismatches. Check recorded extent against every requested window before
interpolation; never let `np.interp` silently extend an endpoint into missing time.
Report missing coverage and missing currents explicitly, preserving a useful
voltage-only record without granting a current-based diagnostic pass.

Fix the old summary helper's one-sample/out-of-range behavior. If it offers a
time mean on an adaptive grid, integrate with the actual time intervals. Any
retained sample percentile must be labeled as such; it is not a time percentile.
Do not estimate input resistance from a partial post-pulse window. Remove the
automatic causal-exclusion interpretation of baseline current shares; report the
measured values as observations.

## Initial scope and execution

Add a small shared module under `docs/evidence` with sibling suffix-style tests;
split extraction and plotting if required by the repository's module-size limits.
Integrate it with `h01_e_current_paths.py` and `h01_e_sahp.py`, reusing the existing
event definitions and NEURON current probes. The first implementation is this E
pipeline; do not claim all H01 campaign runners enforce it until they do.

Run the new analysis on Vast against unchanged B3 at 200, 250, and 310 pA and
against completed SP12 arms. Preserve the approved doses and original registered
score. Write a separate `direct-observation` artifact and interpret its small set
of phase-resolved contrasts in `docs/h01-causal-model.md`. Use the book's next
convergent comparison only when the retained evidence cannot resolve the question.
Rerun a bounded unchanged configuration to collect a missing current or required
time window if needed; check active campaigns and available resources first.
NEURON uses CPU on the Vast GPU host; do not claim GPU kernel execution.

## Tests and acceptance

Write failing regressions first for the actual SP11 defects: a declared 2100-2300 ms
window with only one sample at 2100, and an adaptive grid whose sample mean differs
from its time mean. Test nonuniform current integration against a known capacitor
trajectory; verify density-to-current units and signs with unequal compartment
areas, and retain individual axial branches with opposite directions.

Test zero/one/two/many-spike selection, missing next thresholds, phase offsets
beyond the next spike, incomplete tails, absent current probes, nonfinite and
duplicate timestamps, and event mismatch without silent realignment. Test a pair
of traces with equal means/counts but different recovery trajectories: the direct
records must preserve the difference and must not report causal equivalence.

Target over 90 percent coverage of the new meaningful extraction/validation paths.
Run affected existing tests, check the module-size convention, inspect the generated
plots, and verify source hashes and numerical balance on the retained B3 data.
Report observation validity, registered QC outcome, causal support, and human
qualification separately. A missing observation is never a pass.
