# Active H01 cells and a small E/I circuit

Status: implementation and qualification pending. User authorized the staged
single-cell then circuit approach. Work remains on feat/h01-braincell.

## Objective

Build and validate an active H01 human cortical neuron model, then use
validated excitatory and inhibitory cell models in a small spiking circuit.
Constrain behavior with published human electrophysiology. Use H01 connectivity
only where source identity mapping is reliable; explicitly identify inferred
wiring, borrowed dynamics, and measured geometry separately.

## Approach

1. Pin and inspect Wilbers 2023 released channel mechanisms and experimental
   targets (Dataverse doi:10.34894/L5J0SD). Reproduce channel rate curves and
   reference single-compartment responses before transfer to H01.
2. Investigate the passive H01 equilibrium error with a reproducing test;
   validate electrical discretization and soma/AIS placement rather than
   treating sparse annotation samples as complete compartment boundaries.
3. Implement an explicit active H01 configuration. Preserve the source
   morphology and record every inferred electrical region and parameter.
   Demonstrate threshold, waveform, repeated firing and numerical convergence.
4. Select a published inhibitory model with an explicit species/type and
   electrophysiological validation basis. Do not relabel pyramidal kinetics
   as an inhibitory model merely by changing output sign.
5. Audit H01 partner-ID/connectivity availability. Build a small E/I circuit
   using supported links or explicitly illustrative wiring. Demonstrate
   synaptic causality with disconnected, excitatory and inhibitory controls.

## Required evidence

- Source URLs, versions, file hashes, units, temperature, voltage-reference
  conventions, conductance interpretation and any corrected source behavior.
- Channel edge cases: removable singularities, extreme voltages, gate bounds,
  initialization and temperature scaling; cross-check against independent
  reference equations or simulator outputs.
- Single-cell rest, threshold current, AP amplitude/width, firing response and
  adaptation/recovery where supported by the selected data. Numerical agreement
  alone is not biological validation. Separate reference-model reproduction
  from H01 transfer and held-out electrophysiological validation.
- Time-step and spatial refinement, finite outputs and explicit equilibrium
  tolerance. Record actual results and failures; do not loosen gates silently.
- Circuit excitation/inhibition and delay behavior with isolated-cell and
  removed-connection controls. No unsupported reconstruction claim.
- Co-located meaningful tests exceeding 90% coverage for changed production
  modules, reusable installed-package entry points and a documented runnable
  demonstration. Compiled brainstate loops for repeated simulation.

## Boundaries

H01 does not provide electrophysiology from the imported neurons. A successful
transfer is human-constrained dynamics on measured anatomy, not validation of
the exact donor cell. Do not imply whole-brain, learning-rule, or Example 21
qualification. A new solver is warranted only by a demonstrated limitation.
No merge/publication is implied by this goal.

## Progress and qualification ledger

The goal is NOT complete. Current evidence establishes a source-specific channel
port and an experimental H01 spike, not validated human circuit behavior.

- Implemented human sodium and mixed potassium gates from pinned Dataverse
  files 318563/318566. Preserved temperature, conductance and voltage conventions.
  Corrected removable-singularity evaluation without changing ordinary-voltage
  algebra. Regression tests cover float32 gradient overflow and the analytic limit.
- Corrected the new channels' initialization: inherited HH initialization used
  zero gates, unlike the source equilibrium initialization. Reproducing tests
  initially failed; corrected reference cell now repolarizes after stimulation.
- Reproduced integer initial voltage in the existing H01 helper. Changed
  -65 to -65.0 mV; the same annotated zero-input real H01 run now has maximum
  rest error 7.86e-11 mV (64 bit, 1 ms, dt .025 ms). Historical drift evidence
  is retained, with the correction explained in the usage guide.
- Independent scalar SciPy DOP853 reference, without importing production rate
  functions: at dt .00125 ms BrainCell maximum waveform error .194 mV, RMS .0209
  mV and final error below .002 mV. Waveform error at .005 ms was .792 mV.
  This is numerical reproduction, not an experimental validation result.
- H01 810151953.0 soma labels span 119.872 x 143.744 x 145.662 um; the strict
  AIS region is empty. Added explicitly inferred geodesic neighborhoods rather
  than equating all these sparse labels to a complete active soma/AIS region.
- First H01 transfer: inferred 10 um active neighborhood, remaining cable passive,
  max CV length 10 um, 3151 CVs, dt .005 ms, 20 ms. Default borrowed densities
  produce a subthreshold response at .2 nA and +28.15 mV at .5 nA (3 ms pulse).
  Removing sodium at .5 nA limits voltage to -23.50 mV. This is functional
  spike/channel-causality evidence, not physiological qualification.
- Exploratory H01 densities 25/10 mS/cm2 give peak 46.57 mV, threshold -42.15 mV,
  source-defined rise/fall rates 535.5/-61.6 mV/ms and halfwidth .755 ms.
  These do not yet match all human targets. Do not select parameters on peak alone.
- Feature convention correction: the source fitting routine uses average
  30-70% rise/fall rates and a 40 mV/ms threshold. Earlier exploratory maximum
  derivative comparisons are not the same metric. Using the source extractor,
  the reference compartment gives peak 51.85, threshold -43.64, rise/fall
  399.5/-84.2 and approximate threshold-relative halfwidth .8675 ms.
- Human target medians in source AP1comparison.csv: peak 46.186 mV, threshold
  -44.687 mV, rise/fall 378.190/-76.072 mV/ms, halfwidth .884 ms. Match pulse
  protocol and feature conventions before using these as calibration gates.

## Remaining work

1. Complete H01 time/spatial refinement, settled-rest and repeated-firing checks.
   Establish explicit physiological acceptance intervals and calibration/held-out
  protocols; address AIS placement and active region sensitivity.
   Initial refinement evidence: halving dt from .005 to .0025 ms changes the
   exploratory 25/10 density peak from 46.569 to 46.643 mV. Halving maximum
   CV length from 10 to 5 um changes it to 43.770 mV (3233 CVs). Spatial
   convergence is not established; further refinement is required before
   calibrating against human waveform features. Current dataset suite:
   97 passed, 99.86% coverage; these tests are not the full biological gate.
   The 2.5 um / .0025 ms run has 4087 CVs and peak 42.555 mV; convergence
   remains pending. An installed wheel successfully ran the reference cylinder
   from an isolated Python process outside the source import path. The new
   active-transfer CLI records source assumptions and full traces. These
   package checks do not qualify the inhibitory or circuit components.
   At 1.25 um / .0025 ms (6019 CVs), peak is 41.707 mV, source-defined
   rise/fall rates 430.76/-42.82 mV/ms, threshold -40.711 mV. The falling
   phase remains too slow relative to the human reference distribution;
   peak matching alone is insufficient. Next refinement must hold dt fixed.
   Raw data_fig1.xlsx was downloaded as file 318665. Descriptive 40 Hz human
   first-AP distributions are saved in h01-human-40hz-targets.json (36 rows,
   not asserted to be 36 independent neurons). Raw amplitude is relative
   to threshold; per-row amplitude+threshold reproduces the published CSV
   median peak 46.18614959716797 mV. These quantiles are descriptive, not
   retrospectively selected acceptance gates.
2. Select and reproduce an inhibitory model. Candidate: human putative-PV model
   in ModelDB 267587, commit 82cdd91bc93942ba19315371330a2412e064baf5. Its
   paper uses Allen neuron 528687520 recordings. The source has detailed NaTg,
   Nap, K_P, K_T, Kv3_1, Im, SK, Ca_HVA/LVA and Ih currents; it is NOT yet ported
   or validated here. Source repository GPL-3.0; no source code incorporated.
3. Audit reliable H01 connection-ID mapping, implement the small circuit and
   verify isolated/disconnected/excitatory/inhibitory controls and delays.
4. Complete package/demo documentation, installed-wheel validation and worktree
   closeout. No learning or Example 21 changes are authorized by this milestone.

Sources:
- https://doi.org/10.34894/L5J0SD (API accessible; pinned manifest in evidence)
- https://doi.org/10.1126/sciadv.ade3300
- https://modeldb.science/267587
- https://doi.org/10.1093/cercor/bhac348

## Next numerical investigation

The current uniform MaxCVLen policy can place an active/passive boundary
inside a CV. Investigate explicitly splitting at the inferred active-region
boundaries, retaining the same geometry and conductance densities. Tests must
establish that the policy preserves [0,1] coverage without gaps/overlap and
inserts every requested boundary. Record this as a distinct discretization
policy; do not combine old unaligned and new aligned refinement results as
though they came from the same configuration. Compare against finer aligned
grids before selecting one for physiological calibration.

The first aligned implementation reproduced a degenerate interval caused by
a source cut at 0.9999999999999998 beside 1.0. A regression test now covers
near-coincident cuts, which are merged using the installed BrainCell 1e-9
normalized-coordinate tolerance. Aligned 2.5 um / .0025 ms gives peak 40.699
mV (4171 CVs). This is a distinct mesh from the original unaligned runs.

The inhibitory raw current trace independently confirms a 270--1270 ms pulse
at .19/.23/.27 nA and zero holding current. With height 0 mV / prominence
40 mV detection, those traces contain 12/31/43 spikes. Initial ISIs are
7.60/6.36/6.24 ms; final ISIs 98.76/40.72/26.68 ms. A constant-rate generic
fast-spiking model would not reproduce these recordings. Data are in
.cache/human-pv/{active,passive}.pkl, read with a restricted numerical-array
unpickler and exported to numeric NPZ caches. No foreign executable or pickle
code was executed. Source optimization "std" values are objective weights,
not measured population uncertainty; do not describe them as biological SDs.

## Repeated stimulation and aligned mesh qualification

The aligned 2.5 versus 1.25 um comparison at dt .0025 ms gives maximum
waveform difference .04586 mV, RMS .00365 mV, identical sampled peak times,
and peak difference .00238 mV. This supports using aligned 2.5 um for the
next calibration experiments; temporal refinement is still being checked.
AIS-labelled samples are 12.54--40.39 um along cable from soma anchor 1345;
the current 10 um active region excludes them. Sparse AIS point labels do
not yet define a complete active axon compartment.

Add an explicit finite rectangular pulse train to the active-cell builder:
positive integer pulse count, positive finite period greater than pulse width,
and the existing finite current/onset/width checks. Default one pulse retains
existing behavior. Record pulse count and period with all evidence. Use one
piecewise CurrentClamp and Cell.run's compiled loop, not Python simulation
steps. Test one/multiple pulses, invalid/overlapping protocols, and actual
recorded response. Wilbers source run_cell uses 3 ms pulses every 25 ms;
start with five pulses, then evaluate adaptation against the released human
measurements. This protocol support alone is not biological qualification.

## Direct behavior and causal explanation

The user requires a nested causal explanation, with defined measurement datums.
Maintain it in [h01-causal-model.md](../h01-causal-model.md).
Use short, direct sentences and defined terms under ASD-STE100 guidance.
Keep dated experiments and numerical tables in the evidence folder.
Revise the explanation itself when observations contradict it.

Use raw voltage, applied current, and individual spike times as primary evidence.
Add gate, channel-current, and axial-current observations to test the inner levels.
Population feature distributions do not replace a measured target waveform.
The H01 transfer does not yet match Wilbers' holding-current condition.
Raw pyramidal target traces and factor uncertainties remain missing.
Do not report a complete RSS uncertainty budget without them and covariance checks.

Available paired voltage traces give response-change RSS values, with source
hashes and checked configuration differences. These rank only the tested
interventions. Numerical effects remain separate from physical-factor effects.
The analysis scripts and direct trace figure are under docs/evidence.

The five-pulse configuration (Na25/K20, aligned2.5um, dt.0025ms) produces five
spikes at 3.6525,28.6475,53.6475,78.6475,103.645ms. Peaks fall from39.460 to38.011mV.
Na27/K20 gives first peak41.020mV in a separate single-pulse run.
Neither result establishes human physiological validity.
Current dataset tests: 108 passed, 99.86% coverage. New production modules have
100% statement coverage. Circuit and inhibitory implementation remain pending.

## Observe the channel mechanism directly

Add optional gate and current probes to the active H01 builder and CLI.
Name the sodium and potassium mechanisms explicitly.
Use an active CV midpoint near the soma for these probes.
Record its branch and normalized position, separate from the source soma node.
This avoids comparing a representative-CV voltage with an interpolated point current.
Require aligned boundaries for this observation mode.
Record dimensionless gates, inward-positive current density in uA/cm2,
and membrane voltage in mV, all at the same end-of-step time.
Check current against conductance, gate state, and driving voltage at every sample.
Check gate bounds and finite output. Retain the existing soma voltage probe.
These observations test channel algebra and local dynamics, not full axial balance.

## Independent inhibitory reference

Reproduce the published putative-PV model in NEURON before its BrainCell transfer.
Use the pinned ModelDB source and preserve the original mechanism equations.
Use a separate Linux container because the current NEURON release has no Windows wheel.
Pin NEURON 9.0.2 and record the environment.
Keep downloaded source files in the cache, with hashes and source paths in evidence.
Inspect the source morphology and axon replacement policy before claiming equivalence.
The reference must produce direct voltage traces for the published current steps.
This reference run does not itself qualify a BrainCell inhibitory implementation.

The default NEURON fixed-step reference shows accumulated spike-time changes
under refinement. Also compare NEURON's existing adaptive CVode solver at
two explicit absolute tolerances. Keep the source equations and stimulus fixed.
Record this numerical method separately from the published fixed-step setting.
Do not tune channel parameters against unresolved numerical timing error.

## Current direct-observation evidence

H01 channel observations use branch 281 at normalized position 0.7241921243.
Adding probes changes no soma-voltage samples in the paired real-cell run.
Maximum local current-identity error is 4.55e-13 uA/cm2 for sodium and
1.14e-13 uA/cm2 for potassium. All recorded gates remain within [0,1].
The current dataset suite has 112 passing tests and 99.87% coverage.
Two additional reference-report tests pass after reproducing and fixing
the incorrect sample-weighted baseline mean for adaptive output.

The original published PV circuit cell runs with its unchanged HOC template
and all eleven compiled mechanisms. At the source dt .025 ms, it produces
12/29/46 spikes for .19/.23/.27 nA, versus 12/31/43 in the recordings.
Peaks near 43 mV exceed the recorded peaks near 17 mV. Full traces and
unshifted residuals are saved. This reference is not physiologically qualified.
Refinement to .003125 ms gives 13 spikes at .19 nA. Two tight CVode runs
also give 13 spikes, with sampled peak times agreeing within .000292 ms.
Keep these numerical settings separate from the published fixed-step result.
The independent reference is ready to guide a faithful BrainCell transfer;
that transfer, biological calibration, and the connected circuit remain open.
