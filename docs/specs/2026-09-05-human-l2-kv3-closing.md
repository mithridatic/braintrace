# Test whether residual Kv3 conductance deepens recovery minima

## Observation and hypothesis

In the density-1.3, sodium-opening-factor-2 candidate, each of four
recovery minima is more negative than the matched human minimum.
Recorded local Kv3 current remains outward at each minimum. The source
gate reconstructed from current is above its equilibrium value. These
observations identify a testable path; current size does not prove cause.

Test the hypothesis that faster Kv3 closing reduces each absolute
recovery-minimum voltage error while preserving five complete spikes.
The full voltage response can change. Do not assume that holding the
opening law fixed holds spike peaks or later timing fixed.

## Mechanism preparation

Use a separate directory copied from the checked sodium-activation-source
mechanisms. Do not change any file loaded by an ongoing run. The original
Kv3_1.mod SHA256 is
e636c3437a9d9ccb69a0a7efc09fcfe1f16ad97f0468615b70fdbc352fe0855c.

Add an exposed m_closing_factor with default one. Multiply the source
mTau by this factor only when mInf < m. Keep the source equilibrium,
opening time law, density, reversal, and all other mechanisms unchanged.
The proposed intervention is factor 0.5. This is inferred model kinetics,
not a measured human channel value.

Reject an unexpected source digest. Removing only the declared additions
must restore the original source bytes. Check deterministic preparation.
Use sibling tests. Build and retain all source and library hashes.

## Gate-law check

At fixed voltages -84, -60, -40, -20, 0, and 20 mV, test factors one and
0.5 with opening, closing, and equilibrium initial gate states. Set gbar
to zero to keep voltage fixed. Compare every recorded gate sample with
mInf + (m0-mInf)*exp(-t/tau), using the halved tau for closing only.
Require gate error <=1e-8 and unchanged voltage. Check that all cases
are present. Advance the model with one compiled NEURON call.

## Whole-cell controls

The driver option must reject nonpositive and nonfinite factors before
loading the cache. Assign and read back the value on all somatic Kv3
segments. Record the factor; numerical comparisons must require it to
match, including rejection of one-sided missing metadata.

Before interpreting the intervention, factor one must reproduce the
saved density-1.3, opening-factor-2 control: exact time, voltage, input,
and every recorded soma array. Account for changed source/library hashes
and the new metadata field. All other physical settings must match.
A default-equivalence failure requires diagnosis before intervention.

## Prospective response prediction

For the half-closing-time run, keep calcium factor 1.375, sodium opening
factor 2, sodium density factor 1.3, sodium recovery factor 1, uniform
Ih factor 75, reversal shift -4 mV, source leak, recorded bias, sweep 50,
mesh 9, CVode tolerance 1e-10, and endpoint 2100 ms.

Require five complete positive-peak spikes. For each of four intervals,
select the earliest sampled minimum strictly between the preceding
falling -20 mV crossing and the next rising crossing. Require a smaller
absolute error against its ordinal human minimum than the frozen
control. Both conditions must hold to support the joint prediction.
There is no newly assigned physiological voltage tolerance.

Retain each onset, peak, rising phase, falling phase, duration, interval,
minimum voltage, and minimum delay after the preceding falling crossing.
Record their human residuals and all unmatched events. Improvement in
minima does not override other errors or qualify the cell.

Wrong physical settings, malformed or nonfinite arrays, incorrect input,
inexact raw mapping, and incomplete boundary events invalidate a test.
Missing or extra complete spikes in otherwise valid data reject the
prediction. Require exact endpoint and input plateau error <=1e-12 nA
outside 1e-7 ms around input transitions.

Candidate-specific tolerance, spatial, subthreshold, and reserved-input
checks remain separate. Do not inspect reserved sweep-53 response.
This split tests the total effect of an isolated law change in the model;
it does not identify a unique human mechanism or prove mediation through
one measured local current.
