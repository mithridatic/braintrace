# Causal diagnosis by boundary splits

## Purpose

Explain the measured voltage trajectory through charge flow and internal state.
Use the explanation to constrain the active cell and then the connected circuit.
Do not replace this objective with a sequence of passing diagnostic tests.

The user supplied these methodological references:

- `C:/Users/J/Documents/Projects/comsol-llm-lab/docs/ENERGETIC_FRAMEWORK.md`
- `C:/Users/J/Documents/Projects/comsol-llm-lab/src/gepa_comsol/data/conjugates.yaml`

Use their paired observations, explicit datums, source-load boundaries, and
progressive splits. Do not import their COMSOL bindings into BrainCell.
Do not assume that a nonlinear neuron has only two state variables or that a
fixed linear equivalent identifies its full channel dynamics.

## Document boundary

The living causal document contains physical explanations, supported causal
conclusions, and explicit unresolved links. Exclude run settings, result tables,
chronology, test instructions, and parameter-search history.
Put each test and its decision in evidence. Retain links for traceability.
Revise an earlier explanation when a discriminating result contradicts it.

## Split rules

1. Name the direct behavior, location, time datum, and operating conditions.
2. Name at least two explanations that the observation alone cannot distinguish.
3. Locate a boundary where their predictions differ.
4. Observe both voltage and signed current at that boundary. Retain gate and
   concentration states where they affect the relation.
5. Define an intervention and its unchanged quantities before execution.
6. Define a result that rejects each explanation. Include an unresolved outcome
   when both explanations predict the result or the measurement cannot resolve it.
7. Check the intervention and numerical resolution before judging the claim.
8. Preserve the continuous traces. A binary decision is an interpretation of
   those traces, not a replacement for them.

An intervention may establish a conditional cause in a simulation without
establishing the same mechanism in a human cell. State that boundary explicitly.
RSS helps select informative interventions only when their sizes and sample
grids are stated. It cannot identify mediation or replace uncertainty estimates.

## Next informative boundary

The shorter spike is established conditionally. The route from that change to
the altered train remains unresolved. Possible routes include altered calcium
entry, SK activation, and other potassium currents.

Before another parameter sweep, observe the membrane balance at the same
location: applied current, ionic currents, axial current, and capacitive current.
Use a common clock and consistent signs. Verify the residual against numerical
resolution. Opposing channel currents alone do not close the charge balance.

For the PV soma midpoint, retain voltage in its adjacent soma segments and
each directly attached child segment. Calculate axial current from each
voltage difference divided by its NEURON axial resistance.
Do not define axial current as the residual of the other measured currents.
Include leak, Ih, all ion currents, capacitance, and both applied clamps.
Convert mA/cm2 to nA using the local segment area, not whole-soma area.
Evaluate current balance away from initialization and exact stimulus jumps.
Use 1e-5 nA as an initial diagnostic residual limit and report the measured
maximum. This is a bookkeeping check, not a biological acceptance tolerance.

Then split the proposed mediator from the other paths. A rescue or replay of a
mediator must preserve the intended input and document any external work it adds.
Compare the restored direct trajectory, not only spike count.
Specify the claim and rejection criterion before implementing that experiment.

## Source datums retained outside the causal narrative

The H01 soma anchor is cell 810151953, component 0, node 1345.
Skeleton coordinate scales are 32, 32, and 33 nm per unit.
Synapse coordinate scales are 8, 8, and 33 nm per unit.
Source radius is in nm and does not use those position scales.
BrainCell output time is the end of each step, `(index + 1) * dt`.
Wilbers' channel shift is a source convention, not a universal correction.
The PV export already contains its -14 mV junction correction.
