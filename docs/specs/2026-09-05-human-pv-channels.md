# Human putative-PV channel transfer

## Scope

Implement the mathematical channel model used by the released HL5BN1 cell.
Source: ModelDB 267587, commit 82cdd91bc93942ba19315371330a2412e064baf5.
Keep the measured human recording, fitted parameters, and borrowed kinetics separate.
This work is part of the approved H01 cell-and-circuit goal.

## Required mechanisms

| Mechanism | Gates | Conductance factor |
| --- | --- | --- |
| NaTg | m, h | m³h |
| Nap | m, h | m³h |
| K_P | m, h | m²h |
| K_T | m, h | m⁴h |
| Kv3_1 | m | m |
| Im | m | m |
| Ih | m | m |
| Ca_HVA | m, h | m²h |
| Ca_LVA | m, h | m²h |
| SK | z | z |
| CaDynamics | intracellular calcium | Buffered influx and exponential removal |

Use the published fitted NaTg shifts and slopes, and fitted Ih rate parameters.
The source hard-codes 34 °C corrections in several rate functions.
Preserve that convention. Do not imply an arbitrary-temperature model.
Use voltage in mV, rate time constants in ms, and calcium in mM.
Convert source conductance units explicitly when building the cell.

## Numerical implementation

Implement rate equations as pure JAX functions.
Use analytic limits at removable singularities, without changing membrane voltage.
Retain the source SK cutoff below 1e-4 mM and its 1 ms time constant.
Initialize all gates at equilibrium, as in the source.
Use BrainCell channel and ion integration, with compiled BrainState simulation loops.

The calcium ion uses a dynamic Nernst potential at 34 Celsius, with 2 mM outside.
Its initial and resting inside concentration is 1e-4 mM.
Use signed inward calcium current divided by twice the Faraday constant and
the shell depth, multiplied by the unbuffered fraction gamma.
Add exponential removal toward the resting concentration.
Do not clip outward current: the source equation retains its sign.
Use the fitted removal time for each region. The ion defaults to the soma value
because BrainCell creates a baseline instance before applying region parameters.
The cell builder must set the axon value explicitly.
The source shell depth is 0.1 micrometre and fitted gamma is 0.0005.
Check source-unit conversion, influx direction, removal, Nernst response,
and channel-to-ion current routing in an assembled BrainCell compartment.

## Independent evidence

Expose rate outputs from cached copies of the original NMODL mechanisms.
Add RANGE observation fields only; keep the equations unchanged.
Record the original and instrumented source hashes.
Compile these copies separately from the existing full-cell reference.
Generate reference values by calling the compiled algebraic rate procedures.
No repeated membrane simulation is needed to obtain these tables.

Check all gate steady states and time constants against those independent values.
Also check singular limits, finite derivatives, gate bounds, and positive time constants.
Check calcium-dependent activation at and around the cutoff.
Then check dynamic current responses and whole-cell traces against NEURON.
A passing rate table does not establish human waveform validation.

The source templates and mechanisms stay in the cache.
Retain attribution to the original mechanism references in the provenance document.
Do not rename these mechanisms as human-measured ion-channel subtypes.

## Whole-cell geometry

Export the initialized original template's arc lengths, diameter samples,
parent attachment, child orientation, section length, and membrane area.
Build one BrainCell branch per source section. Preserve tapered pieces.
Use a cylinder only for sections that lack 3-D diameter samples in NEURON.
Do not replace a taper with its average diameter.
Preserve the template's two inferred axon sections and label their provenance.
Compare each branch length and area with the independent NEURON export.
Reject missing parents, disconnected cycles, and unsupported section types.
The exported geometry is from the published reference cell, not H01 anatomy.

Paint passive leak and Ih on all branches. Paint the remaining nine channels
on soma and axon with the exact released conductance densities in S/cm2.
Use Ra 100 ohm cm and Cm 2 uF/cm2 everywhere. Set sodium reversal to 50 mV
and potassium reversal to -85 mV. Initialize voltage at -80 mV.
Place the 270-1270 ms current step and voltage observation at soma position 0.5.
Allow passive-only assembly to check load without active-channel effects.
This switch removes Ih as well as the other active channels.
Record calcium and SK at the soma for later causal checks.
