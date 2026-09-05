# Published PV cell assembly

The BrainCell cell uses the pinned HL5BN1 template geometry.
The [independent export](h01-pv-geometry-reference.json) has 29 sections,
157 original NEURON segments, and 5746.131785301348 square micrometres of membrane.
BrainCell uses its own electrical mesh. The original segment count is not its mesh size.

Every imported branch matches the original section length and membrane area.
Parent attachment and child orientation also match.
The soma has a cylindrical cable representation from the original import.
The two axon sections come from the source template's replacement rule.
They are inferred model geometry. They are not a complete measured axon.
This geometry is from the published reference cell, not the H01 donor.

The source section name `soma[0]` becomes `soma_0` in BrainCell.
The same reversible name conversion applies to each section.
BrainCell requires valid Python identifiers for branch names.
The geometry test caught the initial use of bracketed names.
Future source imports must check naming constraints before assembly.

The builder applies the exact released soma and axon conductance tables.
Source S/cm2 values are multiplied by 1000 for BrainCell mS/cm2.
An assembly test caught the unsupported `u.S` alias; the explicit conversion
now states both the source scale and the runtime scale.
All branches receive passive leak and Ih. Soma and axon receive the other channels.
The fitted calcium removal times remain separate for soma and axon.

A passive-only cell follows the analytic uniform leak relaxation.
This check tests capacitance and leak together. Uniform voltage does not test
axial resistance. A spatially varying passive response still needs comparison.
The active initialization check records finite voltage and calcium, with bounded SK gates.
These checks do not establish the full human spike response.

Run the full trace driver from the worktree root as a module:

```powershell
.cache/validation/Scripts/python.exe -m docs.evidence.h01_pv_braincell_reference --output docs/evidence/h01-pv-braincell-019
```

Module invocation keeps the worktree package on the import path.
Direct script invocation did not find that package in the validation environment.
The driver records voltage, calcium, and SK state for the full published current step.
Use the sample clock and voltage datum without shifting spike times.
Numerical comparison and human waveform validation remain open.

## First full response

At 0.005 ms with maximum compartment length 10 micrometres, the BrainCell cell
produces 12 spikes during the 0.19 nA step. The accurate NEURON reference has 13.
BrainCell spike peaks are near 44 mV.
The first spike occurs at 302.565 ms. Later spike times differ increasingly.
The [saved trace report](h01-pv-braincell-019.json) records each spike.

The [direct comparison](h01-pv-transfer-comparison.json) uses the same clock.
Baseline voltage RMS difference is 0.000300 mV over 200-270 ms.
Stimulus voltage RMS difference is 11.9088 mV over 270-1270 ms.
The large transient residual includes spike timing differences.
No spike alignment or voltage offset was applied.
The transfer does not pass whole-trace agreement.

The next run halves the time step to 0.0025 ms without changing geometry,
conductances, or stimulation. This separates a numerical effect from a
proposed parameter change. Spatial convergence remains a separate check.

The full dataset test suite passes 169 tests with 99.78 percent coverage.
This is implementation evidence, not biological qualification.

## Time and spatial refinement

BrainCell at 0.0025 ms still gives 12 spikes. Its last spike moves 2.2075 ms
relative to the 0.005 ms run. This does not establish time convergence.

The original NEURON mesh has 157 segments. At CVode tolerance 1e-10 it gives
13 spikes. Multiplying each section's count by 3 gives 471 segments and 12 spikes.
Multiplying by 9 gives 1413 segments and 11 spikes.
Physical geometry, channel parameters, temperature, and stimulus do not change.
The original reference is converged in time only. It is not converged in space.
The temporary 12-spike agreement between solvers is not a qualification gate.

The passive current-step comparison differs by at most 0.001895 mV during
270-1270 ms. This supports the passive cable transfer but does not remove
active mesh sensitivity. Calcium and SK are now recorded in both solvers.

The [numerical audit](h01-pv-numerical-audit.json) retains controlled response RSS.
For the same 200000-sample window, NEURON mesh factors 1 to 3 produce RSS
5316.25 mV; factors 3 to 9 produce 4821.22 mV. Halving the BrainCell time step
produces 4312.53 mV. These rank the stated interventions only.
They do not estimate biological uncertainty or prove the dominant microscopic cause.
No voltage shift or spike alignment was used.

At factor 27, NEURON has 4239 segments and still gives 11 spikes.
However, its last spike moves by 8.5298 ms relative to factor 9.
Stable count alone does not establish spatial convergence.
The factor-81 reference and BrainCell maximum-CV-length 5 um runs are the next checks.

## Regional interventions

Each intervention multiplies only the selected region's segment count by 9.
The other regions, physical parameters, current, and solver tolerance stay fixed.

| Region refined | Spikes | Direct voltage-change RSS (mV) |
| --- | ---: | ---: |
| None | 13 | 0 |
| Soma | 13 | 889.33 |
| Axon | 11 | 5102.90 |
| Dendrites | 12 | 5184.32 |

All RSS values use 200000 samples over 270-1270 ms, with no spike alignment.
Dendrite refinement has the largest RSS of these three interventions.
Axon refinement has the largest spike-count change.
Neither statement identifies a dominant biological parameter uncertainty.
The combined effects are not additive.
The [direct plots](h01-pv-mesh-interventions.svg) show the full stimulus window
and a common early-response window.

The factor-81 reference retains 11 spikes. Its last spike moves 1.06387 ms
relative to factor 27. This is smaller than the previous 8.52983 ms change,
but full waveform convergence has not been established.
The high-resolution reference has 12717 segments.

The completed BrainCell 5 um run at 0.0025 ms gives 11 spikes.
The last spike is at 1170.5075 ms, versus 1183.7951 ms for NEURON factor 81.
The [direct comparison](h01-pv-space5-transfer-comparison.json) retains the
voltage, calcium, and SK residuals. Stimulus voltage RMS error is 10.3059 mV.
The matching count does not remove the timing mismatch.
The next BrainCell run uses maximum CV length 2.5 um at the same time step.

The completed 2.5 um run retains 11 spikes. Its last spike is at 1180.8600 ms,
2.9351 ms before the NEURON factor-81 reference. The first spike is at 302.6000 ms,
versus 302.5915 ms in that reference. Late timing error remains larger than early error.
The [direct trace comparison](h01-pv-space2p5-transfer-comparison.json) has
stimulus RMS voltage error 9.84383 mV. Peak timing improvement does not establish
full voltage-trace agreement. The next run halves dt to 0.00125 ms at the same mesh.

The [installed API guide](../h01-pv.md) gives the runnable configuration and
the separate installed-wheel check. Packaging is verified; physiology is not.
