# Frozen E/I spiking checks

## Donor I transfer

The selected I profile reproduces five complete events from 270 to 300 ms on
the donor morphology at 0.27 nA. At dt 0.00125 ms and maximum CV length 2.5 um,
onset and duration pass, but peak error reaches 0.135506 mV and fails the 0.1 mV
limit. With only dt halved to 0.000625 ms, all first-burst gates pass:

| Direct measurement | Largest absolute error | Limit |
| --- | ---: | ---: |
| -20 mV rising crossing | 0.0411484 ms | 0.1 ms |
| Sampled peak voltage | 0.0686396 mV | 0.1 mV |
| Duration above -20 mV | 0.00042575 ms | 0.01 ms |

The saved audits retain every event and signed error. This is first-burst
transfer evidence only. Full-sweep transfer, spatial refinement, and human
waveform qualification remain open. No candidate parameters were fitted.

## H01 response

A 1 nA soma pulse from 2 to 5 ms was applied to each original H01 component.
At dt 0.005 ms, the E soma has a peak of +48.257934 mV at 4.12 ms. The I soma
has a peak of -14.100264 mV at 3.63 ms and no 0 mV crossing. Its final soma
voltage is -47.834077 mV at 10 ms. A -20 mV excursion is not a qualified spike.

For I, reducing dt eightfold to 0.000625 ms while preserving the physical
profile, input, and electrical intervals gives peak -14.072179 mV and final
voltage -47.954307 mV. There is still no 0 mV crossing. This limits a time-step
explanation under the tested conditions; it does not qualify spatial accuracy
or identify the physical cause. At the finer time step, reducing maximum CV
length from 10 to 2.5 um gives peak -14.060704 mV. This spatial change also
fails to restore a 0 mV crossing. With the fine time and space settings, a
2 nA pulse gives peak -1.355538 mV. Stronger input alone is not evidence of
a regenerative spike. See the time and space audits and the two-nA event
record in this directory.

## Circuit emission

The installed Network reduces compartment spike flags with logical any.
Therefore crossings in successive compartments can generate successive output
events. The new output-site helper restricts emission to one declared membrane
CV near a specified source point. Tests verify that other crossings emit no
event, a changed mask shape fails, and a compiled Cell run accepts the helper.
The source-node voltage and selected CV midpoint remain distinct locations.

The helper has 96% statement coverage. Its four tests pass with UserWarning
as error. E/I builder tests also pass with population size (1,), which Network
requires for each distinct morphology. The driver and direct-event audit have
14 passing tests, including empty, missing, shifted, and clipped event cases.

## Connected H01 circuit

The two-cell circuit is implemented with illustrative E-to-I and I-to-E
contacts. Each cell retains its measured source identity. The circuit records
source-soma voltage, output-CV voltage, incoming conductance, and output events.
Four controls retain the same cells and synapses and change only projections.

The real H01 E-only and disconnected runs completed 10 ms at dt 0.005 ms.
Their cell records and time grids match. Both emit one E event at 3.975 ms.
In E-only, I conductance first appears at 4.475 ms: exactly 100 time steps later.
The I voltage is identical to the disconnected control before delivery. At
first delivery, I voltage is 0.273514 mV higher. I reaches -61.770359 mV and
emits no event. The disconnected I conductance remains zero.

The [direct delivery audit](h01-ei-circuit-e-delivery-audit.json) retains file
hashes and each decision. This establishes excitatory delivery for this input
and inferred contact. It does not establish inhibitory feedback or human validity.
The reciprocal run also completed. Every saved trace is exactly equal to
E-only. There is no I event and no incoming conductance in E. The
[feedback audit](h01-ei-circuit-feedback-audit.json) therefore records absent
feedback, not successful inhibition.

The two small-geometry event tests pass with the tighter 100-or-101-step delay
limit. They verify E delivery and I-induced voltage reduction with actual
presynaptic events. They do not qualify the real H01 interneuron. Existing
coverage data show circuit 100%, output helper 96%, and cell builder 97%.
The added midpoint-probe test checks that this probe preserves the output CV.

The real H01 I spike response remains unqualified.

## H01 I pre-pulse split

The 1 nA, 3 ms pulse was delayed from 2 to 270 ms. Candidate, electrical
intervals, initial voltage, dt 0.005 ms, maximum CV length 10 um, and solver
were unchanged. The last pre-pulse sample changed from -81.758462 to
-86.925026 mV. Peak voltage changed from -14.100264 to -14.327054 mV.
Neither response crossed 0 mV. Thus, the longer wait alone does not restore
a positive spike at this input. It does alter the state and response; it is
not evidence that all initial-state effects are absent. The
[pre-pulse audit](h01-i-prepulse-audit.json) retains each direct measurement.

The [electrical area diagnostic](h01-i-electrical-area-diagnostic.json) records
157.019733 um2 of soma, 24.773880 um2 of axon, and 2405.847731 um2 of remaining
dendritic membrane under the current inferred map. These areas are structural
quantities. They do not identify the cause of the observed response.

## Local H01 I currents

Read-only probes at the existing soma source location preserve the complete
baseline voltage trace exactly (maximum absolute difference 0 mV). At the
3.63 ms peak of -14.100264 mV, transient sodium activation m is 0.943119 and
availability h is 0.004951. Transient sodium inward current is 0.143971 mA/cm2;
Kv3 current is -0.174480 mA/cm2 under the inward-positive convention.
At pulse onset h was 0.993398. The
[local datums](h01-i-local-current-datums.json) retain each current and gate
at onset, peak, pulse end, and recovery, with the raw trace hash.

This is observation, not a gate intervention. The sample does not include
axial or applied current, and it cannot establish complete charge balance.
The fall in sodium availability motivates an inactivation intervention;
it does not prove that inactivation alone caused the missing positive spike.
No candidate parameter was changed.

## Local H01 I charge balance

The soma voltage probe maps to CV 144 and is its midpoint. The CV area is
3.838577 um2 and total capacitance is 0.0000767715 nF. Its reduced axial
couplings connect to CVs 143 and 145. A compiled diagnostic loop reproduces
the original voltage trace exactly.

At the 3.63 ms voltage peak, applied current is +1 nA. The two inward-positive
axial currents are -0.792096 and -0.206750 nA. Net channel current is
-0.001573 nA. Thus, the measured axial terms carry almost all the applied
current out of this small injection CV at this instant. These are currents
at one CV, not total currents for the whole soma or cell.

The central-difference charge-balance residual reaches 0.001129 nA away from
initialization and pulse jumps. This fails the stated 0.00001 nA diagnostic
limit. The direct terms are retained, but complete balance is not qualified.
At dt 0.000625 ms, the residual is 0.000141713 nA: about eightfold smaller,
but still above the same limit. Both voltage traces exactly match their
respective uninstrumented baselines. The
[current-balance time audit](h01-i-current-balance-time-audit.json) records
both failures. Discrete voltage and gate evaluation timing needs inspection. See `h01-i-current-balance.json` and
its NPZ for geometry, couplings, direct currents, and the complete residual.

## Discrete current timing

The installed staggered solver advances voltage with a membrane law evaluated
from the old gates, then advances the gates. The earlier central voltage
difference and end-step current sample therefore use different time levels.

The diagnostic now retains that earlier residual and adds a discrete-step
balance. It uses the actual step voltage increment, the pre-update linear
membrane law evaluated at the new voltage, and new-voltage axial currents.
At dt 0.005 ms, the maximum discrete residual over all steps is
2.275668e-11 nA. This passes the unchanged 1e-5 nA bookkeeping limit.
The full voltage trace still matches the baseline exactly. See
`h01-i-current-balance-discrete.json` and its NPZ. The check shares the
solver's current law; it is not an independent validation of physical accuracy.
The regional comparisons also pass: maximum discrete residuals are
2.756962e-11 nA for soma block and 2.275668e-11 nA for axon block. Each
recorder reproduces its corresponding earlier voltage trace exactly.
At the common 3.63 ms sample, axon block reduces net membrane input by
0.014545 nA relative to baseline but reduces axial outflow by 0.017064 nA.
The resulting increase in capacitive current is about 0.002519 nA. This
explains the local rising-voltage difference at that instant through measured
current terms. It does not isolate the upstream axonal segment or mediator.
The [regional current audit](h01-i-regional-current-balance-audit.json) retains
both neighbor contributions at matched time and at each response's own peak.

## Intact source dynamics on H01

The complete published I source profile reaches +36.017608 mV on the same
H01 geometry and 1 nA input where the candidate remains below 0 mV. This
comparison changes several profile parameters, so it does not isolate one.
The finer source run uses dt 0.000625 ms and maximum CV length 2.5 um;
its direct measurements are in the source comparison audit.

A separate one-factor intervention restores only the NaTg h closing-time
factor from 0.15 to the source value 1.0. All other candidate parameters
remain. Its intact dynamic gates produce a +35.036700 mV peak and return to
-81.186204 mV by 10 ms. This is a source-value restoration, not an artificial
availability clamp. Numerical refinement and human waveform qualification
for this single override remain open. The selected default is unchanged.
See the [source comparison](h01-i-source-comparison-audit.json).

## Source closing-time refinement

The single source closing-time restoration retains one positive crossing
under joint time and space refinement. Peak changes from +35.036700 to
+35.213356 mV, and final voltage from -81.186204 to -81.192764 mV.
This supports persistence of the response, not a 0.1 mV precision gate.
The [refinement audit](h01-i-source-closing-refinement-audit.json) retains each
crossing, peak, recovery sample, availability and file hash.

Matched real H01 I-only and disconnected runs completed with this explicit
I override and identical 1 nA pulses. E retains its selected candidate. These
runs test direct inhibitory delivery and response; they do not promote the
override or establish human validity. The circuit edges remain illustrative.

## Direct H01 inhibitory delivery

With the source closing-time restoration, I emits at 3.25 ms in both controls.
I voltage is exactly equal across controls. The I-only projection delivers
conductance to E at 3.75 ms, exactly 0.5 ms after emission. E voltage matches
the disconnected control exactly before delivery. The first delivered sample
reduces E voltage by 0.206212 mV. E emits at 4.010 ms instead of 3.975 ms.
Inhibition delays this event by 0.035 ms; it does not remove it.

Cell records, input, time grid and I override match. Disconnected incoming
conductance is zero. The [delivery audit](h01-i-restored-circuit-delivery-audit.json)
retains all decisions and file hashes. The [direct traces](h01-i-restored-circuit-delivery.png)
show both cell voltages and received conductance. Half-time-step controls
also complete: I emits at 3.245 ms and delivery occurs at 3.745 ms. E emits
at 4.010 ms versus 3.9725 ms disconnected, a 0.0375 ms delay. All paired
delivery checks pass at both time steps. The
[time audit](h01-i-restored-circuit-time-audit.json) records the 0.0025 ms
change in measured delay. Full spatial and reciprocal checks remain open. This is functional inhibition evidence with an explicit diagnostic
override and inferred wiring, not complete human or circuit qualification.

## Four matched circuit controls

All four controls completed at dt 0.0025 ms with identical cell records and
inputs. Disconnected and E-only emit E at 3.9725 ms; I-only and reciprocal
emit E at 4.010 ms. All controls emit I at 3.245 ms. Reciprocal delivery times
are 3.745 ms into E and 4.510 ms into I, each 0.5 ms after its source event.

The E-only source E voltage is identical to disconnected. Receiving I voltage
matches before arrival and initially rises after arrival. Reciprocal E voltage
is identical to I-only, and its pre-arrival voltage matches E-only. These
checks establish the direct effects of both enabled contacts for this input.
The [four-control audit](h01-i-restored-four-control-audit.json) retains all
identities, checks, times and hashes.

Both cells receive external current. I fires before E input reaches it. Thus,
this result does not establish recruitment of I by E, repeated feedback, or
self-sustained activity. Human and full spatial qualification remain open.
