# H01 cell and circuit: causal explanation

This document states the causal explanation for each observed behavior of
the H01 cell and circuit models. A causal explanation names the behavior,
the necessary and sufficient conditions, and the how-why mechanism that
produces it. Each explanation applies only within its stated boundary.
Replace an explanation when stronger evidence contradicts it.

Each behavior has one search tree. The tree records every split that was
run, which branches it eliminated, and which branch remains open. Eliminated
branches are not repeated in the explanation text. Experiment methods,
settings, scores, and decisions belong in [evidence](evidence/).
Numerical qualification of the simulation is reported in its own section,
separate from the physical explanations.
The language uses short sentences and defined terms. Formal ASD-STE100 review is pending.

## What the observations represent

H01 supplies reconstructed anatomy and annotations. It does not supply voltage
recordings from the imported cells. Our active models combine anatomy and
physiology from different cells. They do not reproduce the H01 donor's physiology.

The inhibitory reference is a human-fitted putative PV cell. Its channel laws
include borrowed kinetics. Its direct spike waveform is not yet validated.
The selected excitatory reference uses an Allen human pyramidal cell fit.
Its channel laws include borrowed mechanisms. The Wilbers channel model remains
a separate reference. Direct waveform validation remains incomplete.

H01 labels identify a pyramidal cell and an interneuron for the selected pair.
The interneuron label does not identify a PV subtype. The assigned electrical
regions and PV dynamics remain model assumptions. Neither a cell tag nor a
synapse position supplies a missing partner identity.

The circuit topology sets which cell can supply synaptic current to another
cell. A wrong partner assignment changes that causal path, even when the
channel equations are correct. Proofread-cell IDs and C3 segment IDs are
different identity systems. A measured edge needs a checked mapping from
each synapse endpoint to its cell. The proofreading archive supplies base
segment membership; the synapse export supplies endpoint base IDs. Spatial
checks must also resolve cut or ambiguous segments. These records establish
anatomical support. They do not measure synaptic conductance or delay.

A direct observation retains location, time, input, and response.
A firing rate or spike count cannot explain a voltage trajectory.
Equal counts can conceal different spike times. Equal peaks can conceal different
rising and falling phases. Total spike duration can conceal opposing rising and
falling errors, because it is their sum.

## Datum

| Quantity | Reference |
| --- | --- |
| Voltage | Inside minus outside. Retain the source junction correction. |
| Time | The stimulus clock. Retain pulse onset, end, and sample convention. |
| Position | The source cell, component, node, origin, and coordinate scale. |
| Current | State the direction and whether the value is total current or current density. |
| Initial state | Voltage, channel gates, calcium, temperature, and preceding input. |
| Biological identity | Recording identifier, protocol, and source version. |

BrainCell outputs use the end of each step. Geometry and synapse exports use
different coordinate scales. Radius has its own unit. These conversions define
the observed system; they are not free fitting parameters.

The released PV calibration traces already include the junction correction.
Applying it again changes the datum and invalidates the comparison.
The acquisition bias current is separate from the test pulse. Omitting that
current changes the input. The
[source and datum evidence](evidence/h01-pv-acquisition-audit.md) records
these conventions.

Repeated human threshold trials can have the same measured command and
bias but different spike outcomes. Measured input alone is therefore
insufficient to specify the observed human response at that input. The
[repeated-input evidence](evidence/h01-l2-repeatability200.md) applies at
200 pA; it does not establish uncertainty at the calibration input.

## Physical relations used

For a compartment with constant total capacitance:

\[
C\frac{dV}{dt}=I_{applied}+I_{axial}+I_{synapse}+\sum_k I_k.
\]

All currents in this equation are positive into the compartment.
NEURON channel-current exports use the opposite sign. Convert signs explicitly.
Net current changes stored membrane charge. That charge change changes voltage.
Large opposing currents can produce a small voltage change.
Voltage alone therefore cannot identify which current caused a response.

For a channel branch:

\[
I_k=g_k(E_k-V).
\]

Conductance includes membrane area, channel density, and gate state.
Conductance density times membrane area gives maximum total conductance.
Moving the same density to a different area changes total conductance. Cable
length, radius, and attachment change axial current. Membrane area grows with
radius times length; axial conductance grows with radius squared and falls
with length and resistivity. Matching channel laws therefore does not imply
that two cells have the same soma voltage response.

A channel name does not define its complete law. The Allen persistent sodium
channel has immediate activation and one dynamic inactivation gate. The PV
channel has a dynamic activation gate and uses its third power. These laws give
different currents at the same voltage and gate state. A transfer must retain
the source law, its temperature rule, and its region-specific parameters.

Voltage changes gate states and calcium. These states change current, which
changes the next voltage. A response therefore depends on its preceding state
history. Initial voltage does not specify channel availability; slow gates
retain prior input after voltages become similar. A causal account must
include conditioning and the relevant internal states. In a gate law, the
time constant sets the speed of motion toward the equilibrium gate value and
does not change that equilibrium at fixed voltage.

The reversal potential represents the ion's electrochemical driving condition.
A channel is not a resistor connected only to electrical ground. Ion gradients
supply electrical energy; their maintenance is outside the present
short-duration model when concentrations or reversal potentials are fixed.
For constant capacitance, stored electrical energy is \(CV^2/2\). Channel
dissipation uses the channel driving voltage, \(g_k(V-E_k)^2\), not membrane
voltage alone. A complete energy budget must include the ion-gradient reservoirs.
The source-load framework locates observation boundaries and splits current
paths. A fixed linear source-load equivalent describes a passive subsystem or
a local linearization; it cannot replace the spiking dynamics, and a delayed
voltage-current response does not by itself prove physical inductance.

```mermaid
flowchart LR
    Source[Applied current and ion gradients] --> Currents[Membrane current paths]
    Gates[Channel gate states] --> Currents
    Currents --> Charge[Membrane charge]
    Charge --> Voltage[Local voltage]
    Voltage --> Gates
    Neighbour[Neighbouring compartment voltage] --> Axial[Axial current]
    Axial --> Charge
    Voltage --> Next[Next compartment and receiving cell]
    Boundary([Observation boundary: soma voltage and applied current]) -.-> Voltage
    Boundary -.-> Source
```

| Boundary | Paired direct observations | Additional state needed |
| --- | --- | --- |
| Membrane | Voltage and signed current | Gates, calcium, capacitance |
| Cable connection | Voltage difference and axial current | Geometry and axial resistivity |
| Channel | Driving voltage and channel current | Conductance and gate state |
| Calcium pool | Calcium entry and concentration change | Effective volume, buffering, removal |
| Synapse | Synaptic current and receiving voltage | Conductance, reversal potential, delay |

A voltage-current pair characterizes a boundary. It does not uniquely determine
the hidden states of a nonlinear neuron. Fixed-voltage observations separate
channel response from voltage feedback. Current-driven observations retain that
feedback. Both are needed to distinguish competing explanations.

## Y1. The H01 inhibitory cell fires no positive somatic spike at 1 nA

**Behavior.** With the selected H01 I region map and the tested 1 nA pulse,
the soma voltage never becomes positive.

**Explanation.** Inward sodium current requires two conditions: the activation
gate must open and the availability gate must remain open. In this model the
candidate's accelerated inactivation closes availability before the rise
completes, so sodium conductance falls while activation is still high and the
regenerative rise stops. Restoring only the inactivation closing-time factor to
the published source value is sufficient to produce a positive excursion and
recovery on this H01 geometry. Preventing the availability loss in soma alone
or in axon alone is also sufficient; in the axon-only case soma availability
still falls, and the matched local balance shows less current leaving the soma
injection compartment through axial connections than the increased local
membrane loss, so the remaining current continues to charge the membrane.

```mermaid
flowchart LR
    V[Voltage rises] --> M[Sodium activation increases]
    V --> H[Sodium availability falls]
    M --> G[Sodium conductance]
    H --> G
    G --> I[Inward sodium current]
    I --> V
```

**Prediction confirmed.** Both regional blocks and the source closing-time
restoration were predicted to persist under refinement, and they do: the
blocks persist at a smaller time step and the restoration persists under joint
time and space refinement.

**Steep X.** The sodium inactivation closing-time factor. Its direct voltage
response is much larger than the pre-pulse change, and numerical changes are
smaller still.

**Action.** The source closing-time value is the lever that restores a
positive response in the diagnostic circuit. Time step, compartment size, and
pre-pulse period are not levers for this behavior. This explanation does not
establish a valid human spike, does not justify removing inactivation from the
default model, and does not identify a generally correct closing rate. The
electrical region map remains unqualified.

```mermaid
flowchart TD
    Y1[No positive somatic spike at 1 nA] --> N[Numerical error]
    Y1 --> S[Initial state]
    Y1 --> C[Channel state]
    N --> N1[Time step: eliminated]
    N --> N2[Compartment size: eliminated]
    S --> S1[Short pre-pulse: eliminated]
    C --> C1[Sodium availability loss: sufficient]
    C1 --> C2[Closing-time factor: sufficient]
    C1 --> C3[Axial mediation: open]
```

| Node | Split | Result | Evidence |
| --- | --- | --- | --- |
| N1 | Smaller time steps over the tested range | No positive spike; eliminated | [spike-time audit](evidence/h01-i-defaults-spike-time-audit.json) |
| N2 | Smaller compartments at the fine step | No positive spike; eliminated over the tested range | [spike-space audit](evidence/h01-i-defaults-spike-space-audit.json) |
| S1 | Longer pre-pulse period | Starting voltage changes; no positive spike; eliminated | [pre-pulse audit](evidence/h01-i-prepulse-audit.json) |
| C1 | Block availability loss in soma, axon, or both | Positive excursion in each case; persists at smaller step | [regional audit](evidence/h01-i-inactivation-regions-audit.json) |
| C2 | Restore source closing-time factor only | Positive excursion and recovery; persists under joint refinement | [source-value audit](evidence/h01-i-source-closing-refinement-audit.json) |
| C3 | Matched current balance, axon-only case | Reduced axial outflow explains the local rise; upstream segment not identified | [current balance audit](evidence/h01-i-regional-current-balance-audit.json) |
| rank | Direct response size of each intervention | Inactivation > pre-pulse >> numerical | [factor evidence](evidence/h01-i-factor-response-rss.md) |

## Y2. The transferred PV donor model does not retain its full response

**Behavior.** The frozen inhibitory donor model in BrainCell reproduces the
early spike peaks and widths of its source, but its later rise crossings drift
from event 7 onward (1.2 ms at event 8 over 270-329.5 ms at 0.27 nA). This
behavior concerns transfer on donor anatomy; it is separate from Y1.

**Explanation.** The drift is a spatial discretization difference, not a
simulator difference. The earlier comparisons matched the two meshes only by
mean segment length, so several dendritic branches held fewer BrainCell
compartments than NEURON segments. Copying the NEURON per-branch counts into
BrainCell is sufficient to remove the drift: with equal counts and the same
fixed step, the two simulators agree at every one of the eight events to below
1e-8 ms, and with equal counts against NEURON CVode the largest rise error is
0.044 ms, inside the 0.1 ms gate. Changing the NEURON integration from CVode
to a fixed step at the BrainCell dt moves no event by more than its gate, so
the integration scheme and the gate time level are not measurable contributors
at this dt.

**Prediction confirmed.** Stated before running: if mesh is the dominant
element, both matched-mesh cells pass and both MaxCVLen cells fail on rise
only. That pattern occurred. Halving dt in each simulator at the matched mesh
moved event 8 by 0.002 ms, so the gate is a valid decision limit.

**Steep X.** The per-branch compartment count.

**Action.** Every BrainCell-versus-NEURON comparison must copy the NEURON
per-branch counts (`--mesh-from`) rather than choose a maximum compartment
length. The earlier drift evidence and the axon-only 0.25 um control were read
through unequal meshes. This result qualifies the transfer for the first eight
events at this input and dt only; it does not identify a channel cause, does
not validate the human waveform, and does not cover the full 1270 ms train or
other inputs.

```mermaid
flowchart TD
    Y2[Later spikes drift after transfer] --> W[Short-window agreement: not sufficient]
    Y2 --> T[Halved BrainCell step alone: eliminated]
    Y2 --> I[Implementation differences]
    I --> A[Mesh: per-branch counts unequal]
    I --> B[Integration scheme: CVode vs fixed step]
    A --> A1[Copy NEURON counts: sufficient, both cells pass]
    B --> B1[Scheme swap: moves no event past its gate]
    A --> A0[Axon-only 0.25 um control: consistent, uncontrolled]
```

| Node | Split | Result | Evidence |
| --- | --- | --- | --- |
| W | Full-response comparison against the source | Early spikes agree; later events and counts differ | [full-response comparison](evidence/h01-pv-candidate-transfer-full.md) |
| T | Halve the BrainCell time step, physical parameters unchanged | First failed crossing remains outside the timing limit | [paired comparison](evidence/h01-pv-candidate-transfer-drift-halfdt-audit.json) |
| A0 | Refine only the BrainCell axon to 0.25 um | All gates pass 8/8; not a controlled swap | [axon-only control](evidence/h01-i-axon025-transfer-audit.json) |
| A1, B1 | 2x2 half-split: mesh x integration, everything else held equal | Decision "mesh": matched cells pass, MaxCVLen cells fail on rise; scheme swap within gate | [isolation split](evidence/h01-pv-transfer-isolation.md), [audit](evidence/h01-pv-transfer-isolation-audit.json) |

## Y3. The PV candidate's spike train differs from the human recording

**Behavior.** Against the human recording, the candidate's first spike lasts
too long, its post-spike minimum is too shallow and too late, its approach to
the first spike is too slow at low input, and at high input it does not
sustain late firing.

**Explanation.** Four partial mechanisms are supported; no single sufficient
condition reproduces the whole train.

1. *First-spike duration.* Faster sodium inactivation shortens the first spike
   while preserving spike generation, so inactivation timing contributes
   causally to the excessive duration. During the early fall, sodium
   activation remains high and the driving voltage grows, so inward sodium
   current can increase while the inactivation gate closes; Kv3 current
   opposes it.
2. *Peak.* At the observed soma peak, neighbouring soma membrane supplies
   axial current to the probe compartment and the attached dendrites draw
   current away. Net axial outflow balances most of the applied and net ionic
   inflow, so little current remains to change stored charge and the local
   voltage stops rising. The peak cannot be understood from sodium and
   potassium alone.
3. *Return and minimum.* Reducing somatic Ca_LVA conductance, increasing
   somatic Kv3 conductance, and faster Kv3 gating each advance the voltage
   return. Faster opening and faster closing have opposite effects on the
   falling crossing and the minimum, so the two phases are not
   interchangeable. Total duration can hide these opposing phase errors.
4. *Late firing.* Stronger axonal SK conductance is sufficient to reproduce the
   added first-spike delay seen when SK increases in both regions; the
   soma-only change is not. Calcium removal acts on concentration above rest,
   so slower removal alters later intervals while leaving first onset nearly
   unchanged; this separates a memory of prior activity from an immediate
   conductance change. A steeper inactivation curve restores late firing at
   high input; the raw slope parameter changes both equilibrium availability
   and gate speed, so the rescue is not attributed to either alone.

```mermaid
flowchart LR
    Ca[Calcium enters] --> Pool[Calcium concentration rises]
    Pool --> SK[SK gates open]
    SK --> K[Potassium current leaves]
    K --> V[Voltage response changes]
    Pool --> Removal[Buffering and removal]
```

**Prediction confirmed.** The first-spike shortening from faster inactivation
persists across the two checked meshes. The high-input absence of late spikes
persists on the finer mesh. The slope rescue persists on the finer mesh, with
the first-return mismatch. On fine axonal meshes, the final-onset shift follows
the predicted squared-compartment-length dependence within the declared limit.

**Steep X.** Not separated across the whole train. Sodium inactivation timing
dominates first-spike duration. Early and late timing require separate
constraints; total spike count cannot establish the adaptation history.

**Action.** Calibration must constrain the post-spike minimum voltage and its
time from the peak, and the rising and falling phases separately. Interventions
that only change one of peak, minimum, or interval are not independent
controls: added somatic sodium advances onset and raises the peak; a Kv3
conductance increase also delays onset and narrows the spike; faster recovery
alone and slower recovery alone both fail to restore high-input late firing;
SK is not necessary for the second-spike delay after faster inactivation.
Somatic SK observations alone cannot explain the onset effect. None of the
tested candidates is a human waveform fit. Every Y3 result was obtained in
NEURON; the Y2 split shows that a BrainCell reading of these results requires
the copied per-branch mesh.

```mermaid
flowchart TD
    Y3[PV train differs from human] --> D[First-spike duration]
    Y3 --> P[Peak and onset]
    Y3 --> R[Return and minimum]
    Y3 --> L[Late firing and adaptation]
    D --> D1[Faster inactivation: shortens; contributes]
    D --> D2[Faster recovery alone: does not shorten]
    P --> P1[Axial outflow sets the peak]
    P --> P2[Somatic sodium density: not an independent onset control]
    P --> P3[Applied current alone does not set the local slope]
    R --> R1[Ca_LVA reduction: advances return]
    R --> R2[Kv3 conductance: advances return, shifts onset]
    R --> R3[Kv3 gating speed: opening and closing oppose]
    R --> R4[Minimum depth and delay: located, cause open]
    L --> L1[SK necessity for second-spike delay: eliminated]
    L --> L2[Axonal SK: sufficient for onset delay]
    L --> L3[Calcium removal and entry: change late intervals, not early]
    L --> L4[Recovery speed either direction: not sufficient]
    L --> L5[Steeper inactivation curve: restores high-input firing]
    L --> L6[Late-interval mesh sensitivity: axonal discretization]
```

| Node | Split | Result | Evidence |
| --- | --- | --- | --- |
| D1, D2 | Faster inactivation; faster recovery alone | Shorter first spike; recovery alone does not shorten; increased firing also without recovery change | [inactivation and recovery test](evidence/h01-pv-recovery-causal-test.md), [spatial robustness](evidence/h01-pv-inactivation-mesh-audit.json) |
| P1 | Local charge balance at the peak | Net axial outflow balances inflow | [charge balance](evidence/h01-pv-charge-balance-audit.json), [channel currents](evidence/h01-pv-spike-current-audit.json) |
| P2 | Somatic vs axon-only sodium increase | Somatic advances onset and raises peak; axon-only does not | [regional sodium](evidence/h01-pv-onset-sodium.md) |
| P3 | Independent local balance during the approach | Slower low-input approach consistent with axial and membrane flows | [onset balance](evidence/h01-pv-onset-balance.md) |
| R1 | Reduce somatic Ca_LVA | Deeper, later minimum; reaches falling levels earlier relative to peak; still lags human | [conductance split](evidence/h01-pv-calva-return.md), [falling crossings](evidence/h01-pv-return-crossings.md) |
| R2 | Increase somatic Kv3 conductance | Advances return, deepens minimum, delays onset, narrows spike | [Kv3 split](evidence/h01-pv-kv3-return.md) |
| R3 | Faster Kv3 gating; opening vs closing | Smaller onset shift; opening and closing give opposite return effects; exact equivalence rejected | [timing split](evidence/h01-pv-kv3-fast.md), [phase comparison](evidence/h01-pv-kv3-phases.md) |
| R4 | Direct minima against the recording | Minimum too shallow and too late; cause not isolated | [direct minima](evidence/h01-pv-postspike-minima.md) |
| L1 | Remove SK after faster inactivation | Second-spike delay persists | [SK necessity](evidence/h01-pv-sk-necessity-audit.json) |
| L2 | Axonal vs somatic SK increase | Axonal sufficient for first-spike delay; somatic not | [regional SK](evidence/h01-pv-sk-region-audit.json) |
| L3 | Calcium removal, entry scaling, and their combination | Late intervals change with little first-onset change; entry scaling overshoots and is rejected; early and late timing need separate constraints | [calcium removal](evidence/h01-pv-calcium-candidate-audit.json), [adaptation comparison](evidence/h01-pv-opening-adaptation.md), [entry scaling](evidence/h01-pv-calcium-entry.md), [burst comparison](evidence/h01-pv-burst-adaptation.md), [response-time comparison](evidence/h01-pv-calcium-response.md) |
| L4 | Faster recovery; slower recovery | Neither restores high-input late firing; recovery speed does not test equilibrium sodium | [recovery split](evidence/h01-pv-recovery-rescue.md), [opposite direction](evidence/h01-pv-recovery-slow.md), [gate account](evidence/h01-pv-gate-equilibrium.md) |
| L5 | Steeper inactivation curve | Restores high-input late firing; first return still too slow and too shallow | [slope intervention](evidence/h01-pv-slope5.md) |
| L6 | Refine axon, soma, or dendrites alone | Only axonal refinement reproduces the late-interval change | [regional mesh](evidence/h01-pv-regional-mesh.md) |

## Y4. The layer-2 excitatory candidate fires with wrong timing and recovery

**Behavior.** Under the recorded 110 pA input (sweep 43) the candidate does not fire and
its voltage shows excess depolarization and the wrong post-pulse return. Under
the active input it fires the recorded five spikes. The first interval is
8.32 ms too short and the second is 51.57 ms too long. The fifth onset is
64.48 ms late. Individual recovery minima and spike phases also differ from
the human datum. Count agreement does not establish response agreement.

**Explanation.** No single parameter family is sufficient for the combined
response; two families act on separate observation groups. Calcium handling
(F3, the somatic removal time) is necessary for the five-event count and sets
the late intervals: removing it from the candidate restores a sixth event and
the source's negative late-interval errors, and adding it to the source removes
an extra event. The passive and Ih family (F5, distributed Ih with a shifted
passive reversal) alone sets every subthreshold sample: without it the 1120 ms
residual is 1.76 mV in every run, with it 0.39 mV. Added together to the source
these two families are sufficient for a five-spike train whose interval errors
are all within 10.5 ms and for the candidate's subthreshold residuals, but they
leave the recovery minima at the source's values (first minimum 1.6 mV too
negative, second 4.4 mV). The remaining families, sodium gate law (F1), Kv3
gate law (F2), and somatic sodium density (F4), are what move the minima
toward the human values, and the same three push intervals 2 and 4 late
(+52 ms and +21 ms in the frozen candidate). Somatic sodium density carries the
rising and falling phases. The frozen candidate is therefore a trade: minima
bought with interval error.

**Prediction confirmed.** Stated before the paired swap: adding F3+F5 to the
source gives five events and the candidate's subthreshold residual; removing
them from the candidate gives extra events, negative late intervals, and the
source's subthreshold residual. Five of six checks held. The miss was the
prediction that intervals 2 and 4 would be positive like the candidate's; they
are -0.07 ms and +10.5 ms, which locates the candidate's late intervals in
F1/F2/F4 rather than in F3/F5. At CVode 1e-11 the added-pair run changes no
ranked residual by more than its numerical limit; the removed-pair run changes
two event-3 phase residuals by 0.00044 ms against a 0.00042 ms limit, so those
two phase contrasts are within solver noise while every count, interval,
minimum, and subthreshold contrast is unaffected.

**Steep X.** Per observation group, not pooled: intervals and event count, F3;
subthreshold samples, F5; rising and falling phases, F4; recovery minima, none
separated (F4, F1, F2 within a factor of 1.2 of each other). The pooled
normalized RSS names no family, because the groups have different scales.
Coarse mesh (nseg 3) was rejected as a search setting at Stage 0.

**Action.** Omitted bias, a constant voltage offset, solver tolerance, mesh,
somatic Ih density alone, Ih location alone, uniform leak, and passive reversal
alone are not sufficient levers; each was tested. The family search replaces
single-parameter tuning: the next split is inside F1, F2, and F4, on the
source-plus-F3-plus-F5 base, asking which member deepens the minima without
lengthening intervals 2 and 4. A close late voltage, a correct event count, or
accurate peaks does not establish the correct dynamic response. The
[acceptance table](evidence/h01-l2-acceptance-table.md) fixes every required
observation; no allowance is agreed, so no candidate passes. See the
[campaign result](evidence/h01-l2-campaign-result.md) and the
[validation reassessment](specs/2026-09-05-h01-validation-reassessment.md).

```mermaid
flowchart TD
    Y4[Layer-2 candidate: wrong timing and recovery] --> B[Input and datum]
    Y4 --> Q[Subthreshold shape]
    Y4 --> A[Active train]
    B --> B1[Omitted bias: eliminated]
    B --> B2[Constant voltage offset: eliminated]
    Q --> Q1[Ih density x2: eliminated]
    Q --> Q2[Ih location: eliminated]
    Q --> Q3[Uniform leak: eliminated]
    Q --> Q4[Distributed Ih amount: return sign changes, shape fails]
    Q --> Q5[Passive reversal: changes late direction, amplitude wrong]
    A --> A1[Faster calcium removal: shortens late intervals]
    A --> A2[Slower NaTs opening: lengthens rise, lowers peaks]
    A --> A3[NaTs opening plus density: peaks within 0.40 mV; minima too negative]
    A --> A4[Kv3 closing halved: removes four spikes]
    A --> A5[Kv3 closing minus 10%: five spikes, minima improve]
    A5 --> A6[Calcium removal 657 ms: later intervals closer]
    A --> A7[Sodium recovery while availability rises: worsens intervals 2-3]
    A --> A8[Somatic sodium density 0.9: delays onset, worsens intervals]
    A --> A9[Slower somatic calcium removal: later intervals lengthen]
    Y4 --> FAM[Conditional effects: no single family controls the full response]
    FAM --> FC[Calcium change needed for five spikes in full-candidate context]
    FAM --> FP[Calcium plus Ih/leak gives five spikes in source context]
    FP --> FT[Same count; different second interval]
```

| Node | Split | Result | Evidence |
| --- | --- | --- | --- |
| B1 | Add the recorded negative bias | Delays first spike; extra events remain; initial offset nearly removed in sweep 43 (110 pA) but adaptation and return still wrong | [bias split, active](evidence/h01-l2-recorded-bias-result.md), [bias split, subthreshold](evidence/h01-l2-sweep43-bias-result.md) |
| B2 | Subtract the starting voltage | Excess depolarization and wrong return remain | [subthreshold comparison](evidence/h01-l2-midpoint-sweep43-comparison.md) |
| Q1 | Double somatic Ih density | Adaptation and below-baseline return not restored | [Ih split](evidence/h01-l2-sweep43-ih-result.md) |
| Q2 | Move Ih to uniform soma and dendrite, same total | Shape not restored | [location split](evidence/h01-l2-ih-location-result.md) |
| Q3 | Increase regional passive leak together | Late depolarization reduced; adaptation not restored; two spikes where reference has five | [leak split](evidence/h01-l2-sweep43-leak-result.md), [active check](evidence/h01-l2-leak-active-result.md) |
| Q4 | Larger distributed Ih amount | Return slightly below start; start shifts up; adaptation wrong | [amount split](evidence/h01-l2-distributed-ih75-result.md) |
| Q5 | More negative passive reversal with Q4 fixed | Late response rising to falling; five spikes under active input; every interval too long | [passive-reversal split](evidence/h01-l2-leak-reversal-result.md), [active response](evidence/h01-l2-reversal-active-result.md), [separate phases](evidence/h01-l2-reversal-spike-phases.md) |
| A1 | Faster calcium removal in the Q5 setting | Later intervals shorten; one overshoots; minima unchanged | [removal split](evidence/h01-l2-reversal-calcium125-result.md) |
| A2 | Slower NaTs opening | Rise lengthens; peaks too low; three minima too shallow | [opening split](evidence/h01-l2-sodium-opening-result.md) |
| A3 | Slower opening plus more NaTs conductance | All five peaks within 0.40 mV; minima too negative; later spikes late; subthreshold directions preserved | [density split](evidence/h01-l2-sodium-opening-density130-result.md), [subthreshold response](evidence/h01-l2-density130-subthreshold-result.md) |
| A4 | Halve Kv3 closing time | One spike; sustained depolarized response | [closing split](evidence/h01-l2-kv3-closing-half-result.md) |
| A5 | Reduce Kv3 closing time by 10% | Five spikes; three minima within 0.19 mV; second minimum too negative; first interval too short | [smaller split](evidence/h01-l2-kv3-closing-ninety-result.md) |
| A6 | Removal time 679.277 to 657.046 ms on A5 | Each later interval shorter and closer; first interval changes 0.013 ms; second interval still too long | [calcium split](evidence/h01-l2-kv3-ninety-ca133-result.md) |
| A7 | Slow sodium recovery only while availability rises | First interval lengthens; next two interval errors worsen | [layer-2 sodium split](evidence/h01-l2-sodium-recovery-result.md) |
| A8 | Somatic sodium density 0.9 | First spike delayed and lower; narrower; first three interval errors worsen | [density split](evidence/h01-l2-sodium-density090-result.md) |
| A9 | Slower somatic calcium removal | Later intervals lengthen; first peak and minimum unchanged; first interval still too short | [layer-2 removal split](evidence/h01-l2-calcium-removal-result.md) |
| FAM | Forward and reverse family changes, then paired changes | No single pooled RSS dominant family; five-spike count does not determine the individual intervals | [campaign results](evidence/h01-l2-campaign-result.md), [active tolerance audit](evidence/h01-l2-paired-active-tolerance-audit.json) |

## Y5. Synaptic delivery in the illustrative H01 pair

**Behavior.** In the prior illustrative pair (`810151953`, `678539249`), an E output event is followed by a
delayed conductance increase and a voltage rise in I, and I does not fire. With
the I model using the source closing-time restoration, an I event is followed
by an earlier fall and a later spike in E.

**Explanation.** The E-to-I projection is necessary and sufficient for the
initial I voltage rise: removing only that projection removes the conductance,
and the I voltage is unchanged before delivery and rises when conductance
arrives. The I-to-E projection is necessary and sufficient for the E delay:
removing only that projection preserves the I response and the E voltage
before arrival. A synaptic conductance moves voltage toward its reversal
potential and changes the load seen by other current sources, so an inhibitory
conductance can delay excitation without a negative voltage change. With both
contacts enabled, the delayed E event delays the arrival of excitation at I, so
an inhibitory timing change propagates through the next connection.

```mermaid
flowchart LR
    E[E output event] --> D[Transmission delay]
    D --> G[I synaptic conductance increases]
    G --> V[I voltage moves toward 0 mV]
    V --> S[No I output event at this input]
```

**Prediction confirmed.** The E delay persists at both tested time steps.

**Steep X.** Not applicable; each projection was tested by removal.

**Action.** Delivery alone does not establish a firing response or inhibitory
feedback. Both cells receive external input in the four-way check, and I fires
before E input arrives, so the circuit does not show that E recruits I or that
activity sustains itself. The contacts are inferred; these conclusions apply to
the diagnostic model, not to a measured connection in the donor, and full
numerical and human qualification remain open.

```mermaid
flowchart TD
    Y5[Delivery in the H01 pair] --> E1[E-to-I removed: I rise removed]
    Y5 --> I1[I-to-E removed: E delay removed]
    Y5 --> F1[Both contacts: delay propagates; recruitment open]
```

| Node | Split | Result | Evidence |
| --- | --- | --- | --- |
| E1 | Remove only the E-to-I projection | Conductance and voltage rise removed; no I event either way | [E delivery audit](evidence/h01-ei-circuit-e-delivery-audit.json) |
| I1 | Remove only the I-to-E projection | E fires earlier without inhibition; delay persists at both steps | [I delivery audit](evidence/h01-i-restored-circuit-delivery-audit.json) |
| F1 | Both contacts, both cells driven | Delayed E event delays excitation at I; I fires before E input arrives | [four-control audit](evidence/h01-i-restored-four-control-audit.json) |

The measured default uses a different pair: I `5584343344` to E `4157825456`.
Its source endpoint identities and cable placements are checked. Those checks
establish the anatomical path, not its physiological response. The delivery
results above must not be transferred to this pair without a direct test.
The synthetic wiring fixture confirms the same implementation rule: with E
above the inhibitory reversal potential, a delivered conductance lowers E
voltage; removing only the edge removes that effect. Qualification of the
default profiles' firing and full numerical robustness remains open.

In the measured pair, the recorded I contact voltage stays below -80 mV under
the stated 1 nA soma pulse. It emits no event. The initially empty synapse
therefore receives no event, and its conductance stays zero. This locates the
missing delivery before the synapse response. The recorded I soma reaches
only -29.53 mV. Every one of the 14,685 I compartments remains below
-29.41 mV in the complete sampled 8 ms baseline. There is no hidden positive
excursion elsewhere in that record. The full-compartment recorder preserves
the circuit soma, contact, conductance, and event traces exactly.
[Direct response](evidence/h01-measured-ie-first-response-audit.json),
[all-compartment check](evidence/h01-measured-i-allcv-baseline-audit.json).

More soma depolarization does not ensure an output event in this model.
At 2 nA the I soma reaches -13.92 mV, but the contact-voltage change is at
most 0.0394 mV compared with 1 nA. The contact still emits no event.
E voltage and conductance remain exactly unchanged. Thus, the larger soma
response does not cross the modeled event-delivery boundary. It does not
by itself separate failed spike initiation from failed propagation.
[Direct input split](evidence/h01-measured-ie-i2na-response-audit.json).

At the original 1 nA input, changing only the I sodium-inactivation closing
time from its candidate factor of 0.15 to 1 produces a complete soma
excursion through -20 mV, with peak +34.07 mV. The pre-pulse traces are
exactly unchanged. Thus, the closing-time change controls the loss of this
soma excursion in the tested model context. The measured contact emits at
8.355 ms and peaks at +28.25 mV at 8.465 ms. The first 8 ms of the longer
record match the earlier circuit exactly. Thus, the earlier absence of a
contact event in this condition was caused by ending observation too soon.
This is a diagnostic channel law, not a human-qualified replacement.
[Closing-time response](evidence/h01-measured-ie-source-closing-response-audit.json),
[complete contact arrival](evidence/h01-measured-i-source-closing20-audit.json).

Direct CV traces show the excursion reaching successive locations along
the modeled soma-to-contact path. Each curve below is a local voltage,
not a spatial average. The dashed line marks the earlier observation limit.
[Locations and samples](evidence/h01-measured-i-propagation-audit.json).

![Recorded voltages along the H01 I path](evidence/h01-measured-i-propagation.png)

In the restored-closing condition, the measured edge delivers conductance to
E at 8.855 ms, 0.5 ms after I emits. Removing only this edge removes the
conductance and E voltage effect. I's full trace is exactly unchanged, and
E's local and soma traces are exactly equal before delivery. This establishes
the edge as the cause of the observed E response under the stated input.

The initial local E response is depolarizing: local voltage is below the
borrowed -80 mV reversal. The current pulls that compartment toward -80 mV;
its direction reverses when local voltage crosses the reversal. Soma voltage
alone would give the wrong initial direction because it differs from the
receptor voltage. E emits no event in either control. Thus, delivery is
demonstrated, but suppression of E firing and numerical robustness remain
unverified. The source inhibitory label does not by itself establish the
sign of each local voltage change.
[Measured edge-removal comparison](evidence/h01-measured-delivery20-paired-audit.json).

The absence of E spikes under that short input does not establish that E
cannot spike. With the edge absent, a 5 nA pulse from 8 to 18 ms produces two
complete E excursions with positive peaks. The E channel profile is unchanged.
I's first 20 ms remain exactly equal to the earlier disconnected record.
Thus, E spiking in this condition does not require the I-to-E edge. Amplitude,
onset, and duration changed together, so this result does not identify which
input feature is necessary. Human-response and numerical qualification remain
open. [Direct E response](evidence/h01-measured-E5-overlap-disconnected-audit.json).

With that E input fixed, enabling only the measured edge changes local E
voltage after delivery. I stays exactly unchanged, and E matches before
delivery. The first local current is inward, as predicted from the receptor
voltage below -80 mV. Both E spikes remain, with unchanged sampled event
times at the first two step sizes. A smaller step resolves a one-sample delay
in the second emitted event. The second interpolated onset delay is 0.001124,
0.001140, and 0.001137 ms across the three steps. Thus, the edge causes a local
response and a small onset delay in these discrete runs. It does not remove
an E spike. Both controls pass the declared waveform limits between the two
finer steps. At the finer step, halving maximum compartment length also
preserves the waveform checks and second onset delay: 0.001137 versus
0.001143 ms. I stays identical between edge controls, and E matches before
delivery. The contact coordinates and detector midpoints remain unchanged.
This gives bounded numerical support for the delay in this diagnostic
condition. It does not establish spike removal or human-response accuracy.
[Active E edge comparison](evidence/h01-measured-E5-overlap-paired-audit.json),
[paired half-step comparison](evidence/h01-measured-E5-overlap-paired-halfdt-audit.json).
[Paired finer-step comparison](evidence/h01-measured-E5-overlap-paired-quarterdt-audit.json).
[Paired spatial comparison](evidence/h01-measured-E5-overlap-paired-cv5-audit.json).

![Direct E soma and receptor responses](evidence/h01-measured-E5-overlap-traces.png)

Time-step choice still affects the diagnostic I waveform beyond the declared
limit. Halving the step changes its soma peak by 0.1112 mV, above the 0.1 mV
limit. The E soma and I contact excursions pass their stated checks. This
localizes the failed numerical observation to the I soma peak; it does not
identify a unique channel or integration term as the cause of the error.
[Time-step comparison](evidence/h01-measured-E5-overlap-disconnected-halfdt-audit.json).

Further step refinement reduces this I soma peak difference to 0.0560 mV.
The 0.0025-to-0.00125 ms disconnected comparison passes all declared soma
and contact excursion checks. This supports time-step error as a controllable
source of the failed peak comparison. It does not identify a unique internal
solver term, establish a continuous-time limit, or validate the human response.
[Finer-step comparison](evidence/h01-measured-E5-overlap-disconnected-quarterdt-audit.json).

The electrical map assigns the measured I contact to the dendrite fallback.
That region has Ih but no fast sodium channel. Most of the soma-to-contact
path also uses this fallback. This map is an inference from sparse labels.
No change to this map was needed for the observed contact event with restored
closing time. That arrival does not validate each inferred electrical region.
[Region map](evidence/h01-measured-ie-region-audit.json),
[source path](evidence/h01-measured-i-contact-path-audit.json).

This path also includes six source samples decoded as dendrite and two as
astrocyte by the importer. Filling it with axon parameters would cross these
conflicting labels. Such a map is a model hypothesis, not a measured axon
classification. Keep source codes unchanged.
[Source-label audit](evidence/h01-measured-i-path-label-conflicts.json).

```mermaid
flowchart LR
    G[Diagnostic sodium closing-time restoration] --> P[Soma peak +34.07 mV]
    P --> A
    A[I voltage at the source axon contact] --> B[Modeled output event]
    B --> T[Observed contact event at 8.355 ms]
    B --> C[Borrowed transmission delay]
    C --> D[Conductance at the source E contact]
    D --> E[Current set by conductance and reversal potential]
    E --> F[Local E response observed; firing suppression open]
```

## Measurement function qualification

The simulation is the measurement function. Its numerical error must be small
against the human error it must resolve before a physical split can be read.
The checks below support that discrimination for the named observations only.
They do not prove convergence at every time or input, and a small numerical
change never identifies a channel cause.

Current samples must use the same time level. The staggered solver advances
voltage before it updates channel gates; a current computed from the updated
gates is not the current law used for that voltage step, and mixing the samples
leaves a residual without a failure of the discrete charge balance.

| Candidate and observation | Check | Result | Evidence |
| --- | --- | --- | --- |
| H01 I, voltage-step balance | Matching time levels | Balance closes within its stated limit; bookkeeping only | [balance](evidence/h01-i-current-balance-discrete.json) |
| PV candidate, selected observations | Time step and mesh | Refinement differences recorded | [numerical audit](evidence/h01-pv-numerical-audit.json) |
| PV, first-spike shortening | Two meshes | Effect persists | [spatial robustness](evidence/h01-pv-inactivation-mesh-audit.json) |
| PV, high-input late-spike absence | Finer mesh | Absence persists; full voltage still changes | [failure mesh](evidence/h01-pv-failure-mesh.md) |
| PV, slope rescue | Finer mesh | Rescue and first-return mismatch persist; later times and count change | [candidate spatial check](evidence/h01-pv-slope5-mesh.md) |
| PV, late intervals | Tighter CVode tolerance at fixed mesh | Onsets much closer than under spatial refinement; count preserved | [regional mesh](evidence/h01-pv-regional-mesh.md) |
| PV, final onset | Fine axonal meshes | Shift follows squared compartment length within the limit; event sequence within successive-mesh limits | [fine-mesh qualification](evidence/h01-pv-axon2187-qualification.md) |
| L2 control, 43 pA samples | Tolerance tightened by ten | Error directions persist | [tolerance check](evidence/h01-l2-subthreshold-tolerance-comparison.md) |
| L2 control, 43 pA samples | Successive spatial refinement | Error directions persist with small changes | [spatial refinement](evidence/h01-l2-subthreshold-spatial-comparison.md) |
| L2 reversal candidate, 43 pA | Tighter tolerance; spatial refinement | Directions persist | [tolerance](evidence/h01-l2-reversal-tolerance-result.md), [spatial](evidence/h01-l2-reversal-spatial-result.md) |
| L2 reversal candidate, spike phases | Spatial refinement; tighter tolerance | Error signs persist with small phase changes | [spatial](evidence/h01-l2-reversal-active-spatial-result.md), [tolerance](evidence/h01-l2-reversal-active-tolerance-result.md) |
| L2 slower-opening candidate, phases | Tighter tolerance | Each phase changes below 0.000422 ms | [tolerance check](evidence/h01-l2-sodium-opening-tolerance-result.md) |
| L2 control, minima | Tighter tolerance; two meshes | Below 0.000002658 mV and 0.001330 mV; human errors much larger | [minimum review](evidence/h01-l2-density130-minima-numerical-review.md) |
| L2 density candidate, spikes | Tighter tolerance | Each phase below 0.000417 ms; rising-phase human errors above 0.12 ms persist | [tolerance check](evidence/h01-l2-density130-tolerance-result.md) |
| L2 density candidate, spikes | Segments threefold | Onset changes at most 0.0357 ms; each phase below 0.000347 ms | [spatial comparison](evidence/h01-l2-density130-spatial-result.md) |
| L2 Kv3 half candidate, post-spike voltages | Tighter tolerance | One-spike outcome persists; below 0.000000830 mV | [tolerance comparison](evidence/h01-l2-kv3-half-tolerance-result.md) |
| L2 Kv3 ninety candidate, minima | Tighter tolerance | Spike limits pass; each minimum below 0.000001881 mV | [tolerance comparison](evidence/h01-l2-kv3-ninety-tolerance-result.md) |
| L2 sodium-recovery candidate | Independent solver; successive spatial refinement | Direct events pass at this input | [layer-2 sodium split](evidence/h01-l2-sodium-recovery-result.md) |
| PV transfer, matched mesh, events 1-8 | Halve dt in NEURON and in BrainCell | Event 8 moves 0.002 ms in each; all gates pass | [isolation split](evidence/h01-pv-transfer-isolation.md) |
| PV transfer, matched mesh | NEURON CVode 1e-10 vs fixed dt 0.000625 | Largest rise change 0.044 ms; gates pass | [isolation audit](evidence/h01-pv-transfer-isolation-audit.json) |
| L2 campaign controls | nseg 3 vs nseg 9 at CVode 1e-10, active input | Five phase changes exceed one fifth of the smallest source-candidate-corner contrast; coarse setting rejected | [Stage 0 preservation](evidence/h01-l2-campaign/stage0-active-preservation.json) |
| L2 paired-swap runs | CVode 1e-10 vs 1e-11 at nseg 9 | Added pair: every ranked residual within its limit (max 0.0002 ms); removed pair: two event-3 phase residuals at 0.00044 ms vs 0.00042 ms limit, counts and all other contrasts unchanged | [prediction check](evidence/h01-l2-campaign-fine/stage-b-prediction-check.json) |

The response-size ranking in the [factor evidence](evidence/h01-i-factor-response-rss.md)
compares specified interventions on a fixed observation grid. It selects the
next diagnostic boundary. It does not prove mediation or supply missing
biological uncertainty.

## Unresolved

- Y1: which upstream axonal segment reduces the outflow in the axon-only case;
  whether the electrical region map is correct.
- Y2: transfer of the full 1270 ms train and the other inputs at the copied
  mesh; whether equal counts also equal compartment placement on every branch.
- Y3: the pathway from the shorter first spike to later spike times; the
  channel cause of the shallow, late minimum; the spike initiation site; a
  human recovery constant; the Kv3 candidate's spatial sensitivity; the
  full-cell energy budget; the topology that reproduces the human waveform.
- Y4: which member of F1, F2, or F4 deepens the recovery minima without
  lengthening intervals 2 and 4; the channel source of the remaining interval
  errors; the transition to the sustained depolarized response between the
  two Kv3 closing changes; the separate contributions of entry and removal
  after any coupled intervention; the human cell's densities; every
  physiological allowance.
- Y5: numerical robustness, reciprocal behavior, and qualified cell models.

## Evidence index

Sources: [source and datum evidence](evidence/h01-pv-acquisition-audit.md),
[human pyramidal sources](evidence/h01-active-wilbers-sources.json),
[H01 anatomy](https://h01-release.storage.googleapis.com/landing.html).
The per-behavior tables above index the remaining records.

A yes/no decision applies to a specified causal claim and its tested conditions.
Keep the continuous direct response that supports the decision.
Distinguish a contradicted prediction, an invalid test, and an unresolved explanation.
Only supported conclusions belong in the explanations. Unresolved links stay explicit.
