# H01 E/I starting defaults

The H01 E/I builder now selects the two candidates below. Selection does not
mean that the models pass human waveform validation.

| Role | Default | Reference |
| --- | --- | --- |
| E | `h01-l2-kv3-ninety-ca133:candidate:v1` | Allen specimen 541563728, model 626170538 |
| I | `h01-pv-regional-mesh-axon2187:candidate:v1` | ModelDB 267587, putative PV cell HL5BN1 |

The E choice retains five spikes and improves several later intervals relative
to its Kv3 control. It is not best on every measurement. Onset, rising-phase,
and recovery errors remain. The I choice still has early intervals that are
too short. Its saved full 0.27 nA donor response has 40 events versus 43 in
the human trace, and its first recovery minimum is about 7 mV too high. Neither model is fitted to voltage recordings from the H01 donor.

## Use the defaults

Use `braintrace.datasets.h01_ei_cell.make_h01_ei_cell`. Supply an imported H01
component, its annotations, an explicit E or I role, and an electrical region
map. The map must cover the complete component without gaps or overlap. The I
profile requires soma and axon regions. Explain the map in `region_basis`.
The return value contains the cell and its provenance.

Use `mode="source"` for the published source settings. The donor reference
builders are `h01_l2_cell.make_l2_cell` and `h01_pv_cell.make_pv_cell`. Both select
the candidate by default. The older `h01_active.make_active_cell` remains the
separate Wilbers experiment; it does not implement the Allen candidate.

From the worktree root, run the offline example with the validation Python:

```powershell
.cache/validation/Scripts/python.exe -m examples.h01_ei_candidates --cache .cache/h01 --output .cache/h01/ei-defaults
```

The example uses H01 component 0 from pyramidal cell `810151953` and interneuron
`678539249`. Source labels determine soma and axon samples. Boundaries extend
halfway along adjoining edges. The remaining cable receives basal-dendrite
parameters. This includes unclassified samples. That electrical assignment is
an assumption, not a new anatomical label. Myelin insulation is not modeled.
The source interneuron tag does not establish a PV subtype.

The example applies a 0.1 nA soma pulse from 2 to 5 ms. It saves direct voltage
samples, timing, source identity, profile identity, and region assignments.
It creates two separate cells. It does not claim a measured connection between
them. H01 synapse positions alone do not identify the missing partner cells.

## Evidence and limits

The [frozen profile tests](../braintrace/datasets/h01_ei_profiles_test.py) check
metadata hashes and physical parameters. The channel tests compare source and
candidate laws with 720 independent NEURON cases. The PV source perturbs exact
singular voltages by 0.0001 mV. The transfer uses the analytic limit instead.
The test states a separate tolerance at those source singularities.

The [donor response check](evidence/h01-ei-transfer-check.json) compares the
first 2 ms on both donor geometries. It passes a 0.1 mV soma-error limit for
source and candidate modes. This is an initialization and early-response gate.
It does not qualify spike timing or a full stimulus sweep after transfer.

The [real H01 pulse run](evidence/h01-ei-defaults-response.json) completed
10 ms for both cells at 0.005 ms steps. Both traces remained finite. The pulse
produced subthreshold responses; neither cell crossed 0 mV. This demonstrates
execution of the selected profiles, not spiking or E/I circuit behavior.

The connected diagnostic circuit is available through
`h01_ei_circuit.make_h01_ei_circuit` and `examples.h01_ei_circuit`. It uses
explicit illustrative contacts, not measured H01 partner pairs. A matched
projection-removal control verifies real H01 E-to-I delivery. The I cell does
not fire under that input, so inhibitory feedback remains unverified.

Remaining gates include full spike-response transfer, human waveform checks,
qualified H01 I spikes, and circuit robustness. Selected first-burst and H01
numerical checks are recorded in the [spiking report](evidence/h01-ei-spiking-progress.md).
The reserved human response remains outside this default-selection task.

See the [causal explanation](h01-causal-model.md). Detailed test records belong
in the evidence files, not in that explanation.

See the [full-response audit](evidence/h01-frozen-candidate-full-response.md)
for direct event and recovery residuals. Circuit execution does not remove
these physiological validation gaps.
