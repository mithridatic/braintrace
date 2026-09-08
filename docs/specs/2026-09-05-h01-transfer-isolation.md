# Isolate the PV transfer drift between simulators

## Behavior

The frozen PV candidate in BrainCell reproduces the early NEURON spikes but
its rise crossings drift from event 7 (-0.21 ms) and event 8 (-1.7 ms) over
270-330 ms at 0.27 nA. The existing comparison holds neither mesh nor
integration equal: BrainCell used MaxCVLen 2.5 um and a staggered fixed step;
NEURON used CVode with per-branch counts matched only by mean segment length.
One split (halving the BrainCell step) did not repair event 8. The remaining
space is one undivided lump, "implementation differences".

## Split

Partition that lump with a 2x2 half-split (Dissection) between two elements,
holding everything else equal:

| Element | Level 0 (current) | Level 1 (matched) |
| --- | --- | --- |
| A mesh | BrainCell MaxCVLen 2.5 um | BrainCell CVPerBranchList copied from the NEURON x9 run |
| B integration | NEURON CVode atol 1e-10 | NEURON fixed step at dt 0.000625 ms |

Held equal in every cell: morphology, conductance tables, gate laws
(kv3-phase-reference library), stimulus 270 ms/1000 ms/0.27 nA, initial
voltage -80 mV, 34 C, reversals, Ra, cm, BrainCell dt 0.000625 ms and
staggered solver, duration 330 ms, window 270-329.5 ms (the CVode trace records its last step before 330 ms; no event lies after 323.4 ms), no alignment.
Temperature is held equal only; BrainCell has no temperature knob.

## Reversible isolation first

Halve dt once in each simulator at the matched mesh (NEURON fixed step
0.0003125; BrainCell 0.0003125). Each simulator's own events must pass the
existing gates against its half-step run (rise 0.1 ms, peak 0.1 mV, width
0.01 ms, equal count). Otherwise time-step convergence is insufficient at these limits and the
factorial attribution stops. Keep the limits unchanged.

## Predictions stated before running

- Mesh is the Steep X: both matched-mesh cells pass; both MaxCVLen cells fail
  on rise crossing only.
- Integration is the Steep X: both fixed-step cells pass; both CVode cells fail.
- Both matched and still failing: matching counts and the nominal time step
  is insufficient. Gate update timing is one hypothesis, not a proved cause.
  Equal branch counts do not prove equal compartment placement, cable
  coefficients, state initialization, or stimulus sampling. Isolate a
  specific difference before assigning cause. Y3 remains read through an
  unqualified transfer.
- Only the matched-and-fixed cell passes: the two elements interact; neither
  alone is sufficient.

The audit script evaluates these four patterns literally and records which
one occurred. Any other pattern is reported as "no prediction matched".

## Runs

All 330 ms, candidate mode, 0.27 nA, written to `docs/evidence`:

| Stem | Simulator | Settings |
| --- | --- | --- |
| h01-pv-transfer-isolation-neuron-cvode | NEURON | CVode 1e-10, nseg x9 all regions |
| h01-pv-transfer-isolation-neuron-fixed | NEURON | fixed dt 0.000625, x9 |
| h01-pv-transfer-isolation-neuron-fixed-halfdt | NEURON | fixed dt 0.0003125, x9 |
| h01-pv-transfer-isolation-braincell-maxcv | BrainCell | MaxCVLen 2.5, dt 0.000625 |
| h01-pv-transfer-isolation-braincell-matched | BrainCell | mesh from neuron-cvode json, dt 0.000625 |
| h01-pv-transfer-isolation-braincell-matched-halfdt | BrainCell | mesh from neuron-cvode json, dt 0.0003125 |

Candidate interventions (identical to `h01-pv-calcium-response-004-300-027`):
NaTg x1.1 soma, sodium h tau 0.15, recovery 1, slope 5 mV, somatic Ca_LVA 0.5,
somatic Kv3 tau 0.5, closing 0.5, axon calcium decay 300 ms, gamma 0.004.

NEURON command (Git Bash, worktree root; the library directory is the cwd so
NEURON loads `x86_64/libnrnmech.so`):

```bash
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$(pwd -W)/.cache/human-pv/kv3-phase-reference:/work" \
  -v "$(pwd -W)/docs/evidence:/evidence" -w /work braintrace-h01-neuron:9.0.2 \
  python /evidence/h01_pv_neuron_reference.py --current-na 0.27 --nseg-factor 9 \
  --scale-conductance NaTg --conductance-factor 1.1 --conductance-region soma \
  --sodium-h-tau-factor 0.15 --sodium-h-recovery-factor 1 --sodium-h-slope-mv 5 \
  --somatic-calva-factor 0.5 --somatic-kv3-tau-factor 0.5 --somatic-kv3-close-factor 0.5 \
  --axon-calcium-decay-ms 300 --axon-calcium-gamma 0.004 --duration-ms 330 \
  [--cvode-atol 1e-10 | --dt-ms 0.000625 | --dt-ms 0.0003125] --output /evidence/<stem>
```

BrainCell command (host validation interpreter, worktree root):

```bash
.cache/validation/Scripts/python.exe -m docs.evidence.h01_pv_braincell_reference \
  --current-na 0.27 --duration-ms 330 --dt-ms 0.000625 \
  [--max-cv-um 2.5 | --mesh-from docs/evidence/h01-pv-transfer-isolation-neuron-cvode.json] \
  --output docs/evidence/<stem>
```

Benchmark one NEURON fixed-step run before launching the rest. Quote only
measured durations.

## Code

- `braintrace/datasets/h01_pv_cell.py`: `make_pv_cell(..., cv_policy=None)`;
  a given policy replaces `MaxCVLen`.
- `docs/evidence/h01_pv_braincell_reference.py`: `--mesh-from <neuron json>`
  builds `braincell.CVPerBranchList` from the JSON geometry `nseg` values,
  mapped by the reversible name rule (`soma[0]` to `soma_0`) in morphology
  branch order; the report records the policy, counts, and source hash.
- `docs/evidence/h01_pv_neuron_reference.py`: `--duration-ms` (default 1500)
  replaces the literal; the report records the mechanism library hashes.
- `docs/evidence/h01_pv_transfer_isolation.py`: asserts held-equal metadata,
  checks the matched mesh equals the NEURON counts, runs the two reversible
  checks and the four cells through `compare_spike_transfer` over 270-329.5 ms,
  and writes `h01-pv-transfer-isolation-audit.json` with input hashes and the
  literal decision.

## Decision rule

A cell passes when `compare_spike_transfer` passes (equal nonzero count and
all three gates over 270-329.5 ms). The audit records signed rise errors at
events 7 and 8 for every cell. The causal page names the Steep X only from
the pattern above; a numerical Steep X does not identify a channel cause and
does not validate the human waveform.
