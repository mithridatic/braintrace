# SP2. Transfer isolation on donor geometry (full train, both polarities)

Status: spec written 2026-09-07 before code, under the approved
[population programme](2026-09-07-h01-population-programme.md) (SP0 rules bind:
detached jobs, progress lines, watchdog, measured estimate and approval above 15 min,
prediction and rejection registered before every run, a killed run is untested).
Predecessor: [2026-09-05 isolation split](2026-09-05-h01-transfer-isolation.md), decision
"mesh" over 270-329.5 ms at 0.27 nA only. Standing directive
(`docs/evidence/h01-implementation-status.md`): no serial physiological tuning while the
full-train transfer stands unqualified.

This task delivers tooling, tests, and manifests only. No simulation is launched by any
code written here; every command is registered so that the user launches it by hand as a
detached job.

## Behavior

The frozen PV candidate in BrainCell agrees with NEURON for eight events when the NEURON
per-branch segment counts are copied (`--mesh-from`). Unknown: whether the same copied mesh
carries the full 270-1270 ms train (about 40 events) at 0.19 and 0.27 nA, and whether
equal counts also mean equal compartment placement late in the train. E has never had a
spike-transfer gate at all.

## I side: structural split

Reference for every I arm: the NEURON finalist CVode traces in `docs/evidence/h01-i-energetic/`
(`e-kv3-close2-019`, `e-kv3-close2-027`; `p-close2-023` is the spent holdout, used only
as a control). All are CVode atol 1e-10, nseg factor 9 in every region, 1500 ms.

| Arm | Simulator | Mesh | Integration | Input (nA) | Window (ms) | Runs |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | BrainCell | MaxCVLen 2.5 um (current) | staggered, dt D | 0.27 | 270-1270 | 1 |
| A1 | BrainCell | `--mesh-from` the NEURON x9 geometry JSON of the finalist run | staggered, dt D | 0.19, 0.27, 0.23 (control) | 270-1270 | 3 |
| B1 | NEURON | x9 (same geometry) | fixed step at dt D (`--dt-ms D`, no `--cvode-atol`) | 0.19, 0.27 | 270-1270 | 2 |
| R | BrainCell | copied mesh | dt D/2 | 0.27 | 270-329.5 | 1 |
| R | NEURON | x9 | fixed dt D/2 | 0.27 | 270-329.5 | 1 |

Time step D: 0.005 ms is tried first (unmeasured for BrainCell at the copied mesh). Its
halving pair (R, 330 ms) must pass under half the gate before any full-train arm at that
dt is scored. If R fails at 0.005, D falls back to 0.000625 (measured: 556 s per 330 ms
at the copied mesh, so the full train is derived at about 2,500 s, which needs a measured
estimate and approval before launch; the 0.000625 halving pair already exists,
`h01-pv-transfer-isolation-braincell-matched{,-halfdt}`).

### Fixed-step NEURON

`h01_pv_neuron_reference.py` already integrates with a fixed step whenever `--cvode-atol`
is not given (`h.dt = args.dt_ms`, default 0.025 ms, `h.continuerun`). B1 therefore needs no
new flag: the arm is the finalist candidate JSON with `cvode_atol` removed and `--dt-ms D`.
No `--fixed-dt` option is added. The finalist candidate file sets `cvode_atol` 1e-10 and
`--candidate-json` values are defaults that explicit flags override, but `None` cannot be
passed on the command line, so B1 uses its own candidate file
`docs/evidence/h01-i-transfer/b1-fixed.candidate.json` (same flags, no `cvode_atol`).

### Held equal (asserted by the runner before scoring)

Morphology, conductance tables, gate laws, stimulus 270 ms on / 1000 ms / amplitude,
initial voltage -80 mV, 34 C, reversals, Ra, cm, no alignment. Additionally the runner
asserts that the BrainCell profile's phase controls equal the NEURON finalist flags:
somatic Kv3 `m_close` = `somatic_kv3_close_factor`, `m_open` = `somatic_kv3_tau_factor`,
NaTg `h_close` = `sodium_h_tau_factor`, `h_open` = `sodium_h_recovery_factor`,
`h_slope` = `sodium_h_slope_mv`; somatic NaTg density = source x `conductance_factor`;
somatic Ca_LVA = source x `somatic_calva_factor`; axon calcium decay and gamma.

**Discovered precondition.** Today's I candidate profile (`channel_controls`, Kv3
`m_close` 0.5) is the pre-finalist candidate. The finalist NEURON traces carry
`somatic_kv3_close_factor` 2.0. The alignment assertion therefore fails today and the
runner refuses to score A0/A1 against the finalist. It is released by the registry re-key
(SP6a, another agent) registering a finalist I profile; the manifest field
`braincell_profile_key` is then set to that key and `--mode` (or its successor flag) is
substituted. Until then the only fully held-equal full-train comparison available would
be against a NEURON re-run of the pre-finalist candidate, which is not the finalist and
is not requested.

### Gate and decision limit

Per event over the window: rise crossing 0.1 ms, peak 0.1 mV, time above -20 mV 0.01 ms;
equal nonzero event count (`compare_spike_transfer`). The gate is valid only when the
halving pair R at the same mesh and dt is under half the gate on every event
(0.05 ms, 0.05 mV, 0.005 ms). The runner writes one row per event (index, reference rise,
actual rise, signed errors, in-gate flags) so the first failing event k is named.

### Decision rule (evaluated literally by `decide()`)

| A1 (both 0.19 and 0.27) | B1 (both) | Decision |
| --- | --- | --- |
| passes every event vs CVode and vs B1 | passes vs CVode | `mesh`: copied mesh confirmed for the full train; Y2 closed |
| fails at event k (late) | passes | `implementation_difference_at_event_k`: next split = channel-by-channel state comparison at event k (soma NaTg h, Kv3 m, axon voltage), Y2 reassessed |
| fails | fails | `integration`: time level, not mesh; dt 0.000625 with approval |
| any | any, but R over half the gate | `time_level_open`: stop, report |

A0 is a control: it is expected to fail on rise only, as before. If A0 passes the full
train, the earlier drift attribution to mesh is questioned and reported.

### Prediction and rejection (registered 2026-09-07)

Prediction: A1 passes every event at 0.19 and 0.27 nA against the CVode reference with
max |rise error| <= 0.044 ms (the 8-event value) and equal counts (40 at 0.27; the 0.19
count taken from the reference JSON). Rejection: any A1 event beyond its gate while R is
under half the gate. Unchanged: A0 fails on rise from event 7 as before; 0.23 control
behaves as 0.19/0.27.

### Cost anchors and caps

| Item | Value | Status |
| --- | --- | --- |
| BrainCell copied mesh, 330 ms, dt 0.000625 | 556 s | measured |
| BrainCell copied mesh, 1500 ms, dt 0.000625 | ~2,500 s | derived, needs approval |
| BrainCell copied mesh, dt 0.005 | none | unmeasured; run R first, quote only measured |
| NEURON fixed dt 0.000625, 330 ms | 72 s | measured |
| NEURON CVode, 1500 ms finalist | ~10 s per 330 ms | measured at 330 ms only |

Cap: I 8 BrainCell + 2 NEURON runs (this table uses 6 + 3 including the halving pair; the
0.23 control and B1 count against the caps). E 4 BrainCell runs. Stop rule: R exceeds
half the gate at both dt levels -> "time level open"; any run predicted over 15 min
without a measured estimate is not launched.

## E side

- `make_l2_cell(..., cv_policy=None)`: a given policy replaces `MaxCVLen`; default keeps
  today's behaviour. `current_na` and `duration_ms` also accept equal-length sequences so
  a recorded step protocol replays as one piecewise-constant `CurrentClamp` (scalars keep
  the present single pulse).
- `docs/evidence/h01_l2_braincell_reference.py` (+ sibling test), modelled on the PV
  runner: `--mesh-from` reads the L2 NEURON report's `sections` (`name`, `nseg`; names
  `soma[0]` map to `soma_0`) into `CVPerBranchList` in morphology branch order and records
  policy, counts and source hash; `--sweep NN` loads
  `.cache/human-pyramidal-l2/sweep-NN.npz` (read-only; the cache lives in the sibling
  worktree and is passed by `--cache`), collapses `command_current_na` into constant
  segments, adds `bias_current_na` (the B3 runs used command plus recorded bias), and
  records the segments; `--stop-ms` (default 2100 as the NEURON driver); `--profile-key`
  accepts `candidate` or `source` now and the registered B3 key later.
- `docs/evidence/h01_e_b3_profile_export.py` (+ test): reads
  `h01-e-usable/b3-sk035-ca-decay-sweep53.json` `applied_genome`, `applied_passive_parameters`
  and `ih_distribution`, emits `docs/evidence/h01-e-b3-experimental-profile.py` holding
  `E_B3_EXPERIMENTAL` in the `_h01_ei_parameters` region-tuple format plus the phase
  controls and passive scalars needed for registration. The test asserts density-by-density
  equality with the report. Registration into `_h01_ei_parameters` and the
  `_paint_profile` ion fix (a region with channels but `calcium=None`, the E axon row)
  land after the registry merge; until then the E axon NaTs row cannot be painted.
- Gate identical to the I gate over 1020-2020 ms at sweeps 50 and 53 against the B3 CVode
  reports (`b3-sk035-ca-decay-sweep50`, `-sweep53`), with an E halving pair under half
  the gate. Benchmark 100 ms first: E BrainCell cost at the copied mesh is unmeasured.
  Prediction: registered only after the 100 ms benchmark, in the E manifest.

### B3 densities verified against the report (`b3-sk035-ca-decay-sweep53.json`)

| Stated | Report | Verdict |
| --- | --- | --- |
| soma NaTs 2.934 x 0.9 | `gbar_NaTs` soma 2.6406641498523276 (= 2.9340712776136972 x 0.9) | matches |
| axon NaTs 3.814 S/cm2 inserted | `insert_density_interventions` NaTs:axon:3.814, 2 sections | matches |
| SK x 0.35 | `gbar_SK` soma 0.001049516063519434 (= 0.0029986173243412404 x 0.35) | matches |
| calcium decay factor 1.0 | `decay_CaDynamics` 494.01955262344603 ms (source value; not the 657 ms of `E_CANDIDATE`) | matches |
| distributed Ih x 75 | total 1.542942623630362e-08 S spread uniformly over soma+dend+apic: 9.994138594205759e-05 S/cm2 in each | matches; note the soma density is the area-distributed value, not 75 x the soma source density |
| leak reversal -4 | `e_pas` -87.97993469238281 (= -83.97993469238281 - 4) | matches |
| Kv3 closing 0.9 | `kv3_closing_factor` 0.9 (phase control, not a density) | matches |
| sodium opening 2.0 | `sodium_opening_factor` 2.0 (phase control) | matches |
| not stated | `sodium_recovery_factor` 1.0; Ra 94.62299222737664; cm soma/axon 1.0, dend/apic 2.3031548608821226; input "command plus recorded bias" (-0.003711858945210089 nA); v_init -83.97993469238281 | added to the export |

## Files

| File | Role |
| --- | --- |
| `docs/evidence/h01-transfer-i-manifest.json` | arms, exact commands, stems, reference files, window, gate, caps, prediction |
| `docs/evidence/h01_transfer_isolation_runner.py` (+ test) | manifest check, command listing (`--print-commands`), profile alignment, per-event scoring (`--score`), decision |
| `docs/evidence/h01-i-transfer/b1-fixed.candidate.json` | B1 candidate file (finalist flags, no CVode) |
| `braintrace/datasets/h01_l2_cell.py` (+ test) | `cv_policy`, segment stimulus |
| `docs/evidence/h01_l2_braincell_reference.py` (+ test) | E BrainCell runner |
| `docs/evidence/h01_e_b3_profile_export.py` (+ test), `docs/evidence/h01-e-b3-experimental-profile.py` | B3 profile literal |
| `docs/h01-causal-model.md` | SP2 registered prediction under Y2 |

## Verification

Tests on fixtures only: no NEURON, no BrainCell run longer than a few steps. The runner's
scoring is tested with synthetic traces shaped like the isolation test. Coverage is reported
for `h01_l2_cell.py` and the three evidence scripts.

## Amendment 2026-09-07 (after step 1, before any further code or run): interpolated peak metric

Step 1 (`docs/evidence/h01-i-transfer/step1-decision.json`) showed the two timing gates
(rise crossing, time above -20 mV) inside the halved gate at every event of the dt 0.005 /
0.0025 halving pair, while the peak-sample voltage differed by 0.21-0.23 mV, and by
0.45 mV (dt 0.005) then 0.22 mV (dt 0.0025) against the NEURON CVode finalist. The peak
difference halves with dt: first-order convergence of a **sampling convention**, not of the
solution. BrainCell records the end-of-step voltage (first sample at dt) on a fixed grid,
so the recorded maximum sits up to dt/2 away from the true peak on a waveform that turns
over at about 1e5 mV/s^2, and the sampled maximum is biased low by an amount proportional to
the grid step; CVode samples where its adaptive step lands. The two traces therefore never
sample the same instant near the peak, and comparing sampled maxima measures the grids,
not the membrane.

**Metric registered.** `peak_interpolated_voltage_mv`: the vertex of the parabola through
the three samples around the discrete maximum of each event, computed in the general
three-point (Lagrange) form so that it is exact for a parabola on any grid, uniform or not.
It is applied identically to both traces (BrainCell and the NEURON CVode reference) by the
same function. A flat triple or a vertex outside the three-sample span falls back to the
sampled maximum. The sampled metric (`peak_sample_voltage_mv`) is still computed and
reported beside it on every row; the default `peak_method="sample"` leaves every existing
result reproducible. The runner takes `--peak-method` and records `peak_method` in every
decision JSON. Timing gates are unchanged and were already valid at dt 0.005.

**Prediction (before re-scoring).** With the interpolated metric the existing step-1 traces
give: (a) halving pair `r-braincell-matched-027` vs `-halfdt`, 270-329.5 ms, peak
difference under 0.05 mV at every event; (b) `r-braincell-matched-027` (dt 0.005) vs the
NEURON finalist `e-kv3-close2-027`, peak difference under 0.1 mV at every event.

**Rejection.** Either (a) or (b) fails at any event: SP2 stops as `time_level_open`; no
further run is launched.

**If it holds.** The registered arms run one at a time in manifest order, skipping the two
already done: `a0-maxcv-027`, `a1-matched-027`, `a1-matched-019`, `a1-matched-023`, then
NEURON `b1-fixed-027`, `b1-fixed-019`, `r-neuron-fixed-027-halfdt`. The decision literal
is read on the interpolated metric with the timing gates unchanged; both peak metrics are
reported. The 0.23 nA arm is a spent-holdout control, scored and never used to choose.
The BrainCell full-train cost is derived (about 525 s per run at dt 0.005, an upper bound
from the 330 ms wall clock); NEURON fixed-step full-train cost is unmeasured.
