# SP2 transfer isolation, I side: result (2026-09-07)

Spec: [SP2 transfer isolation](../specs/2026-09-07-h01-transfer-isolation.md), amendment of
2026-09-07 (interpolated peak metric). Step 1: [halving pair](h01-i-transfer-step1.md),
`h01-i-transfer/step1-decision.json`. Step 1b: `h01-i-transfer/step1b-decision.json` (offline
re-score, no run). Runner output: `h01-i-transfer/sp2-i-decision.json`
(`--score --peak-method interpolated --reference-root <sibling worktree>`; the reference
`.npz` traces are tracked on the sibling branch, hash recorded).

**Decision literal: `time_level_open`.** The registered step-1b prediction failed on both
legs, so under its rejection no full-train arm was launched. Runs made in SP2: two
(`r-braincell-matched-027`, `-halfdt`, step 1). Runs untested: `a0-maxcv-027`,
`a1-matched-027`, `a1-matched-019`, `a1-matched-023`, `b1-fixed-027`, `b1-fixed-019`,
`r-neuron-fixed-027-halfdt`. Wall clocks: step 1 only (115.6 s at dt 0.005, 246.0 s at
0.0025, 330 ms each, compile included). No full-train duration was measured; the 525 s
figure remains derived and unmeasured.

## Step 1b: both peak metrics on the existing traces

Peak metric registered before re-scoring: vertex of the parabola through the three samples
around the discrete maximum, same function on both traces (general three-point form, so it is
exact for a parabola on the CVode reference's non-uniform grid). Prediction: (a) halving pair
under 0.05 mV at every event, (b) dt 0.005 vs the NEURON finalist under 0.1 mV at every event.

### Halving pair, `r-braincell-matched-027` (dt 0.005) vs `-halfdt` (dt 0.0025), 270-329.5 ms, half gate 0.05 ms / 0.05 mV / 0.005 ms

| Event | Rise error (ms) | Width error (ms) | Peak, sampled (mV) | Peak, interpolated (mV) | Interpolated in half-gate |
| --- | --- | --- | --- | --- | --- |
| 1 | -0.0043 | -0.0015 | +0.2305 | +0.2134 | no |
| 2 | +0.0076 | -0.0015 | +0.2107 | +0.2111 | no |
| 3 | +0.0088 | -0.0015 | +0.2113 | +0.2109 | no |

Prediction (a): **failed** at every event.

### dt 0.005 vs NEURON finalist `e-kv3-close2-027` (CVode atol 1e-10, x9), full gate 0.1 ms / 0.1 mV / 0.01 ms

| Event | Reference rise (ms) | Rise error (ms) | Width error (ms) | Peak, sampled (mV) | Peak, interpolated (mV) | Interpolated in gate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 281.198 | +0.0085 | +0.0029 | -0.4537 | -0.4349 | no |
| 2 | 297.558 | -0.0151 | +0.0030 | -0.4336 | -0.4343 | no |
| 3 | 314.595 | -0.0176 | +0.0030 | -0.4317 | -0.4323 | no |

Prediction (b): **failed** at every event. Supplementary, dt 0.0025 vs the same reference:
interpolated peak -0.2214 / -0.2232 / -0.2214 mV (sampled -0.2231 / -0.2229 / -0.2204).

### What the metric showed

Interpolation moves each peak by at most 0.018 mV (BrainCell dt 0.005, event 1) and by
about 0.001 mV on the reference, whose adaptive step is 0.00044-0.00052 ms at the peaks. The
sampling convention therefore accounts for none of the 0.21 mV halving difference or the
0.43 mV reference difference: the amendment's premise (sampled maxima measure the grids) is
refuted, and the difference is in the solution's peak amplitude.

Per-event peaks, interpolated: BrainCell dt 0.005 17.505 / 17.375 / 17.263 mV; dt 0.0025
17.718 / 17.586 / 17.474 mV; NEURON 17.940 / 17.809 / 17.696 mV. The BrainCell peak climbs
by 0.21-0.22 mV when dt halves and stands 0.22 mV under NEURON at dt 0.0025: first-order
convergence in dt toward the reference. Derived, not a new run: the first-order Richardson
extrapolation `2 x (dt 0.0025) - (dt 0.005)` gives 17.931 / 17.797 / 17.685 mV, within
0.012 mV of NEURON at every event. The peak amplitude of the staggered BrainCell integration
at this mesh converges first order in the step, and its limit agrees with CVode; at dt 0.005
and 0.0025 the peak is not yet inside the 0.1 mV gate. The timing gates (rise crossing, time
above -20 mV) were inside their halved gates at dt 0.005 and stay so under both metrics.

## Per-input verdicts

| Input (nA) | Arms | Verdict |
| --- | --- | --- |
| 0.27 | halving pair (330 ms), reference comparison | timing gates valid; peak gate `time_level_open` at dt 0.005 and 0.0025 under both metrics |
| 0.27 | `a0-maxcv-027`, `a1-matched-027`, `b1-fixed-027`, `r-neuron-fixed-027-halfdt` | untested (not launched under the rejection) |
| 0.19 | `a1-matched-019`, `b1-fixed-019` | untested |
| 0.23 (spent-holdout control) | `a1-matched-023` | untested |

## What it means for Y2

Y2 is neither closed nor reassessed as an implementation difference. The full 270-1270 ms
train remains unscored at every input. What is new: the peak-voltage gate at 0.1 mV is not
reachable at dt 0.005 or 0.0025 with the present integration, for a reason that is now
located (first-order peak convergence, limit consistent with CVode) rather than attributed to
sampling. The gate's timing components are valid at dt 0.005. Options, each a user decision
and none taken here: (i) run the full-train arms at dt 0.005 reading the decision on the
timing gates alone, with the peak reported but not gated (a spec change); (ii) the spec's
fallback dt 0.000625 (measured 556 s per 330 ms, derived about 2,500 s per full train, needs a
measured estimate and approval); (iii) a higher-order BrainCell integration at the copied
mesh, which would be a new measurement-function qualification before any transfer arm.

Qualification: numerical transfer only on donor anatomy; identifies no channel cause and no
human waveform validity.
