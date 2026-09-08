# SP2 step 1: BrainCell halving pair at the copied mesh, dt 0.005 ms (2026-09-07)

Spec: [SP2 transfer isolation](../specs/2026-09-07-h01-transfer-isolation.md). Manifest:
`h01-transfer-i-manifest.json`. Decision JSON: `h01-i-transfer/step1-decision.json`.
Two runs only (`r-braincell-matched-027`, `r-braincell-matched-027-halfdt`), launched one
after the other as detached jobs with a 600 s watchdog; neither was killed. Profile
alignment (`--check-alignment`): aligned, mode `finalist` (somatic Kv3 closing factor 2.0).
Interpreter: the sibling worktree's `.cache/validation` Python 3.13 with `PYTHONPATH` set
to this worktree (`braintrace.__file__` verified here). Mesh: `--mesh-from`
`h01-i-energetic/e-kv3-close2-027.json` (CVPerBranchList, 1413 compartments).

## Measured

| Arm | dt (ms) | Simulated (ms) | Wall clock (s) | s per simulated ms | Exit |
| --- | --- | --- | --- | --- | --- |
| r-braincell-matched-027 | 0.005 | 330 | 115.6 | 0.35 | 0 |
| r-braincell-matched-027-halfdt | 0.0025 | 330 | 246.0 | 0.745 | 0 |

Wall clock includes interpreter start and XLA compile. Peak RSS was not measured: the
launcher sampled the uv trampoline process (6.4 MB), not the interpreter.

## Halving pair over 270-329.5 ms (gate halved: 0.05 ms / 0.05 mV / 0.005 ms)

Three events in both traces. Rise crossing and time above -20 mV are inside the halved
gate at every event (max |rise| 0.0088 ms, max |width| 0.0015 ms). The peak sample
voltage differs by 0.21-0.23 mV at every event, over the 0.05 mV half-gate. Literal
verdict: **time_level_open** (peak-sample metric only; the two timing metrics are valid).

## dt 0.005 BrainCell vs NEURON finalist CVode (`e-kv3-close2-027`, full gate 0.1 / 0.1 / 0.01)

Equal count (3 = 3). Per event (signed, BrainCell minus NEURON):

| Event | Reference rise (ms) | Rise error (ms) | Peak error (mV) | Width error (ms) | In gate |
| --- | --- | --- | --- | --- | --- |
| 1 | 281.198 | +0.0085 | -0.454 | +0.0029 | rise, width; peak fails |
| 2 | 297.558 | -0.0151 | -0.434 | +0.0030 | rise, width; peak fails |
| 3 | 314.595 | -0.0176 | -0.432 | +0.0030 | rise, width; peak fails |

Supplementary, dt 0.0025 vs the same reference: rise max 0.0087 ms, width 0.0015 ms, peak
-0.220 to -0.223 mV. The peak error halves with dt, as does the pair difference; the
BrainCell peak sample converges toward the NEURON value as the step shrinks, so the
0.1 mV peak-sample gate is not a valid decision limit at dt 0.005 or 0.0025 with this
sampling convention (end-of-step samples, first sample at dt). Rise and width are.

## Derived (not measured) cost of a full 1500 ms run at the copied mesh

Linear scaling of the 330 ms wall clock, compile included, so an upper bound:
dt 0.005: ~525 s (under 15 min); dt 0.0025: ~1,118 s (over 15 min, needs approval).
The 0.27 nA finalist reference holds 37 spikes over the full train.

## Untested

`r-neuron-fixed-027-halfdt`, `a0-maxcv-027`, `a1-matched-027`, `a1-matched-019`,
`a1-matched-023`, `b1-fixed-027`, `b1-fixed-019`. No full-train verdict exists.

Qualification: numerical transfer only on donor anatomy; identifies no channel cause and no
human waveform validity.
