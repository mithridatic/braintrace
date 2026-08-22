# Stage 1: gated and normalized memory reads

## Interface

Add the local public operator:

```text
braintrace.gated_projection(
    values,
    gate_input,
    *,
    gate_weight,
    gate_bias,
    output_weight,
    normalize,
    epsilon,
)
```

It computes `RMSNorm(values)` when `normalize=True`, then returns
`(sigmoid(gate_input @ gate_weight + gate_bias) * values) @ output_weight`.
All three trainable operands belong to one ETP primitive. Add the reusable
`braintrace.nn.GatedProjection` module.

The operator accepts rank-two batched, floating, dimensionless values and gate
inputs. `gate_weight` has shape `(gate_features, value_features)`, `gate_bias`
has shape `(value_features,)`, and `output_weight` has shape
`(value_features, output_features)`. `epsilon` is finite and positive.

The JAX primitive owns eager, lowering, JVP/VJP, and batching behavior. Its
pp-prop rule is exact for one-step finite windows and approximate after temporal
factorization. Its D-RTRL rule retains complete parameter-to-output position
Jacobians and is exact at diagnostic scale; the memory cost is explicit and is
not a recommendation for full-scale D-RTRL.

## Example 21 integration

Add `memory_read_transform` with values `linear`, `gated`, and `gated_rms`,
defaulting to `linear`.

- `linear` uses the existing `memory_read_projection(raw_read)` unchanged.
- `gated` passes `_unit_l2_cap(previous_workspace)` as `gate_input` and skips
  read normalization.
- `gated_rms` uses the same gate input and RMS-normalizes `raw_read` inside the
  fused primitive.

For `gated`, initialize gate weights and bias to exact zero and initialize the
fused output weight to twice the existing read projection. Since
`sigmoid(0)=0.5`, this is function-identical to `linear` at initialization and
retains nonzero gate gradients. `gated_rms` uses the same initialization but is
not claimed initialization-equivalent because normalization changes the read.

The setting round-trips through model and experiment configuration, CLI,
result/report configuration, memory architecture reports, and parameter-tree
checkpoint compatibility. Record gate saturation, per-channel activation,
read RMS, neuron-drive RMS, parameter movement, and matched control effects.

## Tests

- independent forward equation with and without RMS normalization;
- JVP and VJP agreement, JIT, batching, zero/extreme inputs, validation, units;
- gate parameters remain active at zero initialization;
- one-step pp-prop and D-RTRL agreement with BPTT, plus honest finite-window
  pp-prop divergence classification;
- module ownership, initialization, and public exports;
- default linear equivalence and gated initialization equivalence;
- configuration, CLI, report, architecture manifest, reset, and checkpoint
  coverage.

## Promotion

Compare `linear`, `gated`, and `gated_rms` using the shared pilot and full gates.
No Example 21 default changes without full promotion evidence.

