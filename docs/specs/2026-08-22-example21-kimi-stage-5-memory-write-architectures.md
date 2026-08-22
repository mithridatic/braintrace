# Example 21 Kimi Transfer Stage 5: Memory-Write Architectures

## Status

Approved for implementation on `feat/example21-kimi-transfer`. The two arms
are alternatives and remain opt-in pending synthetic and ARC evidence.

## Public Local Operators

### `braintrace.delta_memory_update`

One fused ETP primitive owns key, value, beta, and retention projections. Keys
are L2-normalized and scaled. Value-channel retention is
`exp(min_log_decay * sigmoid(x_value @ Wr + br))`; write strength is scalar
`sigmoid(x_value @ Wb + bb)`. The candidate update is:

```text
decayed = alpha * memory
prediction = decayed^T key
error = value - prediction
updated = decayed + beta * outer(key, error)
```

The caller's existing write gate commits or rejects the candidate. Under
pp-prop this nonlinear error-correcting write is classified approximate beyond
its verified finite window. D-RTRL retains full parameter/output positions.

### `braintrace.situ_glu` and `braintrace.nn.SiTUGLU`

One fused ETP primitive computes:

```text
gate_pre = x @ gate_weight + gate_bias
up_pre = x @ up_weight + up_bias
gate = softcap(gate_pre, gate_beta) * sigmoid(gate_pre)
up = softcap(up_pre, up_beta)
output = (gate * up) @ output_weight
```

Defaults are `gate_beta=4` and `up_beta=25`.

## Example 21 Arms

- `memory_coding="delta_write"` consumes the complete protocol-v2 update
  feature vector for both key and value inputs. Its candidate square memory is
  committed only on the existing write gate; false lanes remain byte-exact.
- `memory_coding="situ_glu_update"` uses hidden width
  `4 * context_memory_width` and output width `context_memory_width ** 2`, then
  enters the existing gated additive recurrence. It bypasses the outer
  `memory_value_softcap` because SiTU-GLU owns both caps.
- Both arms are independent alternatives to `learned_update`.

## Structural and Synthetic Gates

Independent forward/JVP/VJP/JIT/batch references, validation, units, extreme
finite values, pp-prop and D-RTRL classification, reset/snapshot/checkpoint,
false-lane identity, no latent mutation, repeated-key overwrite, orthogonal-key
preservation, collision finiteness, one-sided distinction, shuffled-pairing
sensitivity, and the preregistered finite-window gradient panel.

## Promotion

Compare each arm independently with the accepted `learned_update` stack. If
both pass, select by binding improvement, exact score, rule-at-oracle, and
runtime in that order. Operator qualification does not promote an Example 21
arm or change defaults.
