# Example 21: smooth softcaps for the memory value coding and reasoning query

Date: 2026-08-21
Status: implemented

## Problem

Two hard `tanh` caps in `latent_workspace_model.py` clamp their signals to
`(-1, 1)`:

1. the memory value coding, `code = tanh(features @ value_basis)`
   (`encode_memory_value`), and
2. the iterative reasoning query,
   `tanh(query_encoding + projected_query)` in the latent step.

A `beta = 1` tanh saturates early and flattens gradients for pre-activations
above ~2. The softcap trick (Eq. 12 of the SiTU-GLU formulation — only the
cap, not the GLU/sigmoid-gate structure) generalizes the cap:
`softcap(x, beta) = beta * tanh(x / beta)`, slope one at zero, smoothly
bounded to `(-beta, beta)`.

The row-carrier gate `tanh` is a gate, not a cap, and is deliberately
untouched.

## Decision

- New public `softcap(value, beta)` in `latent_workspace_model.py`.
  `beta = 1.0` reproduces `tanh(value)` bit-exactly (`x / 1.0` and
  `1.0 * y` are exact float operations), pinned by test.
- Two `ModelConfig` knobs, validated positive finite:
  `memory_value_softcap_beta` (default 1.0) and
  `reasoning_query_softcap_beta` (default 1.0). The model layer defaults to
  the legacy-exact caps so every direct `ModelConfig` consumer keeps
  bit-identical behavior; the experiment layer opts into wider caps.
- `ExperimentConfig` knobs with defaults `memory_value_softcap_beta = 4.0`
  and `reasoning_query_softcap_beta = 25.0`, CLI flags
  `--memory-value-softcap-beta` / `--reasoning-query-softcap-beta`, plumbed
  through `_model_config` and reported in the entry's model report (and in
  `to_dict` automatically).
- The binding-gate diagnostic that recomputes the capped query
  (`latent_workspace_binding_gate.py`) mirrors the softcap through
  `model.config.reasoning_query_softcap_beta` so the diagnostic tracks the
  model at any beta.
- The preregistered architecture label `"fixed_tanh_projection"` /
  `"learned_tanh_projection"` is kept: the value map is still a tanh-family
  projection, now scaled; the betas are reported as explicit model-report
  keys instead of a label change, so the binding gate's structural contract
  is untouched.
- Downstream bound: the stored value code is now bounded to `(-beta1, beta1)`
  rather than `(-1, 1)`. The write path (`update_context_memory`) and read
  path are linear in the value and carry no unit-bound assumption; the
  `encode_memory_value` docstring is updated to state the new bound.

## Default sanity measurements (smoke scale)

Measured pre-cap magnitudes on the untrained smoke-scale model (128 neurons,
memory width 2), two embedded fixture tasks, full episode plus 60 latent
steps each (420 calls per site):

- memory value coding pre-cap `|x|`: max 0.4338, mean 0.2116, p95 0.3779.
  Under the legacy `beta = 1` tanh this range was already near-linear
  (~7% compression at the max); `beta1 = 4` removes that residual
  compression and leaves a smooth guard with ~9x headroom.
- reasoning query pre-cap `|x|`: max 2.0582, mean 0.7498, p95 2.0495.
  The legacy `beta = 1` tanh saturated hard here (`tanh(2.06) = 0.968`);
  `beta2 = 25` is near-linear in this regime (<0.1% compression at the
  observed max) and acts purely as a blowup guard. The query encoding
  accumulates across query rows, so trained models can grow into the cap.

Neither default is degenerate (no hard saturation, finite activations
throughout), so 4.0 / 25.0 stand as configured.

## Tests

- `latent_workspace_model_test.py`: `softcap(x, 1.0)` bit-equals
  `jnp.tanh(x)`; softcap formula/bounds at other betas; `ModelConfig`
  rejects nonpositive/nonfinite betas.
- `21-latent-reasoning-in-context_test.py`: entry defaults 4.0/25.0 pinned;
  CLI round-trip; `_model_config` plumbs both betas; nonpositive values
  rejected.
