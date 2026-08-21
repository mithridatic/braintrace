# Example 21: Adam weight-decay policy and decoupled-decay arithmetic pins

Date: 2026-08-21
Status: implemented

## Problem

A review of the Example 21 optimizer weight-decay mechanism found two
discrepancies and one test gap:

1. **Silent ignore of explicit decay on plain Adam.**
   `ExperimentConfig(optimizer="adam", weight_decay=0.05)` validated and stored
   `0.05`, and `_optimizer_policy` reported `weight_decay: 0.05` in the run
   report — but `_make_training_optimizer` constructs
   `braintools.optim.Adam(lr=...)` with no decay argument, so the applied
   optimizer used none. Config, run report, and applied policy disagreed.
2. **braintools `Adam` decay would be coupled if ever routed through.**
   `braintools.optim.Adam.default_tx` places `optax.add_decayed_weights`
   *before* `optax.scale_by_adam` (L2-in-gradient), so "fixing" discrepancy 1
   by forwarding the value would silently produce coupled L2 decay, unlike the
   decoupled decay the docstring promises for AdamW and Muon.
3. **Test gap.** Existing tests pinned config resolution and that updates move
   parameters, but not the decay *arithmetic* — nothing asserted the decoupled
   signature `param -= learning_rate * weight_decay * param` on any path.

## Decision

- Reject a nonzero explicit `weight_decay` when `optimizer="adam"` at config
  validation time with a clear `ValueError`. Plain Adam applies no decoupled
  decay; refusing the value keeps config, docstring, run report, and applied
  policy in agreement. This is safer than forwarding the value into
  `braintools.optim.Adam`, whose decay placement is coupled (finding 2).
  An explicit `weight_decay=0.0` with Adam remains valid.
- Pin the decay arithmetic with a zero-gradient compiled update: with zero
  gradients every optimizer's gradient term vanishes, so one `update` must
  shrink each parameter by exactly `learning_rate * weight_decay * param`
  (factor `1 - lr * wd`) on every decoupled path — braintools AdamW (both
  leaves), Muon's rank-two matrix partition, and Muon's AdamW fallback for
  rank-one leaves — while plain Adam leaves parameters bit-unchanged in value
  (factor 1).

## Non-goals

- No change to braintools (`Adam` coupled-decay placement is upstream).
- No change to the resolved per-optimizer defaults
  (`adam: 0.0, adamw: 0.01, muon: 0.1`) or to the Muon construction.

## Tests

Co-located in `examples/pp_prop/21-latent-reasoning-in-context_test.py`:

- `test_adam_rejects_explicit_nonzero_weight_decay` — constructor and CLI both
  raise; explicit zero stays accepted.
- `test_zero_gradient_update_applies_decoupled_weight_decay` — parametrized
  over `adam`, `adamw`, `muon` with a 2-D matrix leaf and a 1-D vector leaf;
  asserts the exact `1 - lr * wd` shrink factor (1 for Adam).
