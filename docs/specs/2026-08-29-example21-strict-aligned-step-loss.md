# Example 21 strict-aligned PP-Prop step loss

## Problem

Every counted Example 21 update (Gate 4 proof, Gate 5 structural arms) drove
PP-Prop with `step_fn = lambda event: jnp.sum(etrace_evolve(event)[0])`. That
objective is the sum of the 2,048 membrane voltages. The query target never
entered training, the readout weight and bias were never in any gradient, and
the loss mask covered every advancing event instead of the 31 request events.
The recorded shape and row losses were computed after the fact with the NumPy
`request_loss` for reporting only. The model was never trained on ARC.

Evaluation had a second defect: predictions were decoded from the single
`(360,)` readout of the final row-request state, so every decoded grid was a
stack of constant-colour rows. The decoder's `(31, 360)` request path existed
but nothing fed it, and the Gate 4 decoder-timing helper read event 673 (input
end) as the shape request instead of event 674.

## Contract (OpenSpec `Strict-aligned request loss`, design.md §7, PP-Prop paper Eq. 2/22)

PP-Prop's gradient is `Σ_t ∂L^t/∂h^t ∘ (ε_f ⊗ ε_x)`. The traces come from the
compiler; the learning signal `∂L^t/∂h^t` is whatever `step_fn` differentiates.
`step_fn` MUST therefore be `loss(output_t, target_t)` with a target sequence
passed as the second `etrace_grad` sequence, and the readout parameters MUST be
in the differentiated weight set (tutorial: `optimizer.register_trainable_weights(learner.param_states)`
with `w_out` included).

## Requirements

- `encode_targets(task, query_index)` SHALL return an `int32` array of shape
  `(705, 33)`: column 0 is the loss kind (`0` none, `1` shape request, `2` valid
  row request), column 1 is `height - 1`, column 2 is `width - 1`, columns
  3–32 are that row's target colours (zero padded). Only event 674 has kind 1;
  only events `675 + r` for `r < height` have kind 2. A query without a target
  SHALL raise `ValueError`. Targets SHALL never enter a model event; the event
  array from `encode_episode` is unchanged.
- `request_step_loss(logits, target)` SHALL be a JAX function equal to the NumPy
  `request_loss`: kind 1 → height cross-entropy over `logits[:30]` plus width
  cross-entropy over `logits[30:60]`; kind 2 → mean cross-entropy over the
  `width` valid cells of `logits[60:].reshape(30, 10)`; kind 0 → exactly `0.0`.
- `arc_step_loss(learner, model)` SHALL return `step_fn(event, target)` that
  calls the learner exactly once, forms the 360 logits from the returned
  voltage with the model's live `readout_weight` and `readout_bias`
  `ParamState` values, and returns `request_step_loss`.
- `trainable_weights(learner, model)` SHALL be the learner's compiled
  `param_states` plus `readout_weight` and `readout_bias`. The trainer SHALL
  pass it as `weights=` so readout gradients are produced by the same
  `etrace_grad` pass (readout is non-temporal: its gradient is exact autodiff,
  no trace).
- The `etrace_grad` `mask` SHALL be `targets[:, 0] > 0` (`request_loss_mask`).
- `PPPropEpisodeTrainer.update_episode` SHALL accept `targets=` and pass
  `(events, targets)` as the sequences; `weights` is fixed at construction.
- `request_readouts(model, voltages)` SHALL apply the readout to the voltages
  at `REQUEST_EVENT_INDICES = (674, 675, …, 704)` and return `(31, 360)`.
  `predict_episode(model, events, advances)` SHALL decode through that path.
  Gate 4 and the structural runner SHALL use it for every prediction and
  strict Boolean.
- The Gate 4 proof SHALL additionally record `direct_accuracy` for pre and
  post: `shape_correct` (bool), `cells_correct`, `cell_count`. Strict pass is
  `shape_correct and cells_correct == cell_count` (proved equivalent by the
  scorer on 2,000 random grids on 2026-08-29).

## Verification

- `request_step_loss` equals `request_loss` to `1e-5` on random logits for
  shape, row, and non-request targets.
- `encode_targets` marks exactly `1 + height` events; event array unchanged.
- A real compiled episode with `arc_step_loss` yields finite gradients for
  input, recurrent, readout weight, and readout bias; the readout gradient is
  nonzero; repeated updates on one fixed episode lower the masked loss.
- `predict_episode` decodes from the 31 request states and returns the
  decoded shape rather than a constant-row grid.
- Gate 4 rerun in Docker records the pre/post decoded grids and
  `direct_accuracy` under the real loss.
