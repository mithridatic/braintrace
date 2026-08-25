# BRA-7 BrainCell and PP-Prop execution

## Scope

This note maps OpenSpec tasks 4.1-5.3 to the implementation and focused tests.
The implementation is limited to the baseline 2,048-neuron model and its
episode trainer. Structural, Dale, timing, and artifact stages remain outside
this change.

## Review Gate 3 corrections

The episode objective must pass its request mask to `etrace_grad`; masking only
the returned loss does not mask the gradient. The schedule runner must enforce
the declared training task order and reject validation episodes before any
optimizer call. Validation has a separate forward-only entry point. The matched
timestep check must return the selected substep count so the caller can apply
the two-half-step fallback. Tests must include an interspersed false PP-Prop
event and a complete trainer update that changes parameters while retaining
Adam moments across reset.

## Required invariants

- Input CSR has 14,112 entries. Recurrent CSR has 16,384 entries.
- CSR rows identify source neurons. Recurrent pairs are unique and non-self.
- Weight initialization uses `brainstate.random.RandomState(21)` and `(22)`.
- Every cell uses the declared BrainCell 0.1.0 single-compartment values.
- False events preserve voltage, channel gates, spikes, and eligibility state.
- Episode reset clears biological and eligibility state but retains parameters
  and Adam moments.
- Input and recurrent weights have PP-Prop hidden-state relations. Readout
  weights use a finite direct gradient and are not temporal parameters.
- Loss is accumulated only on shape and valid row requests. The combined
  gradient is clipped to norm 1.0 and receives one Adam update per episode.
- Training uses eight updates for proof and 64 updates for ordinary runs.
  Validation is forward-only.

## Verification matrix

The sibling test module must cover topology counts and CSR direction, all
declared cell values, bounded current units, next-event spike transmission,
compiled false-event freezing, reset retention rules, compiler relations,
loss masking, gradient clipping, learning rates, trace decay, task order, and
update counts. Tests must use compiled transform paths for repeated events and
must not call BPTT or synthetic data.

## Edge cases

Reject dense topology, duplicate or self recurrent edges, total-current units,
missing temporal relations, target leakage into model events, extra optimizer
updates after a failed gate, and non-finite optimizer state. A zero advance
must return zero loss and zero gradient while preserving state bitwise.

## Review correction

Schedule metadata (`task_id` and `validation`) is a gate only and must not be
forwarded to `update_episode`. Forward validation must snapshot every
non-parameter state reachable from the learner, including biological hidden
state and eligibility traces, and reject any change with exact array equality.

## Review correction record

The first implementation stopped at compatibility fixtures. The production
path must also expose a compiled sequence driver and an episode trainer. The
driver uses `brainstate.transform.for_loop` for events and
`brainstate.transform.jit` for the outer call. A false event returns a zero
voltage sentinel without touching biological or learner state. The trainer
resets only biological and eligibility state between episodes, accumulates
masked request loss, clips the combined gradient once, and applies one Adam
update. The readout width is a production constant, not a test-module global.

Focused tests must execute these paths, including a false event inserted into
an advancing sequence, retained Adam moments after reset, and exact proof and
ordinary schedule counts. Validation is represented as forward-only by the
trainer API and cannot call the update method.
