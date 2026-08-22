# Example 21 Kimi Transfer Stage 4: Progressive Effort

## Status

Approved for implementation on `feat/example21-kimi-transfer`. Promotion is
not evaluated until matched training evidence exists.

## Objective

Introduce reasoning depths progressively while retaining one model and one
optimizer state across the complete update horizon.

## Configuration

Add `effort_schedule`: `"uniform" | "progressive"`, default `"uniform"`.
The default calls the historical scheduler unchanged.

## Schedule

Split `training_updates` into `len(training_efforts)` contiguous near-equal
phases, assigning an extra update to earlier phases. Phase `i` samples as
evenly as possible from efforts `0..i`. Construct the balanced phase vector
and shuffle only within that phase with `brainstate.random.RandomState`.

For 260 updates and `(0, 30, 60)`, phase boundaries are `[0, 87)`, `[87,
174)`, and `[174, 260)`. The first phase is all effort zero; the second uses
only zero and 30; the final phase uses zero, 30, and 60.

## Compatibility

The schedule changes only prepared training order. It does not rebuild the
model or optimizer between phases. Configuration, CLI, result JSON, training
report, and checkpoint compatibility metadata carry the schedule name.

## Structural Tests

- historical uniform sequence equivalence;
- exact 260-update phase boundaries and admitted effort sets;
- within-phase balance and deterministic replay;
- seed sensitivity confined within phases;
- generic horizons and effort sets;
- chunk-size independence;
- CLI/config/result/report/checkpoint round trips;
- one optimizer instance and uninterrupted optimizer step count.

## Promotion

Compare progressive against globally uniform at identical updates, training
examples, model initialization, data budget, and accepted Stage 3 stack.
