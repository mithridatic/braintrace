# Example 21 documentation backlog

## Scope

This list records documentation findings from source inspection. It does not
change the Gate 4 runner, OpenSpec tasks, or the evidence status of the model.

## B-01: Validation task identifier differs between sources

**Setup.** Compare the declared validation order in the OpenSpec and ARC
contract module with the validation guard in the model module.

**Observation.** The OpenSpec and
`examples/pp_prop/arc_contracts.py` declare `3428a4f5` as the second
validation task. `examples/pp_prop/21-braincell-arc.py` instead contains
`342f8a4f5` in `VALIDATION_TASK_IDS`. `run_fixed_schedule` uses that local
tuple to reject validation episodes from updates.

**Inference.** The data dictionary correctly records the declared contract,
but the duplicate model constant does not match it. A caller that marks the
declared task as validation is protected by the Boolean `validation` field.
The effect when that field is absent is not measured.

**Required owner action.** BRA-34 must decide whether to remove the duplicate
constant or correct it and add a regression test. Do not update OpenSpec task
9.2 or 9.3 from this finding. Re-run the focused schedule tests after the
implementation change.
