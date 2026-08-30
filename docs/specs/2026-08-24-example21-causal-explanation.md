# Example 21 causal explanation

## Scope

This document describes the direct behavior that the current Example 21
replacement code can execute. It separates observations from inferences. It
does not describe a planned experiment as a completed result.

## Acceptance rule

The strict result is the required measure. A task passes only when every query
has the exact predicted height, width, and integer color cells. Partial cell
scores, shape scores, averages, and target-fed decoder checks are not strict
model results.

## Observations

- `load_task` reads one named raw ARC JSON file from a declared training or
  evaluation directory. It validates rectangular grids, dimensions from 1
  through 30, and colors from 0 through 9.
- `encode_episode` returns a fixed Boolean array with shape `(705, 441)` and a
  Boolean advance mask with 705 values. The query target is not in this event
  array.
- `BrainCellArcModel` creates one recurrent layer with 2,048 model-cell
  instances. Its sparse input topology has 14,112 connections. Its sparse
  recurrent topology has 16,384 directed connections. Its direct readout has
  360 values per readout state.
- Each model-cell is a `braincell.SingleCompartment` Hodgkin-Huxley cell with
  sodium, potassium, and leak channels. The default model has no Dale type and
  no deferred chemical mechanism.
- `_advance` uses `braintrace.sparse_matmul` for input and recurrent drives.
  It sends bounded current density to the model-cell population for one
  `0.1 ms` interval.
- `run_event_sequence` uses `brainstate.transform.for_loop` inside
  `brainstate.transform.jit`. A false advance value returns a zero voltage
  sentinel and does not advance the biological state.
- The PP-Prop compiler path uses `braintrace.compile` with `vjp_method` set to
  `single-step` and trace decay `0.95`. The episode trainer clips the combined
  gradient to norm `1.0` and applies one grouped optimizer update.
- The strict decoder selects height and width from 30-value groups and color
  from ten-value groups. It returns an integer grid and performs no repair or
  candidate selection.
- The structural plot topology object uses checkpoint topology and labels
  only. It does not use readout weights, bias values, target grids, or
  prediction inputs. It reports the neuron count, input-connection count,
  recurrent-connection count, Dale groups, and task-owner groups.

## Direct test evidence

The co-located tests in
`examples/pp_prop/21-braincell-arc_test.py` exercise model-cell values,
current units, finite local derivatives, sparse relation discovery, false
event freezing, reset behavior, direct readout gradients, and fixed update
schedules. The tests in
`examples/pp_prop/example21_structural_test.py` exercise sparse topology
counts, checkpoint labels, plot counts, plot groups, and prediction-byte
stability during plotting.

These tests are compatibility and mechanism checks. They do not provide a
real-data strict task count unless a direct ARC data root and an executed
training run supply that count.

## Inferences

The direct code path can represent ARC input as fixed-width events and can
carry state through sparse BrainCell updates. The tests support this limited
inference because they inspect the event contract, sparse relations, finite
cell state, and transformed event execution.

The false-event checks support the inference that padding is intended to be
state-neutral. They do not prove a real ARC prediction improvement.

The direct readout gradient checks support the inference that height, width,
and color readout parameters can receive a finite local gradient. They do not
prove that a full temporal PP-Prop run matches BPTT or solves a real ARC task.

## Boundary

The current repository evidence does not contain an executed real-data
BrainCell checkpoint with a strict task result. Therefore this document makes
no claim of real ARC accuracy, causal task success, Dale-stage success, or
prediction improvement. Such a claim requires a direct executed run and its
recorded prediction and target grids.
