# Example 21 system model

## Scope

This document describes the supported Example 21 code in
`examples/pp_prop/21-braincell-arc.py` and its ARC boundary module,
`examples/pp_prop/arc_contracts.py`. It describes code and focused tests. It
does not report a completed ARC training or evaluation run.

## Model

`BrainCellArcModel` has 2,048 Hodgkin-Huxley cells. It has three trainable
parameter groups:

| Group | Structure | Count |
|---|---|---:|
| Input | 441 source features, 32 CSR entries per source | 14,112 entries |
| Recurrent | 2,048 source cells, 8 CSR entries per source | 16,384 entries |
| Readout | 2,048 by 360 dense weight and a 360-value bias | 737,640 values |

Input and recurrent drives use `braintrace.sparse_matmul`. The model bounds
the combined drive with `tanh`, converts it to current density, and advances
the cells. The next recurrent event uses the emitted spike values from the
previous advance. The readout maps normalized membrane voltage to 360 logits.

The topology code constructs unique, non-self recurrent pairs. It does not
assign Dale signs. It has no stored neuron ownership or mechanism labels.
Those labels occur only in the checkpoint schema and are not set by this
baseline model.

## Event and output path

An ARC episode has 705 Boolean events of width 441 and an equal-length
Boolean advance mask. The encoder includes demonstration inputs and outputs,
the query input, one shape request, and 30 row requests. It does not encode a
query target.

The 360 readout values contain 30 height logits, 30 width logits, and 300
color logits. The production decoder uses 31 request readouts: one shape
readout and 30 row readouts. It selects height and width independently, then
selects one of ten colors for each cell in the selected rectangle. It returns
a `uint8` grid.

## Training boundary

The compiler creates PP-Prop temporal relations for the input and recurrent
weights. Readout weights use direct gradients. An episode update masks loss at
the learner call, clips the combined gradient to global norm 1, and applies
Adam rates of 0.001, 0.0003, and 0.003 for input, recurrent, and readout
groups. Episode reset clears biological and eligibility state while retaining
parameters and Adam moments.

Focused tests verify these code paths, including compiled event sequences,
padding that preserves state, topology counts, direct-readout gradients, and
the fixed proof and ordinary schedules.

## Known execution boundary

The repository has no recorded completed run of this BrainCell model on raw
ARC data. The local Gate 4 note also states that the required ARC files were
not available for its real-data process run. Therefore the code and tests do
not support claims about ARC accuracy, learning speed, causal success, or
BrainCell checkpoint behavior after training.
