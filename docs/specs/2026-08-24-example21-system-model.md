# Example 21 executed system model

## Scope

This document describes the implementation in
`examples/pp_prop/21-braincell-arc.py`,
`examples/pp_prop/arc_contracts.py`, and
`examples/pp_prop/example21_structural.py`. It uses only terms supported by
those modules.

BrainCell supplies the model-cell. BrainState supplies state and compiled
transforms. BrainTrace supplies the PP-Prop compiler and learning path. A
neuron is one model-cell in the recurrent layer. A connection is one sparse
input or recurrent relation. The prediction is the decoded integer grid. The
output-shape is its selected height and width.

## Input and output

An ARC task has demonstration input and output grids and a query input grid.
The loader returns integer grids. The encoder places the demonstrations and
query input in a fixed 705-event sequence. It places the held-out query target
in the loss and scoring path, not in the model event.

The output shape has one height and one width. Height and width each have 30
classes. Each output cell has ten color classes. The direct decoder selects
the largest logit in each required class group and returns a `uint8` integer
grid. A prediction is exact only when its shape and every cell equal the
target.

## Model-cell and layer

`BrainCellArcModel` contains one recurrent layer of 2,048
`braincell.SingleCompartment` model-cells. Each model-cell has one compartment
and sodium, potassium, and leak channels. The model uses a fixed `0.1 ms`
interval by default.

The layer has two sparse connection sets. An input connection maps one of 441
event values to a neuron. A recurrent connection maps a source neuron to a
target neuron. CSR rows identify sources and CSR columns identify targets for
recurrent connections. The layer has 14,112 input connections and 16,384
recurrent connections. The direct readout is a parameter matrix with shape
`(neurons, 360)` and a bias with shape `(360,)`; readout values are not
biological connections.

The default model has no Dale-type assignment. A Dale-type code of `1` means
excitatory, `-1` means inhibitory, and `0` means untyped in the checkpoint
contract. A typed source uses its sign for every outgoing recurrent
connection. The default model has no chemical synaptic mechanism.

## Event execution

For an advancing event, the model calculates sparse input and recurrent drives,
converts them to bounded current density, and updates all model-cells. The
recurrent drive reads the previous spike state. The model then stores the new
voltage and spike state.

For a false advance value, `brainstate.transform.cond` returns a zero sentinel
and leaves the biological state unchanged. Repeated event execution uses
`brainstate.transform.for_loop` and an outer `brainstate.transform.jit`. The
implementation does not use a repeated bare Python loop for model execution.

## Learning state

The PP-Prop compiler discovers input and recurrent parameter relations to
hidden state. It uses the `single-step` VJP method and trace decay `0.95`.
Episode reset clears model-cell and eligibility state. It retains parameters
and optimizer state. The trainer applies one clipped grouped update per
training episode. Validation uses a forward-only path that checks that
parameters, biological state, and eligibility state do not change.

## Structural checkpoint and plot

A format-1 checkpoint stores arrays for neuron identifiers, Dale types,
task-owner codes, sparse CSR topology, readout parameters, optimizer moments,
and step counts. It does not store runtime state, predictions, targets, or
losses.

The explicit `plot` command loads this checkpoint and renders a two-dimensional
Matplotlib image. It places neurons at deterministic circular positions. It
draws every recurrent connection and shows two node-group panels: Dale type
and task owner. The title reports retained neuron, input-connection, and
recurrent-connection counts. The plot command does not run in ordinary
baseline, parent, Dale, pruning, or addition commands.

## Implementation boundary

The repository contains compatibility and focused mechanism tests for this
model path. It does not, by itself, establish a real-data strict task result.
It also does not establish that approximate PP-Prop is equal to BPTT. Those
claims require separate executed evidence.
