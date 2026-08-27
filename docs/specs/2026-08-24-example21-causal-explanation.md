# Example 21 causal explanation

## Observed path

Example 21 encodes an ARC task as a fixed event sequence. The sparse input
relation maps each event to cell drive. The sparse recurrent relation maps
the previous emitted spikes to additional drive. A Hodgkin-Huxley population
updates its state. A dense voltage readout produces shape and color logits.
The decoder converts the logits into one integer grid.

The event encoder does not include a query target. The target is used only by
loss and exact scoring. The result is strict only when every prediction for a
task has the target shape and all target cell values.

## What the tests establish

The focused tests establish that the declared sparse relation sizes are used,
false events do not change biological state, and interspersed padding leaves
an advancing sequence unchanged. They also establish that input and recurrent
weights have compiled PP-Prop relations, direct readout gradients are finite
and nonzero in the fixture, and an episode update can change the declared
parameter groups.

These are component and integration checks. They do not establish that the
model learns ARC transformations.

## Current limit

No completed raw-ARC training or evaluation result for this BrainCell path is
stored in the repository. The available real-data Gate 4 run was not executed
because its ARC files were unavailable. There is therefore no observed model
failure location within ARC task solving. Any statement that the encoder,
recurrent state, or decoder is the main cause of ARC failure would be a
hypothesis, not a finding.

The earlier MiniLSTM v48 measurements describe a retired path. They are not
evidence about this BrainCell baseline.
