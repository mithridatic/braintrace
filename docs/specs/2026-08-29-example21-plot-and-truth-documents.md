# Example 21 plot and implementation-truth documents

## Scope

This specification covers OpenSpec tasks 9.1 through 9.3 for the BrainCell
ARC replacement. It defines the explicit topology plot and the two documents
that describe the executed code.

## Plot contract

The structural module SHALL expose an explicit `plot` command. The command
SHALL load one format-1 checkpoint and write one two-dimensional Matplotlib
image. It SHALL not run during `baseline`, `parent`, `dale`, pruning, or
addition commands.

The plot topology object SHALL use only checkpoint topology and label arrays.
It SHALL not use readout weights, bias values, targets, or prediction inputs.
It SHALL show all recurrent edges and all neurons. It SHALL group neurons by
Dale code and task-owner code. It SHALL show retained neuron,
input-connection, and recurrent-connection counts. The plot operation SHALL
not mutate the checkpoint, model parameters, or prediction inputs.

The plot SHALL use deterministic node positions from neuron identifiers. It
SHALL support headless execution and SHALL close its Matplotlib figure after
the image is written.

## Truth-document contract

`docs/specs/2026-08-24-example21-causal-explanation.md` SHALL state the strict
acceptance rule, direct observations from executed code, and inferences that
are limited to those observations. It SHALL not report planned behavior as
executed behavior. It SHALL state when real ARC prediction evidence is not
available.

`docs/specs/2026-08-24-example21-system-model.md` SHALL describe only the
executed BrainCell, BrainState, BrainTrace, sparse topology, episode reset,
event driver, readout, and strict decoding paths. It SHALL use neuron,
connection, layer, Dale-type, model-cell, prediction, and output-shape terms
consistently. It SHALL separate implementation facts from unsupported claims.

Both documents SHALL use short ASD-STE100 sentences. They SHALL identify
direct evidence by repository-relative file or test path when a measured claim
is made.

## Focused verification

The co-located structural test SHALL verify:

1. plot node and edge counts match the checkpoint topology;
2. owner and Dale groups are represented in the plot;
3. plotting does not change bytes decoded by the real 31-by-360 prediction
   decoder;
4. ordinary structural commands do not create a plot; and
5. both truth documents contain the required observation and implementation
   boundaries.
