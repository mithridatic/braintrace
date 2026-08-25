# Example 21 executed system model

## Scope

This document describes the direct model represented by the executed v48
artifact. It does not describe the planned BrainCell replacement.

## Data path

The model reads an ARC task and its query. One memory updates during the
demonstration phase. A second memory updates during the query and decode
phases. The model joins the two memory states and their elementwise product.
The joined state has 384 values.

Twelve color experts and 16 learned routing programs read this state. Separate
heads produce height, width, and cell scores. The decoder selects one height,
one width, and one color for each output cell. The result is an integer grid.

## Executed model terms

- A MiniLSTM hidden unit is a mathematical state variable.
- A dense weight connects model values. The v48 artifact has 19 trainable
  leaves and 12,533,828 trainable numeric values.
- The artifact does not define a sparse neuron graph or physical connections.
- The artifact does not assign Dale types.
- The output grid is the model prediction. The expected ARC grid is the target.
- The output shape is selected from the height and width heads.

## Limits

The v48 evidence does not execute BrainCell, BrainState event stepping, or a
BrainTrace PP-Prop BrainCell checkpoint. It therefore does not support claims
about Hodgkin-Huxley cells, sparse connection counts, eligibility traces, or
Dale stages. Those claims require the replacement executable and its accepted
checkpoint.
