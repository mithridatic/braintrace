# Example 21 execution map

## Purpose

This map describes the implemented data path. It is not an ARC performance
claim.

```text
raw ARC task
  -> Boolean event stream and advance mask
  -> sparse input and recurrent drive
  -> Hodgkin-Huxley cell state and emitted spikes
  -> 360 voltage-readout logits
  -> shape and row decoder
  -> integer ARC grid
```

## Boundaries

The event encoder uses demonstration inputs and outputs and one query input.
It does not encode the query target. The loss and strict scorer use targets
outside this inference event stream.

Input and recurrent weights are PP-Prop temporal parameters. The readout uses
direct gradients. A false advance returns a zero output and preserves the
model state in the tested driver.

The repository has no completed raw-ARC run for this model. The map therefore
does not identify a cause of ARC success or failure. See the causal
explanation and evidence notes for the tested behavior and its limits.
