# Example 21 causal evidence

## Question

What behavior of the supported BrainCell Example 21 path is established by
the repository?

Two explanations are possible for an absent ARC result: the model has not run
on the required data, or a completed run exists but is not recorded. The
workspace evidence supports the first explanation for the local Gate 4 check.

## Dependency path and checks

`arc_contracts.py` loads and encodes raw ARC data. `21-braincell-arc.py`
defines the sparse BrainCell model, PP-Prop compiler path, sequence drivers,
and episode trainer. The sibling test module exercises these functions.

The focused tests check the high-information boundaries first:

| Boundary | Observation | Inference |
|---|---|---|
| Event encoding | An inference event stream has shape `(705, 441)` and omits the query target. | Input encoding does not directly expose the held-out answer. |
| Model stepping | False events preserve voltage, gates, and previous spikes; padding preserves the advancing result. | The tested driver respects the advance mask. |
| Learning connection | The compiler reports two temporal relations; direct readout fixtures have finite nonzero gradients. | The tested parameter paths reach their declared local objectives. |
| Real-data execution | `2026-08-24-example21-gate4-proof.md` records no local ARC files and no real-data result. | No local BrainCell ARC performance claim is supported. |

## Remaining uncertainty

The checks do not measure training progress, generalization, exact ARC tasks,
or an intervention on a trained checkpoint. They also do not compare PP-Prop
with BPTT. Those questions require a reproducible run with the declared ARC
data, checkpoint, predictions, and strict scorer output.

The former v48 MiniLSTM result files are evidence for that retired path only.
They must not be used to explain the BrainCell baseline.
