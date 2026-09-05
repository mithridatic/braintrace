# Partial PV transfer isolation

The existing MaxCVLen BrainCell run fails the onset limit against both
NEURON integration modes over 270–329.5 ms. Each comparison has eight
complete events. Peak and width gates pass in both comparisons.

| Comparison | Maximum onset difference | Result |
| --- | --- | --- |
| NEURON fixed step versus half step | 0.0221691 ms | All event gates pass |
| BrainCell MaxCVLen versus NEURON fixed step | 1.2277994 ms | Onset fails |
| BrainCell MaxCVLen versus NEURON CVode | 1.2236709 ms | Onset fails |

Switching only the NEURON integration mode is insufficient to repair this
comparison. The matched-mesh BrainCell responses and their half-step check
are not available in this audit. No complete factorial decision follows.
This reference uses nseg x9 in all regions; do not combine these errors with
the axon2187-reference audit as if they used the same reference mesh.

Residual failure would not prove a gate-update cause. Equal branch counts
and nominal steps do not establish equal cable coefficients, compartment
placement, initialization or input sampling. A causal claim needs a split
that isolates the proposed difference.

[Each event and input hashes](h01-pv-transfer-isolation-partial-audit.json).
