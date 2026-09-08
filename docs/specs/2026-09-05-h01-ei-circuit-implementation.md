# Two-cell H01 circuit implementation

Implement the already authorized small E/I circuit with the frozen candidate
profiles. Functional diagnostic runs are permitted before physiological
qualification, but must report that qualification remains open.

Use two distinct H01 source cells and preserve both source tags and region-map
assumptions. Require pyramidal E and interneuron I source tags. Use Network
populations of size one, soma-local ExpSyn contacts, and a single declared
output CV per cell. Weights are positive conductances. E reversal is 0 mV;
I reversal is -80 mV. These synaptic values and all edges are illustrative,
not measured H01 connectivity. Record weights, reversals, decay times, delays,
output sites, and local synaptic conductance and voltage traces.

Support disconnected, E-only, I-only, and reciprocal E/I controls. Retain the
same cell dynamics and placed synapses in all controls; remove only the chosen
projections. Tests must cover exact contact identities, polarity, positive
weights, invalid inputs, single-site events, compiled runtime construction,
and isolated delivery with direct conductance/voltage observations. Missing
presynaptic firing cannot count as successful inhibition. Do not claim final
human validity or complete circuit qualification from construction tests.
