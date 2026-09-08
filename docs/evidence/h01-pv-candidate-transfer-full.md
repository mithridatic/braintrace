# Full inhibitory model transfer

The full transfer fails. From 270 to 1270 ms, the pinned NEURON candidate
has 40 complete events. BrainCell has 42. The human recording has 43;
that count is a separate physiological target.

BrainCell uses the frozen candidate, a 0.27 nA pulse, a 0.000625 ms time
step, and a maximum compartment length of 2.5 um. No parameter was fitted
in this comparison.

The rising-crossing error first exceeds 0.1 ms at event 7. It is
-0.214731 ms there and -1.694908 ms at event 8. The first six crossings
meet that limit. All ordinal paired peak errors are below 0.1 mV and
duration errors are below 0.01 ms, but unequal event counts fail the
complete comparison. Later ordinal pairs do not establish correspondence.

The early-window pass does not establish full-response transfer. Similar
spike peaks can occur with different intervals and a different event count.
This result does not identify whether time stepping, spatial discretization,
or another implementation difference causes the accumulated timing error.

The [audit](h01-pv-candidate-transfer-full-audit.json) retains each event,
the unchanged limits, and hashes of both traces and metadata files.

Next, halve only the BrainCell time step through 330 ms. Compare events
from 270 to 330 ms with the same reference and the completed baseline.
This window includes the first failed crossing and event 8. Keep all
physiological parameters and the spatial mesh fixed. This is a numerical
diagnosis, not a new fitting campaign or a full-response pass.
