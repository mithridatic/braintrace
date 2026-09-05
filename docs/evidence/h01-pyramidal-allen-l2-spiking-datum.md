# Layer-2 human spiking datum

Allen specimen 541563728, sweep 50, contains five complete excursions
above corrected -20 mV during the 1000 ms pulse. Selection used the
[declared stimulus rule](../specs/2026-09-05-human-l2-spiking-datum.md)
before inspection of the voltage response. This is a different donor
from H01. Spiny morphology is an annotation, not a transmitter assay.

The command is 250 pA. The separate bias is -3.711859 pA. The interpreted
total during the pulse is 246.288134 pA. Pulse onset is 1020 ms.
Reported voltage and corrected voltage are retained separately. The
pipeline-1.0 unit handling and -14 mV correction follow the
[source audit](h01-pyramidal-allen-l2-datum.md).

The first upward crossing occurs at 1077.811556 ms. The first sampled
peak is 36.031252 mV at 1078.100000 ms. The first duration above -20 mV
is 0.935864 ms; this is not half-width. The first three intervals are
33.851739, 220.403496, and 307.406282 ms. A mean rate would hide these
individual timing differences.

The 500 ms pre-pulse baseline has mean -84.232297 mV and SD 0.212499 mV.
The last 100 ms mean minus the first 100 ms mean is 0.044968 mV.
The declared baseline, bias, finite-data, clock, and pulse checks pass.
Both pulse boundaries are below the event threshold. This is a partial
quality screen. It does not certify every acquisition quality condition.

The [complete datum record](h01-pyramidal-allen-l2-spiking-datum.json)
retains each event and interval. Arrays are cached in
`.cache/human-pyramidal-l2/sweep-50.npz`. Sweep 43 remains a subthreshold
reference. No model parameters were fitted in this extraction, and no
physiological model gate has passed as a result of this work.

Remaining checks include full source sweep quality and comparison under
matched command, bias, baseline, temperature, and voltage conventions.
Reader regression cases should cover obsolete unit attributes, missing
bias, mismatched clocks, split pulses, no spikes, and incomplete events.
