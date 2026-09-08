# Human layer-2 spiking reference

Use Allen specimen 541563728. Select the lowest available long-square
command at least 40 pA above the reported 200 pA rheobase. Break ties
by sweep number. Selection uses stimulus metadata before voltage inspection.
This rule selects sweep 50 at 250 pA. Retain sweep 43 as subthreshold data.

Use the verified pipeline-1.0 SI convention. Retain reported voltage and
the derived voltage with a -14 mV junction correction. Retain command,
bias, and their sum. Do not adjust these values to improve a model fit.

Before comparison, require finite samples, matching current and voltage
clocks, a contiguous 1000 ms positive pulse, and at least 500 ms of
zero-command baseline immediately before the pulse. Record baseline
noise and the difference between the first and last 100 ms baseline
means. Screen for baseline SD below 0.5 mV, absolute drift below 1 mV,
and absolute bias below 100 pA. These are a partial quality screen;
they do not replace all source acquisition quality checks.

Record every complete corrected -20 mV excursion during the pulse,
its interpolated crossings, sampled peak, and duration above threshold.
Retain individual event times and intervals. No smoothing, peak alignment,
or biological acceptance limits are introduced by this extraction.
If no complete excursion occurs, retain that result and reject this
sweep as a spiking reference. Do not choose another sweep by model fit.

Boundary cases: missing metadata, mismatched clocks, nonfinite samples,
split pulses, unstable baseline, no spikes, and incomplete excursions.
Use existing event extraction. This work reads a source recording; it
does not change model code or establish physiological model validation.
