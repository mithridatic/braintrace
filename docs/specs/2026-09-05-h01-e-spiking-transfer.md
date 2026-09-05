# Frozen E spiking transfer

Extend the approved E donor transfer beyond its first 2 ms. Use the frozen
candidate and pinned donor geometry. Preserve measured command amplitudes,
the test pulse at 5 to 15.02 ms, the main pulse at 1020 to 2020 ms, and the
separate recorded bias from time zero. Obtain these values from the cached
sweep and pinned source metadata; do not round the main command to 0.25 nA.

Run to 1120 ms at dt 0.00125 ms and maximum CV length 2.5 um. Compare every
complete event from 1020 through 1120 ms with the pinned NEURON candidate.
Require matching event count, each rising-crossing error <=0.1 ms, peak error
<=0.1 mV, and duration-above--20 error <=0.01 ms. Keep all samples. This is a
first-two-event transfer check, not full-sweep or human waveform validation.
No physiological parameters or reserved responses are changed.
