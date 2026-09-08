# Active-response check of the leak candidate

Freeze the leak-factor-1.5 candidate used at sweep 43. Apply calibration
sweep 50, including its recorded bias. Keep calcium factor 1.5, source
sodium and Ih, mesh 9, CVode tolerance 1e-10, initial state, temperature,
complete command history, and endpoint 2100 ms. No parameter fitting in
this check. Reserved sweep 53 remains uninspected.

The necessary active-response prediction is five complete positive-peak
events during the main pulse, as in the direct human trace. If this fails,
reject the candidate for the combined subthreshold and active target.
If it passes, retain every event and every matched onset, peak, duration,
and interval error; the count alone cannot qualify the candidate.

Verify setup identity with the subthreshold candidate except for the
selected waveform. Check recorded current, raw mapping, and endpoint.
This is an across-input candidate check, not an isolated causal comparison
against a model run with a different bias. Preserve missing events and
the full voltage trace, even when no spikes occur.
