# Leak candidate fails the combined active target

The frozen leak-factor-1.5 candidate produces two complete positive-peak
events at the bias-included 250 pA command. The human trace has five.
The necessary active-response prediction is rejected. First onset is
129.122409 ms late; the second is 473.266232 ms late. The model's first
interval is 377.995562 ms, compared with 33.851739 ms in the human trace.

The [full audit](h01-l2-leak-active-result.json) retains both model
events, all human events, matched waveform errors, and the three missing
human events. Physical parameters agree with the subthreshold candidate.
Recorded input, raw mapping, and endpoint checks pass. No parameter was
fitted in this active-response check, and reserved sweep 53 was not used.

The candidate's earlier subthreshold amplitude improvement is insufficient
for the combined goal. Do not promote its global leak scaling as a model
repair. The comparison is against the direct human response; it is not
an isolated attribution of spike loss to leak using a control with a
different bias. A later candidate must satisfy subthreshold dynamics and
active behavior together. Candidate numerical qualification is incomplete,
so the recorded errors remain observations at the tested settings.
