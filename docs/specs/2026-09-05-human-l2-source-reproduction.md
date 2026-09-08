# Layer-2 source-model reproduction

Reproduce Allen model 626170538 using its retrieved fit and mechanisms.
Use its specimen morphology, with the two-section axon replacement from
the inspected perisomatic helper. Do not transfer parameters to H01 yet.

Use sweep 50 command samples with no added bias for the source baseline.
Record this choice. A later command-plus-bias run is a separate intervention.
Use the source initial voltage, reversal potentials, and temperature.
Compare raw model voltage to junction-corrected human voltage. Do not
apply the junction correction twice. Save the applied current as a trace.

The source helper selects the least common multiple of the 50 kHz input
and 40 kHz target, which is 200 kHz (dt 0.005 ms). Use fixed-step NEURON
at that step for the first reproduction. Advance once with continuerun.
Static section setup may use Python loops; time stepping must not.

Check finite arrays, increasing time, trace endpoint, input identity,
loaded mechanism parameters, region properties, and soma attachment.
Record every complete -20 mV excursion during the pulse. Report direct
residuals against sweep 50 without fitting. A source reproduction is not
a claim of physiological validation or spatial convergence.

Then halve the time step with all other settings unchanged. The declared
time check requires the same complete-event count and peak-sign class,
each event onset difference <=0.1 ms, each sampled peak difference
<=0.1 mV, and each duration-above-threshold difference <=0.01 ms.
Missing events make the comparison invalid; differing counts reject it.
These numerical limits do not define human physiological acceptance.

Boundary cases include missing region names, wrong source hashes,
nonfinite command values, nonuniform input times, invalid time steps,
incomplete events, and absent spikes. No new physiological fit is allowed
in this reproduction. Preserve source mismatch and numerical failure.

If the first halving fails, halve again with the same limits and compare
successive runs. Keep the full source waveform and all fitted parameters
fixed. Do not shorten the pre-pulse state history to make a run faster.

The repeatable audit must reject changed model metadata, nonfinite or
nonmonotonic traces, insufficient pulse coverage, and a pulse boundary
already above the event threshold as invalid comparisons. No complete
event in either trace is invalid; different nonzero counts reject the
time gate. Preserve every paired residual and all unmatched events.
Test these cases with direct synthetic traces before using the audit.
