# Faster calcium removal in the fixed reversal candidate

The numerically checked candidate has four intervals longer than the
human values. Test a decrease of somatic calcium-removal time from
source factor 1.5 to 1.25, or 741.029329 to 617.524441 ms. This is an
inferred diagnostic setting, not a measured human parameter.

At fixed calcium above its resting value, a shorter removal time makes
the removal term larger. In the coupled cell this can change SK current,
calcium reversal, and voltage-dependent currents. Do not interpret the
full response as an isolated SK effect. Earlier removal tests used a
different operating condition and do not establish this result.

Control: h01-l2-sweep50-ih75-epasminus4. Keep distributed Ih factor 75,
passive reversal shift -4 mV, source sodium and Ih kinetics, all densities,
recorded bias, initial state, temperature, geometry, mesh 9, CVode 1e-10,
full sweep-50 command history, current recording, and endpoint 2100 ms.
Change only the somatic decay_CaDynamics value through the existing helper.

Before the run, require all these conditions:

- Retain exactly five complete positive-peak events in the main pulse.
- Each of intervals 2, 3, and 4 has smaller absolute human error than
  its control counterpart.
- The first interval's absolute human error does not increase.

Missing or extra events reject this combined prediction. Bad input,
unintended setup changes, nonfinite traces, incomplete boundary events,
or an invalid recording make the test invalid. Keep these outcomes distinct.

Retain every event, interval, rise-to-peak time, peak-to-fall time,
interspike voltage minimum, and human residual. Report first peak and
phase changes separately. Passing interval conditions does not override
waveform errors or establish a full cell fit. Preserve all traces even
on rejection. The reserved sweep remains excluded.

Verify the applied genome differs in exactly one intended row. Verify
raw-to-final mapping, source waveform hash, recorded current, and endpoint.
Candidate-specific numerical checks and a subthreshold check are still
required if this setting is kept for further physiological validation.
