# Test calcium removal after the smaller Kv3 recovery improvement

The Kv3 closing-factor-0.9 candidate has five spikes and improved recovery
minima, but interval errors are -8.311265, +58.870067, +8.799043, and
+29.184656 ms. Earlier calcium-removal changes shortened later intervals
without materially correcting minimum voltages. Those results came from
a different channel setting and do not establish the present response.

Change only calcium removal factor from 1.375 to 1.33. This small inferred
step tests whether later timing can improve without losing the recovery
benefit. The third interval is only 8.799043 ms too long, so a large change
can overshoot it. No linear response or measured human decay is assumed.

Freeze all other settings: Kv3 closing factor 0.9, sodium opening 2,
sodium density 1.3, sodium recovery 1, distributed Ih 75, passive reversal
shift -4 mV, original leak, temperature, geometry, initial voltage, sweep
50 with recorded bias, mesh 9, CVode 1e-10, endpoint 2100 ms. Use the
checked kv3-closing-source build.

Require five complete positive-peak spikes. Require every later interval
(2, 3, 4) to have smaller absolute human error than the frozen Kv3-0.9
control. Require the first interval to change by no more than 0.1 ms;
this preserves its current behavior for a separate diagnosis, not a
physiological acceptance. Require each minimum's absolute human error
to remain smaller than in the Kv3-factor-one density-1.3 reference.
All conditions must pass for the joint prediction to be supported.

Keep the minimum definition from the Kv3 split: earliest sample strictly
between prior fall and next rise through -20 mV. Retain all events,
phases, intervals, minimum voltages and delays, human residuals, and
unmatched events. Passing this prediction cannot establish a complete
fit while first-interval and waveform errors remain.

In parallel, repeat the unchanged Kv3-0.9 control at CVode 1e-11. Require
equal event counts and positive peak signs, onset changes <=0.1 ms,
peak changes <=0.1 mV, duration changes <=0.01 ms, and each separate
phase change <=0.01 ms. Record all four minimum changes descriptively;
do not create an extrema-specific acceptance threshold from the output.

For both runs, require finite data, exact raw mapping, correct input
plateaus within 1e-12 nA outside 1e-7 ms of command edges, and exact
2100 ms endpoint. The calcium split must change exactly one somatic
decay_CaDynamics genome row and its intervention record; other physical
metadata must match. The tolerance run may change tolerance only.
Allow sample counts and integration time to differ. Invalid setup or
malformed data invalidates the comparison. Valid response failures reject
it. Reserved sweep-53 response must remain unopened.
