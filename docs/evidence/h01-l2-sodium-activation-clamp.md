# Selective sodium opening passes the gate-law check

All 36 fixed-voltage cases pass the 1e-8 gate-error limit. Cases cover
six voltages, factors one and two, and opening, closing, and equilibrium
initial states. The set includes the -40 mV rate limit.

Maximum activation error is 1.196820e-13. Maximum inactivation error is
2.871592e-13. Voltage stays exactly fixed in every case. The equations
match an opening time constant doubled only for factor-two opening
cases. Closing and equilibrium behavior remain unchanged. Inactivation
matches the original analytic law with recovery factor one.

The test advances all independent zero-conductance sections in one
NEURON continuerun call. Temperature is 34 degrees C, step is 0.00005 ms,
and duration is 2 ms. This isolates gate dynamics from membrane feedback.
The [result](h01-l2-sodium-activation-clamp.json) records case values and
all mechanism and compiled-library hashes. Complete gate and voltage
trajectories are retained in the matching NPZ file.

Source preparation tests also pass: three tests with 100 percent helper
statement coverage. This does not measure clamp-driver coverage. The
gate experiment is direct numerical verification of the intended law.
Driver integration, invalid-factor handling, default whole-cell
equivalence, and physiological intervention tests remain open.
