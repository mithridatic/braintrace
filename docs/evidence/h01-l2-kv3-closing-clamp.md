# Selective Kv3 closing passes the gate-law check

All 36 declared cases pass: six fixed voltages, factors one and 0.5,
and opening, closing, and equilibrium initial states. The maximum gate
error against the analytic exponential solution is 8.01471e-13, below
the 1e-8 limit. Voltage remains exactly fixed in every case.

The half factor changes closing only. Opening and equilibrium retain
the source law. The simulation uses zero channel conductance, fixed-step
cnexp with dt 0.00005 ms, and one compiled advance to 2 ms. This validates
the implemented gate intervention; it does not validate a human channel.

All eleven input mechanism hashes match the checked activation build.
Only Kv3_1.mod changes in the new source directory. All compiled-run
source hashes match the preparation manifest, and the saved library hash
matches the built library. Source tests pass: three tests, 100% statement
coverage of the twelve-statement preparation helper. That coverage does
not apply to the NEURON clamp driver.

The [full gate result](h01-l2-kv3-closing-clamp.json) retains each direct
error and all hashes; the NPZ retains every gate and voltage trajectory.
Driver integration, default whole-cell equivalence, and the physiological
recovery prediction remain open. Useful remaining edge checks include
invalid driver factors and changed or missing factor metadata in numerical
comparisons.
