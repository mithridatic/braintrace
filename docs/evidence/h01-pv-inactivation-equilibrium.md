# Fixed-voltage inactivation-slope check

The initialized source gates agree with the independent logistic equation
at all six specified voltages for slopes 6 and 5 mV. Below the -56 mV
midpoint, slope 5 gives more available sodium channels. Above it, slope 5
gives fewer. The activation gate remains unchanged.

At -70 mV, h rises from 0.911600 to 0.942676. At -30 mV, h falls from
0.012954 to 0.005486. With fixed conductance and driving voltage, calculated
stationary sodium current at -30 mV changes from -0.267181 to -0.113159
mA/cm2, with outward positive. At -70 mV it changes from -0.003855 to
-0.003986 mA/cm2. These are model quantities, not measured human currents.

At the midpoint, the source's 0.0001 mV perturbation produces errors of
4.17e-6 and 5.00e-6 in h against one half. They pass the predeclared 1e-5
limit. Away from that pole, errors are below 1e-10.

The raw slope parameter also changes the source rate sum. Its calculated
closing time constant at -70 mV increases from 0.199128 to 0.214162 ms.
Thus, this is not a selective equilibrium-only intervention. No whole-cell
rescue follows from this fixed-voltage result.

The first diagnostic attempted to read hidden ASSIGNED variables from the
compiled mechanism and failed. A regression reproduced the interface error.
The corrected diagnostic reads exposed gate states after static initialization.
Time constants and stationary currents are explicitly calculated from the
source equations, not claimed as directly exposed mechanism outputs.
Inspect variable exposure before using compiled intermediates in future probes.

The container regression now passes all analytic and directional assertions.
It does not advance repeated simulations in a Python loop. The
[JSON](h01-pv-inactivation-equilibrium.json) retains all voltages, gates,
currents, and calculated time constants. The source mechanisms are unchanged.
