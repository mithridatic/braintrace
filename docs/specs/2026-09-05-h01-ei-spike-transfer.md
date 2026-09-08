# Frozen E/I spike transfer

Continue the approved source/candidate transfer checks. Do not fit parameters.
First run the I candidate on donor geometry at 0.27 nA through 300 ms.
Compare every event in the complete shared window with the pinned NEURON
h01-pv-regional-mesh-axon2187 response. Keep the original 270 ms pulse onset.

Use direct -20 mV rising crossings, peak voltage, and duration above -20 mV.
Require identical event count, onset error <=0.1 ms, peak error <=0.1 mV,
and duration error <=0.01 ms. Numerical failures do not justify relaxing human
targets. Retain all voltage samples. Record candidate identity, mode, source
metadata hash, dt, mesh, stimulus, and window. Refine numerical settings only
if a direct transfer error requires it. No reserved human response is opened.

The initial window is a first-burst diagnostic, not full-sweep qualification.
Extend to the full input only after resolving a failed first-burst transfer.

## Circuit emission boundary

The installed Network reduces the cell's compartment spike array with logical
any. A propagating action potential can cross different compartment thresholds
on different steps. Before connecting H01 cells, restrict emitted events to one
explicit soma-region CV and record that location. Test that a threshold crossing
elsewhere does not emit a second event. Retain the soma-node voltage probe
separately from the chosen CV voltage. Do not call either location a measured
axon terminal. Use population size one for each distinct H01 morphology.

Circuit wiring must remain explicitly illustrative unless H01 partner mapping
is qualified. Required paired controls are disconnected, E-only, and E-plus-I,
with identical starting state, stimulus, delays, and unchanged physical profiles.
