# H01 I local charge balance

Use the unchanged candidate and original 1 nA pulse. Locate the electrical
CV used by the soma voltage probe and record its source point, area and total
capacitance. Verify whether that point is the CV midpoint. Do not combine
currents sampled at another location with that CV voltage.

Record the soma CV voltage and each neighbor voltage in one compiled loop.
Calculate each inward axial current from the reduced axial coupling matrix
and the neighbor-minus-local voltage. Do not estimate axial current from a
charge-balance residual. Record total membrane current and applied clamp
current separately at the same CV. Channel current is their difference.

Compare the complete voltage trace with the existing baseline. Calculate
capacitive current from central voltage differences, then compare it with
membrane plus axial current. Exclude initialization and samples within two
time steps of pulse jumps. Use 1e-5 nA as an initial diagnostic residual limit;
report failures rather than call this biological validation. End-step gate
sampling and the staggered solver can require smaller dt for this comparison.
Only after this boundary is qualified compare the regional interventions.

## Discrete-step timing check

The inspected solver uses a linearized implicit Euler voltage step, followed
by channel updates. Record the pre-update membrane linear and constant terms,
then evaluate that law at the resulting voltage. Compare total capacitance
times the step voltage increment divided by dt with that membrane current
and new-voltage axial currents. Keep the earlier central-difference residual
as a separate observation. This checks discrete bookkeeping, not independent
physical accuracy. Require the same 1e-5 nA residual limit and exact baseline
voltage. Do not change the solver or claim that this proves human validity.
