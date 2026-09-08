# Local charge balance

The probe is the middle segment of soma[0], with mesh factor 9.
Its area is 154.666011525 um2. Do not use whole-soma area for this balance.
The unchanged model runs at 0.19 nA with zero bias and CVode atol 1e-10.
The additional probes leave both time and voltage arrays unchanged.

Axial inflow is calculated independently for eight neighbours:
two adjacent soma segments and six attached dendrites.
Each current is `(V_neighbour - V_probe) / resistance` in nA.
Resistance comes from NEURON ri in megohms.
The topology is restricted to this root soma midpoint with forward child sections.
The calculation is not a general arbitrary-tree axial-current implementation.

Ion, leak, Ih, and capacitive currents are recorded as densities.
Multiply mA/cm2 by local area in um2 and by 0.01 to obtain nA.
The balance is applied inflow plus axial inflow minus ionic outflow minus
capacitive current. Initialization and 0.001 ms around stimulus jumps are excluded.
The maximum residual is 4.44e-15 nA, below the specified 1e-5 nA check.
If axial current is omitted, the maximum residual becomes 1.18808 nA.
That omission is an accounting check, not a physical cable-removal experiment.

At the first sampled peak, applied inflow is 0.19 nA and net ionic inflow
is 0.748235 nA. Net axial outflow is 0.939076 nA.
Capacitive current is -0.000841 nA; the sampled peak is not an exact derivative root.
Adjacent soma segments supply current while the attached dendrites draw it away.
The [JSON audit](h01-pv-charge-balance-audit.json) retains each path separately.

An additional check integrates capacitive current over the actual samples
near the first spike. Its charge agrees with C times the voltage change within
5.35e-7 pC. This checks the relation to voltage without differentiating noisy samples.
It is one integral check, not proof of a complete continuous trajectory error bound.

The [NEURON geometry reference](https://www.neuron.yale.edu/neuron/static/new_doc/modelspec/programmatic/topology/geometry.html)
defines ri and segment area. The
[mechanism reference](https://neuron.yale.edu/neuron/static/new_doc/modelspec/programmatic/mechanisms/mech.html)
defines the recorded capacitive current.
Run `python -m docs.evidence.h01_pv_charge_balance_audit` to repeat analysis.
The NPZ residual file retains all current terms on their original clock.

The result closes one model compartment's charge balance. It does not validate
human physiology, energy balance, or the proposed explanation of later spikes.
