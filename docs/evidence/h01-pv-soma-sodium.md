# Soma sodium intervention

The source model retains its original axonal channels in these checks.
Only somatic NaTg conductance changes. The current is 0.19 nA.
The initial voltage is -80 mV. The mesh factor is 9.
CVode uses an absolute tolerance of 1e-10.

| Soma conductance factor | Somatic spike count | First peak, mV |
| --- | ---: | ---: |
| 1.00, reference | 11 | 44.25 |
| 0.90 | 9 | 43.50 |
| 0.75 | 3 | 42.03 |
| 0.25 | 0 | No spike |
| 0.10 | 0 | No spike |
| 0.05 | 0 | No spike |

The recorded axon has the same upward crossing count at -20 mV.
For factors 0.25 and below, its maximum voltage is below -72 mV.
These cases show loss of spike generation in both recorded regions.
They do not support failed propagation to an otherwise silent soma.
Two voltage probes cannot establish the spatial site of initiation.

The human trace has 12 spikes. Its first peak is 19.5625 mV.
The smaller reductions lose spikes before they correct the high model peaks.
Do not promote any of these candidates. Stop this one-parameter reduction path.
Further calibration must consider the coupled inward and outward currents,
with the original conditioning and recording protocols checked first.

The JSON audit contains direct-trace extrema, event times, and response-change
RSS. Each RSS uses 200000 samples at 0.005 ms from 270 to 1270 ms.
For each region, interpolate the changed and reference traces onto this grid.
Subtract the reference, square each difference, sum, then take the square root.
The reference is `h01-pv-neuron-initial80-gates.npz`.
RSS is an intervention effect. It is not propagated biological uncertainty.
No voltage offset or spike alignment is used.

The driver is `h01_pv_neuron_reference.py`. Use the pinned NEURON container
and the existing source mounts. The common arguments are:

```text
--current-na .19 --cvode-atol 1e-10 --nseg-factor 9
--scale-conductance NaTg --conductance-region soma
--conductance-factor FACTOR --output /evidence/h01-pv-neuron-soma-na-SUFFIX
```

The five factors and suffixes are .05/005, .10/010, .25/025, .75/075, .90/090.
Each output retains both direct voltage traces and calcium state.
These are diagnostic simulations, not physiological validation.
