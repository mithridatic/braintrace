# Spike errors persist under spatial refinement

Both meshes produce five complete positive-peak spikes. Increasing the
compartment count from 1239 to 3717 passes the specified event limits.

| Measurement | Largest absolute change | Limit |
| --- | --- | --- |
| Onset | 0.0270524 ms | 0.1 ms |
| Peak voltage | 0.0112833 mV | 0.1 mV |
| Time above -20 mV | 0.0000805674 ms | 0.01 ms |

The separate phase errors relative to the human recording retain their
signs in all five events. The largest phase change is 0.000370953 ms.
These phase observations are descriptive for spatial refinement; the
additional prospective phase gate was specified for tolerance refinement.

Section counts triple, parent connections agree, and physical section
values pass the established rtol=atol=1e-10 geometry check. All other
model settings agree. Both recordings match indexed raw samples and
have zero input plateau error, excluding 1e-7 ms around transitions.
Both traces are finite and end at 2100 ms.

The [full result](h01-l2-reversal-active-spatial-result.json) retains
every event, phase, difference, audit result, and artifact hash. The
large human timing and waveform errors remain. This is a spatial
numerical result at one input, not physiological validation.
