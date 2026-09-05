# Axonal calcium removal candidate

The candidate increases axonal removal time from 370.641079 to 1000 ms.
Closure factor stays 0.18, recovery factor 1, and SK density stays unchanged.
Both calibration runs completed. The observed 0.19 nA control reproduces
the previous voltage and time arrays exactly after adding axonal probes.

At 0.19 nA, the first rising crossing changes by 0.00000903 ms.
That is much smaller than the 2.702903 ms change from doubled axonal SK.
Do not interpret this tiny difference as a resolved physiological onset effect.
The predicted separation from the density intervention is supported here.

The last complete interval changes from 43.403855 to 76.671443 ms at 0.19 nA.
At 0.27 nA it changes from 12.635506 to 20.429850 ms.
These compare each trace's last interval, not the same event index.
The JSON retains all events and their individual intervals.
Counts remain too high: 24 versus 12 human events at 0.19 nA, and 70 versus
43 at 0.27 nA. The low-current first onset remains about 12.2 ms late.
First-spike shape is preserved, but no complete model is validated.

The axonal calcium pool is near its resting value at stimulus onset.
Its removal term is proportional to the excess above rest, so changing removal
has little initial effect. The audit retains axonal calcium, SK gate, voltage,
and current at every somatic rising crossing. These are different clocks in
different runs; they are not a matched-voltage clamp or mediation proof.
Calcium also affects its Nernst reversal potential. Do not attribute the
whole response change solely to SK without a further split.

Axonal SK current matches density times gate times driving voltage within
1e-10 mA/cm2 throughout each recorded trace. Four evidence tests pass.
They verify reporting and event datums, not physiological acceptance.

The [JSON audit](h01-pv-calcium-candidate-audit.json) records checks and events.
Run `python -m docs.evidence.h01_pv_calcium_candidate_audit` to repeat it.
The reserved 0.23 nA trace remains unused.
