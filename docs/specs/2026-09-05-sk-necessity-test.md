# SK necessity for the sodium timing effect

## Claim

The change in later spike timing after faster sodium inactivation requires
current through SK channels. This is a necessity claim in the reference model.
It is not a claim about human channel expression.

## Split

Compare original and faster sodium inactivation with SK present, then with
SK conductance set to zero in soma and axon. Preserve calcium dynamics and
all other channels. Keep recovery at its original speed.
Use 0.19 nA, zero bias, -80 mV initial voltage, mesh factor 9, CVode atol 1e-10.
Observe all channel currents and each complete voltage event.

## Decisions defined before simulation

Verify SK current is zero in both removal cases. A failed intervention check
invalidates the test. Check source-control identity through saved conditions.
Use the second spike's rising -20 mV crossing as a named later event.
If both removal cases retain at least two complete spikes and their second
crossing times differ by more than 1 ms, reject SK necessity at this condition.
This is a diagnostic separation margin, not a biological acceptance tolerance.
Retain the first and all later event times as well; do not infer equality from
equal counts. Inspect the first interspike interval separately from absolute time.
If either case loses the second spike, or the difference is below the margin,
the necessity claim remains unresolved. An absent effect may be masked by
the operating-point change caused by removing SK.

If the difference survives, investigate other potassium currents and sodium
state before treating SK density as a complete correction for the firing error.
The test does not estimate the fraction of the effect mediated by SK.
Numerical robustness is required before promoting any fitted parameters.
