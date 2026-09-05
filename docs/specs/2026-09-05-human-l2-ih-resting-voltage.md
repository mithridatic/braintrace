# Leak reversal split on the larger-Ih candidate

Keep the uniform distributed-Ih factor-75 candidate fixed except for
passive reversal potential. Shift the source e_pas by -4 mV. This
prospective diagnostic value is motivated by the observed +3.215 mV
starting-voltage displacement, with no assumption of unit sensitivity.
It is inferred, not measured. Do not change recorded bias or initial
voltage to force the response.

Use sweep 43, original leak density, calcium factor 1.5, original sodium
and Ih kinetics, mesh 9, CVode 1e-10, complete recorded-bias input, and
endpoint 2100 ms. Preserve source fit and applied reversal metadata.

Require a smaller absolute error at the predefined 1019 ms voltage
sample than the factor-75 control. Separately assess the established
adaptation, below-baseline return, and no-spike conditions. Retain all
12 observations, deflections, and errors. Starting-voltage improvement
does not override a failed shape condition or qualify the cell.

The helper must change exactly one passive e_pas entry, reject multiple
passive entries or genome e_pas overrides, and reject nonfinite source,
shift, or output values. Default zero shift must preserve the fit.
This tests one electrical parameter; it does not identify a biological
leak ion or establish a unique mechanism for the human waveform.

Numerical comparison audits must reject changed or one-sided missing
applied passive parameters and reversal-shift metadata. Legacy pairs
without these fields retain only their original evidence scope.
