# Test the calcium-removal bracket midpoint

The third interval errors at removal factors 1.5 and 1.25 have opposite
signs: +23.839866 and -24.440005 ms. Test their parameter midpoint,
factor 1.375. This does not assume a linear interval response.
The factor is inferred from calibration results, not measured biology.

Use the exact setup from the reversal calcium split: distributed Ih 75,
passive reversal shift -4 mV, source sodium and Ih kinetics, recorded
bias, sweep 50, mesh 9, CVode 1e-10, soma-current recording, and endpoint
2100 ms. Change only somatic decay_CaDynamics. Preserve the source fit.

Require all conditions before accepting the bracket prediction:

- Five complete positive-peak spikes remain.
- The third interval's absolute human error is smaller than at both
  bracket endpoints.
- Each of intervals 2 and 4 has smaller absolute human error than the
  factor-1.5 control.
- The first interval's absolute human error does not exceed the
  factor-1.5 control error.

Retain each event, interval, phase, and recovery minimum with human
residuals. Missing or extra events reject the prediction. An unintended
setup change, bad input, nonfinite trace, incomplete boundary event, or
invalid recording makes the test invalid. Verify the one-row genome
change, raw mapping, command-plus-bias current, and endpoint.

This is one midpoint test, not an automatic parameter search. A passing
interval prediction does not accept the cell: rise, fall, recovery
depth, subthreshold amplitude, and reserved-input checks remain separate.
Candidate-specific numerical checks are required before promotion.
Do not inspect the reserved response during this calibration.
