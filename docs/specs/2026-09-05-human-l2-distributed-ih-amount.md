# Distributed Ih amount split

Use the completed uniform-location candidate as control. Keep placement,
original rate law, original leak, calcium factor 1.5, sodium parameters,
mesh 9, CVode 1e-10, initial voltage, temperature, recorded-bias sweep 43,
and endpoint 2100 ms fixed. Increase total maximum Ih by factor 75.

This factor is a rounded diagnostic scale from the pinned published
model densities applied hypothetically to the current geometry. It is
not fitted to the current waveform or measured in this neuron. Expected
uniform density is about 1e-4 S/cm2 and total maximum conductance about
15.43 nS. It does not reproduce the published modified rate law or its
cell-specific passive fit.

Require the same three sufficiency conditions: late pulse voltage at
2019 ms below 1120 ms, voltage at 2099 ms below 1019 ms, and no complete
pulse spikes. Preserve every one of the 12 direct voltage samples,
absolute and deflection errors, and baseline shift relative to control.
A shape pass does not override a worsened resting voltage or qualify
the human waveform. Verify every non-Ih genome row is unchanged, all
three Ih densities are multiplied by 75, and integrated maximum
conductance changes by that factor to relative tolerance 1e-12.

Do not alter bias or passive reversal to compensate for any baseline
shift in this experiment. Such compensation would be another factor.
Reserved input and candidate promotion remain gated on full validation.
