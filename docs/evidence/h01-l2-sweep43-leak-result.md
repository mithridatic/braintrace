# Leak scaling improves amplitude but not response shape

The amplitude prediction passes and the shape prediction fails. Scaling
all four passive leak densities by 1.5 reduces both selected absolute
deflection errors, but does not restore adaptation or the below-baseline
post-pulse return. There are no complete pulse spikes.

| Time, ms | Human deflection, mV | Control, mV | Leak factor 1.5, mV |
|---|---:|---:|---:|
| 1120 | 10.3125 | 12.088560 | 8.776974 |
| 2019 | 8.8125 | 12.932369 | 8.927384 |
| 2099 | -1.3125 | 1.432744 | 0.343657 |

At 2019 ms, deflection error falls from 4.119869 to 0.114884 mV. At
1120 ms, its absolute value falls from 1.776060 to 1.535526 mV, but
the sign reverses. The candidate starts 0.150382 mV above the human
1019 ms voltage. The [full audit](h01-l2-sweep43-leak-result.json)
retains each of the 12 direct samples and both prediction decisions.

Only the four declared leak rows change. Other fitted parameters,
geometry, source build, initial state, temperature, input, and solver
remain consistent. Raw mapping, endpoint, and input checks pass. Helper
and driver guards pass 35 tests with 100 percent statement coverage of
the leak helper. Source-fit default identity and four-row isolation also
pass against the retrieved fit.

The tested common leak increase controls much of the excess late rise,
but is insufficient for the time-dependent response. It does not identify
human leak densities or prove that leak caused the original biological
discrepancy. Candidate numerical and active-response checks remain open.
No candidate is promoted from this partial improvement.

The subsequent [frozen active-response check](h01-l2-leak-active-result.md)
produces two spikes instead of five. This candidate fails the combined
target despite its subthreshold amplitude improvement.
