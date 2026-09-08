# Somatic Kv3 density and first voltage return

The intervention doubles somatic Kv3_1 conductance on the half-Ca_LVA
candidate. Axonal Kv3_1, sodium settings, and all other parameters stay
fixed. Both inputs use mesh factor 9 and CVode tolerance 1e-10.

The directional test passes at both inputs: the first downward -65 mV
crossing occurs more than 0.1 ms earlier relative to the first peak, and
the first two peaks remain positive.

| Input, nA | Advance at -65 mV, ms | New minimum, mV | Time from peak to minimum, ms |
| --- | ---: | ---: | ---: |
| 0.19 | 0.257145 | -78.104705 | 1.884033 |
| 0.27 | 0.287577 | -76.744663 | 1.675362 |

The new -65 mV times from peak are about 0.520527 and 0.530631 ms.
Human times are 0.339762 and 0.337857 ms. The return improves but remains
late. Human minimum times are 0.760000 and 0.700000 ms.

First onset errors are +14.025879 and -0.178275 ms. First peak errors are
-2.201975 and -1.724375 mV. Duration-above--20-mV errors are -0.054425
and -0.048472 ms. Thus, stronger Kv3 conductance helps the voltage return
but worsens the low-input onset and makes the first spikes too narrow.
It is not an independent return-time control.

The trains have 24 and 80 complete events, versus human 12 and 43.
The [audit](h01-pv-kv3-return-audit.json) retains all event times, intervals,
minima, and falling crossings. No aggregate count establishes a fit.
Numerical refinement of this intervention remains open. No candidate is
promoted, and the reserved trace was not read.

Run `python -m docs.evidence.h01_pv_kv3_return_audit` to repeat the analysis.
Twenty-four driver, datum, and crossing tests pass. The driver rejects
negative and nonfinite Kv3 factors, accepts zero, and preserves factor 1
when omitted. Missing positive spikes or falling crossings invalidate the
directional comparison rather than passing it.
