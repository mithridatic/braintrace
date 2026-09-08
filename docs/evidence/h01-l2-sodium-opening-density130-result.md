# More sodium restores peaks but does not restore the full response

The predefined joint prediction passes. Five complete positive-peak spikes
remain. With opening factor two fixed, increasing NaTs density from one
to 1.3 reduces every absolute peak error. Each rising-phase error remains
more than 0.01 ms smaller than in the fast-opening reference.

| Spike | Onset error (ms) | Peak error (mV) | Rise-phase error (ms) | Duration error (ms) |
| --- | --- | --- | --- | --- |
| 1 | +0.901594 | -0.071490 | -0.121668 | +0.024776 |
| 2 | +1.328053 | +0.170729 | -0.148963 | -0.179830 |
| 3 | +32.650842 | +0.100067 | -0.145065 | -0.025515 |
| 4 | +36.100554 | +0.181610 | -0.138678 | -0.022538 |
| 5 | +60.271546 | +0.389983 | -0.146242 | -0.014894 |

Errors are model minus human. Duration is time above -20 mV. Interval
errors are +0.426459, +31.322789, +3.449712, and +24.170992 ms.
Recovery-minimum errors are -2.094337, -4.966488, -1.914093, and
-1.872330 mV. All four minima are too negative. Rising phases remain too
short; longer falling phases partly hide those errors in total duration.

Only sodium density and its one soma genome row change among physical
setup fields. Raw mappings are exact, both input plateau errors are zero,
and all arrays are finite. Both traces end at 2100 ms. The
[full result](h01-l2-sodium-opening-density130-result.json) retains individual
responses, residuals, setup changes, and hashes.

This result supports compatible peak and rising-phase improvements under
the tested intervention. It does not establish a full human cell fit or
measured sodium density. Numerical checks for this density candidate,
subthreshold validation, and reserved-input validation remain open.
