# Faster calcium removal overshoots one interval target

The combined prediction is rejected. Five complete positive-peak events
remain. The first interval error does not increase, but the third
interval's absolute error increases. The separate interval errors are:

| Interval | Control error (ms) | Faster-removal error (ms) |
| --- | --- | --- |
| 1 | +3.354386 | +3.210483 |
| 2 | +21.702419 | -10.465990 |
| 3 | +23.839866 | -24.440005 |
| 4 | +41.737076 | -4.293383 |

Errors are model minus human. Later intervals shorten by tens of
milliseconds. The first peak changes by only +0.000027494 mV. The
four recovery minima change by +0.000059377, +0.001522548,
+0.002499891, and +0.001305283 mV. They remain too negative relative
to the human trace. This setting does not repair recovery depth.

Metadata isolation passes: only the somatic decay_CaDynamics value
changes, from 741.029329 to 617.524441 ms. Other differences are its
intervention record, output size, and execution time. Raw mappings are
exact; both inputs have zero plateau error excluding 1e-7 ms around
transitions. Both traces are finite and end at 2100 ms.

The [full result](h01-l2-reversal-calcium125-result.json) retains each
event, interval, phase, minimum, residual, audit result, and source hash.
This is a coupled calcium-removal intervention, not an isolated SK test.
Candidate numerical and subthreshold checks remain open. No full human
fit or reserved-input validation is established.
