# Active check of the fixed passive-reversal candidate

Use the completed sweep-43 candidate with distributed Ih factor 75,
passive reversal shift -4 mV, calcium removal factor 1.5, and original
sodium and Ih kinetics. Keep all parameters fixed. Apply calibration
sweep 50 with its recorded bias and complete command history. Use mesh
9, CVode tolerance 1e-10, soma current recording, and endpoint 2100 ms.
Reserved sweep 53 remains excluded from fitting and this check.

Require five complete positive-peak events within the main pulse as a
necessary condition. Retain each event onset, peak, duration above -20 mV,
and each interspike interval. Compare each with its human counterpart.
Matching the event count alone does not qualify the candidate. Missing
or extra events reject the combined response target at these settings.

Check setup identity against the subthreshold run except for waveform,
sample count, and execution time. Verify the raw-to-final sample mapping,
input current on plateaus, finite observations, and exact endpoint.
This checks behavior across inputs. It does not isolate a channel effect.

If the candidate remains useful, tighten tolerance to 1e-11 and compare
mesh factors 3 and 9. Preserve the established numerical gates: selected
subthreshold voltage differences at most 0.01 mV; matched spike onset
differences at most 0.1 ms, peak voltage differences at most 0.1 mV, and
duration-above-threshold differences at most 0.01 ms. Require equal event
counts and matched peak signs. Retain every event difference.
Numerical consistency does not establish agreement with human physiology.

Before reading the tighter-tolerance spike output, add a separate check
for the phase diagnosis derived from the completed coarse run. For each
matched event, compare time from the -20 mV rise to the sampled peak and
time from that peak to the -20 mV fall. Require each phase difference to
be at most 0.01 ms and each model-minus-human phase error to retain its
sign. Retain all values. This additional numerical limit is not a human
physiological acceptance limit and does not replace the earlier gates.
