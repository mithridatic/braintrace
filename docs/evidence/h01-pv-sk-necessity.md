# SK necessity result

The [specification](../specs/2026-09-05-sk-necessity-test.md) defines the claim
and rejection threshold before the two removal simulations.
The removal sets SK density to zero in soma and axon.
Recorded somatic SK current is exactly zero. Calcium dynamics remain active.

With SK removed, the second rising -20 mV crossing occurs 2.734115 ms later
after faster sodium inactivation. This exceeds the predeclared 1 ms margin.
Both cases retain complete spikes. The stated necessity claim is rejected
at these conditions: SK is not required for this specific later timing effect.

The first interspike interval is 8.471573 ms without accelerated inactivation
and 10.976462 ms with it when SK is absent. Thus, the second-spike difference
is not just a uniform shift of the whole response caused by first-spike latency.
With SK present, those intervals are 8.999825 and 12.030478 ms.

Do not generalize this result to the full train. With SK present, faster
inactivation changes the count from 11 to 19. With SK absent, it changes from
42 to 38. The aggregate direction reverses, but this does not identify a
unique mechanism for any later event. Each event is retained in the JSON audit.

The removal changes the operating point and does not measure a mediation
fraction. Other channel states and cable currents remain possible routes.
No changed model is accepted as a human fit. The direct first-spike waveform
and the overall response still need physiological calibration.

Run `python -m docs.evidence.h01_pv_sk_necessity_audit` to repeat the decision.
The [JSON audit](h01-pv-sk-necessity-audit.json) retains complete crossing and
peak records for all four cases, with intervention checks.
