# Waveform candidate screen

The candidates use faster sodium inactivation with source recovery speed.
Other parameters remain fixed. Both new runs finished successfully.
The reserved 0.23 nA trace was not used.

| Closure factor | First peak, mV | First duration above -20 mV, ms | Complete -20 mV excursions |
| --- | ---: | ---: | ---: |
| Human | 19.5625 | 0.273395 | 12 |
| 1.0 | 44.2482 | 0.774893 | 11 |
| 0.5 | 41.1906 | 0.451292 | 19 |
| 0.25 | 32.0173 | 0.295184 | 26 |
| 0.10 | -3.2372 | 0.288703 | 1 |

The quarter-time candidate improves both first-event errors substantially.
It remains too high and produces an incorrect train. It is not validated.
The tenth-time candidate has one above-threshold excursion, but its peak never
reaches zero. Do not call this a preserved human-like spike train.
The original positive-peak detector reports no spikes in that case.
Threshold excursions and detected action potentials must remain separate fields.

The two-error screening rule alone is insufficient: the tenth-time case also
reduces both absolute errors relative to the poor original waveform.
It still fails the intended spiking behavior. The saved report preserves the
screen result and separately records positive-peak events. No candidate is promoted.

The target peak lies between these two candidate peaks. This is a bracket for
further inference, not proof of a valid intermediate fit. Any new candidate
must retain repeated spikes and satisfy individual event durations and times.
Coupled channel parameters and the full conditioning protocol remain relevant.

The [JSON report](h01-pv-waveform-candidates.json) retains every event, signed
first-event errors, and unaligned trace RMS. No traces are shifted or rescaled.
Run `python -m docs.evidence.h01_pv_waveform_candidates` to repeat the analysis.
The numerical and multi-current validation requirements remain open.
