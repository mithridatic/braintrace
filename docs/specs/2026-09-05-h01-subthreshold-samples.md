# Fixed direct subthreshold samples

Use human E sweep 43 and released human I passive traces 0 and 1, whose
existing mapping is -0.11 and -0.05 nA. Use the established clocks and voltage
corrections. Do not read reserved active traces.

Before extracting values, fix these sample times: pulse onset minus 1 ms;
onset plus 1, 5, 20, 100 and 500 ms; pulse end minus 1 ms; pulse end plus 1,
20 and 100 ms. Select the nearest existing sample and retain its actual time
and timing error. Require all requested times to be inside the recording and
the error no larger than half the largest sample interval. Keep raw hashes.
These are direct targets, not fitted summaries. Do not assign physiological
allowances from baseline noise or declare model validation from extraction.
