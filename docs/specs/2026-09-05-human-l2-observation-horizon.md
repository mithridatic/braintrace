# Preserve the response prefix with an earlier observation endpoint

The source waveform lasts 8019.98 ms. The selected physiological
observations end with the main pulse at 2020 ms. Test an optional
2100 ms simulation endpoint for subsequent mechanism experiments.
Retain the complete command vector and all pre-pulse state history.
Do not stop or restart the ongoing factor-nine full run.

Before using the shorter endpoint, run the unchanged source at its
0.005 ms step through 2100 ms. Compare every saved time, voltage, and
applied-current sample with the prefix of the existing full source run.
Require exact array equality, not a tolerance or a spike-count proxy.
The only intended change is the termination condition after the pulse.
If prefix equality fails, do not use this optimization for mechanism work.

Default behavior retains the complete recording duration. Reject an
endpoint before 2020 ms, beyond the source duration, or nonfinite.
Record the requested and observed endpoints. A prefix pass establishes
equivalence only through the tested endpoint; it makes no claim about
the uncomputed post-pulse tail. Future tests of recovery beyond 2100 ms
must use a longer horizon.
