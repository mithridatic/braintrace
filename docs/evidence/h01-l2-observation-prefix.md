# Exact preservation of the response prefix

The source run through 2100 ms has 420001 saved samples at dt 0.005 ms.
Every time, voltage, and applied-current sample equals the corresponding
prefix of the full 1603997-sample source run exactly. The
[comparison record](h01-l2-observation-prefix-check.json) satisfies the
predeclared exact-equality requirement.

The input vector remains complete. Only the simulation endpoint changes.
All pre-pulse state history and the complete main pulse are retained.
The default driver endpoint remains the full source duration. Eleven
driver guard tests pass, including invalid or too-early endpoints.

Use the 2100 ms endpoint for subsequent experiments whose observations
end at the 2020 ms main-pulse boundary. This qualification is for the
tested source setup; retain the endpoint in each new run's metadata.
It does not claim a physiological result or predict the uncomputed tail.
Experiments about later recovery must use the required longer endpoint.
The ongoing factor-nine spatial run was not stopped or restarted.
