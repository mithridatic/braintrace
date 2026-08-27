# Example 21 executable Gate 4 runner

## Scope

`examples/pp_prop/example21_gate4.py` must provide an executable real-data
proof runner for the immutable ARC image data. The existing validators remain
the single place for acceptance rules; the runner produces the measurements
that those validators consume.

## Execution contract

- The command starts two child processes, one with CPU-only JAX and one with
  GPU-enabled JAX. Both load the same first `d631b094` query and its 705-event
  encoded episode. Each child performs one warm PP-Prop gradient call and then
  three calls timed after explicit synchronization.
- The parent compares prediction bytes before selecting a backend. It selects
  the literal lower valid median, with CPU winning an exact tie. Invalid timing
  or non-finite evidence cannot win.
- The selected backend runs five warmed decoder calls for each fixed-validation
  request and rejects any call over 100 ms.
- The proof training arm performs exactly eight updates, all on `d631b094`.
  `46f33fce` is evaluated forward-only. It records prediction bytes, targets,
  shape and row loss components, recurrent-weight movement, and six state
  interventions: voltage, sodium gates, potassium gates, spikes, all state,
  and null.
- Evidence is written to a durable sibling JSON document and a human-readable
  report. `result.json` is reserved for the bounded prediction/result schema.
  The runner fails if the total elapsed time exceeds 180 seconds.

## Test contract

Focused tests must cover process argument construction, child-result parsing,
strict median selection, decoder timing, eight-update/data isolation, state
invariance, direct evidence, intervention evidence, and durable evidence
writing. They must be co-located with the runner and finish within 60 seconds.
