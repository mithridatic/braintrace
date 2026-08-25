# Example 21 Gate 4 proof contract

This note records the implementation boundary for OpenSpec tasks 6.1–6.4.

- CPU and GPU probes run in separate processes against the same first
  `d631b094` query and full 705-event episode. Each probe warms once, records
  three synchronized calls, rejects non-finite results, and reports medians.
- Backend selection uses the literal lower valid median; an exact tie selects
  CPU. The selected backend is frozen for all matched proof arms.
- Decoder timing uses the 31 executed request readouts. It records five calls
  per fixed validation query and fails above 100 ms per call.
- The temporary proof performs eight updates on `d631b094` only. `46f33fce`
  is forward-only and cannot alter parameters, optimizer state, or eligibility.
  It records direct predictions, targets, loss components, recurrent-weight
  movement, and voltage-only, sodium-gates-only, potassium-gates-only,
  spikes-only, all-state, and null interventions.
- Proof runtime is at most 180 seconds. The co-located Example 21 focused
  module is at most 60 seconds and must exceed 90 percent meaningful coverage.

Acceptance evidence must be written outside `result.json` for timing fields and
must include the direct prediction/intervention records required by Gate 4.

## Local verification record

The Gate 4 helper has focused coverage of 97 percent. Its 14 tests finish in
1.79 seconds. The complete co-located Example 21 selection has 44 passing
tests and finished in 56.74 seconds on the CPU worktree environment.

The backend guard rejects non-finite timings, selects the lower valid median,
and rejects mismatched prediction bytes. The proof guard requires Boolean
change records for all six interventions, requires the null intervention to
remain unchanged, and requires at least one non-null direct change.

The real-data CPU/GPU process run remains an environment-owned execution step:
the local checkout has no `/datasets/arc` or equivalent `d631b094` and
`46f33fce` files. No real-data result is claimed from this worktree run.
