# Example 21 Kimi Transfer Final Compatibility

## Gate C2 Snapshot Inventory

Periodic memory reads add `ShortTermState` cadence diagnostics to model
snapshots. Gate C2's H0 evidence is explicitly a semantic `HiddenState`
inventory and must not absorb counters or booleans merely because snapshot
serialization carries both state classes.

`_gate_c2_snapshot_arrays` therefore excludes only the three cadence
diagnostics. Boundary replay retains every true `HiddenState`, including the
derived `memory_drive` cache. Raw H0 evidence continues selecting only its
preregistered digest subset, so the v1 Gate C2 digest domain stays stable while
complete semantic snapshots remain replayable.

## Verification

The existing raw-H0 fixture must reproduce the failure before the change and
pass afterward. Add a focused diagnostic-state exclusion regression, then run
the complete co-located Example 21 numerical/oracle gate again.
