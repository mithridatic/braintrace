# Real event and learning profile

Continue the full default 104-cell evolution under-30-second objective without
changing dt, corpus, rounds, update count, or learning semantics. The previous
scoring-only optimization does not satisfy that objective.

Measure four anatomical cells from verified network evidence using current
numerical settings. Separately retain cProfile data and synchronized timings
for construction, initialization, model setup, cold event, warm event, sparse
graph construction, cold Muon update, and warm update. The one-event synthetic
loss characterizes runtime cost only; it is not ARC quality evidence.

Use a five-minute external process cap and retain each completed phase so a
timeout cannot be mistaken for completion. Do not attach to, stop, or restart
the other session's active evolution. Its container denied profiler attachment
because SYS_PTRACE is unavailable. The output directory had no checkpoints
after approximately 58 minutes, so checkpoint serialization is not yet an
evidenced bottleneck for that run.

Candidate: padded DHS tree entries currently scatter zero into the same valid
sentinel row. Place invalid parent indices one past the array end and use
explicit drop-mode scatter. Preserve child padding, valid edge indices, update
order, sentinel state, forward solves, transpose solves, and implicit gradients.
Measure a deliberately uneven tree on GPU before and after; require dense and
installed-solver oracles and existing real-cell solver tests with xdist.

Next candidate: place the existing 32-wide elimination schedule inside one
Pallas/Triton GPU kernel, retaining masked atomic additions and a barrier after
each level. Use only inside custom_linear_solve primal/transpose callbacks so
derivatives remain the exact implicit linear derivatives. Platform dispatch
must preserve CPU execution; unsupported widths retain the existing scan.
Before activation require input non-mutation, multiple RHS batches, vmap,
forward/reverse Jacobians against dense solves, and real-cell learning tests.
