# Worktree assessment at the six-hour user checkpoint

The requested result has not been achieved. None of the six population values has
increased during this campaign. There is no newly qualified human donor, no new
qualified 104-cell driven window and no qualified physiological transfer onto the
corrected anatomy. Existing construction and isolated timestep results do not
qualify the final human-only population.

## Work worth preserving

- The historical diagnostic anatomy conversion was shown to be malformed. The
  corrected export preserves 11,524 source nodes and 11,523 edges, with coordinate
  and radius agreement checked. This repairs the interpretation of the old failure;
  it does not demonstrate physiological transfer. See conversion-correction.json.
- Human recordings, protocol clocks, units and source identities were recovered
  and preserved. The voltage-clamp exporter now retains recorded amplifier filter
  fields. Ambiguous ABF pulse reconstruction has explicit rejection behavior.
- The recorded four-pole observation filter passed its specified small-control
  prediction gate, reducing the excluded early-control error by 65.1 percent.
  This is a usable conditional observation correction, not a human channel model.
- Source provenance was corrected: the published potassium model's pooled human
  and mouse kinetics cannot be counted as human-only. The production-package diff
  since the recorded execution baseline 23ed242 is a provenance docstring change
  in h01_wilbers.py; most campaign implementation is diagnostic code under
  docs/evidence, not new production capability.
- Failed candidates are reproducible and retain their original raw observations,
  parameter bounds, decisions and test results. The newest fit improved training
  error but remains rejected; weak local parameter distinguishability now gives
  a concrete reason to avoid another unconstrained fitting iteration.

## Value and limits

This is a research/evidence worktree with reusable corrections and tested numerical
tools. It is not a finished human-only simulator or a qualified release. Passing
helper tests does not change that assessment. It should be reviewed selectively
before integration; the volume of artifacts and commits is not a measure of
progress toward 1.000.

The six-hour expense has not produced the requested score improvement. Further
work needs a discriminating data/model decision tied to qualification, not another
run selected merely because it lowers a training objective. At this checkpoint
the fit, full-sample review and sensitivity processes are all terminal. No further
experiment was launched after the user's cost concern.
