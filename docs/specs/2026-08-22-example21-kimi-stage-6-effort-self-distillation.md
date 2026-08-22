# Example 21 Kimi Transfer Stage 6: Effort Self-Distillation Gate

## Status

The implementation gate is closed. No accepted-stack evidence currently shows
that effort 60 is no worse than effort 30 for both pixel and rule-at-oracle,
with at least one improving by 0.005, in two of the three preregistered full
reduced-topology seeds.

## Configuration Surface

Add `effort_distillation_weight`, a finite nonnegative float with default
`0.0`, to configuration, CLI, reports, result JSON, checkpoint compatibility,
and architecture manifests. The zero default is behavior-identical to the
historical supervised objective.

A positive value fails closed during configuration validation while the
teacher gate is unavailable. It must not silently omit the loss or distill an
unqualified deeper trajectory.

## Deferred Mechanism

Once retained evidence opens the teacher gate, a later stage may add the
specified stopped-gradient, per-example better-teacher KL over shape and valid
cell-color outputs. Until then, no distillation loss, extra decoder trajectory,
or teacher selection is implemented.

## Evidence

The stage evidence manifest records the missing teacher qualification and the
tests proving default equivalence, validation, CLI/report/result round trips,
and checkpoint compatibility.
