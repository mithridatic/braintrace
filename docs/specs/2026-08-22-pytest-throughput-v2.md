# Pytest throughput v2

## Status

Approved implementation specification for `pytest-throughput-v2`.

## Objective

Reduce the current Example 21 and latent-workspace pytest gate while retaining
the complete default selection and finite-window learning-rule coverage. The
live selection contains 2,208 tests and took 913.42 s with six xdist workers
on Python 3.14.6, JAX 0.11.1, brainstate 0.5.4, and pytest-xdist 3.8.0. The
recovery target is a green median no greater than 435 s.

## Scope and invariants

1. Use `compiled_scan=True` only in tolerance-based finite-window numerical
   probes. The oracle default remains `compiled_scan=False`; byte-identity and
   host-order tests retain the legacy path.
2. Profile repeated ARC adaptation compilation and Example 21 setup before
   changing fixtures or production drivers. Reuse must reset mutable model,
   optimizer, and eligibility-trace state at current test boundaries.
3. Tune xdist only from fresh measurements. Keep all 2,208 nodes in the
   default gate and preserve the JAX cache-clear safety policy unless memory
   evidence justifies a documented change.
4. Preserve public APIs, JSON schemas, production source pins, and scientific
   artifacts. Compiled numerical probes may differ only within their existing
   documented tolerances.

## Verification

Run changed numerical probes and oracle regressions first. Run the complete
selection in three fresh processes with the chosen bounded xdist configuration,
recording pass count, wall time, durations, and worker memory. Run coverage on
the same scope and compare its meaningful line total with the prior 95% total.

## Measured hotspot follow-up

One representative Gate C2 validator call spends about 37.5 s in 2,328
floating-difference records because each 512-element aggregate is checked by
Python scalar loops. Optimize this path with array operations only after
retaining strict input types, schemas, recomputed aggregates, and failure
behavior.
