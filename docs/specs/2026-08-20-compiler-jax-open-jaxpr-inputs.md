# Compiler/JAX open-jaxpr input compatibility

## Problem

The compiler constructs transition jaxprs whose external runtime values are
represented as a logical constvar prefix followed by hidden-state inputs. JAX
0.11 unified `Jaxpr` storage: when a jaxpr is constructed without attached
constant values, the former `constvars` are exposed through `invars` instead.
Consumers that read only `jaxpr.constvars` therefore drop the external runtime
values and call `jax.core.eval_jaxpr` with too few arguments.

This currently breaks the compiler's grouped-state transitions and the
`y -> hidden` transition used by D-RTRL. The observed failures are the cond
and vmapped D-RTRL parity tests plus the first three grouped SNN transition
cases, all reporting `foreach() argument 2 is shorter than argument 1`.

## Contract

Provide one compatibility interpretation for an open transition jaxpr:

- On JAX versions with distinct open `constvars`, return those constvars.
- On JAX 0.11+, recover the leading external-input prefix from the unified
  input list when the known runtime invars form its suffix.
- Preserve the existing ordering so callers pass values to
  `jax.core.eval_jaxpr` as `external_values, runtime_invars`.
- Do not change the compiler's mathematical graph, state grouping, or
  approximation semantics.

## Acceptance gates

1. A co-located compatibility regression covers both unattached JAX 0.11
   open jaxprs and ordinary jaxprs with attached constants.
2. The five currently failing compiler tests pass without weakening their
   numerical assertions.
3. The complete `braintrace/_compiler` suite passes.
4. The broader test gate is run and any unrelated failures are reported
   separately.
