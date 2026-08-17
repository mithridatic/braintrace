# Latent-reasoning expressivity: Dale recurrence and a latent clock

Status: implementing
Supersedes nothing. Extends `2026-08-16-pp-prop-latent-reasoning.md` and
`2026-08-16-latent-readout-consistency.md`.

## Problem

Example 21 transports a contextual memory read across latent ticks but does not
compute with it. Measured on the readout-consistency build:

| metric | R=0 | R=1 | R=2 | R=4 | R=8 |
|---|---|---|---|---|---|
| terminal accuracy | 0.429 | 0.429 | 0.393 | 0.357 | 0.357 |
| participation ratio (of 32) | 3.82 | 3.35 | 1.42 | 0.72 | 0.69 |

Workspace decodability peaks at R=2 (0.357) and never reaches the 0.571 that a
single memory read already supplies. Depth is not buying computation.

## Diagnosis: two independent causes

Both are structural properties of the current model, confirmed by reading it.

1. **Nonnegative recurrence.** `LatentWorkspaceModel.__init__` initializes `Wf`
   as `uniform(0, 1) * bernoulli(connectivity) + 0.01 I`, rescaled to spectral
   radius 0.9. A nonnegative matrix has, by Perron-Frobenius, a real dominant
   eigenvalue with a nonnegative eigenvector, and repeated application projects
   any nonnegative state onto it. Participation ratio 3.82 -> 0.69 over eight
   ticks is that projection.

2. **The latent map is autonomous.** On any latent tick after the seed the
   phase gates give `demo = query = seed = 0`, so `query_drive`, `key_rows`,
   and `value_rows` are identically zero. Every latent tick applies the same
   map `H <- f(Wf H, read(H))` against a frozen memory. An iterated autonomous
   nonlinear map converges to a fixed point *regardless of weight sign*.

Cause 2 is why fixing cause 1 alone is not sufficient: mixed-sign weights under
an autonomous map buy oscillation or a different fixed point, not computation.
Both levers move together or the change is not testable.

## Lever A: Dale-structured recurrence

Partition the `latent_width` presynaptic columns into an excitatory majority
and an inhibitory minority (`dale_inhibitory_fraction`, default 0.25). Sample
magnitudes exactly as today, then negate inhibitory columns and rescale to the
configured spectral radius. The self-leak `0.01 I` stays positive.

Dale's law is a constraint on the *sign pattern*, so it must survive training.
`Wf` remains a free-sign `ParamState` (the ETP compiler sees an unmodified
parameter feeding `braintrace.matmul`; transforming the value before the
primitive would break parameter identification). The sign pattern is restored
by projecting after each optimizer update, outside the ETP graph:

    Wf <- dale_signs * abs(Wf)

`dale_signs` is a non-parameter constant of shape `(1, latent_width)`
broadcasting over rows, so the constraint is per-presynaptic-neuron as Dale's
law requires, not per-synapse.

## Lever B: latent clock drive

Give successive latent ticks distinct external drive so the map is no longer
autonomous, without introducing a counter hidden state (which would enlarge the
hidden Jacobian).

`TaskConfig` gains a `clock_width` channel bank (default 4, must be even and
positive) appended after the phase vector. On latent tick `i` (zero-based
within the latent span) the bank holds a fixed Fourier phase code

    [sin(2 pi i / P_k), cos(2 pi i / P_k)] for k in 0 .. clock_width/2 - 1
    P_k = 2 ** (k + 1)

and is zero on every non-latent tick. A Fourier code rather than a one-hot over
latent index is deliberate: its width is independent of `latent_steps`, so a
model trained at one depth is evaluated at another without a shape change.

The model gains `Wc : (clock_width, latent_width)`, a trainable `ParamState`
consumed by a `braintrace.matmul` whose input is row-masked onto the workspace
row *before* the primitive, exactly as `query_drive` already is. This keeps the
write position-preserving: no scatter and no reshape of an ETP output reaches a
hidden state. The result is added to `parameter_drive`.

Because the clock is external input rather than hidden state, it adds no
hidden->hidden contraction. The hidden-group transition jaxpr must still lower
to exactly two `dot_general` equations (the two memory-read contractions).

## Success criterion

Stated before measurement, scored after.

- **Necessary:** participation ratio holds near its R=0 value (~3.8 of 32)
  across depth instead of collapsing to ~0.7.
- **Sufficient:** workspace decodability rises past the 0.571 single-read
  baseline at some R > 0.

Participation ratio recovered with decodability flat is reported as a partial
result, not as latent reasoning. Accuracy is not a gate: per-K cells are n=4
and the decodability probe is n=14, so accuracy cannot resolve the difference.

## Non-goals

- Learning the write path. `write_mode` stays `"fixed_random"`.
- Changing the seeding rule, the readout carrier, or the phase-channel split
  established by the readout-consistency change.

## Edge cases

- `clock_width` odd, zero, or negative -> `ValueError` at `TaskConfig`
  construction.
- `latent_steps = 0` -> the clock bank is all zero and `Wc` contributes
  nothing; the R=0 path must stay byte-identical to the pre-change R=0 path
  apart from the widened input row.
- `dale_inhibitory_fraction` outside `(0, 1)` -> `ValueError`.
- A `latent_width` small enough that the inhibitory partition rounds to zero
  neurons -> at least one inhibitory neuron is forced, so Dale structure is
  never silently vacuous.
- Spectral-radius rescaling uses `max(abs(eigvals))` on a now-complex spectrum;
  the signed matrix must still be rescaled to the configured radius.
- Dale projection is idempotent: applying it twice equals applying it once.

## Test plan

- `TaskConfig` validation for every `clock_width` and fraction edge case.
- Clock code is zero outside the latent span, nonzero inside, and differs
  between latent tick `i` and `j` for `i != j` within one period set.
- `input_width` accounting matches the emitted row width.
- Signed `Wf` has both signs present, uniform sign per column, and the
  configured spectral radius.
- Dale projection restores the sign pattern after an arbitrary perturbation and
  is idempotent.
- Hidden-group transition still lowers to exactly two `dot_general` equations.
- `Wc` appears in the compiled ETP weight relations.
- `latent_steps = 0` still runs and reads the query terminal.
