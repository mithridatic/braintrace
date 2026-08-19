# An online-update sequence driver

Status: implemented; 86 sequence-driver tests pass
Date: 2026-08-19
Branch: `feat/example21-row-refinement`
Extends `2026-07-27-sequence-driver-api.md`.

## 1. The gap

pp-prop's learning rule is

```
grad_theta L = sum over t' in T of  (dL^{t'} / dh^{t'})  hadamard  eps^{t'}
```

Every term in that sum is complete at its own timestep. Nothing in
`(dL^{t'}/dh^{t'}) ∘ eps^{t'}` refers to a later step — that is the entire point
of an eligibility trace, and the property that separates these algorithms from
backpropagation through time.

`etrace_grad` computes exactly those per-step terms. Its scan body calls
`grad_fn` once per step and then adds the result into a carry:

```python
def body(carry, xs_t):
    slices, weight = xs_t
    grads, (loss, aux) = grad_fn(slices, weight)
    return jax.tree.map(jnp.add, carry, grads), (loss, aux)
```

The accumulator is the only thing standing between this library and online
learning, and it is a choice rather than a constraint. Summing the terms and
applying one update at the end is the *batch* specialization of an online
algorithm. The published tutorials teach only that specialization, so a user
reaching for braintrace to do online learning currently has to write the loop
themselves and get the trace advance, the loss gating, and the optimizer state
right by hand.

## 2. Change

`SequenceDriverMixin` gains `etrace_online`. It drives the sequence exactly as
`etrace_grad` does — same step function, same trace advance, same per-step
gradient — and applies the optimizer at each step instead of accumulating.

```python
learner.etrace_online(
    inputs, targets,
    step_fn=step_loss,
    optimizer=opt,
    mask=loss_mask,
)
```

The model therefore learns *within* the sequence: step `t + 1` runs under
parameters that step `t` already moved. Weight changes and the eligibility
trace interact, which is the regime these algorithms exist for and which no
accumulate-then-update path can express.

### 2.1 What `mask` means here, and why it differs

In `etrace_grad`, `mask` gates the loss only, and the documentation says so
loudly. In `etrace_online` it gates the loss **and the update**.

The difference is not cosmetic. A masked-out step produces an identically zero
gradient, and under a stateful optimizer a zero gradient is not a no-op: Adam
decays its moment estimates and takes a nonzero step from surviving momentum.
Over a sequence whose supervised window is a small fraction of its length —
Example 21 supervises 60 of 180 ticks — that momentum bleed would dominate the
learning signal. So a zero-weight step drives the model and the trace, exactly
as in `etrace_grad`, and leaves the parameters and the optimizer state alone.

Non-binary weights keep their `etrace_grad` meaning: the weight scales the
gradient. A step whose weight is zero is skipped; every other step updates with
a gradient scaled by its weight.

### 2.2 What it does not take

`reduction` is absent. In `etrace_grad` it divides an accumulated gradient by
the total mask weight; there is no accumulator here to divide, and an argument
that silently means something else is worse than an argument that is missing.
Per-step losses are returned unreduced and the caller reduces them.

`chunk_size` keeps its `etrace_grad` meaning and becomes the update-frequency
knob: `k` steps produce one gradient and one update. A window updates when any
step in it carries nonzero weight.

### 2.3 Gradient transform

`transform`, an optional callable applied to the gradient before the optimizer
sees it. Online updates are applied hundreds of times per sequence rather than
once, so clipping is not optional in practice, and requiring the caller to
reach inside a compiled scan to do it would defeat the driver.

## 3. Protocol

This adds a driver. It changes no numerics, no estimator, no per-primitive
rule, and no existing method. `etrace_grad` and `etrace_evolve` are untouched.

## 4. Tests

- one step with an all-ones mask moves the parameters exactly as a hand-written
  `grad_fn` + `optimizer.update` pair does;
- a `T`-step online run differs from `etrace_grad` + one update on the same
  sequence, and the two agree when `T == 1` — the online path is a real
  mechanism, not a formatting choice;
- a zero-weight step leaves both parameters and optimizer state untouched while
  still advancing the trace, shown by a later supervised step landing where an
  `etrace_evolve` prefix would put it;
- an all-zero mask performs no update at all and stays finite;
- a non-binary weight scales the applied gradient;
- `transform` reaches the optimizer, verified by a transform that zeroes the
  gradient, and scales the step under SGD over a single step;
- a scaled step **compounds** rather than scaling the whole run, which is the
  online path's content and the reason the linearity claim is only well posed
  at `T == 1`;
- window mode applies one update per window and refuses the same inputs
  `etrace_grad` refuses;
- the same argument validation as `etrace_grad`: empty sequences, disagreeing
  lengths, wrong mask shape, bad chunk size, wrapper arguments;
- an optimizer that was never given the learner's weights fails loudly.

## 5. Gate

Accepted when the tests above pass and the equivalence test in §4 line 2 shows
`T == 1` agreement together with `T > 1` divergence. No claim about task
performance is made by this specification; it adds an expressible regime, not a
better one.

## 6. Two things the tests had to be rewritten to see

Both were instrument errors rather than defects, and both are worth recording
because they will recur in any online measurement here.

**Adam cannot see a gradient scaling.** It divides by its own second moment, so
halving a gradient changes the step only through epsilon — the first attempt at
the `transform` test measured a 2.7e-7 difference and called it absence. Any
assertion about gradient magnitude must be made under SGD; Adam is the wrong
instrument for it.

**An online run is not linear in its step size.** The first scaling test drove
six steps and expected a doubled transform to double the total displacement. It
does not: a doubled step at `t` leaves the model somewhere else at `t + 1`, so
the gradient it meets there is not the doubled one. That compounding *is* the
mechanism this driver adds, so it is now asserted directly rather than being
mistaken for numerical error.
