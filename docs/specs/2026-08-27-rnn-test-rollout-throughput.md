# RNN test rollout throughput

## Status

Proposed implementation specification for BRA-106. Implementation requires
explicit approval after review of this specification and a runnable baseline.

## Objective

Reduce the runtime of `braintrace/nn/_rnn_test.py` without weakening its RNN
cell coverage or changing production behavior. The change targets repeated
recurrent execution currently driven from Python and lowers each fixed rollout
as one BrainState transform.

## Evidence and baseline

Static inspection on Python 3.12.3 found 11 fixed-length Python loops that drive
RNN cells for 154 total recurrent steps across a focused module containing 67
tests. The rollout inventory is:

| Test path | Steps |
| --- | ---: |
| ValinaRNNCell sequential updates | 5 |
| GRUCell sequential updates | 10 |
| MGUCell sequential updates | 5 |
| LSTMCell sequential updates | 15 |
| MinimalRNNCell sequential updates | 7 |
| MiniGRU sequential updates | 12 |
| MiniLSTM sequential updates | 10 |
| CFNCell sequential updates | 10 |
| LRUCell sequential updates | 20 |
| Full-sequence LSTM integration | 50 |
| GRU reset-during-sequence setup | 10 |

Repository examples and compiler tests already exercise the intended API form:
`brainstate.transform.for_loop(cell_or_step, stacked_inputs)`. This reduces API
uncertainty, but it is not evidence of a speedup in this test module.

This execution image does not contain pytest, JAX, BrainState, BrainUnit,
Braintools, or NumPy. No matching virtual environment or cached wheels are
available, and the installed Docker client cannot access the Docker daemon.
Consequently, this image cannot produce an honest runtime baseline. No code
will be changed until the following measurements run in a dependency-complete
CPU environment:

1. Record Python, JAX, BrainState, pytest, CPU backend, and worker count.
2. Run the focused module in five fresh processes with
   `pytest braintrace/nn/_rnn_test.py -q --durations=0`.
3. Record collection time, total wall time, and the median and range across the
   five runs.
4. Time one representative LSTM rollout separately, synchronizing the cold
   transformed call and a second warm call so compile and steady-state execution
   are not conflated.

The execution worktree is on the required
`paperclip/BRA-106-speed-up-code-and-tests` branch, but its Git metadata is
mounted read-only. The specification therefore remains an untracked workspace
file until the harness provides a writable index; unrelated pre-existing changes
remain untouched.

The implementation is accepted only when the focused median improves by at
least 10 percent or 0.5 seconds, whichever threshold is smaller, without a
meaningful regression in collection time. Otherwise the proposed code change
is abandoned and only the measurements are retained.

## Proposed change

For each fixed sequential-rollout test:

1. Generate the complete input tensor once with `brainstate.random`.
2. Execute the cell with `brainstate.transform.for_loop`, collecting its stacked
   outputs.
3. Assert the same step count and per-step output shapes as the current test.

The reset test uses a no-output loop body if needed, but still executes all ten
state transitions through `brainstate.transform.for_loop`. Tests that vary
batch geometry remain outside this change because they are construction/schema
checks rather than a uniform recurrent rollout.

## Correctness invariants

- Use `brainstate.random`; never call `jax.random` directly.
- Preserve the exact tested cell classes, batch sizes, feature sizes, step
  counts, reset boundaries, and shape assertions.
- Preserve stateful sequential semantics: step `t + 1` must consume state from
  step `t`.
- Do not relax assertions, tolerances, markers, or default test selection.
- Do not change production modules or public APIs.
- Keep the test beside `braintrace/nn/_rnn.py` under the existing suffix naming
  convention.

## Regression tests and verification

Before the rewrite, add a focused equivalence regression that compares a short
Python reference rollout with the transformed rollout from identically seeded
cells and inputs. The Python loop is permitted only as the one-shot correctness
oracle; the optimized test path itself must use the transform. Compare every
stacked output and the terminal hidden state, including both LSTM hidden states.

After implementation:

1. Run the new equivalence regression and all of
   `braintrace/nn/_rnn_test.py`.
2. Re-run the five-process focused timing protocol and report median, range,
   cold compile time, and warm execution time.
3. Run coverage for `braintrace/nn/_rnn.py`; retain more than 90 percent
   meaningful line coverage.
4. Run the smallest broader neural-network test selection that catches shared
   state/transform regressions.

## Edge cases and tradeoffs

- Stacked transform outputs have a leading step axis rather than a Python list;
  assertions must check that axis explicitly.
- Random generation is moved outside the transform, so equivalence is defined
  over the same pre-generated inputs rather than generator call ordering.
- A transform may increase cold compile cost for very short rollouts. The
  benchmark gate prevents landing the rewrite if that cost outweighs reduced
  dispatch overhead in this test module.
- Shape-varying batch tests cannot share one compiled loop and remain unchanged.

## Correction notes

- The first baseline command assumed a `python` shim and pytest installation.
  Future benchmark commands will resolve and record the interpreter and required
  package versions before timing.
- The initial temporary-file command trusted an unset scratch variable. Future
  scratch paths will first validate the Paperclip scratch variable and otherwise
  use a directory created by `mktemp -d`. This mistake recurred when the current
  adapter again omitted the variable; the fallback is now mandatory rather than
  advisory.
- A later control-plane status attempt constructed temporary payload files with
  shell redirection despite the workspace editing rule. Future API payloads will
  be encoded in memory and passed directly to the client; project-file changes
  continue to use `apply_patch` exclusively.
