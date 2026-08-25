# BrainCell compatibility fixtures

## Scope

OpenSpec tasks 1.1–1.3 establish the BrainCell 0.1.0 compatibility boundary
before the Example 21 replacement is implemented. The fixture module is
`examples/pp_prop/21-braincell-arc.py` and its sibling test module.

## Dependency contract

Every declared development, example, testing, CPU, CUDA, TPU, and Example 21
image environment declares `braincell==0.1.0`. The fixture reports the imported
BrainCell, BrainState, BrainUnit, JAX, and BrainTrace modules. It does not accept
an unpinned BrainCell dependency.

## Compatibility contract

The fixture constructs one four-cell Hodgkin–Huxley population with the exact
BrainCell constructor values in the approved change design: `10 um` length,
`5 um` radius, `1 uF/cm²` capacitance, `0 mV` threshold, constant `-65 mV`
initial voltage, `ReluGrad(alpha=0.3, width=1.0)`, and `ind_exp_euler`. It uses
the SodiumFixed, PotassiumFixed, HH1952, and leak mechanisms with the declared
conductances and reversal potentials.

The PP-Prop fixture uses one `1 x 4` CSR input relation with indices
`[0, 1, 2, 3]`, row pointer `[0, 4]`, and raw values `[0.1, 0, 0, 0]`.
It checks the production bounded current-density path, one compiled `0.1 ms`
step, reset-state isolation for plus and minus perturbations, finite gates and
voltage, zero spikes, and finite nonzero gradients. The centered derivative
uses `epsilon=1e-3` and the declared tolerance
`1e-5 + 1e-2 * max(abs(a), abs(b))`.

A separate spike fixture starts from reset gates, sets voltage to `-0.001 mV`,
uses zero previous spikes, and applies input drive `20` through the same bounded
path. It requires a deterministic threshold crossing, finite gradients, and at
least one nonzero spike-path gradient. Direct readout gradients are checked for
finiteness and nonzero values in both output heads.

## Oracle boundary

The finite-difference fixture is a one-step local derivative check only. It
does not call BPTT, a multi-step VJP, or any temporal BPTT oracle. PP-Prop
relations are checked through the compiled single-step relation report and
direct finite gradients. A temporal PP-Prop result is never labelled a BPTT
gradient.

## Repeated execution

Repeated model execution uses `brainstate.transform.jit` for one step and
`brainstate.transform.for_loop` for sequences. Random values use
`brainstate.random`. No repeated model execution uses a bare Python loop.
