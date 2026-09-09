# Sparse multicompartment pp-prop extension

## Intent and correction

Continue the approved 104-cell ARC lifecycle through its learning blocker. The
previous forward-only closeout was premature: compiler rejection identifies work
to implement, not completion of the requested training/evolution lifecycle.

The current IO factorization requires output and hidden positions to coincide.
H01 instead maps one current to many cable/channel states and placed contacts to
delay queues and receptor states. Shape filtering must not discard these paths.
Removing the position-preservation guard without changing representation and
contractions is incorrect.

## Representation and execution

Add an explicit sparse influence representation: each state block stores the ETP
output positions that can affect it, and an output-side factor for each such
position. Derive conservative support from the actual transition program; an
unknown operation widens support, never deletes it. Preserve constant indexing
and per-cell separation where provable. Analyze scans through their bodies and
carry dependencies. Report support width, factor elements and byte estimates
before allocation, with a hard configurable ceiling.

Compute instantaneous output-to-state derivatives with JVPs and contract the
learning signal back to the original ETP output positions before applying the
registered parameter VJP rule. Retain input-side filtering, output-side smoothing,
bias correction and direct readout gradients. Propagate output factors through
the actual cable transition with matrix-free JVPs; do not construct a dense
compartment-squared Jacobian or replace the cable state by soma voltage. Sparse
support must close over temporal dependencies, including delivery queues.

The extension is explicit and leaves existing pp-prop defaults and rejection
tests intact. It is still an IO-factorized approximation: exactness is claimed
only in guaranteed regimes. Unsupported analysis must fail before training, with
the implicated operation or allocation reported.

## Staged validation and continuation

First reproduce mismatched-shape and mixed-position tail failures with sibling
tests. Test conservative support on indexing, branching, repeated scans, disconnected
blocks, recurrent coupling, empty support and allocation limits. Measure derivative
columns against AD on small systems and learning-rule behavior through finite-window
`chunked_online_param_gradients`, including controlled descent and nonzero input/contact
updates on actual cable cells. Padding and reset must preserve/clear eligibility
along with all physical state.

After the small real-cell learning gates pass, progress through bounded population
allocation/learning probes, then complete the original evolution, checkpoint and
104-cell encoded ARC episode gates. Retain the existing 15-minute approval rule;
the prior 104-cell forward approval does not authorize an unbounded learning run.
Initial topology remains two anatomical contacts unless the user chooses the
separately proposed synthetic scaffold. Source E/I identity and provenance remain
immutable through either topology choice.

## Derivative working memory

The first four-source-cell compiled update completed, but its second call failed
requesting a 10.0 GiB temporary. Factor allocation alone is therefore not an
adequate runtime-memory gate. Preserve the forward DHS equations and ordering,
but differentiate its linear system implicitly with the same tree solve for
tangents and the transposed tree solve for adjoints. Verify both derivatives
against a dense nonsymmetric system. Rematerialize cable-event substeps where
needed; verify forward equality and retain measured executable working memory.
An incoming floating state proven unread by the compiled program has no temporal
influence. Preserve its native updates and resets but omit its redundant factor
block, so overwritten aggregate drive vectors do not force disconnected cells
into separate derivative colors. Retain all read cable, channel and queue state.
