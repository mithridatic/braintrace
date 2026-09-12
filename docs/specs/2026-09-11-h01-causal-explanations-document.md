# H01 causal explanation document correction

## Scope

Revise `docs/h01-causal-model.md` on `campaign/h01-currents-2026-09-11`
in the main checkout, as requested. Evidence baseline: `56240d4`, including
SP10, the SP11 whole-soma erratum, and approved SP12 registration.
This is a documentation correction; it does not run or change a campaign.

The user's subsequent instruction authorizes redoing diagnostic work on Vast when
aggregates or unsupported inference undermine the explanation. Reanalyze the saved
B3 trace there, preserving per-time observations and provenance. Check SP12's active
status before any execution so an existing run is not duplicated. New simulation
is needed only if the required observation is absent; this audit can establish
the aggregate/window defects from retained data without another model run.

## Required result

Lead with the explanations of specific observed responses. For each explanation,
state the system and operating conditions, the mechanism linking those conditions
to the response, the supporting observation or intervention, and what remains open.
Distinguish an isolated model mechanism, a measured current balance, a numerical
failure route, a proposed mechanism, and an unexplained response difference.

Preserve per-input and per-cycle contrasts, spatial boundaries, current signs,
clock conventions, and links to the records. Counts and voltage means may summarize
an observation or size a registered dose; they cannot establish a mechanism.
Apply the SP11 erratum ahead of older statements in its decision files. Do not turn
unrun Nap/Im arms or the registered SP12 prediction into observed results. Explain
the actual smooth KsAHP gate rather than claiming an exact spike switch.

Retain the existing Y1-Y6 section anchors and qualification boundaries. Put the
general equations, code map, and diagnostic guidance after the response explanations.
Cite Hartshorne's definition and diagnostic distinction using the local page scans.
Treat the older Reasoning Studio graph as a secondary map where it predates SP11's
erratum and SP12 approval.

## Verification

Review every changed numerical claim against the saved decision, manifest, trace
summary, or mechanism definition. Check local Markdown links, duplicate headings,
Git diff whitespace, and the final changed-file scope. No simulation or new software
test is needed for this documentation-only change. Unresolved physiological claims
must remain explicitly unresolved; failed tested doses must not exclude a family.
Save the direct trace audit as JSON and Markdown with a plot, verify the local plot
inputs against the hashes recorded by the Vast reanalysis, and inspect the plot.
