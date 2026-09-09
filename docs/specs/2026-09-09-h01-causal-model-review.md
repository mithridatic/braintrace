# H01 causal model review

## Purpose

Revise `docs/h01-causal-model.md` from the local Hartshorne book,
the H01 source code, and the saved evidence.
The review starts from `feat/h01-braincell` at `41eb735`.

## Scope

This change updates documents only. It does not change code, model parameters,
acceptance limits, or experimental permissions. It launches no simulation.
The existing user approval rule applies before any later code change.

## Document design

- Define the main concepts once.
- Separate physical relations, model assumptions, observations, and hypotheses.
- State the conditions for each claim. Do not present a finite test as a universal proof.
- Map each physical function to its source module.
- Retain the Y1 to Y6 identifiers, with one current account per behavior.
- Separate anatomy, construction, initialization, runtime, transfer, and human agreement.
- Keep the previous document in a dated history file with the same relative link base.
- Use short sentences and consistent technical terms, following ASD-STE100 writing rules.

## Required corrections

- The B3 source already has an M current.
- Failed Ih and leak doses do not exclude all combinations or all doses.
- A state observation does not prove that an untested current is absent or necessary.
- The SP2 accepted identity decision differs from its failed strict prediction.
- Corrected all-104 construction and initialization pass in the saved decisions.
- Corrected all-104 compiled runtime has no completed trace verdict.
- Negative calcium explains a numerical failure route on the historical component.
- Corrected component selection changes anatomy; its results need their own boundary.

## Review checks

Check each current claim against code or a saved result.
Check all new local file links, Markdown whitespace, and prose sentence length.
Check that the history body is an exact copy of the original document.
Check that only the intended documents changed.
No model tests are required because model behavior does not change.

## Evidence limits

The review reads saved decisions. It does not repeat their simulations.
Some decisions refer to local cache files or traces that are no longer tracked.
The review must not claim independent reproduction of those measurements.
Sentence checks do not certify full ASD-STE100 dictionary compliance.
