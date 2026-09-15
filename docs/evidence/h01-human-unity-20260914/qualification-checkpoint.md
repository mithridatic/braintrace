# Qualification checkpoint after rejected sodium transfer

The objective remains unmet. No six-term score improved in this checkpoint.
This is a decision record and an unsent data request, not a new experiment.

| Term | Retained ledger value | Present qualification boundary |
| --- | ---: | --- |
| donor_accuracy | 0.7178289854126294 | B3's ten-element comparison only; no human-only replacement is qualified. |
| type_coverage | 55/104 | No newly qualified donor has filled a remaining layer/class gap. |
| construction | 1 | Historical synthetic construction; must be repeated with final qualified inputs. |
| driven_window | 0 | No qualifying complete 104-cell physiological window and controls. |
| timestep | 1 | Historical isolated-cell result; not final-population convergence. |
| anatomy_transfer | 0 | Geometry repair and checks do not establish human response transfer. |

The [unchanged completion contract](../../specs/2026-09-14-h01-human-unity.md)
requires all six gates, all 104 cells and unrounded donor accuracy. A promise to
reach those values through more code or parameter fitting is not supported by
the evidence. Exact metric agreement would also not by itself establish actual
physiology of the H01 donor.

## What prevents the next promotion

The retained [ten-element donor comparison](../h01-topographic/human-vs-rodent-accuracy.json)
localizes most of B3's deficit to the rise speed and threshold accumulation:
570.417 versus 347.656 V/s for the rise, -0.0231 versus 1.3750 mV for climb2,
and 0.2358 versus 3.6250 mV for climb5. A different channel-amplitude RMSE is not
this donor score. The historical alternative donors retain unqualified biological
assumptions or fail their reproduction checks; their nominal labels do not make
them eligible human-only replacements.

The [latest executed sodium change](sodium-thermal-transfer-r2/README.md) improved
combined validation amplitude RMSE by 23.16%, but four of nine recordings worsened
by more than the registered 5% limit. It was rejected and no parameters were
installed. The two-scalar temperature-scaling branch remains closed. Its 6.384 s
runtime removes that experiment's execution bottleneck; it does not repair the
biological discrepancy.

For L1, the [initial-state repair](human-data/pax6-initial-state-840043481/README.md)
does not improve the second-pulse error. The original human currents are present;
this is an unsuccessful model, not a missing-data excuse. The state-only branch
remains closed. The retained [control analysis](human-data/pax6-controls-840043481/README.md)
also rules out its specified uniform-gain correction. It does not establish a
unique cause or exclude every possible model.

For the pyramidal potassium branch, the experimental conditioning command and
original-current measurement operator remain unresolved. The
[specific unsent request](human-channel-data-request.md) is ready for review.
Receiving those inputs could enable a new human-constrained comparison; it would
not guarantee success or close the donor/population gates.

## Execution correction

Too many supporting fits and audits were allowed to consume checkpoints while
the acceptance scores stayed fixed. Repeating the same rejected branches, widening
their bounds or moving validation observations into fitting would not address that
mistake. A new experiment needs a distinct hypothesis, its available human inputs,
a frozen rejection rule, and an explicit path to a donor qualification gate.

There is no new model run or code change in this checkpoint. No reserved external
L1 responses were opened, no human observations were removed, and no request was
sent. This record does not establish a project-wide external blocker: broader
model research remains possible, while this review found no demonstrated repair
that can presently justify a score promotion.
