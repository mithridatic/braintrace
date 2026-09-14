# H01 constructor preview optimization

The complete default 104-cell evolution in 30 seconds is still not achieved.
This change removes discarded construction work; it does not alter dt, precision,
compartments, pp-prop, training updates or the full-workload acceptance criterion.

BrainCell 0.1.0 ends Cell.__init__ by requesting cvs solely for eager validation.
H01 painting invalidates this initial default discretization. H01Cell now resolves
the policy with default paint rules and builds validated geometry during that
discarded access, deferring mechanism/CV/node assembly until a real preview or
initialization is requested. The constructor flag is cleared even on failure.
Public cvs behavior after construction is unchanged. This relies on the installed
parent constructor discarding that validation result, covered by a regression test.

## Matched bounded profiles

Vast RTX 4090, four anatomical cells, 75,605 compartments, float64, dt=0.000625 ms,
160 substeps/event. Both profiles ran sequentially without the old evolution job
or tests competing for GPU resources. Baseline is cad955c; source identities and
settings are recorded in adjacent report.json files. Each is one cProfile sample,
not a statistical benchmark or complete ARC episode.

| Phase | Baseline | Changed |
| --- | ---: | ---: |
| Construction | 32.899 s | 25.899 s |
| Initialization | 24.120 s | 20.571 s |
| Combined startup | 57.019 s | 46.470 s |
| Cold event | 7.918 s | 8.646 s |
| Warm event | 0.392485 s | 0.392132 s |
| Peak host RSS | 2,299,464 KiB | 2,296,436 KiB |

Combined construction/initialization was 18.5% faster. Peak host RAM differed by
only 0.13%, so no material RAM reduction is claimed. Largest recorded event-voltage
difference was 9.678e-12 mV. Runtime and ARC quality are not established by this
forward-only probe. Startup alone still exceeds the full-command target.

## Validation

The new regression failed before implementation because the constructor built a
node tree; eager policy failure propagation already passed. After implementation,
33 tests passed in 33.44 seconds with pytest -n 2 across h01_construction_test.py,
h01_dhs_gpu_test.py and h01_network_step_test.py. This includes full public-preview
parity, voltage parity with BrainCell, cache ownership, batching, solver checks,
and branched-cell sparse pp-prop parameter/optimizer/eligibility parity. All 13
executable statements reported in the added constructor/property range were
covered; no whole-module 100% coverage claim is made.

Raw cProfile files and coverage JSON remain on Vast under /tmp/h01-startup-*
and /tmp/h01-startup-coverage.json. An SSH observation disconnected after baseline
completion; its terminal forward_pass report and absence of a live profiling
process were inspected before accepting the result. The command was not restarted.
