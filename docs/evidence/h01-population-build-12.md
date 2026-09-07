# H01 population: 12-cell construction check (SP8 stage 2)

Generated 2026-09-07T15:40 from [h01-population-build-12.json](h01-population-build-12.json).
Construction only (`--build`, no run), `--include-isolated --cells 12 --control ei`, the
evidence topology (2 construction-ready contacts, 4 incident cells) plus the 8 smallest isolated
cells by largest-component node count. **Measured under load** (other jobs were running).
Cost and construction qualification only; no physiology claim.

| Quantity | Value |
| --- | --- |
| Status | **failed** at 332.6 s wall (limit 900 s; not killed) |
| Construction seconds | untested (build did not complete) |
| Compartments | untested (registration stage never reached) |
| Peak RSS | 649 MB at failure (before discretization, not comparable to the SP1 1,350 MB peak) |
| Cells built before failure | 4 incident + 3 isolated (`2530864375`, `3178243558`, `3751392341`) |
| Failing cell | `2451406889`, isolated, largest component 0 of 3,750 nodes (fraction 0.970), 8th of 12 |

Failure: `ValueError("Invalid electrical region interval.")` in `h01_ei_cell._validate_regions`.
A read-only evaluation of that component afterwards found one soma interval on branch 162 of
`(0.9902931374228255, 1.0000000000000002)`: floating-point rounding past 1 by 2e-16, rejected by
the strict `hi <= 1` test while the same validator's coverage test tolerates 1e-9. Fixed in this
commit by snapping ends within 1e-9 of 0 or 1 onto the branch bounds (a rounding fix; no region,
channel, dt, solver, or synapse change) with a reproducing unit test. The build was not re-run
here (one construction check allowed); the next 12-cell check is the first task of the next
SP8 step.

Partial timing under load: the four incident cells finished `make_h01_ei_cell` by 318 s (SP1
alone: 239-280 s for the whole 4-cell construction); the three small isolated cells took
1.5-2.2 s each.

## Derived projections (SP1 proportional rule; not validated by this check)

| Stage | Largest-component nodes | Ratio to 4 cells | Construction s | Peak RSS GB |
| --- | ---: | ---: | ---: | ---: |
| 4 (measured, SP1) | 245,769 | 1.00 | 239-280 | 1.35 |
| 12 (derived) | 276,064 | 1.12 | ~275 | ~1.5 |
| 40 (derived) | 537,152 | 2.19 | ~540 | ~3.0 |
| 104 (derived) | 2,804,549 | 11.4 | ~2,800 | ~15.4 |

Every derived row is proportional extrapolation; the 12-cell measurement that was to anchor it
is untested, so 40 and 104 remain unapproved.
