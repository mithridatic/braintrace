# Compact contraction and discretization ownership

The full default 104-cell evolution under 30 seconds remains **not achieved**.
Contraction remains opt-in. The production change removes strong global
ownership of obsolete discretization snapshots; live cells retain their own
current snapshot and the shared lookup uses weak values.

## Compact solver

Replace the unrolled contraction stages with forward/reverse numerical scans.
Retain eliminated rows in the existing coefficient arrays, avoiding a stacked
history. Padded writes are dropped. The independent sentinel row is solved
separately, with nonexistent child coefficients masked out.

Three dense forward/Jacobian fixtures and all four exported anatomical-tree
solves passed. Cold anatomical solve times were 0.439–0.875 s, versus
1.099–1.566 s for unrolled contraction. Warm solves remain 3.6–8.2x faster than
the production GPU solver on well-conditioned synthetic coefficients.

The first real four-cell compact probe completed two finite synthetic one-event
Muon updates. It retained float64, dt=0.000625 ms and 160 substeps per event.

| Measurement | Production GPU probe | Unrolled contraction | Compact scan |
| --- | ---: | ---: | ---: |
| First compile/update | 107.653 s | 139.897 s | 94.741 s |
| Warm one-event update | 24.862 s | 3.295 s | 4.750 s |
| Peak host RSS | 4,728,436 KiB | 5,579,628 KiB | 5,031,140 KiB |

The compact solver reduces the unrolled prototype's peak RSS by 9.8%, but this
sample remains 6.4% above production. These profiles are small samples on a
shared Vast RTX 4090; tests overlapped portions of the first compact profile.
This is not an isolated statistical benchmark or full ARC-quality evidence.

With weak discretization ownership, the subsequent bounded run completed with
95.622 s first compile/update, 4.770 s warm update, and peak RSS 4,920,904 KiB.
No extra tests overlapped this run; the user's original evolution remained live.
This is 11.8% below the unrolled prototype, 2.2% below the first compact run,
and still 4.1% above the earlier production probe. The two synthetic losses and
gradient norms remain finite and differ from the production probe by less than
1.9e-12 and 2.1e-13 respectively. The resource target therefore remains unmet;
the ownership fix ships independently, while contraction stays experimental.

The fixed-width index schedule still pads every stage to the largest stage.
Grouping stages by width is a remaining candidate for reducing constant and
working-buffer memory without returning to a fully unrolled solver.

## Validation and corrections

- Fifteen compact-solver tests passed with `pytest -n 2` in 19.61 s, with
  39/39 covered statements (100%; script entry-point statements excluded).
- The actual branched-cell finite-window learning oracle compares losses,
  gradients, all parameter groups, Muon state, eligibility factors and voltages
  with the scan solver at rtol=1e-10/atol=1e-11. A counter confirms the candidate
  callback executes.
- A new nonzero-sentinel test first failed, then passed with both Jacobian
  modes after treating the sentinel as an independent row. Do not generalize
  physical zero-sentinel evidence into arbitrary-RHS solver correctness.
- Two ownership regressions first failed against strong global caching. After
  the weak-cache change, the combined construction/network/compact-learning
  gate passed 39 tests in 33.94 s with xdist.

The first compact physical profile predates the general sentinel correction;
its zero sentinel follows unchanged numerical arithmetic. Its original source
is retained at `/tmp/h01-contraction-scan-as-profiled.py` on Vast. The subsequent
combined ownership profile uses the corrected implementation. Before production
solver activation, add explicit static-schedule validation and bounded cache
ownership rather than promoting the experimental driver's unrestricted cache.
