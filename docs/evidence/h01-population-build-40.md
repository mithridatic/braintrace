# H01 population: 40-cell staged build (SP8 stage 3) — stopped after a failed construction

Generated 2026-09-07T22:12 from [h01-population-build-40.json](h01-population-build-40.json). One build-only launch
(`--build --include-isolated --cells 40 --control ei --current-na 1`, solver `h01_staggered_scan`, merged tree 0488c84) was made after
the registered equivalence step held ([pinned split](h01-network-equivalence-post-merge-pinned.json)). It **failed at
82.6 s wall** (peak RSS 563 MB, 36 cells loaded, none discretized). The user then stopped all
remaining runs: stages 2-5 (init-only, 1 ms ei, 1 ms disconnected, 3 ms ei) are **untested**. No 40-cell number exists; nothing
below is measured at 40 cells. Load label: measured beside one NEURON container (Stage G); system CPU mean 36 % of 20 logical cores; other processes above 0.25 cores: SearchIndexer.exe 4.88 cores; WindowsTerminal.exe 1.41 cores; System 1.65 cores; dwm.exe 1.40 cores.

| Run | Status | Construction s | Compartments | init_state s | compile+run s | Wall s | Peak RSS MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| build-only, control ei | failed (ValueError in H01Archive.load) | untested | untested | - | - | 82.6 | 563 |
| init-only | untested (not launched) | - | - | - | - | - | - |
| 1 ms, dt 0.005, ei | untested (not launched) | - | - | - | - | - | - |
| 1 ms, dt 0.005, disconnected | untested (not launched) | - | - | - | - | - | - |
| 3 ms, dt 0.005, ei | untested (not launched) | - | - | - | - | - | - |

## Failure

Failing cell: 5805562981 (isolated, largest component 0 of 10,603 nodes, fraction 0.720; 30th of 40 in cell_order). Exception: `ValueError: from_points() requires at least two points. (braincell.morph.branch.Branch.from_points, reached from braincell.io.swc.reader._make_branch via H01Archive.load -> Morphology.from_swc(mode='neuromorpho'))`.

```
Traceback (most recent call last):
  ...
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\J\Documents\Projects\connectome-agent\braintrace-source\.venv\Lib\site-packages\braincell\morph\branch.py", line 375, in from_points
    raise ValueError("from_points() requires at least two points.")
ValueError: from_points() requires at least two points.
```

Diagnosis (no fix committed): Three leaf nodes of that component (original ids 3329, 7348, 7900) sit exactly one skeleton voxel (0.032 um) from a parent that is a branch point (two children) with the same radius. BrainCell's SWC reader decides whether to copy the attach point into the child branch with np.allclose (rtol 1e-5), which at coordinates near 3,000 um tolerates about 0.03 um, so the leaf is judged coincident with its parent and the child branch is built from a single point. braintrace's _h01_swc.normalize keeps every node as released (it invents no repairs), so nothing removed the zero-length twig.

Scope: Offline load of all 104 largest components with the merged code (H01Archive.load only; no simulation) reproduces the failure on 7 of 104: 5965472721 (13th in cell_order), 5805562981 (30th), 4365276903 (58th), 3111823553 (68th), 5687162964 (73rd), 4641147055 (90th), 5136107765 (92nd). Two of them are inside the 40-cell prefix; the 12-cell prefix is unaffected (it built three times today). In 5965472721 the coincident node is itself a branch point (two children), not a leaf, so a leaf-only repair would not cover it.

The user stopped all runs; no repair was committed. A candidate repair (collapse a node that is within the reader's tolerance of a branch-point parent, has the same radius and has 0 or >= 2 children) was drafted and reverted untested. The reader tolerance is braincell's; the 12-cell build passed on the merged tree in the equivalence reruns, and the failing components were never constructed before.

## Predictions registered for this stage (addendum 18:35) — all untested

Compartments ~163,600; construction ~590 s; init_state ~450 s; compile + 1 ms ~60 s; peak RSS ~3.8 GB; finite traces; disconnected
probes zero. None was reached. The "inside 2x" test has no measurement to apply to.

## 104-cell projection (derived from the 12-cell point only)

| Stage | Nodes | Ratio to 12 | Construction s | init_state s | compile+run (1 ms) s | Peak RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 (derived) | 537,152 | 1.946 | 590 | 450 | 60 | 3.80 |
| 104 (derived) | 2,804,549 | 10.16 | 3,094 | 2,370 | 302 | 19.84 |

derived from the quiet-machine 12-cell point (attempt 3) by the proportional rule 2,804,549 / 276,064 = 10.16x; not measured; no 40-cell measurement exists to re-derive it. not evaluable from a 40-cell measurement; from the 12-cell point the derived 104-cell RSS (19.8 GB) exceeded the headroom rule at 38.9 GB available RAM (psutil.virtual_memory, 18:30), so the clause is live; no N is proposed while 7 of 104 components fail to load.

**Population deliverable: the measured 12-cell network (h01-population-build-12.json, attempt 3); 40 and 104 remain derived.**
