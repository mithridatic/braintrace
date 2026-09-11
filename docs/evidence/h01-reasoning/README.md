# H01 model vs human recordings: Current Reality Tree

A Logical Thinking Process Current Reality Tree (Dettmer 2007) for the H01
human-constrained neuron campaign, built 2026-09-11 in the Reasoning Studio
Document format. It records which contract rows fail, the mechanisms that have
been measured, the candidate causes that remain open, and the two running tests
as proposed injections. Nothing on the map is inferred beyond the evidence file
named on each node.

## Files

| File | What it is |
| --- | --- |
| `h01-current-reality.reasoning.json` | The document (21 nodes, 20 edges). This is the source of truth; Reasoning Studio opens it directly. |
| `h01-current-reality.svg` | Export of the document. |
| `h01-current-reality.png` | Export of the document. |
| `build_map.py` | Builder. Writes the JSON, then opens, validates and exports it through the CLI. |

## How it was built

1. Read the content sources: `docs/h01-causal-model.md` (Y3, Y4, Y5, Y6,
   Qualification rules), `docs/evidence/h01-e-gain/stage-close-decision.json`,
   `docs/evidence/h01-i-sk/s0-finalist-usable.json`,
   `docs/specs/2026-09-11-h01-i-sk-isolation.md` and
   `docs/specs/2026-09-11-h01-e-current-measurement.md`.
2. `build_map.py` writes the Document as a full JSON file (custom class names
   such as `Undesirable Effect` only work via a full document or `set_graph`,
   never through incremental commands). The builder checks that every evidence
   path named on a node exists in the repository.
3. The Reasoning Studio CLI is driven without the desktop app. Each request is a
   JSON payload on stdin to `node reasoning-studio\dist\engine.cjs request`, run
   from `C:\Users\J\Documents\Projects\reasoning-studio`:

   ```text
   {"action": "open",     "path": "<abs path>\\h01-current-reality.reasoning.json"}
   {"action": "validate", "document": "<id returned by open>"}
   {"action": "export",   "document": "<id>", "path": "<abs path>\\h01-current-reality.svg", "format": "svg"}
   {"action": "export",   "document": "<id>", "path": "<abs path>\\h01-current-reality.png", "format": "png"}
   ```

   Export does not create directories, so the builder creates this folder first.
   Do not edit the JSON on disk while a session has it open (EXTERNAL_CHANGE).

Command used:

```text
python docs/evidence/h01-reasoning/build_map.py
```

Validate result: `[]` (no issues at any severity).

## Reading the map

- Arrows point from cause to effect, downward. `causes` edges are causal
  claims; `describes` edges carry context (why a cause is still open, what a
  test targets) and are not causal.
- Ellipses of joint sufficiency are drawn as AND junctions (`j_e`, `j_i_late`,
  `j_i_count`).
- Node status `supported` means the numbers on the node are read from the named
  file; `unverified` means a candidate cause or a test that has not run.
- Every `causes` and `describes` edge has an open relationship review
  (Cause sufficiency or Entity existence) for the reader to close.

## Node list

Evidence paths are repository-relative; each node also carries them as
Evidence entries with file URIs.

### Undesirable Effects (contract rows that fail)

| Id | Statement | Evidence |
| --- | --- | --- |
| `ude_e_lowdrive` | E cell (frozen B3): counts 4 and 8 at 200 and 250 pA vs human 1 and 5; 310 pA 10/10; 350 pA 12 vs 13. | `docs/evidence/h01-e-gain/stage-close-decision.json`; `docs/h01-causal-model.md` (Y4) |
| `ude_i_count` | I cell finalist: 14 vs 12 at 0.19 nA; 37 vs 43 at 0.27 nA; 26 vs 31 at 0.23 nA. | `docs/evidence/h01-i-sk/s0-finalist-usable.json`; `docs/evidence/h01-i-reserve/stage-1-decision.json` |
| `ude_i_burst` | Early burst absent: human cycles 2-4 at 0.27 nA are 6.3/5.9/7.0 ms; model cycles start 13.6 ms and lengthen monotonically. | `docs/evidence/h01-i-sk/s0-finalist-usable.json`; `docs/h01-causal-model.md` (Y3) |
| `ude_i_latecycle` | At 0.27 nA the human settles to ~25 ms cycles (24.2-26.7); the model drifts to 30.4 ms. At 0.19 nA the sign differs (human 96-117, model 82-84 ms). | `docs/evidence/h01-i-sk/s0-finalist-usable.json` |
| `ude_i_width` | Widths ~0.06 ms narrow: model 0.221-0.224 ms vs human 0.267-0.300 ms. | `docs/evidence/h01-i-sk/s0-finalist-usable.json`; `docs/evidence/h01-multivari-i.png`; `docs/h01-causal-model.md` (Y3) |
| `ude_pop` | No H01 physiology qualified; corrected 104-cell population built and initialised but the compiled 10 ms runtime ended without a trace verdict. | `docs/evidence/h01-104-corrected-initialization-decision.json`; `docs/h01-causal-model.md` (Y5, Qualification rules) |

### Intermediate Effects (measured mechanisms)

| Id | Statement | Evidence |
| --- | --- | --- |
| `ie_axial` | Axial outflow 0.18058 nA, ~95% of applied current, 6 ms after the first trough at 0.19 nA; 0.00198 nA left to charge the soma; sodium availability 0.986. | `docs/h01-causal-model.md` (Y3); `docs/evidence/h01-causal-current-observations-2026-09-09.md`; `docs/evidence/h01-i-energetic-result.md` |
| `ie_sk` | Along the 0.23 nA train axonal calcium 0.000120 to 0.000199 mM, axonal SK gate 0.0022 to 0.024, trough-to-threshold delay 17.6 to 40.3 ms. | `docs/specs/2026-09-11-h01-i-sk-isolation.md`; `docs/h01-causal-model.md` (Y3); `docs/evidence/h01-causal-current-observations-2026-09-09.md` |
| `ie_sk_confounded` | The earlier "no axonal SK" arm (57/138) also changed somatic NaTg and Kv3 closing factor; no isolated SK effect exists for the finalist. | `docs/evidence/h01-causal-current-observations-2026-09-09.md`; `docs/h01-causal-model.md` (Y3) |
| `ie_kv3` | Sodium duration and retained Kv3 activation have a supported role in the spike fall and trough; somatic Kv3 removal gave depolarisation block. | `docs/h01-causal-model.md` (Y3); `docs/evidence/h01-i-energetic-result.md` |
| `ie_passive` | Passive levers move the f-I curve and the sweep-43 return in a fixed ratio (Ih half: 8 -> 8 at 250 pA, rest -1.73 mV; leak x1.5: 0/0/6, return -3.06 to -3.78 mV); SK translates the curve but cannot rotate it; g3 registered, not run. | `docs/evidence/h01-e-gain/stage-close-decision.json` (`established`, `arms`); `docs/evidence/h01-e-gain/stage-g-decision.json` |
| `ie_b3_notrace` | Saved B3 traces hold voltage only (`record_soma_currents` false); the current carrying the three excess spikes has never been measured; B3 already contains Im at 0.0003009 S/cm2. | `docs/specs/2026-09-11-h01-e-current-measurement.md`; `docs/h01-causal-model.md` (Y4) |
| `ie_pop_runtime` | Runtime stages are separate; the corrected population has construction and initialisation passes but no completed driven runtime record; the timeout has no stage record. | `docs/evidence/h01-104-corrected-initialization-decision.json`; `docs/h01-causal-model.md` (Y5, Qualification rules) |

### Root Causes (candidates that remain open)

| Id | Statement | Evidence |
| --- | --- | --- |
| `rc_vdep` | The E low-drive current must be voltage- or use-dependent, outside the passive family; candidates Nap / slow sodium inactivation, Kv7/M kinetics, a second human L2/3 donor. | `docs/evidence/h01-e-gain/stage-close-decision.json` (`unresolved_link`, `reassessment`); `docs/h01-causal-model.md` (Y4) |
| `rc_burst` | The current balance that restores early refiring and the later I train changes is unidentified. | `docs/h01-causal-model.md` (Y3 Open); `docs/specs/2026-09-11-h01-i-sk-isolation.md` |
| `rc_donor` | Fitted electrical properties are a donor's (Allen 541563728, kv3-closing-source library), not H01's; HL5MN1 and Allen 527952884 fits also miss recorded counts. | `docs/h01-causal-model.md` (Y6, Qualification rules); `docs/evidence/h01-e-gain/stage-close-decision.json` (`what_a_user_gets`); `docs/evidence/h01-donors/stage-hl5mn1-decision.json`; `docs/evidence/h01-donors/stage-allen-l4-decision.json` |

### Injections (running tests, pre-registered)

| Id | Statement | Evidence |
| --- | --- | --- |
| `inj_sp10` | SP10: s0-finalist then s1-sk-axon0 at 0.19 and 0.27 nA. Prediction: late delay shortens > 12.8 ms at both inputs, counts rise above 14 and 37, cycle 2 moves < 12.8 ms. Rejection: no shortening, lost spikes, or depolarisation block. | `docs/specs/2026-09-11-h01-i-sk-isolation.md`; `docs/evidence/h01-i-sk-manifest.json` |
| `inj_sp11` | SP11: stage 0 records unchanged B3 currents at 200/250/310 pA (counts 4, 8, 10 within 0.1 ms / 0.1 mV, axon-first); stage 1 one lever (Nap down or Im up) with bands 200 pA 1-2, 250 pA 5-7, 310 pA 9-10 within 1.5 Hz, sweep-43 rows < 1 mV. | `docs/specs/2026-09-11-h01-e-current-measurement.md`; `docs/evidence/h01-e-currents-manifest.json` |

### Junctions

`j_e` (rc_vdep AND ie_passive -> ude_e_lowdrive), `j_i_late` (ie_axial AND
ie_sk -> ude_i_latecycle), `j_i_count` (ude_i_burst AND ude_i_latecycle ->
ude_i_count).

## Edges

| Source | Target | Relation | Label |
| --- | --- | --- | --- |
| rc_vdep | j_e | input | |
| ie_passive | j_e | input | |
| j_e | ude_e_lowdrive | causes | candidate cause; passive family excluded |
| ie_b3_notrace | rc_vdep | describes | why the current is still unidentified |
| rc_donor | ude_e_lowdrive | describes | candidate: second human L2/3 donor (SP6b lead) |
| ie_axial | j_i_late | input | |
| ie_sk | j_i_late | input | |
| j_i_late | ude_i_latecycle | causes | measured together; not isolated |
| ie_sk_confounded | ie_sk | describes | why SK is not isolated |
| rc_burst | ude_i_burst | causes | unidentified balance |
| ude_i_burst | j_i_count | input | |
| ude_i_latecycle | j_i_count | input | |
| j_i_count | ude_i_count | causes | count = cycles that fit the pulse |
| ie_kv3 | ude_i_width | causes | supported role; not a unique mechanism |
| rc_donor | ude_pop | causes | |
| ie_pop_runtime | ude_pop | causes | |
| inj_sp10 | ie_sk | describes | tests whether axonal SK is the direct path |
| inj_sp10 | rc_burst | describes | prediction excludes the early refire |
| inj_sp11 | ie_b3_notrace | describes | stage 0 removes this gap |
| inj_sp11 | rc_vdep | describes | stage 1 tests Nap or Im |

## Rebuild

```text
python docs/evidence/h01-reasoning/build_map.py            # write, validate, export
python docs/evidence/h01-reasoning/build_map.py --no-export # write the JSON only
```
