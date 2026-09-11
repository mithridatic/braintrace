# H01 model vs human recordings: Current Reality Tree

A Logical Thinking Process Current Reality Tree (Dettmer 2007) for the H01
human-constrained neuron campaign, in the Reasoning Studio Document format.
Built 2026-09-11; updated the same day with the closed SP10 and SP11 results and
the SP12 draft. It records which contract rows fail, the mechanisms that have
been measured, the candidate causes that remain open, and the tests (executed
and proposed) as injections. Nothing on the map is inferred beyond the evidence
file named on each node.

## Files

| File | What it is |
| --- | --- |
| `h01-current-reality.reasoning.json` | The document (26 nodes, 27 edges). This is the source of truth; Reasoning Studio opens it directly. |
| `h01-current-reality.svg` | Export of the document. |
| `h01-current-reality.png` | Export of the document. |
| `build_map.py` | Builder. Writes the JSON, then opens, validates and exports it through the CLI. |

## How it was built

1. Read the content sources: `docs/h01-causal-model.md` (Y3, Y4, Y5, Y6,
   Qualification rules), `docs/evidence/h01-e-gain/stage-close-decision.json`,
   `docs/evidence/h01-i-sk/s0-finalist-usable.json`, the SP10 and SP11 specs,
   and (update) `docs/evidence/h01-i-sk-result.md`,
   `docs/evidence/h01-i-sk/stage-1-decision.json`,
   `docs/evidence/h01-e-currents-result.md`,
   `docs/evidence/h01-e-currents/stage-close-decision.json`,
   `docs/specs/2026-09-11-h01-e-spike-triggered-outward.md`.
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
   The CLI auto-starts a persistent engine that keeps an opened document in
   memory: after rewriting the JSON on disk, send `{"action": "shutdown"}` once
   before re-running the builder, or the exports will be of the old document
   (observed during the update: identical byte sizes, no new nodes).

Commands used:

```text
echo {"action":"shutdown"} | node dist\engine.cjs request     # from the reasoning-studio folder, only after a rebuild
python docs/evidence/h01-reasoning/build_map.py                # write, validate, export
```

Validate result: `[]` (no issues at any severity) on both the first build and
the update.

## Reading the map

- Arrows point from cause to effect, downward. `causes` edges are causal
  claims; `describes` edges carry context (what a test measured or targets,
  what constrains an open cause) and are not causal.
- Ellipses of joint sufficiency are drawn as AND junctions (`j_e`, `j_i_late`,
  `j_i_count`).
- Node status `supported` means the numbers on the node are read from the named
  file (this includes the two executed injections); `unverified` means a
  candidate cause or a test that has not run.
- Every `causes` and `describes` edge has an open relationship review
  (Cause insufficiency or Entity existence, the app's own category names) for
  the reader to close.

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
| `ie_axial` | IE1. Axial outflow 0.18058 nA, ~95% of applied current, 6 ms after the first trough at 0.19 nA; 0.00198 nA left to charge the soma; sodium availability 0.986. | `docs/h01-causal-model.md` (Y3); `docs/evidence/h01-causal-current-observations-2026-09-09.md`; `docs/evidence/h01-i-energetic-result.md` |
| `ie_sk` | IE2 (established by INJ1). Axonal SK carries the finalist's late delay: SK 0 gives 0.19 nA count 14 -> 29, last delay 80.8 -> 32.2 ms; 0.27 nA 37 -> 59, 27.2 -> 14.1 ms; 22/22 bands. Not a repair: both counts move away from the human. | `docs/evidence/h01-i-sk/stage-1-decision.json`; `docs/evidence/h01-i-sk-result.md`; `docs/h01-causal-model.md` (Y3) |
| `ie_sk_confounded` | IE3 (superseded). The earlier "no axonal SK" arm (57/138) also changed somatic NaTg and Kv3 closing factor; that confound is why INJ1 was needed. | `docs/evidence/h01-causal-current-observations-2026-09-09.md`; `docs/h01-causal-model.md` (Y3) |
| `ie_kv3` | IE4. Sodium duration and retained Kv3 activation have a supported role in the spike fall and trough; somatic Kv3 removal gave depolarisation block. | `docs/h01-causal-model.md` (Y3); `docs/evidence/h01-i-energetic-result.md` |
| `ie_passive` | IE5. Passive levers move the f-I curve and the sweep-43 return in a fixed ratio (Ih half: 8 -> 8 at 250 pA; leak x1.5: 0/0/6, return -3.06 to -3.78 mV); SK translates the curve but cannot rotate it; g3 registered, not run. | `docs/evidence/h01-e-gain/stage-close-decision.json`; `docs/evidence/h01-e-gain/stage-g-decision.json` |
| `ie_e_passthrough` | IE6 (measured by INJ2). The B3 soma is a pass-through: 96% of the 200 pA input leaves axially; soma ionic terms sum to ~-9 pA (leak -6.6, Kv3 -3.0, SK -1.7, NaTs +1.5, Nap +1.5, Im -0.1 pA). Nap (max 1.7 pA) and Im (0.2 pA) cannot carry the 20-50 pA offset; only the axial boundary is admissible. | `docs/evidence/h01-e-currents/stage-close-decision.json`; `docs/evidence/h01-e-currents-result.md` |
| `ie_pop_runtime` | IE7. Runtime stages are separate; the corrected population has construction and initialisation passes but no completed driven runtime record. | `docs/evidence/h01-104-corrected-initialization-decision.json`; `docs/h01-causal-model.md` (Y5) |
| `ie_i_late_fi` | IE8. Human late cycle ~100 ms at 0.19 nA and ~25 ms at 0.27 nA: between the two model settings at 0.27, beyond both at 0.19. Late-rate f-I: human 8.2 -> 39.4 Hz (0.39 Hz/pA, zero near 169 pA); SK-free 28.3 -> 58.5 (0.38, zero near 115 pA); finalist 12.0 -> 32.9 (0.26, zero near 144 pA). | `docs/evidence/h01-i-sk-result.md`; `docs/evidence/h01-i-sk/stage-1-decision.json` |
| `ie_e_postspike` | IE9 (measured by INJ2). At 200 pA the human sits at -67.1 mV late / -67.7 mid after its single spike; the model -64.8 late / -61.9 mid. At 110 pA with no spike they match within 0.4-0.9 mV, so the offset appears only after a spike. | `docs/evidence/h01-e-currents/stage-close-decision.json`; `docs/evidence/h01-e-currents-result.md`; `docs/specs/2026-09-11-h01-e-spike-triggered-outward.md` |
| `ie_e_threshold` | IE10. Human threshold climbs along the train (-56.4 to -52.8 mV at 310 pA; -55.8 to -54.3 at 250 pA); the model holds -57.2 mV at every input. | `docs/specs/2026-09-11-h01-e-spike-triggered-outward.md`; `docs/evidence/h01-e-currents-result.md` |
| `ie_cross` | IE11 (observed pattern, not a cause). Both humans show a higher rheobase and steeper late gain than their donor models (E 1/5/10/13 vs 4/8/10/12; I late rate zero near 169 vs 144 pA), and a threshold that climbs along the train while both models hold a fixed threshold. | `docs/evidence/h01-e-gain/stage-close-decision.json`; `docs/evidence/h01-i-sk-result.md`; `docs/evidence/h01-e-currents-result.md`; `docs/specs/2026-09-11-h01-e-spike-triggered-outward.md` |

### Root Causes (candidates that remain open)

| Id | Statement | Evidence |
| --- | --- | --- |
| `rc_vdep` | RC1. The E low-drive current is voltage- or use-dependent, outside the passive family; after INJ2 it is a current that holds the human 2-6 mV below the model after a spike, silent without a spike, not in the soma genome, carried through the cable boundary. Candidates: sAHP/KNa (SP12 draft), dendritic or axonal placement of an existing outward conductance, a second human L2/3 donor. Nap-down and Im-up excluded. | `docs/evidence/h01-e-currents/stage-close-decision.json`; `docs/evidence/h01-e-gain/stage-close-decision.json`; `docs/h01-causal-model.md` (Y4) |
| `rc_burst` | RC2. The current balance that restores early refiring is unidentified; the early burst is absent with and without axonal SK (INJ1 moved cycle 2 by -3.9 and -0.4 ms). | `docs/evidence/h01-i-sk-result.md`; `docs/evidence/h01-i-sk/stage-1-decision.json`; `docs/h01-causal-model.md` (Y3) |
| `rc_donor` | RC3. Fitted electrical properties are a donor's (Allen 541563728, kv3-closing-source library), not H01's; HL5MN1 and Allen 527952884 fits also miss recorded counts. | `docs/h01-causal-model.md` (Y6, Qualification rules); `docs/evidence/h01-e-gain/stage-close-decision.json`; `docs/evidence/h01-donors/stage-hl5mn1-decision.json`; `docs/evidence/h01-donors/stage-allen-l4-decision.json` |

### Injections

| Id | Statement | Evidence |
| --- | --- | --- |
| `inj_sp10` | INJ1 (executed, CLOSED PASS). SP10 stage 0 reproduced the container (14, 37; 12/12); stage 1 s1-sk-axon0 held 22/22 bands: delay shortened 48.7 and 13.1 ms, counts 29 and 59, cycle 2 changed -3.9 and -0.4 ms, widths/troughs/axon-first unchanged, no depolarisation block. | `docs/evidence/h01-i-sk/stage-1-decision.json`; `docs/evidence/h01-i-sk-result.md`; `docs/specs/2026-09-11-h01-i-sk-isolation.md`; `docs/evidence/h01-i-sk-manifest.json` |
| `inj_sp11` | INJ2 (stage 0 executed PASS 12/12; stage 1 registered, not spent; CLOSED). Unchanged B3 with soma currents reproduced g0-b3 exactly. Stage 1 (Nap-down / Im-up) not spent under fail-fast: neither can carry the 0.020-0.05 nA offset; the Im dose that does removes the 310 pA train. | `docs/evidence/h01-e-currents/stage-close-decision.json`; `docs/evidence/h01-e-currents-result.md`; `docs/specs/2026-09-11-h01-e-current-measurement.md`; `docs/evidence/h01-e-currents-manifest.json` |
| `inj_sp12` | INJ3 (proposed, DRAFT, not approved, not run). SP12: spike-triggered slow outward current (sAHP or KNa) on the E cell; needs a new mod file and user approval. Registered predictions: 200 pA post-spike p50 within 1 mV of -67.4 mV with count 1; then 250 pA 5-7 with cycles 4-5 of 250-350 ms, 310 pA 9-10 with late cycles within 12.8 ms of 120 ms, sweep-43 onset rows within 1 mV. | `docs/specs/2026-09-11-h01-e-spike-triggered-outward.md`; `docs/evidence/h01-e-currents/stage-close-decision.json` |

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
| ie_e_passthrough | rc_vdep | describes | soma levers excluded; cable boundary only |
| ie_e_postspike | rc_vdep | describes | offset appears only after a spike |
| ie_e_threshold | rc_vdep | describes | not supplied by sAHP/KNa alone |
| rc_donor | ude_e_lowdrive | describes | candidate: second human L2/3 donor (SP6b lead) |
| ie_cross | ude_e_lowdrive | describes | shared pattern; not a cause |
| ie_cross | ude_i_count | describes | shared pattern; not a cause |
| ie_axial | j_i_late | input | |
| ie_sk | j_i_late | input | |
| j_i_late | ude_i_latecycle | causes | SK path established by INJ1; axial path measured |
| ie_sk_confounded | ie_sk | describes | earlier confound, superseded by INJ1 |
| ie_i_late_fi | ude_i_latecycle | describes | human late cycle vs both model settings |
| rc_burst | ude_i_burst | causes | unidentified balance |
| ude_i_burst | j_i_count | input | |
| ude_i_latecycle | j_i_count | input | |
| j_i_count | ude_i_count | causes | count = cycles that fit the pulse |
| ie_kv3 | ude_i_width | causes | supported role; not a unique mechanism |
| rc_donor | ude_pop | causes | |
| ie_pop_runtime | ude_pop | causes | |
| inj_sp10 | ie_sk | describes | executed: established the SK path (22/22) |
| inj_sp10 | rc_burst | describes | early burst absent at both settings |
| inj_sp11 | ie_e_passthrough | describes | stage 0 measured this |
| inj_sp11 | ie_e_postspike | describes | stage 0 measured this |
| inj_sp11 | rc_vdep | describes | stage 1 not spent: Nap/Im cannot carry the offset |
| inj_sp12 | rc_vdep | describes | proposed test (draft, needs approval) |

## Rebuild

```text
python docs/evidence/h01-reasoning/build_map.py            # write, validate, export
python docs/evidence/h01-reasoning/build_map.py --no-export # write the JSON only
```

If the JSON changed since the engine last opened it, send the engine a
`{"action":"shutdown"}` request first (see above).
