"""Build, validate and export the H01 Current Reality Tree with the Reasoning Studio CLI.

Writes ``h01-current-reality.reasoning.json`` next to this script (Reasoning Studio
Document format, see reasoning-studio/packages/core/model.ts), then opens, validates and
exports it to SVG and PNG through ``node reasoning-studio/dist/engine.cjs request``.
No app is needed. Run with ``python build_map.py`` from any directory.

Every node carries the path of its evidence file (repo-relative, in ``notes`` and as an
Evidence entry with a file URI). Numbers are copied from those files; nothing is inferred.
"""
from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "h01-current-reality.reasoning.json"
STUDIO = Path(r"C:\Users\J\Documents\Projects\reasoning-studio")
ENGINE = STUDIO / "dist" / "engine.cjs"
DATE = "2026-09-12"
ORIGIN = f"H01 campaign Current Reality Tree; built {DATE} from the evidence files named on each node"
TITLE = "H01 model vs human recordings"

# Evidence files (repo-relative). Each is checked to exist at build time.
CM = "docs/h01-causal-model.md"
E_CLOSE = "docs/evidence/h01-e-gain/stage-close-decision.json"
E_STAGE_G = "docs/evidence/h01-e-gain/stage-g-decision.json"
I_S0 = "docs/evidence/h01-i-sk/s0-finalist-usable.json"
I_CURRENTS = "docs/evidence/h01-causal-current-observations-2026-09-09.md"
I_ENERGETIC = "docs/evidence/h01-i-energetic-result.md"
I_RESERVE = "docs/evidence/h01-i-reserve/stage-1-decision.json"
I_CHART = "docs/evidence/h01-multivari-i.png"
POP_INIT = "docs/evidence/h01-104-corrected-initialization-decision.json"
DONOR_HL5 = "docs/evidence/h01-donors/stage-hl5mn1-decision.json"
DONOR_L4 = "docs/evidence/h01-donors/stage-allen-l4-decision.json"
SP10 = "docs/specs/2026-09-11-h01-i-sk-isolation.md"
SP11 = "docs/specs/2026-09-11-h01-e-current-measurement.md"
SP10_MAN = "docs/evidence/h01-i-sk-manifest.json"
SP11_MAN = "docs/evidence/h01-e-currents-manifest.json"
SP10_RESULT = "docs/evidence/h01-i-sk-result.md"
SP10_S1 = "docs/evidence/h01-i-sk/stage-1-decision.json"
SP11_RESULT = "docs/evidence/h01-e-currents-result.md"
SP11_CLOSE = "docs/evidence/h01-e-currents/stage-close-decision.json"
SP12 = "docs/specs/2026-09-11-h01-e-spike-triggered-outward.md"
SP12_DEC = "docs/evidence/h01-e-sahp/stage-0-reanalysis.json"
SP12_RESULT = "docs/evidence/h01-e-sahp-result.md"
SP12_MAN = "docs/evidence/h01-e-sahp-manifest.json"
SP13 = "docs/specs/2026-09-12-h01-e-sahp-tail.md"
SP13_MAN = "docs/evidence/h01-e-sahp-tail-manifest.json"

COLORS = {
    "Undesirable Effect": "#b35f48",
    "Intermediate Effect": "#697b8a",
    "Root Cause": "#9b6947",
    "Injection": "#4f76a3",
    "operator": "#697b8a",
}


def uri(rel: str) -> str:
    return "file:///" + str(REPO / rel).replace("\\", "/").replace(" ", "%20")


def node(nid: str, ntype: str, text: str, notes: str, files: list[str], status: str) -> dict:
    for f in files:
        if not (REPO / f).exists():
            raise FileNotFoundError(f"{nid}: evidence file missing: {f}")
    return {
        "id": nid,
        "type": ntype,
        "text": text,
        "status": status,
        "confidence": None,
        "notes": notes + "\n\nEvidence: " + "; ".join(files),
        "evidence": [
            {
                "id": f"{nid}_ev{i}",
                "title": f,
                "text": "Source file for this node; numbers on the node are copied from it.",
                "uri": uri(f),
                "stance": "supports" if status == "supported" else "context",
                "origin": f"repository file inspected {DATE}",
            }
            for i, f in enumerate(files)
        ],
        "origin": ORIGIN,
    }


def junction(nid: str) -> dict:
    j = node(
        nid,
        "operator",
        "All of these hold together",
        "Dettmer ellipse (joint sufficiency) drawn as the app's AND junction. No probability is computed.",
        [],
        "unverified",
    )
    j["operator"] = "AND"
    j["mode"] = "boolean"
    return j


def edge(src: str, dst: str, relation: str, label: str | None = None) -> dict:
    e = {"id": f"{src}__{dst}", "source": src, "target": dst, "relation": relation}
    if label:
        e["label"] = label
    return e


def build_nodes() -> list[dict]:
    UDE, IE, RC, INJ = "Undesirable Effect", "Intermediate Effect", "Root Cause", "Injection"
    return [
        # ---------------------------------------------------------------- UDEs
        node("ude_e_lowdrive", UDE,
             "UDE1. E cell (frozen B3): low-drive spike counts are 4 and 8 at 200 and 250 pA against the human 1 and 5; 310 pA matches (10/10); 350 pA is 12 vs 13.",
             "Count error changes sign across inputs (+3, +3, 0, -1), so no constant offset describes it. Usable-tier verdicts: 200 pA count FAIL; 250 pA count and rate (7.73 vs 4.79 Hz) FAIL; 310 pA count and rate pass. B3 stays unpromoted.",
             [E_CLOSE, CM], "supported"),
        node("ude_i_count", UDE,
             "UDE2. I cell (Kv3 close factor 2.0 finalist): counts are 14 vs human 12 at 0.19 nA and 37 vs human 43 at 0.27 nA.",
             "Rates 14.63 vs 12.19 Hz (limit 1.83) and 36.58 vs 43.63 Hz: both rate rows fail. At 0.23 nA (reserved input) the finalist gave 26 spikes against 31.",
             [I_S0, I_RESERVE], "supported"),
        node("ude_i_burst", UDE,
             "UDE3. I cell: the human early burst is absent. At 0.27 nA the human fires cycles 2-4 at 6.3, 5.9 and 7.0 ms (cycle 5 at 12.1 ms); the model's cycles start at 13.6 ms and lengthen monotonically (16.4, 17.1, 17.9 ms).",
             "At 0.19 nA the same pattern: human cycles 2-3 are 7.6 and 10.2 ms; model cycles 2-3 are 34.8 and 39.2 ms. Values are cycle_ms from the usable-tier tables.",
             [I_S0, CM], "supported"),
        node("ude_i_latecycle", UDE,
             "UDE4. I cell at 0.27 nA: the human train settles to a ~25 ms plateau (24.2-26.7 ms over the last five cycles); the model drifts to a 30.4 ms plateau.",
             "At 0.19 nA the sign differs: human late cycles are 96-117 ms, the model's 82-84 ms. The model's steady late cycle is therefore not a fixed offset from the human's.",
             [I_S0], "supported"),
        node("ude_i_width", UDE,
             "UDE5. I cell spike widths are about 0.06 ms narrow: model 0.221-0.224 ms against human 0.267-0.300 ms at both inputs.",
             "Usable-tier width rows fail on cycles 2-12 at 0.19 nA and cycles 23-37 at 0.27 nA (the earlier 0.27 nA cycles pass on the wider limit). The I response chart shows the same ~0.06 ms difference and a different rise-rate change along the train.",
             [I_S0, I_CHART, CM], "supported"),
        node("ude_pop", UDE,
             "UDE6. Population: no H01 physiology is qualified. The corrected 104-cell population built (807,588 compartments, two contacts) and initialised, but the compiled 10 ms runtime ended without a trace verdict.",
             "H01 donor qualification would need matching H01 physiological evidence that this project does not have (Qualification rules). The single-cell fits use donor anatomy in NEURON, not an H01 transfer test.",
             [POP_INIT, CM], "supported"),
        # ------------------------------------------------------ intermediates
        node("ie_axial", IE,
             "IE1. I cell: axial outflow carries 0.18058 nA, about 95% of the applied current, at 6 ms after the first trough at 0.19 nA; the signed balance leaves 0.00198 nA to charge the soma. Sodium availability there is 0.986.",
             "All eight recorded soma neighbours sit below the soma voltage. Recovered sodium therefore does not imply rapid soma charging. The same path appears in early and late samples at 0.19 and 0.23 nA. Downstream cable currents remain unresolved.",
             [CM, I_CURRENTS, I_ENERGETIC], "supported"),
        node("ie_sk", IE,
             "IE2 (established by INJ1). I cell: the calcium-activated axonal SK current carries the finalist's late delay. With axonal SK density zero as the single change: 0.19 nA count 14 -> 29, last trough-to-threshold delay 80.8 -> 32.2 ms; 0.27 nA count 37 -> 59, delay 27.2 -> 14.1 ms; 22 of 22 bands held.",
             "Along the finalist's 0.23 nA train the axonal SK gate rises 0.0022 to 0.024 as the delay grows 17.6 to 40.3 ms. Without SK the cycle stops growing after cycle 2; widths (0.221 ms), peaks, troughs 1-3 (within 0.04 mV) and axon-first initiation are unchanged; no depolarisation block. This is a mechanism result, not a repair: both counts move away from the human (12, 43).",
             [SP10_S1, SP10_RESULT, CM], "supported"),
        node("ie_sk_confounded", IE,
             "IE3 (superseded). I cell: the earlier 'no axonal SK' arm (57/138 spikes) was not a single-change comparison; it also returned somatic NaTg to 1.0 and used Kv3 closing factor 0.5 (the finalist uses 2.0).",
             "That confound is why the SK path needed INJ1. Increasing axonal NaTg density 1.5x failed its registered recovery prediction; that dose does not exclude all axonal density changes.",
             [I_CURRENTS, CM], "supported"),
        node("ie_i_late_fi", IE,
             "IE8. I cell late cycle and late-rate f-I: the human's late cycle is ~100 ms at 0.19 nA (96-117) and ~25 ms at 0.27 nA (24-26); the finalist gives 84 and 30.4 ms, the SK-free model 33-35 and 17.1 ms. The human lies between the two model settings at 0.27 nA and beyond both at 0.19 nA.",
             "Late-rate f-I from the mean of the last five cycles: human 8.2 -> 39.4 Hz (0.39 Hz/pA, zero near 169 pA); s1-sk-axon0 28.3 -> 58.5 Hz (0.38 Hz/pA, zero near 115 pA); s0-finalist 12.0 -> 32.9 Hz (0.26 Hz/pA, zero near 144 pA). The SK-free model has the human's late gain and a rheobase about 55 pA too low; the SK dose lowers the gain without moving the rheobase to the human's. A brake that accumulates with calcium grows with the input; the human's late slowing shrinks with the input. Three lines through two points each, not a mechanism.",
             [SP10_RESULT, SP10_S1], "supported"),
        node("ie_kv3", IE,
             "IE4. I cell: earlier comparisons support a role for sodium duration and retained Kv3 activation in the spike fall and trough; removing somatic Kv3 produced depolarisation block in that earlier candidate.",
             "A repeated width difference does not establish one unique mechanism (Y3). The finalist's somatic Kv3 close factor is 2.0.",
             [CM, I_ENERGETIC], "supported"),
        node("ie_passive", IE,
             "IE5. E cell: the passive levers move the f-I curve and the sweep-43 subthreshold return (F5) in a fixed ratio. Ih half: 250 pA count 8 -> 8, rest -1.73 mV, return -0.7 mV. Leak x1.5: counts 0/0/6 at 200/250/310 pA, return -3.06 to -3.78 mV, past the donor.",
             "The 250 pA count is a step in the leak, not a line: no leak dose lands it at 5 while the 310 pA rate stays within 1.5 Hz. Somatic SK (0.35) with calcium decay 1.0 sets the late 310 pA interval (10.05 vs 10.00 Hz) but translates the whole curve and cannot rotate it. The combined arm (g3) was registered and not run because its additive prediction fails every band.",
             [E_CLOSE, E_STAGE_G], "supported"),
        node("ie_e_passthrough", IE,
             "IE6 (measured by INJ2 stage 0; corrected by the 2026-09-11 erratum). E cell: the recorded balance is that of the middle soma segment, one of nine (65.8 of 592 um2). Whole-soma interspike estimates at 200 pA: leak -0.058, Kv3 -0.017, SK -0.016, NaTs +0.010, Nap +0.012, Im -0.0002 nA, summing to -0.069 nA (35 percent of the applied 0.196 nA). Of the segment's 0.188 nA axial outflow, 0.061 nA enters the neighbouring soma segments and 0.128 nA the cable. 'The soma is a pass-through' is withdrawn.",
             "Whole-soma Nap (+0.012 nA) is 59 percent of the 0.020 nA p50 offset; the earlier '8 percent' and the 'not spent by construction' reasoning for the Nap arm are withdrawn (erratum in the SP11 close decision). A baseline current does not bound an intervention effect: removing Nap silenced 200 pA entirely (INJ3 control). B3 reproduced exactly with currents recorded (counts 4/8/10, peak times and troughs identical to g0-b3, axon-first).",
             [SP11_CLOSE, SP11_RESULT], "supported"),
        node("ie_e_postspike", IE,
             "IE9 (measured by INJ2 stage 0). E cell at 200 pA: after its single spike (206 ms) the human sits at -67.1 mV (late pulse) and -67.7 mV (mid pulse); the model averages -64.8 mV late and -61.9 mV mid while firing four times. At 110 pA with no spike the same model matches the human onset rows within 0.4-0.9 mV, so the offset appears only after a spike.",
             "Offset 2.0 mV by p50 (0.020 nA at the model's 98 MOhm), 2.3 mV by mean late, 5.8 mV by mean mid pulse. Unresolved link: a current that holds the human 2-6 mV below the model after a spike, silent without a spike, not present in the soma genome, carried through the cable boundary in the model.",
             [SP11_CLOSE, SP11_RESULT, SP12], "supported"),
        node("ie_e_threshold", IE,
             "IE10. E cell: the human's spike threshold climbs along the train (-56.4 to -52.8 mV at 310 pA; -55.8 to -54.3 mV at 250 pA); the model's stays at -57.2 mV at every input.",
             "A rising threshold is not supplied by the sAHP/KNa family alone; the SP12 draft pairs it with sodium slow inactivation as a separate hypothesis.",
             [SP12, SP11_RESULT], "supported"),
        node("ie_cross", IE,
             "IE11 (observed pattern, not a cause). Both humans show a higher rheobase and a steeper late gain than their donor models: E counts 1/5/10/13 vs model 4/8/10/12 at 200/250/310/350 pA; I late rate 8.2 -> 39.4 Hz (zero near 169 pA) vs finalist 12.0 -> 32.9 Hz (zero near 144 pA). Both humans show a threshold that climbs along the train; both models hold a fixed threshold.",
             "A shared pattern across two cells and two fits. It does not identify a mechanism and is recorded here as a constraint on candidate causes.",
             [E_CLOSE, SP10_RESULT, SP11_RESULT, SP12], "supported"),
        node("ie_pop_runtime", IE,
             "IE7. Population: construction, initialisation, compilation, stepping and trace analysis are separate runtime stages; the corrected 104-cell run has a construction and initialisation pass but no completed driven runtime record.",
             "The runtime timeout does not establish numerical failure or memory exhaustion; a timeout needs a stage record before it can support a performance explanation.",
             [POP_INIT, CM], "supported"),
        # -------------------------------------------------------- root causes
        node("rc_vdep", RC,
             "RC1 (open). E cell: the current that removes three spikes at 200-250 pA while leaving the 310 pA train and the sweep-43 return fixed must be voltage- or use-dependent and lies outside the fit's passive family.",
             "After INJ2 the unresolved link is narrowed: the current holds the human 2-6 mV below the model after a spike at 200 pA, is silent without a spike (110 pA rows), is not in the soma genome, and is carried through the cable boundary. After INJ3 the post-spike part of the link has a sufficient mechanism (a spike-triggered outward gate) whose tail must be longer than 1 s; INJ4 tests 5 s. The pre-spike part (B3 fires 78 ms early at 200 pA) is not addressed by that gate; Nap is a necessary condition for that spike (IE12) and a partial Nap dose is a candidate for it. Im-up stays unspent (needs 100x). Other candidates: dendritic or axonal placement of an existing outward conductance; a second human L2/3 donor (SP6b lead).",
             [SP12_DEC, SP11_CLOSE, E_CLOSE, CM], "unverified"),
        node("rc_burst", RC,
             "RC2 (open). I cell: the current balance that would restore early refiring (cycle 2 at or below 22 ms) is unidentified. The human's early burst (cycles 2-4 at 6-10 ms) is absent with and without axonal SK.",
             "INJ1 isolated the late-delay path (axonal SK) but changed cycle 2 by only -3.9 ms at 0.19 nA and -0.4 ms at 0.27 nA. The finalist remains experimental; both tiers fail count and rate at both inputs for both arms.",
             [SP10_RESULT, SP10_S1, CM], "unverified"),
        node("rc_donor", RC,
             "RC3 (open). The fitted electrical properties are a donor's, not H01's: B3 runs on Allen 541563728 geometry with the kv3-closing-source mechanism library, and no H01 physiological recording exists to qualify against.",
             "Additional donor fits also miss recorded counts: HL5MN1 gave 16 and 30 against 14 and 34; Allen 527952884 gave 19 and 8 against 20 and 12 (Y6). These runs do not establish a missing biological mechanism; the condition behind each count difference is unidentified.",
             [CM, E_CLOSE, DONOR_HL5, DONOR_L4], "unverified"),
        # --------------------------------------------------------- injections
        node("inj_sp10", INJ,
             "INJ1 (executed, CLOSED PASS). SP10: isolate the axonal SK effect on the I finalist. Stage 0 s0-finalist reproduced the container (14, 37; 12/12 bands). Stage 1 s1-sk-axon0 (--scale SK:axon:0, single change) held 22 of 22 bands; rejection not met.",
             "Pre-registered prediction: removing axonal SK shortens the late trough-to-threshold delay by more than 12.8 ms at both inputs and raises counts above 14 and 37, while cycle 2 moves less than 12.8 ms and widths, peaks and early troughs stay within stage-0 bands. Observed: delay shortened 48.7 ms at 0.19 nA and 13.1 ms at 0.27 nA; counts 29 and 59; cycle 2 changed -3.9 and -0.4 ms; widths 0.221 ms, troughs within 0.04 mV, axon-first, longest time above -20 mV 0.22 ms. Not a repair: both counts move away from the human; the finalist stays experimental. Two evaluations spent of two; sweep 48 stays sealed. Causal model Y3 updated in commit 8869c22.",
             [SP10_S1, SP10_RESULT, SP10, SP10_MAN], "supported"),
        node("inj_sp11", INJ,
             "INJ2 (stage 0 executed PASS 12/12; stage 1: Nap arm spent as the INJ3 control, Im arm not spent; CLOSED with erratum). SP11: unchanged B3 rerun with record_soma_currents at 200, 250, 310 pA reproduced g0-b3 exactly (counts 4/8/10, identical peak times and troughs, axon-first) with every soma current recorded.",
             "Stage 0 prediction held: counts 4, 8, 10; crossings within 0.1 ms; troughs within 0.1 mV; axon-first. Erratum 2026-09-11: the current tables are one soma segment of nine; whole-soma Nap is +0.012 nA (59 percent of the p50 offset), so the 'by construction' reason for not spending the Nap arm is withdrawn; it was spent as the INJ3 comparison arm and silenced 200 pA. The Im arm (0.0002 nA whole soma; 100x dose acts at the 310 pA threshold) stays unspent. One evaluation spent of three. Runs took 885, 981 and 1015 s. B3 stays frozen and unpromoted; causal model Y4 updated in the same commit.",
             [SP11_CLOSE, SP11_RESULT, SP11, SP11_MAN], "supported"),
        node("inj_sp12", INJ,
             "INJ3 (executed 2026-09-12, stage 0 FAIL under the registered bands; 3 of 4 evaluations; stage 1 not opened). SP12: a phenomenological spike-triggered gate (KsAHP.mod: opens above -20 mV with a 1 ms rise, closes with a 1000 ms tail) on the B3 soma at two doses fixed from the retained trace (3.50e-4 and 1.03e-3 S/cm2), plus a Nap-removal comparison arm. Dose A: count 1 at 200 pA with B3's first spike and pre-spike trace unchanged (0.000 mV), mid-pulse p50 within 0.5 mV of the human, late p50 +1.26 mV (band 1 mV): 5 of 6 bands. Dose B overshoots (mid -73.5, late -70.3 mV). Nap zero: no spike at all.",
             "Reading: dose A follows the human within 0.5 mV from 1240 to 1700 ms and then rises 1.4 mV as the 1 s gate decays; the human's post-spike shift does not decay within the pulse. So a spike-triggered outward current is sufficient to remove the three excess 200 pA spikes without touching the pre-spike response; the 1 s tail is not sufficient for the level over the whole pulse. The gate carrier is not the fit's calcium (soma cai 1.0e-4 to 2.8e-4 mM, not spike-shaped). The comparison arm broke its prediction the other way: Nap is a rheobase lever (4 to 0 spikes), its signature differs from the human's one spike. Not addressed: B3's first spike 78 ms early. Runs 635 and 656 s on the Vast box; library recompiled with KsAHP.mod.",
             [SP12_DEC, SP12_RESULT, SP12, SP12_MAN], "supported"),
        node("inj_sp13", INJ,
             "INJ4 (executed 2026-09-12; stage 0 PASS 6/6, stage 1 FAIL 11/16; CLOSED, 2 of 2 evaluations). SP13: the same gate with a 5 s tail at 3.12e-4 S/cm2, sized from the two SP12 doses. 200 pA: one spike at B3's time, mid -67.94 and late -67.32 mV (human -67.47, -67.06; derived prediction -67.43, -67.06), every 100 ms median within 1 mV. 110 pA: five onset rows equal B3 to 0.000 mV. 250 pA: count 3 (doublet, then 442 ms silence) vs human 5. 310 pA: count 7, late cycles 184 ms vs human 10 at 120 ms; gate 0.83 and 0.96.",
             "Inference: a spike-triggered brake that persists for seconds is sufficient for the 200 pA response and silent without a spike, but a single non-saturating brake cannot serve 200, 250 and 310 pA together because it accumulates while the human's 310 pA cycles equal unchanged B3's. Not separated by this test: (a) the human's brake saturates after one spike and B3 lands 310 pA only through a compensating low gain (IE11), or (b) the 200 pA shift is not a brake that persists into a train. The 250 pA doublet-then-silence points to (a). Draft SP14: saturating gate plus one gain lever, not approved.",
             [SP13, SP13_MAN, SP12_DEC, "docs/evidence/h01-e-sahp-tail/stage-1-decision.json", "docs/evidence/h01-e-sahp-tail-result.md"], "supported"),
        node("ie_e_nap", IE,
             "IE12 (measured by INJ3 control). E cell: with soma Nap density 0, B3 gives no spike at 200 pA (pulse maximum -63.0 mV; level 2.0 mV below B3 before B3's first spike). Nap is a necessary condition for the first spike under B3's settings at this input.",
             "Its baseline current (+0.012 nA whole soma) did not bound the intervention effect. Nap removal does not reproduce the human (one spike, then a lower level); partial Nap doses are not excluded and would act before the first spike, where B3 is 78 ms early.",
             [SP12_DEC, SP12_RESULT], "supported"),
        junction("j_e"),
        junction("j_i_late"),
        junction("j_i_count"),
    ]


def build_edges() -> list[dict]:
    return [
        # E cell branch
        edge("rc_vdep", "j_e", "input"),
        edge("ie_passive", "j_e", "input"),
        edge("j_e", "ude_e_lowdrive", "causes", "candidate cause; passive family excluded"),
        edge("ie_e_passthrough", "rc_vdep", "describes", "soma levers excluded; cable boundary only"),
        edge("ie_e_postspike", "rc_vdep", "describes", "offset appears only after a spike"),
        edge("ie_e_threshold", "rc_vdep", "describes", "not supplied by sAHP/KNa alone"),
        edge("rc_donor", "ude_e_lowdrive", "describes", "candidate: second human L2/3 donor (SP6b lead)"),
        edge("ie_cross", "ude_e_lowdrive", "describes", "shared pattern; not a cause"),
        edge("ie_cross", "ude_i_count", "describes", "shared pattern; not a cause"),
        # I cell branch
        edge("ie_axial", "j_i_late", "input"),
        edge("ie_sk", "j_i_late", "input"),
        edge("j_i_late", "ude_i_latecycle", "causes", "SK path established by INJ1; axial path measured"),
        edge("ie_sk_confounded", "ie_sk", "describes", "earlier confound, superseded by INJ1"),
        edge("ie_i_late_fi", "ude_i_latecycle", "describes", "human late cycle vs both model settings"),
        edge("rc_burst", "ude_i_burst", "causes", "unidentified balance"),
        edge("ude_i_burst", "j_i_count", "input"),
        edge("ude_i_latecycle", "j_i_count", "input"),
        edge("j_i_count", "ude_i_count", "causes", "count = cycles that fit the pulse"),
        edge("ie_kv3", "ude_i_width", "causes", "supported role; not a unique mechanism"),
        # population branch
        edge("rc_donor", "ude_pop", "causes"),
        edge("ie_pop_runtime", "ude_pop", "causes"),
        # injections
        edge("inj_sp10", "ie_sk", "describes", "executed: established the SK path (22/22)"),
        edge("inj_sp10", "rc_burst", "describes", "early burst absent at both settings"),
        edge("inj_sp11", "ie_e_passthrough", "describes", "stage 0 measured this"),
        edge("inj_sp11", "ie_e_postspike", "describes", "stage 0 measured this"),
        edge("inj_sp11", "rc_vdep", "describes", "Nap arm spent as INJ3 control; Im arm not spent"),
        edge("inj_sp12", "ie_e_postspike", "describes", "sufficient to remove the excess spikes; 1 s tail too short"),
        edge("inj_sp12", "ie_e_nap", "describes", "control arm measured this"),
        edge("ie_e_nap", "rc_vdep", "describes", "pre-spike part: Nap is necessary for the first spike"),
        edge("inj_sp13", "rc_vdep", "describes", "executed: sufficient at 200 pA, over-brakes 250/310 pA"),
        edge("inj_sp13", "ie_cross", "describes", "points to a compensating gain error in B3 at 310 pA"),
    ]


REVIEW_TEXT = {
    "causes": ("Cause insufficiency", "Is the stated cause enough on its own, or is an unlisted condition needed? Record it before asserting the link."),
    "describes": ("Entity existence", "Does the described condition exist as stated in the named evidence file, or is it assumed?"),
}


def reviews(edges: list[dict]) -> list[dict]:
    out = []
    for e in edges:
        if e["relation"] == "input":
            continue
        cat, concern = REVIEW_TEXT[e["relation"]]
        out.append({
            "id": "review_" + e["id"],
            "edgeId": e["id"],
            "assumptions": "The link is a claim to scrutinise against the evidence files named on both nodes, not an observed result.",
            "evidence": [],
            "reservations": [{
                "id": "reservation_" + e["id"],
                "category": cat,
                "concern": concern,
                "status": "open",
                "rationale": "",
                "responses": [],
            }],
        })
    return out


def explanation(nodes: list[dict]) -> dict:
    return {
        "id": "crt_account",
        "title": "Current Reality Tree: H01 single-cell fits and population against the human recordings",
        "effect": TITLE,
        "conditions": (
            "Two single-cell fits (E: frozen B3 on Allen 541563728 geometry; I: the Kv3 close factor 2.0 "
            "energetic-search finalist) are compared with human recordings under the approved 1 mV contract "
            "and the separate usable tier. The 104-cell H01 population builds and initialises but has no "
            "completed driven runtime qualification. Two tests ran on the Vast.ai box under pre-registered "
            "predictions and are closed: SP10 (PASS, axonal SK is the I finalist's late-delay path) and SP11 "
            "(stage 0 PASS with soma currents recorded; stage 1 not spent). SP12 is a draft awaiting approval."
        ),
        "narrative": (
            "Undesirable Effects are the contract rows that fail. Intermediate Effects are measured "
            "mechanisms copied from the decision files; IE11 is a cross-cell pattern recorded as a constraint, "
            "not a cause. Root Causes are the candidate causes the campaign leaves open; none is established. "
            "Injections are the two executed tests (with prediction, observation and verdict) and one proposed "
            "test. Arrows point from cause to effect (downward); 'describes' links carry context, not "
            "causation. Every node names its evidence file. Status 'supported' means the numbers on the node "
            "are read directly from that file; 'unverified' means the node is a candidate or a not-yet-run test."
        ),
        "uncertainties": (
            "The E post-spike offset (2-6 mV at 200 pA) is carried through the cable boundary and is not in "
            "the soma genome; its mechanism (sAHP/KNa, dendritic or axonal placement, or a donor difference) "
            "is untested. The I early burst is absent with and without axonal SK and has no candidate "
            "mechanism. Neither executed test produced a contract pass or a promotion. The population "
            "runtime timeout has no stage record. Whether any H01 cell's physiology matches the donor fits "
            "is untestable with the data held."
        ),
        "predictions": (
            "SP12 (if approved): a spike-triggered slow outward current lands the 200 pA post-spike p50 "
            "within 1 mV of -67.4 mV with count 1, then 250 pA count 5-7 with cycles 4-5 of 250-350 ms, "
            "310 pA count 9-10 with late cycles within 12.8 ms of 120 ms, and sweep-43 onset rows within "
            "1 mV of g0-b3; otherwise the family is rejected."
        ),
        "nodeIds": [n["id"] for n in nodes],
        "testIds": [],
        "evidence": [],
        "status": "unverified",
        "origin": ORIGIN,
    }


def document() -> dict:
    nodes = build_nodes()
    edges = build_edges()
    ids = {n["id"] for n in nodes}
    for e in edges:
        assert e["source"] in ids and e["target"] in ids, e
    classes = []
    for n in nodes:
        if n["type"] not in [c["id"] for c in classes]:
            classes.append({"id": n["type"], "label": n["type"], "color": COLORS[n["type"]]})
    graph = {
        "classes": classes,
        "direction": "DOWN",
        "edges": edges,
        "events": [],
        "explanations": [explanation(nodes)],
        "nodes": nodes,
        "project": {"calendar": {"exceptions": [], "hours": 8, "weekdays": [1, 2, 3, 4, 5]},
                    "date": DATE, "direction": "forward", "resources": []},
        "relationshipReviews": reviews(edges),
        "simulation": {"end": 10000, "step": 100},
        "tests": [],
        "summaries": [],
        "characteristics": [],
        "verifications": [],
    }
    return {
        "id": "doc_" + str(uuid.uuid4())[:12],
        "title": TITLE,
        "graph": graph,
        "scenarios": [],
        "revision": 0,
        "history": [],
        "undo": [],
        "redo": [],
        "proposals": [],
        "jobs": [],
        "access": {"direct": True},
    }


def request(payload: dict) -> object:
    proc = subprocess.run(["node", str(ENGINE), "request"], input=json.dumps(payload), text=True,
                          encoding="utf-8", capture_output=True, cwd=str(STUDIO))
    if proc.returncode != 0:
        raise RuntimeError(f"{payload.get('action')} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout)


def main() -> None:
    doc = document()
    HERE.mkdir(parents=True, exist_ok=True)  # export does not mkdir
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT, len(doc["graph"]["nodes"]), "nodes", len(doc["graph"]["edges"]), "edges")
    if "--no-export" in sys.argv:
        return
    # The engine caches documents by path across restarts; open a uniquely named copy so the export is fresh.
    fresh = OUT.with_name(f"h01-current-reality.{uuid.uuid4().hex[:8]}.reasoning.json")
    fresh.write_text(OUT.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        opened = request({"action": "open", "path": str(fresh)})
    finally:
        fresh.unlink(missing_ok=True)
    doc_id = opened["id"]
    issues = request({"action": "validate", "document": doc_id})
    print("validate:", json.dumps(issues, indent=1))
    if any(x.get("severity") == "error" for x in issues):
        raise SystemExit("validation errors; not exporting")
    for fmt in ("svg", "png"):
        target = HERE / f"h01-current-reality.{fmt}"
        request({"action": "export", "document": doc_id, "path": str(target), "format": fmt})
        print("exported", target, target.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
