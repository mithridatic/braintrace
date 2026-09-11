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
DATE = "2026-09-11"
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
             "IE2. I cell: along the 0.23 nA train the axonal calcium rises 0.000120 to 0.000199 mM, the axonal SK gate rises 0.0022 to 0.024, and the trough-to-threshold delay grows 17.6 to 40.3 ms.",
             "These observations locate current and state changes; they do not isolate the cause of the timing error (Y3).",
             [SP10, CM, I_CURRENTS], "supported"),
        node("ie_sk_confounded", IE,
             "IE3. I cell: the earlier 'no axonal SK' arm (57/138 spikes) was not a single-change comparison; it also returned somatic NaTg to 1.0 and used Kv3 closing factor 0.5 (the finalist uses 2.0).",
             "So no isolated axonal-SK effect exists for the finalist. Increasing axonal NaTg density 1.5x failed its registered recovery prediction; that dose does not exclude all axonal density changes.",
             [I_CURRENTS, CM], "supported"),
        node("ie_kv3", IE,
             "IE4. I cell: earlier comparisons support a role for sodium duration and retained Kv3 activation in the spike fall and trough; removing somatic Kv3 produced depolarisation block in that earlier candidate.",
             "A repeated width difference does not establish one unique mechanism (Y3). The finalist's somatic Kv3 close factor is 2.0.",
             [CM, I_ENERGETIC], "supported"),
        node("ie_passive", IE,
             "IE5. E cell: the passive levers move the f-I curve and the sweep-43 subthreshold return (F5) in a fixed ratio. Ih half: 250 pA count 8 -> 8, rest -1.73 mV, return -0.7 mV. Leak x1.5: counts 0/0/6 at 200/250/310 pA, return -3.06 to -3.78 mV, past the donor.",
             "The 250 pA count is a step in the leak, not a line: no leak dose lands it at 5 while the 310 pA rate stays within 1.5 Hz. Somatic SK (0.35) with calcium decay 1.0 sets the late 310 pA interval (10.05 vs 10.00 Hz) but translates the whole curve and cannot rotate it. The combined arm (g3) was registered and not run because its additive prediction fails every band.",
             [E_CLOSE, E_STAGE_G], "supported"),
        node("ie_b3_notrace", IE,
             "IE6. E cell: the saved B3 traces contain soma and axon voltage but no channel currents or calcium (record_soma_currents false). The current that carries the three excess low-drive spikes has never been measured.",
             "The older E current study used different sodium, calcium and regional settings, so its budgets cannot explain B3. B3 already contains a somatic M current, Im, at 0.0003009 S/cm2 (source correction in Y4).",
             [SP11, CM], "supported"),
        node("ie_pop_runtime", IE,
             "IE7. Population: construction, initialisation, compilation, stepping and trace analysis are separate runtime stages; the corrected 104-cell run has a construction and initialisation pass but no completed driven runtime record.",
             "The runtime timeout does not establish numerical failure or memory exhaustion; a timeout needs a stage record before it can support a performance explanation.",
             [POP_INIT, CM], "supported"),
        # -------------------------------------------------------- root causes
        node("rc_vdep", RC,
             "RC1 (open). E cell: the current that removes three spikes at 200-250 pA while leaving the 310 pA train and the sweep-43 return fixed must be voltage- or use-dependent and lies outside the fit's passive family.",
             "Reassessment candidates: a slow sodium inactivation absent from the Allen genome; Kv7/M kinetics absent from the Allen genome; a second human L2/3 donor with a recorded f-I curve (SP6b lead). No candidate has been run; any is a new spec with its own cap.",
             [E_CLOSE, CM], "unverified"),
        node("rc_burst", RC,
             "RC2 (open). I cell: the current balance that would restore early refiring (cycle 2 at or below 22 ms) and the later train changes is unidentified.",
             "Y3 'Open': a matched SK test must retain the other finalist settings and measure the current change before the timing change. The finalist remains experimental.",
             [CM, SP10], "unverified"),
        node("rc_donor", RC,
             "RC3 (open). The fitted electrical properties are a donor's, not H01's: B3 runs on Allen 541563728 geometry with the kv3-closing-source mechanism library, and no H01 physiological recording exists to qualify against.",
             "Additional donor fits also miss recorded counts: HL5MN1 gave 16 and 30 against 14 and 34; Allen 527952884 gave 19 and 8 against 20 and 12 (Y6). These runs do not establish a missing biological mechanism; the condition behind each count difference is unidentified.",
             [CM, E_CLOSE, DONOR_HL5, DONOR_L4], "unverified"),
        # --------------------------------------------------------- injections
        node("inj_sp10", INJ,
             "INJ1 (running). SP10: isolate the axonal SK effect on the I finalist. Stage 0 s0-finalist (byte-identical) at 0.19 and 0.27 nA, then stage 1 s1-sk-axon0 (--scale SK:axon:0) as the single change.",
             "Pre-registered prediction (stage 1): if axonal SK is the direct path of the growing delay, removing it shortens the late trough-to-threshold delay by more than 12.8 ms at both inputs and raises counts above 14 and 37, while cycle 2 moves less than 12.8 ms and widths, peaks and early troughs stay within stage-0 bands. Rejection: delay not shortened beyond 12.8 ms; or spikes lost; or depolarisation block (above -20 mV for more than 5 ms). A count rise away from the human (12, 43) is a mechanism observation, not a promotion. Two evaluations, 1800 s abort, no holdout opened. Stage 0 must reproduce counts 14 and 37, cycle 2 (34.76, 16.37 ms) within 0.1 ms and troughs 1-3 within 0.1 mV.",
             [SP10, SP10_MAN], "unverified"),
        node("inj_sp11", INJ,
             "INJ2 (running). SP11: measure the B3 current paths (stage 0, unchanged B3 with record_soma_currents at 200, 250, 310 pA), then one voltage-dependent lever (stage 1: Nap density reduced or Im density raised).",
             "Stage 0 prediction: counts 4, 8, 10; every rise crossing within 0.1 ms and every trough within 0.1 mV of g0-b3; axon-first. Rejection: any count or crossing difference (stop). Lever selection rule from stage-0 numbers: (a) its interspike current at 200 pA is at least the net drift current there; (b) its share at 310 pA is smaller than at 200 pA. Stage 1 bands: 200 pA count 1 (1-2); 250 pA 5-7; 310 pA 9-10 with rate within 1.5 Hz of 10.0 Hz and late cycles 110-124 ms within 12.8 ms; sweep-43 rows move less than 1 mV. Rejection: 310 pA count below 9 or late cycle moves more than 12.8 ms; or sweep-43 rows move 1 mV or more. Cap three evaluations, 2400 s abort; sweep 54 sealed. A pass is a gain-split result, not a promotion.",
             [SP11, SP11_MAN], "unverified"),
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
        edge("ie_b3_notrace", "rc_vdep", "describes", "why the current is still unidentified"),
        edge("rc_donor", "ude_e_lowdrive", "describes", "candidate: second human L2/3 donor (SP6b lead)"),
        # I cell branch
        edge("ie_axial", "j_i_late", "input"),
        edge("ie_sk", "j_i_late", "input"),
        edge("j_i_late", "ude_i_latecycle", "causes", "measured together; not isolated"),
        edge("ie_sk_confounded", "ie_sk", "describes", "why SK is not isolated"),
        edge("rc_burst", "ude_i_burst", "causes", "unidentified balance"),
        edge("ude_i_burst", "j_i_count", "input"),
        edge("ude_i_latecycle", "j_i_count", "input"),
        edge("j_i_count", "ude_i_count", "causes", "count = cycles that fit the pulse"),
        edge("ie_kv3", "ude_i_width", "causes", "supported role; not a unique mechanism"),
        # population branch
        edge("rc_donor", "ude_pop", "causes"),
        edge("ie_pop_runtime", "ude_pop", "causes"),
        # injections
        edge("inj_sp10", "ie_sk", "describes", "tests whether axonal SK is the direct path"),
        edge("inj_sp10", "rc_burst", "describes", "prediction excludes the early refire"),
        edge("inj_sp11", "ie_b3_notrace", "describes", "stage 0 removes this gap"),
        edge("inj_sp11", "rc_vdep", "describes", "stage 1 tests Nap or Im"),
    ]


REVIEW_TEXT = {
    "causes": ("Cause sufficiency", "Is the stated cause enough on its own, or is an unlisted condition needed? Record it before asserting the link."),
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
            "completed driven runtime qualification. Two tests (SP10, SP11) are running on the Vast.ai box "
            "under pre-registered predictions."
        ),
        "narrative": (
            "Undesirable Effects are the contract rows that fail. Intermediate Effects are measured "
            "mechanisms copied from the decision files. Root Causes are the candidate causes the campaign "
            "leaves open; none is established. Injections are the two running tests, with their "
            "pre-registered predictions and rejection clauses. Arrows point from cause to effect "
            "(downward); 'describes' links carry context, not causation. Every node names its evidence file. "
            "Status 'supported' means the numbers on the node are read directly from that file; "
            "'unverified' means the node is a candidate or a not-yet-run test."
        ),
        "uncertainties": (
            "The E low-drive current has never been measured (SP11 stage 0 fixes that). The axonal SK "
            "effect in the I finalist has no single-change comparison (SP10 stage 1 fixes that). Neither "
            "test can produce a contract pass or a promotion. The population runtime timeout has no stage "
            "record. Whether any H01 cell's physiology matches the donor fits is untestable with the data "
            "held."
        ),
        "predictions": (
            "SP10 stage 1: late trough-to-threshold delay shortens by more than 12.8 ms at 0.19 and 0.27 nA "
            "with counts above 14 and 37, or the axonal-SK path is rejected. SP11 stage 0: counts 4, 8, 10 "
            "reproduced within 0.1 ms / 0.1 mV; stage 1 lever lands 200 pA at 1-2 and 250 pA at 5-7 while "
            "310 pA stays 9-10 within 1.5 Hz and sweep-43 rows move less than 1 mV, or the lever is rejected."
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
    opened = request({"action": "open", "path": str(OUT)})
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
