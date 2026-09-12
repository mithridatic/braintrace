"""Build the H01 progressive-search tree (Hartshorne) as a Reasoning Studio document.

Companion to build_map.py (the Current Reality Tree). This map is the search tree of
docs/specs/2026-09-12-h01-topographic-strategy.md: each question is a binary split of
the remaining search space; branches are marked eliminated, retained or open, with the
evidence file that closed them. Reuses the CLI helpers of build_map.py.
"""

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_map as cm  # noqa: E402

OUT = cm.HERE / "h01-search-tree.reasoning.json"
TITLE = "H01 progressive search tree"
DATE = "2026-09-12"
ORIGIN = f"H01 progressive search tree (Hartshorne splits); built {DATE} from the evidence files named on each node"

STRATEGY = "docs/specs/2026-09-12-h01-topographic-strategy.md"
SP13_S1 = "docs/evidence/h01-e-sahp-tail/stage-1-decision.json"
SP15_S0 = "docs/evidence/h01-topographic/stage-0-decision.json"
SP12_DEC = "docs/evidence/h01-e-sahp/stage-0-reanalysis.json"
E_USABLE = "docs/evidence/h01-e-gain/g0-b3-usable.json"
I_USABLE = "docs/evidence/h01-i-sk/s0-finalist-usable.json"
E_G0 = "docs/evidence/h01-e-gain/stage-g0-decision.json"
SP10_RESULT = "docs/evidence/h01-i-sk-result.md"
SP11_CLOSE = "docs/evidence/h01-e-currents/stage-close-decision.json"
E_CLOSE = "docs/evidence/h01-e-gain/stage-close-decision.json"
CM = "docs/h01-causal-model.md"
SP16 = "docs/specs/2026-09-12-h01-sodium-slow-inactivation.md"
CYCLES = "docs/evidence/h01-topographic/cycle-tables.json"

COLORS = {
    "Search question": "#4f76a3",
    "Branch retained": "#b35f48",
    "Branch eliminated": "#8a8f94",
    "Branch open": "#c9a227",
    "Policy": "#5b8a5a",
    "Next test": "#7a5c9e",
}
Q, R, X, O, P, T = COLORS  # type names


def n(nid, ntype, text, notes, files, status):
    d = cm.node(nid, ntype, text, notes, files, status)
    d["origin"] = ORIGIN
    return d


def build_nodes():
    return [
        n("y", Q, "Y. Human-model contrast in the recorded spike trains of two cells (E: L2/3 pyramidal; I: PV interneuron), read against each human's own repeat envelope.",
          "Repeat envelope at 200 pA (E): count 1 in 5 of 5 sweeps; first spike 153-209 ms (57 ms spread); threshold -54.5 to -55.2 mV; AHP -69.5 to -70.2 mV. Contrasts smaller than about 3 sigma of this envelope are not searched.",
          [E_USABLE, I_USABLE, STRATEGY], "supported"),
        # Q1 Matryoshka
        n("q1", Q, "Q1 (Matryoshka). In which family does the contrast live?", "Families: elemental (within one spike cycle), cyclical (spike to spike), structural (cell to cell), temporal (sweep to sweep). Answered from retained traces; no run.", [STRATEGY], "unverified"),
        n("q1_elemental", R, "Elemental: retained (SP15 stage 0). E: spike upstroke 332 vs 598 V/s (177 sigma of the repeats) at every cycle; post-spike level at 200 pA 2.1 mV (6.9 sigma). I: post-spike level 4.7 mV (28 sigma); upstroke equal but the human leaves threshold with a kink the model soma lacks.",
          "Phase planes and load curves drawn from retained traces: docs/evidence/h01-topographic/. First spike (E, 78 ms early) is 3.3 sigma, marginal.", [SP15_S0, SP12_DEC, E_G0], "supported"),
        n("q1_cyclical", R, "Cyclical: retained (SP15 stage 0). Threshold climb over the train: E human +2.1 mV vs model +0.1 (8.1 sigma); I human +4.8 vs +0.9 (10 sigma); the I human's upstroke falls to 450 V/s by the last cycle while the model holds 590. The I early burst (cycle 2: 6.3 vs 16.4 ms) is 0.8 of the 12.8 ms repeat limit: not a steep X.",
          "Use-dependent signatures that accumulate along the train.", [SP15_S0, E_USABLE, I_USABLE], "supported"),
        n("q1_structural", R, "Structural: retained and strongest. Both humans share one deviation from both fits: higher rheobase, steeper late gain, climbing threshold, post-spike slowing that shrinks with input.",
          "Two different cells sharing one deviation from two different fits places the steep X in what the fits share.", [SP10_RESULT, SP11_CLOSE, CM], "supported"),
        n("q1_temporal", X, "Temporal: eliminated (SP15 stage 0). Repeat sigma: E level 0.3 mV, threshold 0.25 mV, rise 1.5 V/s, first spike 23 ms; I level 0.17 mV, threshold 0.38 mV, rise 5 V/s, first spike 8 ms.",
          "These are the decision limits every contrast above is divided by.", [SP15_S0, E_USABLE, I_USABLE], "supported"),
        # Q2 Isolation
        n("q2", Q, "Q2 (Isolation). Does the contrast come in on the inputs (anatomy, passive cable, per-cell densities) or live in the function (shared active kinetics)?", "Both branches must be phrased on Y only.", [STRATEGY], "unverified"),
        n("q2_inputs", X, "Inputs below threshold: eliminated. At 110 pA without a spike the model matches the human onset rows within 0.4-0.9 mV; Ih and leak doses moved the protected rows past their bands.",
          "The passive family cannot carry the low-drive count while holding the subthreshold return (E close).", [E_G0, E_CLOSE], "supported"),
        n("q2_function", R, "Function, use-dependent: retained. The offset appears only after a spike; a spike-triggered gate silent before the spike reproduces the 200 pA response exactly (SP13 stage 0) and the same signature is shared by both cells.",
          "Whether the shared function element is an added outward current or a lost inward one is Q3's job.", [SP13_S1, SP12_DEC, SP11_CLOSE], "supported"),
        n("q2_swap", T, "Open tactic: hold the function, swap the inputs (the same genome on the H01 morphology and on the donor morphology, one run each). If the signature stays, it follows the function.",
          "One evaluation pair; registers before it runs.", [STRATEGY], "unverified"),
        # Q3 z-strategy
        n("q3", Q, "Q3 (z-strategy). Where in voltage and in time after a spike do the human's and the model's net-current load curves separate?", "Pipette = flow source (Norton); cell = load. I_net(t) = I_inj - C dV/dt from retained traces; C from the 110 pA onset transient.", [STRATEGY], "unverified"),
        n("q3_v", X, "Separation in voltage: eliminated (SP15 stage 0). After the last spike the human and model load curves overlap at the same voltages in both cells; the onset capacitance is equal in E (128 vs 125 pF).", "I onset capacitance differs by 19 percent (70 vs 57 pF): a candidate inputs contrast not yet referred to a repeat limit.", [SP15_S0], "supported"),
        n("q3_t", R, "Separation in time after the spike: retained (SP15 stage 0). Load curves converge after the last spike in both cells and differ only early after a spike: E human absorbs the whole 0.196 nA at -70 to -65 mV (holds) while B3 absorbs 0.09 nA; I human absorbs 0 to -0.1 nA at -78 to -65 mV (net inward, refires in 6-8 ms) while the finalist absorbs 0.16-0.25 nA.", "Opposite sign early in the two cells, same convergence late; with the climbing threshold and the falling I upstroke this favours use-dependent sodium availability plus an early post-spike inward balance over a plateau conductance.", [SP15_S0, SP13_S1, STRATEGY], "supported"),
        # eliminated levers (recorded, not to be re-run)
        n("levers", X, "Named levers dosed before a split (eliminated as a method, not as physics): Ih half, leak x1.5, SK 0.35, Nap zero (silences 200 pA), Im x100 (kills 310 pA), KsAHP 1 s and 5 s tails (over-brake 250/310 pA).",
          "Each was sized from counts, medians or an offset divided by an input resistance: lossy transforms of Y. The book's rule: locate the family and the branch first, then name a mechanism once.", [E_CLOSE, SP11_CLOSE, SP12_DEC, SP13_S1], "supported"),
        # next test
        n("q4", Q, "Q4 (cyclical Matryoshka, retained traces, no run). What is the SHAPE of the use-dependence, cycle by cycle: a step or an accumulation; does it recover between sweeps?",
          "Stage 0 compressed the cyclical family to one number (threshold first-to-last). Hartshorne ch.3: compare the same position in consecutive cycles; ch.5: a single compressed value loses the answer to the strategic question.", [CYCLES], "supported"),
        n("q4_e", R, "E human, 230-350 pA (7 sweeps, one per drive): threshold steps +1.2 to +3.6 mV between spike 1 and spike 2 (inside the first interval) and then holds for the rest of the second at every drive; rise steps from 345-352 V/s to 0.86-0.93 of that, then drifts 5 percent. Spike 1 is identical in every sweep (rise sd 2.7 V/s): full recovery between sweeps. At 200 pA: one spike then a hold below threshold for 800 ms (0 spikes in 2 of 7 repeats).",
          "Recovery bracket: longer than 800 ms, shorter than the inter-sweep gap.", [CYCLES], "supported"),
        n("q4_i", R, "I human: at 0.27 nA the threshold steps +4 mV over the first 5 spikes (43 ms) then holds near -58.4 for 40 more spikes; rise 602 to 550 then holds. At 0.19 nA the climb is gradual across the whole second (-60.5 to -52, rise 597 to 452) as the intervals lengthen. Both fits: flat.",
          "Two shapes in one cell: fast-then-flat at high drive, gradual at low drive.", [CYCLES], "supported"),
        n("next", T, "Next (SP16, amended): one sodium gate with fast entry above its half-point and slow recovery below, applied to both fits; predictions per cycle with decision limits (E 1.0 mV, I 1.5 mV; spike-1 rise within 3.5 percent); E at 310 and 200 pA, I at 0.19 and 0.27 nA; eight evaluations.",
          "The first draft's single 1 s time constant and -60 mV half-point were withdrawn before any run: the constant time constant cannot make the step, and the half-point removed 12 percent of sodium at the 200 pA plateau. SP14 (brake plus gain lever) deferred.", [SP16], "unverified"),
        n("q2_upstroke", O, "E upstroke split (SP15 stage 1, executed): B3 genome on the H01 skeleton 955432427 beside the donor anatomy at nseg 1. Mesh control held (653 vs 598 V/s, 9 percent; count, first spike, threshold unchanged). The H01 anatomy did not spike at 200 pA (plateau -78 mV, onset capacitance 785 vs 125 pF, input resistance about 38 vs 98 MOhm): registered no-reading outcome.",
          "The cable load moves the low-input response by more than any tested channel change, but the change was too large to read the upstroke. Next: a graded cable change on the donor anatomy (membrane area x1.5, x2), and a check of the H01 conversion against the surface mesh.", [SP15_S0, "docs/evidence/h01-e-morphology/stage-1-decision.json", STRATEGY], "supported"),
        # policies
        n("policy", P, "Policies: contrasts > 3 sigma of the repeat envelope; samples of three; Y only; every split written before its data; fail fast under the cap; both tiers; sweeps 54 and 48 sealed.",
          "From Hartshorne chapters 1-5 and the campaign's own AGENTS rules.", [STRATEGY], "supported"),
    ]


def build_edges():
    return [
        cm.edge("y", "q1", "describes", "first split"),
        cm.edge("q1", "q1_elemental", "describes", "branch"),
        cm.edge("q1", "q1_cyclical", "describes", "branch"),
        cm.edge("q1", "q1_structural", "describes", "branch"),
        cm.edge("q1", "q1_temporal", "describes", "branch"),
        cm.edge("q1_structural", "q2", "describes", "shared by both cells: split inputs from function"),
        cm.edge("q1_elemental", "q2", "describes", "post-spike only: split by spike history"),
        cm.edge("q2", "q2_inputs", "describes", "branch"),
        cm.edge("q2", "q2_function", "describes", "branch"),
        cm.edge("q2_function", "q2_swap", "describes", "tactic to close the split"),
        cm.edge("q1_elemental", "q2_upstroke", "describes", "E upstroke: first split to run"),
        cm.edge("q2_function", "q3", "describes", "locate the function element energetically"),
        cm.edge("q3", "q3_v", "describes", "branch"),
        cm.edge("q3", "q3_t", "describes", "branch"),
        cm.edge("q3_t", "q4", "describes", "characterise the cycle before naming"),
        cm.edge("q4", "q4_e", "describes", "E"),
        cm.edge("q4", "q4_i", "describes", "I"),
        cm.edge("q4_e", "next", "describes", "fixes entry and recovery separately"),
        cm.edge("q4_i", "next", "describes", "fixes entry and recovery separately"),
        cm.edge("levers", "q3", "describes", "why the dose scans stop here"),
        cm.edge("policy", "q1", "describes", "applies to every split"),
    ]


def document():
    nodes, edges = build_nodes(), build_edges()
    ids = {x["id"] for x in nodes}
    for e in edges:
        assert e["source"] in ids and e["target"] in ids, e
    classes = [{"id": k, "label": k, "color": v} for k, v in COLORS.items()]
    graph = {
        "classes": classes, "direction": "DOWN", "edges": edges, "events": [],
        "explanations": [{
            "id": "search_tree", "title": TITLE, "effect": "Where the human-model contrast lives",
            "conditions": "Two human cells with repeats; two donor fits; retained traces from SP3-SP13 on the Vast executor.",
            "narrative": "A Hartshorne progressive search: Matryoshka family split, then inputs-against-function isolation, then the z-strategy load-curve split, before any mechanism is named. Grey branches are eliminated with the file that closed them; orange are retained; yellow are open; purple are the next tests; green are the policies.",
            "uncertainties": "SP16 is a hypothesis about the shared function with its time course fixed by the recordings, not a result.",
            "predictions": "Q1 names elemental and cyclical as the families, structural as the shared pattern; Q3 separates in time after the spike; Q4 gives a step inside one interval that holds and recovers between sweeps; SP16 moves both cells with one change in that shape.",
            "nodeIds": [x["id"] for x in nodes], "testIds": [], "evidence": [], "status": "unverified", "origin": ORIGIN}],
        "nodes": nodes,
        "project": {"calendar": {"exceptions": [], "hours": 8, "weekdays": [1, 2, 3, 4, 5]}, "date": DATE, "direction": "forward", "resources": []},
        "relationshipReviews": cm.reviews(edges),
        "simulation": {"end": 10000, "step": 100}, "tests": [], "summaries": [], "characteristics": [], "verifications": [],
    }
    return {"id": "doc_" + str(uuid.uuid4())[:12], "title": TITLE, "graph": graph, "scenarios": [], "revision": 0,
            "history": [], "undo": [], "redo": [], "proposals": [], "jobs": [], "access": {"direct": True}}


def main():
    doc = document()
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT, len(doc["graph"]["nodes"]), "nodes", len(doc["graph"]["edges"]), "edges")
    if "--no-export" in sys.argv:
        return
    # The engine caches documents by path across restarts; open a uniquely named copy so the export is fresh.
    fresh = OUT.with_name(f"h01-search-tree.{uuid.uuid4().hex[:8]}.reasoning.json")
    fresh.write_text(OUT.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        opened = cm.request({"action": "open", "path": str(fresh)})
    finally:
        fresh.unlink(missing_ok=True)
    issues = cm.request({"action": "validate", "document": opened["id"]})
    print("validate:", json.dumps(issues, indent=1))
    if any(x.get("severity") == "error" for x in issues):
        raise SystemExit("validation errors; not exporting")
    for fmt in ("svg", "png"):
        target = cm.HERE / f"h01-search-tree.{fmt}"
        cm.request({"action": "export", "document": opened["id"], "path": str(target), "format": fmt})
        print("exported", target, target.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
