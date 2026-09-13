"""Build the H01 accuracy roadmap as a Reasoning Studio document.

Companion to build_map.py (Current Reality Tree) and build_search_tree.py (the Hartshorne
progressive-search tree). This map answers one question the other two do not: how far is the
model from the recordings, what has already been eliminated as a route to closing the gap, what
is running, and what stands between today's single-cell score and a scored population of the
104 H01 cells. Every node carries a confidence, and the goal node states plainly what can and
cannot be measured.

Confidence convention, stated once so the numbers mean the same thing everywhere:
  * on a CLOSED branch it is the confidence that the branch stays closed (the refutation holds);
  * on an OPEN or planned task it is the confidence that the task achieves its stated target;
  * on a state node it is the confidence that the stated number is what it claims to be.
They are the author's calibrated judgement from the evidence named on each node, not outputs of
a statistical procedure, and they are recorded so that they can be scored later against what
actually happens.

Reuses the CLI helpers of build_map.py.
"""

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_map as cm  # noqa: E402

OUT = cm.HERE / "h01-accuracy-tree.reasoning.json"
TITLE = "H01 accuracy roadmap"
DATE = "2026-09-13"
ORIGIN = (f"H01 accuracy roadmap; built {DATE} from the evidence files named on each node. "
          "Confidence = probability the branch stays closed (closed nodes) or that the task hits "
          "its stated target (open/planned nodes); author's calibrated judgement, recorded to be scored later.")

STRATEGY = "docs/specs/2026-09-12-h01-topographic-strategy.md"
CM = "docs/h01-causal-model.md"
STAGE0 = "docs/evidence/h01-topographic/stage-0-decision.json"
STAGE9 = "docs/evidence/h01-topographic/stage-9.json"
STAGE10 = "docs/evidence/h01-topographic/stage-10.json"
STAGE11 = "docs/evidence/h01-topographic/stage-11.json"
STAGE12 = "docs/evidence/h01-topographic/stage-12.json"
STAGE13 = "docs/evidence/h01-topographic/stage-13.json"
STAGE14 = "docs/evidence/h01-topographic/stage-14.json"
STAGE15 = "docs/evidence/h01-topographic/stage-15.json"
ACC = "docs/evidence/h01-topographic/human-vs-rodent-accuracy.json"
OBS16 = "docs/evidence/h01-topographic/stage-16-observation.json"
CLIMB_MF = "docs/evidence/h01-e-climb-manifest.json"
ONSET = "docs/evidence/h01-topographic/onset-shape.json"
SP16_E = "docs/evidence/h01-e-sodium-slow/stage-1-decision.json"
POP = "docs/evidence/h01-population-status.md"
TYPES = "docs/evidence/h01-population-types.md"
BUILD40 = "docs/evidence/h01-population-build-40.json"
BUILD12 = "docs/evidence/h01-population-build-12.json"

COLORS = {
    "Current state": "#2f6f9f",
    "Closed branch": "#8a8f94",
    "Open test": "#c9a227",
    "Planned step": "#7a5c9e",
    "Goal": "#3f7f5f",
    "Blocker": "#b35f48",
}
STATE, CLOSED, OPEN, PLAN, GOAL, BLOCK = COLORS


def n(nid, ntype, text, notes, files, status, confidence):
    d = cm.node(nid, ntype, text, notes, files, status)
    d["origin"] = ORIGIN
    d["confidence"] = confidence/100. if confidence is not None and confidence > 1 else confidence
    return d


def build_nodes():
    return [
        # ---------------- where we are -------------------------------------------------
        n("state", STATE,
          "TODAY: 71.8 percent. That is ONE cell, ONE drive, TEN elements. B3 (the Allen perisomatic fit "
          "of human L2/3 cell 541563728) scored against that cell's own recording at 310 pA, every fast "
          "quantity read through the measured recording chain.",
          "Per element: count 100, rest 99.9, threshold 99.1, take-off 99.1, peak 97.2, fall 90.6, "
          "rise-ratio 89.5, rise 35.9, climb spike-5 6.5, climb spike-2 0.0. Seven of ten elements are "
          "already 90-100 percent; the whole deficit is three numbers. Caveat carried on the number: "
          "voltage levels are scored against a stated 100 mV span, so they score high by construction and "
          "flatter the mean; the element table, not the mean, is the reading.",
          [STAGE15, ACC, CM], "supported", 95),
        n("state_scope", BLOCK,
          "SCOPE WARNING: 71.8 percent is NOT a score on the 104 H01 cells. The H01 cells are electron-"
          "microscopy reconstructions with no electrophysiology; there is no recording to score them "
          "against. Every accuracy figure in this campaign is against Allen/Krembil patch recordings from "
          "OTHER human cells.",
          "So 'accuracy for the 104 cells' can only ever be INHERITED: a donor model validated against a "
          "real recording is transferred onto an H01 anatomy. The honest chain is (model vs its own "
          "recording) x (is this donor the right type for this H01 cell) x (does this anatomy build and "
          "run). Each link is measurable; their product is what 'population accuracy' can mean.",
          [POP, TYPES], "supported", 92),
        n("headroom", STATE,
          "The remaining 28.2 points are concentrated: threshold climb 19.4, rise 6.4, everything else 2.4.",
          "Computed from the same ten elements: climb spike-2 carries 10.0 points of the mean, climb "
          "spike-5 9.4, rise 6.4; the other seven elements carry 2.4 between them. Ceilings: fix the climb "
          "-> 91.1; fix the rise -> 78.2; fix both -> 97.5.",
          [STAGE15, CM], "supported", 93),

        # ---------------- what is already closed ---------------------------------------
        n("closed_passive", CLOSED,
          "CLOSED: the passive/cable family. Dendritic membrane area x1.5/x2/x3 silences the cell before "
          "it moves the rise; the human's own passive load is only 1.13x conductance and 1.03x capacitance.",
          "Stages 2-5. A cable-load account must work at 1.13x, not 1.5x, and at 1.13x there is no room.",
          [STRATEGY, CM], "supported", 90),
        n("closed_onset", CLOSED,
          "CLOSED: onset shape as a discriminator. The fit's somatic turn-over equals the recording's "
          "(5.1 vs 5.0 mV over 10-100 V/s) even though its axon fired 0.28 ms earlier; and stage 12 showed "
          "the span is chain-fragile.",
          "Stage 8, qualified by stage 12. The registered 'imposed take-off = kink' prediction was refuted.",
          [ONSET, STAGE12], "supported", 85),
        n("closed_slowna", CLOSED,
          "CLOSED: slow inactivation of the somatic sodium as the climb's mechanism. It lowers later rises "
          "but moves the threshold only +0.4 mV where the recording gives +1.4 with the rise kept.",
          "SP16 stage 1 (E). Other time courses are not excluded, but this family is spent.",
          [SP16_E], "supported", 85),
        n("closed_chain", CLOSED,
          "CLOSED: the measurement chain as the explanation of the rise gap. Measured from the recording's "
          "own step edges and noise floor, the chain takes only 5-15 percent off the rise; 220-270 V/s of "
          "the gap is the cell's.",
          "Stage 12. A first pass with a free corner fitted 2.65 kHz and would have blamed the whole gap on "
          "the chain; refuted by the flat noise floor and the two-sample edge jump, and recorded.",
          [STAGE12], "supported", 90),
        n("closed_base", CLOSED,
          "CLOSED: the fitting pipeline / the base. A second, independently fitted human L2/3 model "
          "(Toronto HL23PYR) rises at 592-600 V/s through the chain, like B3's 570, against the recorded 348.",
          "Stage 13. Two bases differing in densities, shifts, anatomy and fitting pipeline agree on the "
          "rise, so the rise does not follow the pipeline.",
          [STAGE13], "supported", 90),
        n("closed_lineage", CLOSED,
          "CLOSED: the rodent-vs-human kinetic lineage. Swapping the somatic rodent NaTs for the human NaTg "
          "with its human voltage shifts, over a 78-fold dose range, does not lower the rise; and the fully "
          "human-fitted model scores 72.3 against B3's 71.8 - half a point.",
          "Stage 15. Below B3's own conductance the cell is silent (plateau -49 mV, above its own -57 mV "
          "take-off; 8x density moves it 0.8 mV): the +13 mV activation shift is fitted against the Toronto "
          "potassium set and never activates against Allen's. Kinetics are fitted as a SET, not as "
          "interchangeable parts. At matched conductance the rise goes UP (625, then 921 V/s).",
          [STAGE15, ACC], "supported", 92),
        n("closed_somatic_climb", CLOSED,
          "CLOSED: any somatic conductance as the climb's lever. Somatic outward current accumulates across "
          "the train (+22.5 percent by spike 2, doubling by spike 4) while the take-off does not move at all "
          "(-57.20, -57.42, -57.16, -57.20, -57.20 mV).",
          "Stage 16 part 1, on retained traces. At the take-off the axon delivers +0.33 nA against a total "
          "accumulated somatic brake of ~0.012 nA - 3 percent of the trigger. B3's somatic threshold is the "
          "ARRIVAL TIME of the axonal spike (stages 9, 10, 14 from three directions). The registered somatic "
          "dose was refused before it ran; four evaluations saved.",
          [OBS16, STAGE14, STAGE9], "supported", 95),

        # ---------------- what is running ----------------------------------------------
        n("open_site", OPEN,
          "RUNNING: the climb as accommodation at the INITIATION SITE, read through a coupling that lets "
          "the soma see it. Axonal sodium recovery dosed 0.5 / 0.25 / 0.125 on the stage-10 thickened-stub "
          "geometry, plus the strongest dose on the default 1 um stub as the coupling control.",
          "Stage 16 part 2, four runs at 310 pA. Target: at least half the recorded first-interval step "
          "(+0.69 of +1.38 mV) with the count in 5-15 and the spike-1 rise within 15 percent of the arm's "
          "control. Confidence 40 percent: the mechanism is the right place (the site sets the threshold) "
          "and HL23PYR proves a human L2/3 model can produce the step, but SP16 showed sodium availability "
          "is a weak threshold lever at a dense fast site, and this arm's own rise (655 V/s) starts further "
          "from the recording than plain B3's.",
          [CLIMB_MF, STAGE10], "unverified", 40),

        # ---------------- the plan, single cell ----------------------------------------
        n("plan_climb", PLAN,
          "STEP 1 - close the climb on the E cell: 71.8 -> 91.1 percent. Worth 19.4 points, the largest "
          "single gain available anywhere in the campaign.",
          "If stage 16 part 2 lands, this is done. If it does not, the remaining candidates are a slow "
          "process at the site with a longer time constant than SP16 tested, or the dendritic/axial route "
          "that stage 14 showed carries most of the take-off current. Confidence 45 percent that the climb "
          "reaches at least half the recorded step within two more stages.",
          [STAGE15, CM], "unverified", 45),
        n("plan_rise", PLAN,
          "STEP 2 - close the rise on the E cell: +6.4 points to ~97.5 percent. Five stages have now failed "
          "to move it.",
          "Stage 14 localised it: above -40 mV the excess is the soma's own sodium current, and the "
          "recording needs roughly half of it, with the count and take-off held. Stage 11 showed the "
          "somatic density dose reaches 505 V/s at 0.5x before the count breaks, and the site-delivered "
          "floor is ~320 V/s. Confidence 20 percent: the constraint set may simply have no solution in this "
          "model family, and the honest outcome may be to record the residual rather than repair it.",
          [STAGE14, STAGE11], "unverified", 20),
        n("plan_holdout", PLAN,
          "STEP 3 - spend the sealed holdout. Sweep 54 (330 pA) has never been read; it is the only "
          "untouched test of whether a closed element generalises beyond the drive it was fitted at.",
          "Until this is spent, every score is at drives the model was tuned against. Confidence 60 percent "
          "that a model that closes the climb at 310 pA also holds at 330 pA: the climb is a train "
          "property and the drives are adjacent, but nothing in the campaign has yet been tested "
          "out-of-sample.",
          [STRATEGY], "unverified", 60),
        n("plan_second_cell", PLAN,
          "STEP 4 - the second human cell. The I cell (aspiny, Krembil/Allen 528687520) has its own "
          "recording and its own fitted model, and the campaign's I-side findings are already logged.",
          "This is the first true cross-cell test: a mechanism that closes the climb on the E cell should "
          "not need refitting to appear on the I cell, whose recording also climbs (+4 mV over 5 spikes). "
          "Confidence 35 percent that the same mechanism transfers without refitting.",
          [STAGE0, CM], "unverified", 35),

        # ---------------- the plan, population ------------------------------------------
        n("pop_typematch", PLAN,
          "STEP 5a - donor type match: 55 of 104 H01 cells currently have a type-matched donor model "
          "(53 percent). The other 49 have no validated donor of their class.",
          "Every unmatched cell would inherit a model fitted to a different cell class, which the campaign "
          "has already shown is worth little: HL23PYR on the E cell's own class scores 72.3, and it is a "
          "same-class human model. Confidence 55 percent that type coverage can be raised materially "
          "without a new fitting programme per class.",
          [TYPES, POP], "unverified", 55),
        n("pop_build", BLOCK,
          "STEP 5b - BLOCKER: the population does not build. The 40-cell build fails on one-point SWC "
          "branches (7 of the 104 largest components fail to load); 12 cells have been measured and run "
          "1 ms; 92 remain unsimulated; 104 derived is ~19.8 GB.",
          "This is an engineering blocker, not a science one, and it gates every population number. "
          "Confidence 75 percent that it is fixable (the failure is a two-point requirement on leaf "
          "branches 0.032 um from a branch point, i.e. a mesh repair, not a modelling problem).",
          [BUILD40, BUILD12, POP], "supported", 75),
        n("pop_noground", BLOCK,
          "STEP 5c - BLOCKER, unfixable by simulation: there is no H01 electrophysiology. No H01 cell can "
          "be scored against its own recording, ever, from this data.",
          "So a population 'accuracy' is a transfer claim, not a measurement. The defensible form is: "
          "donor accuracy against real recordings (measurable, today 71.8), times type-match coverage "
          "(measurable, 53 percent), times build/run success (measurable, currently 12 of 104). Anything "
          "stated as a single population accuracy percentage without those three factors named is not "
          "supportable. Confidence 95 percent that this limitation stands.",
          [POP, TYPES], "supported", 95),

        # ---------------- the goal --------------------------------------------------------
        n("goal", GOAL,
          "GOAL: a scored model set over the 104 H01 cells. Reachable form: donor accuracy x type coverage "
          "x build success. Today that product is 71.8 percent x 53 percent x 12/104 = about 4 percent of "
          "the stated goal; with the climb closed, full type coverage and a fixed build it would be ~91 "
          "percent of the reachable ceiling.",
          "The 1-to-100 scale the goal asks for is therefore best read as this product, with the three "
          "factors reported separately so no single number hides a blocker. Confidence 25 percent that the "
          "full product exceeds 80 percent within this campaign's scope: the climb is plausible, the build "
          "is fixable, but type coverage for 49 cells and the absence of H01 ground truth are the hard "
          "limits, and the second of those cannot be removed.",
          [POP, TYPES, STAGE15], "unverified", 25),
        n("policy", PLAN,
          "STOPPING RULE, stated in advance: after the climb resolves, the remaining measurable gain on the "
          "E cell is 6.4 points (rise) plus 2.4 (everything else). That is the point to stop single-cell "
          "work and either fix the population build or stop.",
          "The campaign has already spent five stages on the rise. Sparsity of effects says chase the "
          "steep X; the accuracy ledger says the steep X for SCORE is the climb, and after it there is no "
          "third thing worth a run on this cell.",
          [CM, STAGE15], "supported", 85),
    ]


def build_edges():
    e = cm.edge
    return [
        e("state", "headroom", "describes", "where the 28.2 missing points are"),
        e("state", "state_scope", "describes", "what the number is NOT"),
        e("headroom", "closed_passive", "describes", "eliminated route"),
        e("headroom", "closed_onset", "describes", "eliminated route"),
        e("headroom", "closed_chain", "describes", "eliminated route"),
        e("headroom", "closed_base", "describes", "eliminated route"),
        e("headroom", "closed_lineage", "describes", "eliminated route (human vs mouse)"),
        e("headroom", "closed_slowna", "describes", "eliminated route (climb)"),
        e("headroom", "closed_somatic_climb", "describes", "eliminated route (climb)"),
        e("closed_somatic_climb", "open_site", "describes", "so the climb must be dosed at the site"),
        e("open_site", "plan_climb", "describes", "if it lands, +19.4 points"),
        e("plan_climb", "plan_rise", "describes", "then the only element left"),
        e("plan_climb", "plan_holdout", "describes", "generalisation test"),
        e("plan_holdout", "plan_second_cell", "describes", "cross-cell test"),
        e("plan_second_cell", "pop_typematch", "describes", "scale to the population"),
        e("pop_typematch", "goal", "describes", "coverage factor"),
        e("pop_build", "goal", "describes", "build factor (blocker)"),
        e("pop_noground", "goal", "describes", "no H01 ground truth (hard limit)"),
        e("state_scope", "pop_noground", "describes", "why population accuracy is inherited"),
        e("plan_rise", "policy", "describes", "stopping rule"),
        e("policy", "goal", "describes", "when to stop single-cell work"),
    ]


def explanation(nodes):
    return {
        "id": "acc_account",
        "title": "How far the model is from the recordings, and what stands between that and 104 cells",
        "effect": TITLE,
        "conditions": (
            "One human L2/3 cell (Allen 541563728) with its own recording, its own reconstruction and an "
            "Allen perisomatic fit of itself (B3), scored at 310 pA on ten elements read through the "
            "measurement chain that stage 12 measured from the recording's own step edges and noise floor. "
            "The 104 H01 cells are electron-microscopy reconstructions with no electrophysiology, so none "
            "of them can be scored against its own recording."
        ),
        "narrative": (
            "Seven of the ten elements are already between 90 and 100 percent; the whole remaining gap is "
            "three numbers - the threshold climb, which carries 19.4 points of the mean, and the spike "
            "upstroke rate, which carries 6.4. Seven routes to closing them have been opened and closed on "
            "evidence: the passive cable, the onset shape, the measurement chain, the fitting pipeline, the "
            "rodent-versus-human kinetic lineage, slow inactivation of the somatic sodium, and any somatic "
            "conductance at all for the climb. The last is the sharpest: somatic outward current accumulates "
            "across the train while the take-off does not move, because at the take-off the axon delivers "
            "about thirty times the accumulated somatic brake. The threshold is the arrival time of the "
            "axonal spike, so the climb can only be reached at the initiation site through a coupling wide "
            "enough for the soma to see it, which is the test now running. Confidence on each node is the "
            "probability that a closed branch stays closed, or that an open or planned task hits its stated "
            "target."
        ),
        "uncertainties": (
            "Population accuracy is inherited, not measured: no H01 cell has a recording. Its defensible "
            "form is the product of three separately measurable factors - donor accuracy against a real "
            "recording (71.8 percent today), donor type-match coverage (55 of 104) and build success (12 of "
            "104 built and run; the 40-cell build fails on one-point SWC branches). The sealed holdout "
            "(sweep 54, 330 pA) has never been spent, so no element has been tested out of sample. Whether "
            "a mechanism that closes the climb on this cell transfers to another cell without refitting is "
            "untested. The ten-element mean scores voltage levels against a stated 100 mV span, which "
            "flatters it; the element table is the reading."
        ),
        "predictions": (
            "Stage 16 part 2 (running): on the thickened-stub geometry the first-interval threshold step "
            "grows monotonically as the axonal sodium recovery factor falls and some dose reaches at least "
            "+0.69 mV with the count in 5 to 15 and the spike-1 rise within 15 percent of the arm's control, "
            "while the same dose on the default 1 um stub gives at most a fraction of it. If that lands the "
            "E cell reaches about 91 percent; the rise would then be the only element left, at 20 percent "
            "confidence of closing."
        ),
        "nodeIds": [x["id"] for x in nodes],
        "testIds": [],
        "evidence": [],
        "status": "unverified",
        "origin": ORIGIN,
    }


def document():
    nodes = build_nodes()
    edges = build_edges()
    ids = {x["id"] for x in nodes}
    for x in edges:
        assert x["source"] in ids and x["target"] in ids, x
    classes = []
    for x in nodes:
        if x["type"] not in [c["id"] for c in classes]:
            classes.append({"id": x["type"], "label": x["type"], "color": COLORS[x["type"]]})
    graph = {
        "classes": classes, "direction": "DOWN", "edges": edges, "events": [],
        "explanations": [explanation(nodes)], "nodes": nodes,
        "project": {"calendar": {"exceptions": [], "hours": 8, "weekdays": [1, 2, 3, 4, 5]},
                    "date": DATE, "direction": "forward", "resources": []},
        "relationshipReviews": cm.reviews(edges), "simulation": {"end": 10000, "step": 100},
        "tests": [], "summaries": [], "characteristics": [], "verifications": [],
    }
    return {"id": "doc_" + str(uuid.uuid4())[:12], "title": TITLE, "graph": graph, "scenarios": [],
            "revision": 0, "history": [], "undo": [], "redo": [], "proposals": [], "jobs": [],
            "access": {"direct": True}}


def main():
    doc = document()
    cm.HERE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT, len(doc["graph"]["nodes"]), "nodes", len(doc["graph"]["edges"]), "edges")
    if "--no-export" in sys.argv:
        return
    fresh = OUT.with_name(f"h01-accuracy-tree.{uuid.uuid4().hex[:8]}.reasoning.json")
    fresh.write_text(OUT.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        opened = cm.request({"action": "open", "path": str(fresh)})
    finally:
        fresh.unlink(missing_ok=True)
    doc_id = opened["id"]
    issues = cm.request({"action": "validate", "document": doc_id})
    print("validate:", json.dumps(issues, indent=1))
    if any(x.get("severity") == "error" for x in issues):
        raise SystemExit("validation errors; not exporting")
    for fmt in ("svg", "png"):
        target = cm.HERE / f"h01-accuracy-tree.{fmt}"
        cm.request({"action": "export", "document": doc_id, "path": str(target), "format": fmt})
        print("exported", target, target.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
