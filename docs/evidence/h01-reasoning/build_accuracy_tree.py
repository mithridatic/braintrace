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
STAGE16 = "docs/evidence/h01-topographic/stage-16.json"
ONSET = "docs/evidence/h01-topographic/onset-shape.json"
SP16_E = "docs/evidence/h01-e-sodium-slow/stage-1-decision.json"
POP = "docs/evidence/h01-population-status.md"
TYPES = "docs/evidence/h01-population-types.md"
BUILD40 = "docs/evidence/h01-population-build-40.json"
BUILD12 = "docs/evidence/h01-population-build-12.json"
LEDGER = "docs/evidence/h01-population-accuracy-ledger.md"
LEDGER_JSON = "docs/evidence/h01-population-accuracy-ledger.json"
IMPORT104 = "docs/evidence/h01-population-import-104.json"
PROBE104 = "docs/evidence/h01-arc-probe-104.json"
DRIVEN104 = "docs/evidence/h01-ready-104-ei-10ms-decision.json"
DTLADDER = "docs/evidence/h01-ready-cell7196644737-implicit-decision.json"
MORPH1 = "docs/evidence/h01-e-morphology/stage-1-decision.json"
LADDER = "docs/evidence/h01-timestep-ladder.json"

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
          "RESULT (stage 16 part 2): the climb IS reachable at the initiation site - and only by "
          "crippling the cell. Axonal sodium recovery 0.25 gives a first-interval step of +0.89 mV, past "
          "the +0.69 target that no somatic conductance could reach at all; but the train collapses from "
          "9 spikes to 2, and the series is non-monotone (-0.07, -3.28, +0.89 mV at recovery 1.0, 0.5, 0.25).",
          "Registered cells (c) and (d) both fail as written: not monotone, and the count leaves the 5-15 "
          "band at every dose that moves the threshold. Scored over the same ten elements the best arm is "
          "the CONTROL at 69.1 percent, BELOW plain B3's 71.8, because this coupled geometry starts with a "
          "worse rise (609 against 570 V/s) and accommodation only costs more. Confidence lowered from 40 "
          "to 15 percent: the mechanism is real and correctly located, but no dose separates the threshold "
          "step from the spike count in this model family.",
          [STAGE16, CLIMB_MF, STAGE10], "supported", 15),

        n("coupling_confirmed", CLOSED,
          "CONFIRMED (not eliminated): the soma-site coupling is real and load-bearing. The same axonal "
          "accommodation dose gives +2.39 mV of threshold step through a 2 um stub and -2.56 mV - the "
          "opposite sign - through the default 1 um Allen stub.",
          "Stage 16 cell (f), the one cell of the stage that passed. This is the first direct test of "
          "stage 10's account of the take-off rather than an inference from it: the site's state reaches "
          "the soma only when the stub is thickened. It also means the Allen 1 um replacement stub, not "
          "any channel, is what makes B3's threshold immovable.",
          [STAGE16, STAGE10], "supported", 88),

        # ---------------- the plan, single cell ----------------------------------------
        n("plan_climb", PLAN,
          "STEP 1 - close the climb on the E cell: 71.8 -> 91.1 percent. Still the largest single gain "
          "available (19.4 points), but the direct route is now measured and costly.",
          "Stage 16 located the mechanism (the site, not the soma) and showed it moves the threshold the "
          "right way and by roughly the right amount, while taking the spike count with it. What is left "
          "is to separate the two: a slower site process (longer recovery time constant than the 13.5 ms "
          "interspike interval, which the dose family tested here does not provide), or accommodation "
          "placed where it reaches threshold without gating the train. Confidence lowered from 45 to 25 "
          "percent that the climb reaches at least half the recorded step with the count held, within two "
          "more stages.",
          [STAGE15, CM], "unverified", 25),
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
        n("pop_build", CLOSED,
          "STEP 5b - CLOSED, and it was closed before the campaign quoted it: the population DOES build. "
          "All 104 components import with zero missing segments, construct to 808,495 compartments with "
          "no failures, initialise, and complete a forward pass. The one-point SWC branch was repaired on "
          "2026-09-08; the 12-of-104 figure quoted on 2026-09-13 was six days stale.",
          "This is the correction that occasioned the ledger. Raising this term from 0.115 to 1.0 does "
          "not raise the product, because the same audit exposes three terms that were never in the "
          "product and are zero. Confidence 95 percent that the construction gates stay passed: they are "
          "hash-pinned audits over committed inputs, not timing-sensitive runs.",
          [IMPORT104, PROBE104, LEDGER], "supported", 95),
        n("pop_driven", BLOCK,
          "STEP 5c - BLOCKER: no driven window at a physiological duration. The forward pass that passes "
          "drives an all-ones synthetic 441-feature probe and records physiology as unqualified. The 10 ms "
          "explicit run died nonfinite on cell 7196644737 after 2,705 s; the implicit retry aborted at the "
          "wall cap after 21,286 s with no result.",
          "Construction is not simulation: until a driven window of physiological length completes, no "
          "population number describes anything the cells do. With the step now qualified this is a pure "
          "cost problem, not an accuracy one - the r2 attempt was already running at the qualified "
          "0.000625 ms and still burned 5.9 hours for no result. What is needed is a cheaper window "
          "(fewer cells, a shorter duration, or a coarser step justified by a per-cell ladder), not a "
          "fourth full-population attempt. Confidence 30 percent that a driven window completes within "
          "this campaign's cost policy.",
          [DRIVEN104, LADDER, LEDGER], "supported", 30),
        n("pop_dt", CLOSED,
          "STEP 5d - CLOSED 2026-09-14: the integration step IS qualified, at dt 0.000625 ms. The four "
          "adjacent pairs give 6.365, 3.471, 1.820 and 0.933 mV, so the last halving lands inside the 1 mV "
          "contract; the ratios 1.83/1.91/1.95 are first order in dt, matching what the I-cell transfer "
          "study found independently.",
          "The node previously said the ladder could not close because the three finer rungs' traces had "
          "been deleted. They had not: all five survived in the 2026-09-09 worktree-recovery stash, which "
          "was restored while cleaning up the stale worktrees, and the ladder closed as arithmetic with no "
          "new simulation. The earlier FAIL verdict had compared only the two coarsest rungs. Note that "
          "0.000625 ms is exactly the step the failed r2 run was already using: that run died on cost, not "
          "on accuracy. Confidence 90 percent that this stays closed - it is a convergence reading on "
          "committed traces, and the only way it reopens is if this one isolated cell is unrepresentative "
          "of the other 103.",
          [LADDER, DTLADDER, LEDGER], "supported", 90),
        n("pop_anatomy", BLOCK,
          "STEP 5e - BLOCKER, the binding one, and previously unnamed: a donor's fitted conductances do "
          "not survive the move onto H01 anatomy. B3 fires 4 spikes at 200 pA on the donor's own "
          "reconstruction and 0 on the H01 skeleton, sitting on a -78.2 mV plateau; onset capacitance 785 "
          "pF against 125, input resistance about 38 MOhm against about 98.",
          "This is the only term in the ledger with a NEGATIVE reading rather than a missing one, and it "
          "makes the others moot: donor accuracy, type coverage and a working build all describe a model "
          "that is silent on the anatomy it is meant to run on. Whether the load is physical or an "
          "artefact of the conversion (skeleton radii used as cable radii) was left unestablished. "
          "Confidence 40 percent that a graded load change restores a readable response without "
          "refitting.",
          [MORPH1, LEDGER], "supported", 40),
        n("pop_noground", BLOCK,
          "STEP 5c - BLOCKER, unfixable by simulation: there is no H01 electrophysiology. No H01 cell can "
          "be scored against its own recording, ever, from this data.",
          "So a population 'accuracy' is a transfer claim, not a measurement. The defensible form is a "
          "product of six separately sourced terms: donor accuracy against a real recording (0.718, and "
          "measured for the 28 cells B3 donates to, not for all 104), type-match coverage (0.529), build "
          "(1.000), a driven physiological window (0.000), a qualified timestep (1.000, closed at dt "
          "0.000625 ms) and anatomy "
          "transfer (0.000, measured and negative). A term with no evidence scores zero and is never "
          "dropped - dropping it raises the score of exactly the case that fails it, which is the bug "
          "the single-cell scorer already made once. Anything stated as one percentage without those six "
          "named is not supportable. Confidence 95 percent that this limitation stands.",
          [POP, TYPES, LEDGER, LEDGER_JSON], "supported", 95),

        # ---------------- the goal --------------------------------------------------------
        n("goal", GOAL,
          "GOAL: a scored model set over the 104 H01 cells. Reachable form: the product of the six ledger "
          "terms. As measured that product is 0 percent, because three terms have no positive evidence. "
          "The figure obtained by counting construction as a working simulation and assuming the other three "
          "terms away is 38 percent (0.718 x 0.529), and it is quotable only with those assumptions "
          "attached. The earlier 'about 4 percent' was arithmetic on a stale build term and is withdrawn.",
          "The 1-to-100 scale the goal asks for is therefore best read as this product, with all six "
          "factors reported separately so no single number hides a zero. Confidence lowered 15 -> 8 "
          "percent that the full product exceeds 80 percent within this campaign's scope. The reason is "
          "not the climb: it is that the ledger correction moved the binding constraint from an "
          "engineering blocker believed 75 percent fixable to anatomy transfer, which has a measured "
          "negative reading, and to a driven window that has twice failed to complete. The absence of H01 "
          "ground truth remains the one limit that cannot be removed at all.",
          [POP, TYPES, STAGE15, LEDGER], "unverified", 8),
        n("policy", PLAN,
          "STOPPING RULE, amended by the ledger: the remaining measurable gain on the E cell is 6.4 points "
          "(rise) plus 2.4 (everything else), and it applies to a score over 28 cells that is conditional "
          "on a transfer which currently fails. Single-cell work stops here; the next run belongs to "
          "anatomy transfer, which is also a single-cell run on the donor and needs no population.",
          "The original rule said 'stop single-cell work and either fix the population build or stop'. "
          "The build was already fixed, so that branch is void. Sparsity of effects still applies but the "
          "steep X has moved: a term with a measured negative reading dominates two terms that are merely "
          "small. The stage-1 record already names the split - check the H01 conversion's membrane area "
          "against the H01 surface mesh, and scale the donor's dendritic load by 1.5 and 2 to see whether "
          "the response degrades gradually or falls off a cliff.",
          [CM, STAGE15, MORPH1, LEDGER], "supported", 85),
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
        e("open_site", "coupling_confirmed", "describes", "the cell that passed"),
        e("coupling_confirmed", "plan_climb", "describes", "mechanism located, not yet usable"),
        e("plan_climb", "plan_rise", "describes", "then the only element left"),
        e("plan_climb", "plan_holdout", "describes", "generalisation test"),
        e("plan_holdout", "plan_second_cell", "describes", "cross-cell test"),
        e("plan_second_cell", "pop_typematch", "describes", "scale to the population"),
        e("pop_typematch", "goal", "describes", "coverage factor"),
        e("pop_typematch", "pop_build", "describes", "and the matched cells have to build"),
        e("pop_build", "pop_dt", "describes", "and the step has to be qualified"),
        e("pop_typematch", "pop_anatomy", "describes",
          "and the matched donor's fit has to survive H01 anatomy"),
        e("pop_build", "goal", "describes", "construction factor (closed: 104/104)"),
        e("pop_build", "pop_driven", "describes", "construction is not simulation"),
        e("pop_dt", "pop_driven", "describes", "the step is qualified; the cost is not"),
        e("pop_driven", "goal", "describes", "driven-window factor (zero)"),
        e("pop_dt", "goal", "describes", "timestep factor (closed: dt 0.000625 ms)"),
        e("pop_anatomy", "goal", "describes", "anatomy-transfer factor (zero, measured negative)"),
        e("pop_anatomy", "policy", "describes", "where the next run belongs"),
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
            "form is the product of six separately sourced factors - donor accuracy against a real "
            "recording (0.718, and measured only for the 28 cells B3 donates to), type-match coverage "
            "(55 of 104), build (1.000: all 104 import, construct to 808,495 compartments, initialise and "
            "complete a forward pass; the one-point SWC branch was repaired on 2026-09-08 and the earlier "
            "12-of-104 figure is withdrawn), a driven physiological window (0.000: two attempts, one "
            "nonfinite and one dead at the wall cap), a qualified timestep (1.000: closed on 2026-09-14 at "
            "dt 0.000625 ms from five recovered traces, 0.933 mV to the next halving) and anatomy "
            "transfer (0.000, and measured NEGATIVE: "
            "B3's fit fires 4 spikes at 200 pA on the donor's own reconstruction and 0 on the H01 "
            "skeleton). As measured the product is 0 percent; the 38 percent obtainable by assuming the "
            "two unqualified terms away is quotable only with those assumptions attached. The sealed "
            "holdout "
            "(sweep 54, 330 pA) has never been spent, so no element has been tested out of sample. Whether "
            "a mechanism that closes the climb on this cell transfers to another cell without refitting is "
            "untested. The ten-element mean scores voltage levels against a stated 100 mV span, which "
            "flatters it; the element table is the reading."
        ),
        "predictions": (
            "Stage 16 is closed: the threshold step did grow with falling axonal recovery and reached "
            "+2.39 mV against a recorded +1.38, and the same dose through the default 1 um stub gave "
            "-2.56 mV, the opposite sign - so the coupling is load-bearing and the mechanism is located. "
            "But every dose that moved the threshold collapsed the train from 9 spikes to 2, both "
            "registered cells failed, and the best arm scored 69.1 over ten elements, below B3's 71.8. "
            "The next registrable prediction is not on this cell's score: scaling the donor's dendritic "
            "load by 1.5 and 2 will show whether the H01 anatomy's silence at 200 pA is a graded "
            "consequence of a six-fold larger load (response degrades smoothly, so a load correction or a "
            "conversion fix restores it) or a cliff (so the conversion itself is suspect). That is the "
            "only split whose outcome changes the population ledger."
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
