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
SP16_E = "docs/evidence/h01-e-sodium-slow/stage-1-decision.json"
SP16_I = "docs/evidence/h01-i-sodium-slow/stage-2-decision.json"
TAKEOFF = "docs/evidence/h01-topographic/threshold-approach.json"
ONSET = "docs/evidence/h01-topographic/onset-shape.json"
STAGE9 = "docs/evidence/h01-topographic/stage-9.json"
STAGE10 = "docs/evidence/h01-topographic/stage-10.json"
STAGE11 = "docs/evidence/h01-topographic/stage-11.json"
STAGE12 = "docs/evidence/h01-topographic/stage-12.json"
STAGE13 = "docs/evidence/h01-topographic/stage-13.json"
STAGE14 = "docs/evidence/h01-topographic/stage-14.json"
STAGE15 = "docs/evidence/h01-topographic/stage-15.json"
ACCJSON = "docs/evidence/h01-topographic/human-vs-rodent-accuracy.json"
OBS16 = "docs/evidence/h01-topographic/stage-16-observation.json"
STAGE16 = "docs/evidence/h01-topographic/stage-16.json"
KEEPDROP = "docs/evidence/h01-keep-drop/decision.json"
Y7 = "docs/evidence/h01-driven-window-20260915/anatomy-transfer-decision.json"
C3SPEC = "docs/specs/2026-09-16-h01-c3-partner-expansion.md"
# Closing agents swap C3SPEC for these once the evidence files exist:
C3GRAPH = "docs/evidence/h01-c3-candidates.json"
C3KEPTGRAPH = "docs/evidence/h01-c3-kept-partner-graph.json.gz"
C3GRAPH_C3ID = "docs/evidence/h01-c3-candidates-c3id-only.json"
C3IMPORT = "docs/evidence/h01-c3-candidates-components.json"
C3B_RESULT = ("Result 2026-09-16 (h01-c3-candidates-import.json, -construct.json, -components.json, -archive.json): the pool from A is 32 (20 E + 12 I), not 40; 32 of 32 import (0 missing / 0 extra segments, 28 s), 32 of 32 construct and initialise with the deployed donor profile (316-12,995 compartments, 90 s). Archive sha256 1b4d232b: 704 components from 1,165,714 vertices (21 singletons dropped, 0 cycle edges), 352 mm cable, every cell's largest component carries the soma (19 have more than one soma-bearing component). 152 vertices carried the out-of-vocabulary label 99 and were written as unclassified. Export 30 min on the box (10-149 s per cell; label-volume chunks, mip 0). Control 1684504313 (h01-c3-control-1684504313.json): C3 / proofread = components 24 / 267, nodes 28,642 / 63,526, cable 8.6 / 18.2 mm, soma-labelled nodes 197 / 48 (the label volume marks a contiguous soma blob; the proofread SWC annotates 48 nodes); imports and constructs (5,151 compartments). The prediction held on every candidate that exists; the 40 was the pool size, refuted in A.")
C3KEEP = "docs/evidence/h01-c3-keep-drop/decision.json"
C3NET = "docs/evidence/h01-c3-verified-network.json"

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
        n("sp16", X, "SP16 (executed, 2x2 dissection {E, I} x {fitted sodium, slow-inactivating sodium}): REFUTED in both cells. Removing 40 percent of sodium availability along the train takes the rise to 0.69-0.72 of spike 1 and moves the threshold +0.4 mV at spike 2 (E) and +1.9 mV (I), where the recordings move +1.4 and +4 mV with the rise at 0.86-0.91. Sodium availability is a rise lever in both fits, 12-19 percent of rise per mV; the recorded cells move threshold at 2-10 percent per mV.",
          "Stage 0 reproduced both libraries exactly. The 200 pA rows (count 4, level) are below the discrimination line: the recorded cell's own repeats give 0 or 1 there. A gate fitted to a response cannot name a channel.", [SP16, SP16_E, SP16_I], "supported"),
        n("q5", Q, "Q5 (elemental, retained traces, no run). Is the take-off a fixed voltage, or is it set by the trajectory that reaches it? Threshold under two definitions against the approach rate (up to 10 ms before it), every spike, both humans and both fits; the 200 pA twins; the subthreshold staircase. Sweep 54 withdrawn from the stage-6 tables.",
          "Hartshorne ch.6: sweep the state space you already hold before buying a new observation; ch.3: tightly connected pairs (approach, take-off) constrain everything that is not the difference.", [TAKEOFF, STRATEGY], "supported"),
        n("q5_e", R, "E human: spike 1 takes off at -54.5 to -55.2 mV at 0.2 mV/ms (200 pA), -57.2 at 0.75 (350 pA), -61.9 under the 3 ms short squares at 7 mV/ms; 2.2 mV slide over the long pulses (9 sigma) under both definitions; all 41 later spikes sit 1.9 mV (median, 8 sigma) above the spike-1 relation. E fit: -57.0 to -57.2 mV at every spike of every train, 0.23 to 1.16 mV/ms (0.8 sigma). Twins: one spike or none at 200 pA, same level 800 ms later (+0.16 mV, 0.5 sigma).",
          "The recorded take-off is set locally by the trajectory; the fit's is imposed from the axonal initiation. The Y4 'level after a spike' row was a steady-state I-V difference (recorded Rin 75 MOhm below 70 pA, 91 MOhm from 90 to 190 pA). This is the relation that made SP16 predictable on paper.", [TAKEOFF, CM], "supported"),
        n("q5_i", R, "I human: spike 1 at -58 mV (120 pA, 0.7 mV/ms), -60.5 (0.19 nA, 1.1), -61.7 (0.27 nA, 1.7): 4 mV slide, 10 sigma of the spike-1 spread, both definitions. I fit: slides 0.7 mV over 0.44 to 1.72 mV/ms and holds within 0.3 mV along each train: not fixed to 1 sigma, a fifth of the recorded slide. Later spikes +1.0 mV median (inside 3 sigma); the I along-train rows have no repeat spread (the four repeats are single-spike 120 pA sweeps).",
          "Registered prediction 2 fails for the I fit; the reading for I is partial.", [TAKEOFF], "supported"),
        n("q6", Q, "Q6 (SP15 stage 8, retained traces, no run): soft or hard take-off. Onset span from 10 to 100 V/s and phase-plane rapidness at every spike; both recordings at the soma, both fits at the soma and at their retained axon point.",
          "Registered prediction: each fit's somatic onset is a kink because its take-off is imposed from the axon. Refuted in both.", [ONSET, STRATEGY], "supported"),
        n("q6_e", R, "E: recording and fit have the SAME somatic onset (5.0 vs 5.1 mV span, 0.4 sigma; rapidness 34 vs 41/ms), both gradual, although the fit's axon point fires 0.28 ms first and is at -40 mV when the soma starts. No discrimination by shape. The fit's take-off is set at or proximal to the axon point, 45 um out (-54.3 to -55.2 mV there at every spike); under the SP16 gate at depth 0.6 it climbs only 2.1 mV by spike 10 for 40 percent availability lost.",
          "A dense fast site crosses its own threshold at nearly the same voltage whatever its availability; the recording moves 2.2 mV with approach and 1.9 mV more after one spike: more than availability at such a site gives.", [ONSET, CM], "supported"),
        n("q6_i", R, "I (PROVISIONAL: the registered rejection fired, the four repeats spread 1.16 mV in span; read as the ratio of spans, 5.2, not the absolute bands): the recorded onset covers 2.7 mV (52/ms); the fit's is a 14.2 mV whole-cell turnover (5/ms), 20 sigma; the fit's soma and first axon section rise together (spike 1 within 0.06 ms, later spikes soma first). The finalist has NO sharp initiation site; the earlier axon-first readings were -20 mV crossings (upstroke). Largest elemental contrast after the count; a contrast of the initiation site, an input.",
          "Also why the I fit's take-off answers to the approach at a fifth of the recorded rate: no site whose own threshold the approach can move.", [ONSET, CM], "supported"),
        n("q7", Q, "Q7 (SP15 stage 9, executed: 3 E ramps + 1 I probe run, plus retained traces): the take-off's own variables. E recording: is the post-spike +1.9 mV a function of time since the spike? E fit: does the take-off move under ramps 0.37-2.53 mV/ms at fixed availability, read at soma, axon[0], axon[1]? I fit: where does the spike begin (five points)?",
          "Registered before the runs; the first E launch (2 nA ramp over the whole second) was killed unread after 30 min and the ramps were re-registered to end after the first spike.", [STAGE9, STRATEGY], "supported"),
        n("q7_e", R, "E: (a) the recorded post-spike step is UNRESOLVED between fixed and slowly recovering (+2.0/+2.2/+1.5 mV after intervals <30/30-100/>100 ms, not monotone, spread 0.69 vs 0.75 limit); it does not recover measurably inside the pulse and is gone by the next sweep. (b) The fit's SOMATIC take-off is fixed under ramps (0.07 mV, 0.37-2.53 mV/ms) but its AXONAL take-off slides 4.1 mV (axon[1]) / 2.4 mV (axon[0]) in the recorded direction and size; the axon leads by 0.30 ms at every rate; 1 ms before its take-off it is +1.2/+0.5/-0.5 mV relative to the soma (leads under the slow ramp, is driven under the fast one). Confound: the ramps also span 0.24-1.08 nA at take-off (x4.4) where the recording's steps span x1.75.",
          "The recorded approach signature exists in the fit AT THE SITE and is hidden from the soma by the 45 um x 1 um stub: the soma reads the arrival of the axonal spike at one voltage. Along trains the fit's axon take-off is a function of approach only: the recorded post-spike step has no counterpart at either site.", [STAGE9, CM], "supported"),
        n("q7_i", R, "I: confirmed at five points - no probe leads the soma by more than 0.08 ms, every span >= 8.9 mV; the model's axon ends at axon[1] (registered rejection fired by the letter: earliest point axon[1] by four samples, nothing beyond it). The finalist's spike is a whole-cell turnover; it has no initiation site.",
          "The stage-8 provisional I reading is now confirmed on a second run.", [STAGE9, CM], "supported"),
        n("q8", Q, "Q8 (SP15 stage 10, executed, 11 runs): an INPUT split - the coupling between the soma and the site, every channel held. Stub diameter x2 (total NaTs held), x2 (density held), x4 (total held); ramps 1.1/11/110 pA/ms and the 310 pA step.",
          "Prediction: the somatic take-off slides in the recorded direction with the coupling. Both registered rejections fired by the letter (x2 arms differ by 0.8 mV; onset span leaves 3.5-5.7 mV in every arm).", [STAGE10, STRATEGY], "supported"),
        n("q8_e", R, "The somatic take-off slides with the approach once the stub is thicker: control -0.07, x2 total -2.9, x2 density -2.1, x4 -3.4 mV (recorded -2.2); axon lead 0.30 -> 0.16/0.20/0.04 ms; 310 pA counts 10 and 9. The x2 density-held arm is the campaign's closest run to the recorded take-off AS A SLOPE (-2.8 mV per decade of approach vs the recording's -3.8; ranges overlap only at 0.45-0.75 mV/ms where the arm sits ~1 mV above; span 4.1-6.0 vs 3.5-5.7). No arm moves the rise (579/655 vs 639; recorded 348) or produces the post-spike +1.9 mV.",
          "The approach signature is an input (soma-site coupling), not a channel; the coupling also reshapes the onset (total-held arms merge the site into the soma, 6-11 mV spans). Sufficient for the approach signature under these conditions, but every dose also took the onset out of the recorded band: the split did not separate the reading from the spike. Does not name the recorded geometry.", [STAGE10, CM], "supported"),
        n("q9", Q, "Q9 (SP15 stage 11, executed, 2 runs + retained traces, PASS): the two remaining elements split apart. Rise: on the x2 density-held geometry, somatic NaTs density 0.9/0.7/0.5 at 310 pA. Post-spike step: the recording's short squares 27-31 (27 unprimed after five silent sweeps, 28-30 each 2.7-4.2 s after one spike, 31 15.8 s after) and the later-spike residual by spike index.",
          "The registered next-long-square read could not fire (173 s to the first spiking long square; recorded as a fired no-reading before the traces were opened) and was replaced by the primed/unprimed short-square series.", [STAGE11, STRATEGY], "supported"),
        n("q9_rise", R, "Rise follows the somatic density monotonically: 655/586/505 V/s at 0.9/0.7/0.5 (about 375 V/s per unit, read over 0.5-0.9 only); the somatic take-off holds (-55.46/-55.36/-55.32 mV at 0.63/0.61/0.59 mV/ms, corrected moves +0.07/+0.06); count 9 at every dose; onset span 4.1/5.0/5.4 mV, in band. The recorded 348 V/s lies 157 V/s below the lowest dose.",
          "On this geometry the somatic density is a rise lever and nothing else, as the causal model predicted; the slope is not carried past its range.", [STAGE11, CM], "supported"),
        n("q9_step", R, "The post-spike step is a per-spike step of +1.9 mV: full after one spike (spike 2 residual +1.83 mV, n 6), not growing with the count (spikes 6+ +2.23, difference +0.40 mV, limit 0.75), not recovering measurably within 0.73 s (stage 9) and gone within 2.7 s (primed short squares -0.16 to +0.56 mV from the unprimed, inside the independent 3 sigma of 0.75). Limits: nothing inside the first 2.7 s; assumes the step is as visible at 7 mV/ms as at 0.2-0.75.",
          "A process a single spike starts, holding about a second and clearing within three, raising the take-off without costing more than a few percent of rise; no element of the fit produces it (SP16 slow inactivation recovers on the wrong timescale and costs rise).", [STAGE11, CM], "supported"),
        n("q10", Q, "Q10 (SP15 stage 12, executed, retained traces only): the recording's MEASUREMENT CHAIN, the split the book (ch. 4 Table 3, ch. 6 observation point) says comes first and the campaign never made. Pipette pole from the short-square step edges (transient area), corner from the noise floor and an assumed conservative 10 kHz; applied to the model traces; every quantity re-read.",
          "First pass with the corner free returned 2.65 kHz and would have read the whole rise gap as the chain; refuted by the flat noise floor and the two-sample edge jump (the step carries the command's own pole). Recorded, then amended.", [STAGE12, STRATEGY], "supported"),
        n("q10_chain", R, "Chain: pipette pole 5 us (E) / 2 us (I) at best, bound 30 us; neutralisation live in practice (a live 5 pF pole would put 9 mV at every edge, none is there); bridge mismatch +0.24 / -0.96 MOhm; corner >= 20 kHz if the noise floor is pipette noise; an assumed 10 kHz applied as the conservative case (no filter setting in the file). E fit rise 655 -> 609 (10 kHz) / 638 (20 kHz), control 639 -> 570 / 615: 5-15 percent of the gap to 348; residual 260-290 V/s real. Peak bound would cover 55 percent (fired by the letter at a bound an order of magnitude above the measured pole). Fall: recording -104 vs fit -91/-95 while rising slower - no chain does that.",
          "Slow set holds (take-off constant 0.13 mV shift, slide to 0.01 mV; approach 1 percent; lead 0.02 ms; count): stages 7-11 stand. The 10-100 V/s onset span is chain-fragile (E 4.1 -> 5.5/5.7, non-monotone across ramps): stage-8 E cell qualified; I span survives (12.8 vs 1.9). I fit rise 592 -> 530 through its chain: the raw equality with the recording's 597 was a coincidence of observation points.", [STAGE12, CM], "supported"),
        n("q11", Q, "Q11 (SP15 stage 13, executed, 4 runs): the BASE as a sub-system, a Dissection half-split. J: every stage dosed one Allen fit inside a frame never questioned; the two largest contrasts are sodium-kinetic signatures. Facts: B3 is Allen 626170538, the perisomatic fit of THIS human cell (541563728) under the rodent-lineage kinetic set; the I base HL5BN1 is already human-fitted and misses the climb too. System B: Toronto HL23PYR (ModelDB 267595), driven as the recording is driven, read through the chain.",
          "Registered cells: rise within 15 percent of 348; climb at least half the recorded; count 5-15 at 0.31; the second half (HL23PYR biophysics on this anatomy) only if the first passes.", [STAGE13, STRATEGY], "supported"),
        n("q11_rise", R, "Rise is NOT the fitting pipeline: HL23PYR 592-600 V/s through the chain at every drive (B3 545-570, recorded 332-351). Two independently fitted human L2/3 models on different anatomy agree; they share the sodium equation family, the ideal clamp and 34 C. Elemental picture: the three spike-1 loops coincide to about -40 mV and separate above it - the excess is in the somatic phase, not the site-delivered one. Fall: HL23PYR -80, B3 -94, recorded -104.",
          "Base stays B3 for the rise; the shared sodium equations are the sub-system the next split swaps or doses as a characteristic curve.", [STAGE13, CM], "supported"),
        n("q12", R, "Q12 (SP15 stage 14, re-analysis of a retained trace, no run): the rise decomposed by observation point. On m0-b3-currents-sweep53 the soma-midpoint spike-1 upstroke divides into the currents that carry it; the balance closes on the solver's native grid (the injected clamp is one term - the resampled grid breaks the near-cancelling axial legs). Above -40 mV the somatic-total drive (local NaTs/Nap + the two 0.01 MOhm intra-soma legs that carry the rest of the soma's sodium) is the majority of dV/dt, the axon[0] site-delivered leg the minority; below -40 mV axon[0] carries it. Recorded/fit dV/dt through the chain: 0.96 at -40 (the hand-over, loops match), 0.74/0.64/0.51 at -20/0/+20 - a somatic-phase inward deficit that widens with voltage.",
          "The rise excess is the soma's own sodium above -40 mV, not the site-delivered leg; the shared sodium is named by the carrying current, not merely by what the bases share. The dose is justified as the finishing step; had axon[0] been the majority it would be refused for the coupling.", [STAGE14, CM], "supported"),
        n("q11_climb", R, "Climb has a positive control: HL23PYR +1.38 mV at spike 2 (the recorded step exactly) / +1.78 at spike 5 at 0.31 nA, rise kept 0.95; B3 -0.02 / +0.24; recording +1.38 / +3.63 at 0.86. Shape: HL23PYR steps in the first interval then holds; the recording steps then accumulates +2.2 mV more. Excitability is the wrong cell's: rest -74 vs -84, counts 16/18/20/24 vs 1/5/10 (cell e fired). Take-off relation read over 1.1-1.9 mV/ms, disjoint from 0.2-0.75, not compared; 0.4 nA take-off is a reader failure (excluded).",
          "A human-fitted sodium reproduces the step, neither base the accumulation.", [STAGE13, CM], "supported"),
        n("q13", R, "Q13 (SP15 stage 15, executed, 7 runs): the rodent-vs-human KINETIC LINEAGE, the last unquestioned input in the frame. B3's somatic rodent NaTs (gbar 2.641) zeroed and the Toronto human NaTg inserted with HL23PYR's human voltage shifts (vshiftm 13, vshifth 15, slopem 7), on this cell's own anatomy and densities, dosed 0.068 to 5.282 S/cm2 (78-fold).",
          "REFUTED. Below B3's own conductance the cell is SILENT (plateau -49 mV, ABOVE its own -57 mV take-off; 8x density moves it 0.8 mV): the +13 mV activation shift is fitted against the Toronto K set and never activates against Allen's, which was fitted to a sodium activating 13 mV lower. Kinetics are fitted as a SET, not interchangeable parts. At matched conductance the rise goes UP (625, then 921 V/s) against the recorded 348. Accuracy: B3 71.8, best human-sodium arm 62.0, fully human-fitted HL23PYR 72.3 - 'human rather than mouse' is worth half a point.", [STAGE15, ACCJSON], "supported"),
        n("q14", R, "Q14 (SP15 stage 16 part 1, observation on retained traces, no run): can ANY somatic conductance move the threshold? Somatic outward current at a common subthreshold voltage before each spike, against the take-off along the train.",
          "NO, and the registered somatic dose was REFUSED before it ran (4 evaluations saved). Outward current accumulates (+22.5 percent by spike 2, doubling by spike 4) while the take-off does not move at all (-57.20, -57.42, -57.16, -57.20, -57.20 mV). At the take-off the axon[0] leg delivers +0.33 nA against a total accumulated somatic brake of ~0.012 nA - 3 percent of the trigger. Carriers at take-off: leak 69 percent, Kv3_1 30 percent, Im 0.4 percent. B3's somatic threshold is the ARRIVAL TIME of the axonal spike.", [OBS16, STAGE14], "supported"),
        n("q15", R, "Q15 (SP15 stage 16 part 2, executed, 4 runs): the climb dosed at the INITIATION SITE. Axonal sodium recovery 0.5/0.25/0.125 on the stage-10 thickened-stub arm, plus the strongest dose on the default 1 um stub as the coupling control.",
          "MECHANISM CONFIRMED, NOT USABLE. The first-interval step moves -0.07 -> +0.89 -> +2.39 mV, PAST the recorded +1.38: the site can move the threshold by more than the recording asks, where no somatic conductance moved it at all. The coupling control passed: the same dose on the 1 um stub gives -2.56 mV, the OPPOSITE sign, so accommodation reaches the soma only through the thickened coupling (stage 10 confirmed, not assumed). But every dose that moves the threshold collapses the train (9 -> 3 -> 2 -> 2 spikes); the registered guard fired at every dose and the series is non-monotone. Over the same ten elements the best arm is the CONTROL at 69.1 percent, below B3's 71.8.",
          [STAGE16, CM], "supported"),
        n("next", T, "Next: not registered. The climb needs a site process SLOWER than the 13.5 ms interspike interval, able to accumulate a threshold without gating the next spike; the recovery-factor family acts on the interval's own timescale and cannot do both. Remaining: spend the sealed holdout (sweep 54), then the second human cell, then the population build. Accuracy ledger: climb 19.4 points, rise 6.4, all else 2.4; best measured model remains B3 at 71.8 percent.",
          "Y before levers; register the outcome cells first.", [STRATEGY], "unverified"),
        n("q2_upstroke", O, "E upstroke split (SP15 stage 1, executed): B3 genome on the H01 skeleton 955432427 beside the donor anatomy at nseg 1. Mesh control held (653 vs 598 V/s, 9 percent; count, first spike, threshold unchanged). The H01 anatomy did not spike at 200 pA (plateau -78 mV, onset capacitance 785 vs 125 pF, input resistance about 38 vs 98 MOhm): registered no-reading outcome.",
          "The cable load moves the low-input response by more than any tested channel change, but the change was too large to read the upstroke. Next: a graded cable change on the donor anatomy (membrane area x1.5, x2), and a check of the H01 conversion against the surface mesh.", [SP15_S0, "docs/evidence/h01-e-morphology/stage-1-decision.json", STRATEGY], "supported"),
        # policies
        n("policy", P, "Policies: contrasts > 3 sigma of the repeat envelope; samples of three; Y only; every split written before its data; fail fast under the cap; both tiers; sweeps 54 and 48 sealed.",
          "From Hartshorne chapters 1-5 and the campaign's own AGENTS rules.", [STRATEGY], "supported"),
        # Q-C3: the population branch after keep/drop (registered 2026-09-16, before any run)
        n("qc3", Q, "Q-C3 (registered 2026-09-16, no run yet). Keep/drop left 17 pyramidal cells, 0 interneurons, 0 contacts (Y7: donor densities do not transfer onto 2-4e3 um2 components). Does the non-proofread C3 release supply partners of the kept 17 that hold a test-to-failure gate at the deployed donor densities, and anatomical synapses among them? A dissection over 40 more anatomies with every profile held; no donor, no fit, no tuning.",
          "Gate: firing range (model rheobase within one human sweep step of the human's; still fires at the human's highest recorded amplitude; count at the primary input inside the human repeat range), rest within 3 sd of the donor level, dt-half repeat. The former 10 mV / 30 percent bands are recorded beside it, not scored.", [C3SPEC, KEEPDROP, Y7], "unverified"),
        n("qc3_graph", O, "C3-A: partners exist. Prediction: every kept cell has a neuron partner in the 166-shard synapse export (NSI 300-2,300 per kept cell); at least 40 partners with >= 2 synapses to the kept set. Falsifier: fewer than 20 partners in total. Measured under two joins, refuted under both.",
          "Join 1, C3 neuron_id on both sides (h01-c3-candidates-c3id-only.json, 114,126 neuron-to-neuron rows): 85 partners, 4 with >= 2 synapses to the kept set (15 either direction), one kept cell without a partner; 20 E + 12 I. That join was defective: a C3 soma-bearing segment carries almost none of its own axon, so every kept axon's output was unseen; checked against join 2, its 11 highest-ranked candidates rest on soma- or dendrite-presynaptic rows only and 21 of its 32 have no AXON-presynaptic row. Join 2, kept cells by proofread base-segment ownership (h01-c3-kept-partner-graph.json.gz, 378 rows, 17.2 min box time): kept -> tabulated 167 rows (102 AXON-pre), tabulated -> kept 90 (27 AXON-pre), kept -> kept 0, same-cell 121; the kept axons place a further 910 synapses on untabulated segments and 32,823 untabulated axon fragments synapse onto the kept cells. Counting AXON-pre rows only: 91 partners (falsifier not tripped), 2 with >= 2 synapses to the kept set (14 either direction); kept cells 2001418787 and 3571083397 have no AXON-pre partner; eligible E 61, I 28; candidates 20 E + 20 I (h01-c3-candidates.json). The first verdict was an artefact of the join; the second measures the biology available through the tabulated segments.", [C3GRAPH, C3KEPTGRAPH, C3GRAPH_C3ID, C3SPEC], "refuted"),
        n("qc3_import", O, "C3-B: candidate skeletons import through the unchanged proofread SWC path (subcompartment labels = SWC type codes + 100). Prediction: 40 of 40 import, construct, initialise. Control: kept cell 1684504313 exported from C3 beside its proofread SWC (recorded, not a gate).",
          "Non-proofread segments carry merge and split errors; the import audit and the gate are the filter. " + C3B_RESULT, [C3IMPORT, C3SPEC], "supported"),
        n("qc3_e", O, "C3-C (E): 1-4 of 20 pyramidal candidates pass the test-to-failure gate (the former bands alone would pass 3-6, the proofread rate). Kept 17 under the same ramp: about half the 13 L2 show a block edge under 3x input; the 4 L4 hold. A kept cell that fails is reported to J, not dropped.",
          "Y7 named the somatic active profile on a 107-315 um2 soma region as the surviving condition; this branch measures how often that condition falls in the transferring range.", [C3SPEC, Y7], "unverified"),
        n("qc3_i", O, "C3-C (I): 0-2 of 20 interneuron candidates pass at the HL5BN1 / HL5MN1 densities. Falsifier for the interneuron route: 0 of 20 -> a fit-pilot decision request is written for J (per-cell human fit spec, 2026-09-15); no tolerance is widened and no cell is re-run.",
          "PV was silent on all 18 proofread cells at 0.19 nA; SST fired before the pulse on all 7.", [C3SPEC, Y7], "unverified"),
        n("qc3_net", O, "C3-D: at least one construction-ready contact among the passing cells, each with a delivery weight sweep (unresolvable edge under 3 sd of the post cell's rest noise; fires-alone edge; literature pin 3.1 nS, 0.5-1 mV; I->E swept with the post cell held depolarised because the -80 mV reversal sits above the -84 mV rest). Falsifier: none -> the campaign stops at the network JSON.",
          "Y5: identity, placement and electrical parameters are three separate checks; an anatomical contact alone predicts neither suppression nor recruitment.", [C3SPEC], "unverified"),
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
        cm.edge("q4_e", "sp16", "describes", "one gate, both fits"),
        cm.edge("q4_i", "sp16", "describes", "one gate, both fits"),
        cm.edge("sp16", "q5", "describes", "back to Y: where is the take-off set"),
        cm.edge("q5", "q5_e", "describes", "E"),
        cm.edge("q5", "q5_i", "describes", "I"),
        cm.edge("q5_e", "q6", "describes", "isolation before any lever"),
        cm.edge("q5_i", "q6", "describes", "isolation before any lever"),
        cm.edge("q6", "q6_e", "describes", "E"),
        cm.edge("q6", "q6_i", "describes", "I"),
        cm.edge("q6_e", "q7", "describes", "the take-off's variables"),
        cm.edge("q6_i", "q7", "describes", "locate the initiation"),
        cm.edge("q7", "q7_e", "describes", "E"),
        cm.edge("q7", "q7_i", "describes", "I"),
        cm.edge("q7_e", "q8", "describes", "change the coupling only"),
        cm.edge("q8", "q8_e", "describes", "E"),
        cm.edge("q8_e", "q9", "describes", "rise and post-spike step, apart"),
        cm.edge("q9", "q9_rise", "describes", "rise"),
        cm.edge("q9", "q9_step", "describes", "post-spike step"),
        cm.edge("q9_rise", "q10", "describes", "measure the chain before sizing the rise"),
        cm.edge("q10", "q10_chain", "describes", "the chain"),
        cm.edge("q10_chain", "q11", "describes", "the frame itself as a sub-system"),
        cm.edge("q11", "q11_rise", "describes", "rise"),
        cm.edge("q11", "q11_climb", "describes", "climb"),
        cm.edge("q11_rise", "q12", "describes", "decompose the shared rise by observation point"),
        cm.edge("q12", "q13", "describes", "dose or swap the sodium equations"),
        cm.edge("q13", "q14", "describes", "lineage refuted; the climb is where the accuracy is"),
        cm.edge("q14", "q15", "describes", "no somatic lever; dose the site"),
        cm.edge("q15", "next", "describes", "site moves the threshold but takes the train with it"),
        cm.edge("q11_climb", "next", "describes", "the step has a control; the accumulation has none"),
        cm.edge("q9_step", "next", "describes", "a 1-3 s per-spike process"),
        cm.edge("levers", "q3", "describes", "why the dose scans stop here"),
        cm.edge("policy", "q1", "describes", "applies to every split"),
        cm.edge("next", "qc3", "describes", "the population build, on the cells the keep rule left"),
        cm.edge("qc3", "qc3_graph", "describes", "A: partners"),
        cm.edge("qc3", "qc3_import", "describes", "B: skeletons"),
        cm.edge("qc3_graph", "qc3_e", "describes", "C: E candidates"),
        cm.edge("qc3_graph", "qc3_i", "describes", "C: I candidates"),
        cm.edge("qc3_import", "qc3_e", "describes", "C: E candidates"),
        cm.edge("qc3_import", "qc3_i", "describes", "C: I candidates"),
        cm.edge("qc3_e", "qc3_net", "describes", "D: contacts and manifest"),
        cm.edge("qc3_i", "qc3_net", "describes", "D: contacts and manifest"),
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
