"""Build the C3 partner-expansion campaign prerequisite tree as a Reasoning Studio document.

Companion to build_search_tree.py (the Q-C3 split lives there). This map is the
prerequisite tree of docs/specs/2026-09-16-h01-c3-partner-expansion.md: the goal, the
obstacles, the intermediate objectives (workstreams A-D) and the failure-edge datums that
define acceptance, all registered before any run. Reuses the CLI helpers of build_map.py.
Closing agents set a node's status and add its evidence file when a stage closes.
"""

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_map as cm  # noqa: E402

OUT = cm.HERE / "h01-c3-campaign.reasoning.json"
TITLE = "H01 C3 partner expansion: campaign prerequisites"
DATE = "2026-09-16"
ORIGIN = f"H01 C3 partner-expansion prerequisite tree; registered {DATE} before any run"

SPEC = "docs/specs/2026-09-16-h01-c3-partner-expansion.md"
KEEPDROP = "docs/evidence/h01-keep-drop/decision.json"
Y7 = "docs/evidence/h01-driven-window-20260915/anatomy-transfer-decision.json"
EVOLVE = "docs/evidence/h01-keep-drop/evolve/evolve-record.json"
REST = "docs/evidence/h01-keep-drop/donor-rest.json"
DATUMS = "docs/evidence/h01-c3-keep-drop/human-datums.json"
ALLEN = "docs/evidence/h01_allen_type_population.py"
SYN = "docs/evidence/h01-ie-synapse-literature.json"
MAPPING = "docs/evidence/h01-cellmatrix-mapping-audit.json"
JOIN = "docs/evidence/h01-full-export-join-progress.json"
C3CAND = "docs/evidence/h01-c3-candidates.json"
C3IMPORT = "docs/evidence/h01-c3-candidates-components.json"
C3CTRL = "docs/evidence/h01-c3-control-1684504313.json"
C3B_RESULT = ("Result 2026-09-16 on the kept-partner list of 40 (h01-c3-candidates-import.json, -construct.json, -components.json, -archive.json sha 75d15798): 40 of 40 import (0 missing / 0 extra segments, 34 s), 40 of 40 construct and initialise with the deployed donor profile (20 E, 20 I; 316-16,732 compartments; 105 s). Archive: 668 components from 1,434,425 vertices (24 singletons dropped, 0 cycle edges), 433 mm cable; every cell's largest component carries the soma (14 have more than one soma-bearing component); 152 vertices carried the out-of-vocabulary label 99 and were written as unclassified. Export 30 min on the box for the 29 new cells (8-178 s per cell); the 11 cells shared with the superseded list were reused from the stage. An earlier 32 of 32 on the C3-id-only list (archive 1b4d232b, h01-c3-candidates-c3id-only.json) is superseded and recorded as such in the provenance. Control 1684504313 (h01-c3-control-1684504313.json): C3 / proofread = components 24 / 267, nodes 28,642 / 63,526, cable 8.6 / 18.2 mm, soma-labelled nodes 197 / 48 (the label volume marks a contiguous soma blob; the proofread SWC annotates 48 nodes); imports and constructs (5,151 compartments). Prediction B held.")
C3KEPTGRAPH = "docs/evidence/h01-c3-kept-partner-graph.json.gz"
C3CAND_C3ID = "docs/evidence/h01-c3-candidates-c3id-only.json"
CM = "docs/h01-causal-model.md"

COLORS = {
    "Goal": "#477a67",
    "Intermediate objective": "#5b79ab",
    "Datum": "#83709c",
    "Obstacle": "#b86e52",
}
G, IO, D, OB = COLORS


def n(nid, ntype, text, notes, files, status):
    d = cm.node(nid, ntype, text, notes, files, status)
    d["origin"] = ORIGIN
    return d


def build_nodes():
    return [
        n("goal", G, "Goal: Example 21's H01 backend starts from a connected human E/I core: kept pyramidal cells, interneurons that hold a test-to-failure gate at the deployed donor densities, and anatomical synapses among them, in one two-archive manifest.",
          "Outcome condition only: it does not claim Example 21 learns and it does not repair the Y7 anatomy-transfer condition. No donor is imported, nothing is fitted, no density or anatomy is tuned; the 87 dropped cells stay dropped.", [SPEC, KEEPDROP, CM], "unverified"),
        n("io_graph", IO, "A: C3 synapse graph and ranked partners of the kept 17 (box, /workspace/venv-c3). h01_c3_cell_table.py; h01_c3_neuron_graph.py (C3 neuron_id join, superseded) and h01_c3_kept_partner_graph.py (kept cells by proofread base-segment ownership) over all 166 export shards; h01_c3_partner_rank.py -> top 20 E + top 20 I by AXON-presynaptic synapse count with the kept set.",
          "Prediction: every kept cell has a neuron partner (NSI 300-2,300); at least 40 partners with two or more synapses to the kept set. Falsifier: fewer than 20 partners in total. Measured twice. Join 1 (C3 neuron_id both sides; h01-c3-candidates-c3id-only.json): 85 partners, 4 with >= 2 to the kept set, one kept cell without a partner, 20 E + 12 I; defective because a C3 soma-bearing segment carries almost none of its own axon, so no kept axon output was counted; checked against join 2, its 11 highest-ranked candidates rest on soma- or dendrite-presynaptic rows only and 21 of its 32 have no AXON-presynaptic row. Join 2 (base-segment ownership; h01-c3-kept-partner-graph.json.gz, 378 rows, 17.2 min): kept -> tabulated 167 rows (102 AXON-pre), tabulated -> kept 90 (27 AXON-pre), kept -> kept 0; 910 kept-axon synapses land on untabulated segments and 32,823 untabulated fragments synapse onto the kept cells. AXON-pre only: 91 partners (falsifier not tripped), 2 with >= 2 to the kept set (14 either direction), kept cells 2001418787 and 3571083397 without a partner; eligible E 61 / I 28 -> 20 E + 20 I candidates (h01-c3-candidates.json). Refuted under both joins; the pool handed to B and C is the join-2 list.", [C3CAND, C3KEPTGRAPH, C3CAND_C3ID, SPEC, JOIN, MAPPING], "refuted"),
        n("io_skel", IO, "B: candidate C3 skeletons written as proofread-format SWC (type = subcompartment label - 100; x,y,z in 32/32/33 nm voxels; radius nm), archived with sha256 and non-proofread provenance. H01Archive accepts an explicit digest; H01ArchiveSet resolves a source by archive_sha256. Import audit through the unchanged path.",
          "Prediction: 40 of 40 import, construct, initialise. Control: kept cell 1684504313 exported from C3 beside its proofread SWC (28,643 vs 63,526 nodes), recorded, not a gate. " + C3B_RESULT, [C3IMPORT, C3CTRL, SPEC], "supported"),
        n("io_gate", IO, "C: test-to-failure on the 40 candidates and the 17 kept cells (box, two chains): the donor step run, a ramp run (--ramp-na, 0 to 3x the donor test current) reading rheobase and block current, the dt-half repeat. decision.json under docs/evidence/h01-c3-keep-drop/.",
          "Predictions: E 1-4 of 20 pass (the former bands alone would pass 3-6); I 0-2 of 20; about half the 13 kept L2 show a block edge under 3x, the 4 L4 hold. Falsifier for the interneuron route: 0 of 20 -> a fit-pilot decision request to J; no tolerance widened, no re-run. A kept cell that fails is reported, not dropped.", [SPEC, Y7], "unverified"),
        n("io_net", IO, "D: contacts among passing cells from the C3 graph (identity by neuron_id; placement as recorded projection distance, blocker beyond 3 um on the proofread 17; delivery weight sweep per contact), probe evidence, two-archive manifest, one receipted evolve launch under timeout 3600.",
          "Prediction: at least one construction-ready contact. Falsifier: none -> stop at the network JSON and report. Y5: identity, placement and electrical parameters are three separate checks; an anatomical contact alone predicts neither suppression nor recruitment.", [SPEC, CM], "unverified"),
        n("datum_fire", D, "Hard failure edges (J, 2026-09-16, gate split while the chains ran; runs unchanged): finite traces; no -20 mV crossing before the pulse; recruitable_in_human_range = the 1 s ramp rheobase at or below the human's highest still-firing long-square amplitude (long-square-index.json / NWB stimulus, LJP-corrected) and the step run at the primary drive firing at least once; the ramp still firing at the human's highest amplitude; no depolarisation block at or below it; the dt-half repeat reproducing the count. rheobase_in_step (ramp rheobase within one sweep step of the human long-square rheobase) is recorded, not scored (coordinator 2026-09-16, flagged to J): a 570 pA/s ramp and a step are different measurement chains, and the donors' own Allen recordings document a step-to-ramp offset (threshold_i_long_square -> threshold_i_ramp: L2 200 -> 236, L4 50 -> 52, PV 120 -> 192, SST 50 -> 77 pA; human-datums.json allen_ramp_threshold, pull allen-donor-ephys-features.json a701d082); both human thresholds sit beside the model reading in every receipt. "
          "Plausibility rule, count: the step count at the primary input against the donor's own count there (L2 10, L4 12, PV 12, SST 13.25), tolerance 2 across-cell sd of num_spikes on the type population's 1 s long-square sweeps within one step of the primary drive (human-datums.json type_population.count: spiny L2/3 at 310 pA 14.2 over 147 cells; spiny L4 at 90 pA 20.1 over 37; aspiny at 190 pA 62.2 over 67; aspiny at 100 pA 43.1 over 77). The single-donor count band (L2 290/310/330 pA -> [9, 12]; L4 90/110 -> [12, 17]; PV 170/190/210 -> [4, 23]; SST 90/100 x4/110 -> [12, 21]) is recorded beside it and is not the gate.",
          "Qualification rules: a dose outside the recorded cell's range measures the model, not the difference from the recording, so the block edge above the human range is recorded and not scored; at rheobase the human repeats give 0 or 1, the information there is the rheobase and the level. Type populations from the Allen Cell Types API (tag__dendrite_type, structure__layer): no fast-spiking label exists for human cells, so both interneuron donors use the aspiny population (79 cells).", [DATUMS, ALLEN, CM, SPEC], "unverified"),
        n("datum_rest", D, "Plausibility rules, levels (J, 2026-09-16): pre-pulse rest = mean over the 100 ms before the pulse against the donor's family mean (human-datums.json rest_repeat_mean_mv: L2 -84.01, L4 -80.60, PV -87.12, SST -77.50 mV), tolerance 2 across-cell sd of ipfx pre_vm_mv (500 ms before onset, -14 mV LJP) over the type population's matched sweeps: spiny L2/3 7.7, spiny L4 8.3, aspiny 9.4 / 9.8 mV. Post-pulse level = mean over the 10 ms ending 200 ms after the pulse offset (the runner's return_mv window) against the donor's family mean over that window (after_repeat_mean_mv: L2 -84.94, L4 -81.03, PV -87.18, SST -79.18 mV; every sweep of all four families reaches it), tolerance 2 across-cell sd of ipfx post_vm_mv over the same matched sweeps: L2/3 7.6, L4 8.2, aspiny 9.3 / 9.7 mV. "
          "Named fallback: the Allen table has no field for the 10 ms ending 200 ms after offset; post_vm_mv is the mean over the last 500 ms of the recording (5-7 s after offset), the closest documented post-stimulus level. Failure: any -20 mV crossing before the pulse (hard edge).",
          "Comparison columns, not the gate: the single-donor bands (3 across-sweep sd of each leg's own family reading: pre-pulse 0.92 / 1.42 / 1.15 / 1.16 mV; post-pulse 2.87 / 5.49 / 1.34 / 6.82 mV; the single-sweep donor-rest.json value L2 -84.26, L4 -81.01, PV -86.76, SST -77.15 mV feeds the legacy verdict) and the former 10 mV band and max(2, 30 percent) count band.", [DATUMS, ALLEN, REST, SPEC], "unverified"),
        n("datum_syn", D, "Datum, delivery: literature pin 3.1 nS (1.4-3.9), decay 4.18 ms, delay 2.3 ms, reversal about -75 mV, somatic PSP about 0.5-1 mV. Edges from a weight sweep per contact at the post soma: unresolvable (PSP under 3 sd of the post cell's rest noise) and fires-alone. The manifest weight must lie inside the resolvable range.",
          "Manifest kinetics are forced by H01Topology._validate (E 0 mV / 2 ms, I -80 mV / 5 ms, delay 0.5 ms); only the weight is chosen. I->E contacts are swept with the post cell held depolarised because -80 mV sits above the -84 mV model rest.", [SYN, SPEC], "unverified"),
        n("ob_archive", OB, "Obstacle: the shipped runtime knows one sha-pinned archive of 104 proofread cells; a C3 candidate has no path into make_h01_ei_cell or the manifest. Addressed by B; the implementation_sha256 pin then forces a manifest rebuild.",
          "Existence: H01Archive._verify pins ARCHIVE_SHA256 3e0534df...; no H01ArchiveSet exists.", [SPEC], "supported"),
        n("ob_partners", OB, "Obstacle: no partner index exists. The released per-cell synapse CSV lists partner_neuron_id as missing, and the earlier 166-shard scan kept only pairs among the 104. Addressed by A.",
          "Seven kept cells carry C3 ids that differ from their released ids (cellmatrix mapping audit); ranking uses the C3 ids.", [JOIN, MAPPING], "supported"),
        n("ob_box", OB, "Obstacle: the box has no cloud-volume/fastavro venv (venv314 checked 2026-09-16), a 7,680-thread cgroup ceiling, and at most two population processes at a time. Addressed by A step 0 (/workspace/venv-c3, separate from the pinned venv314) and the two-chain rule in C.",
          "128 GB disk: shards are deleted after scanning; keep ps -eo nlwp under ~4,200 before launching.", [SPEC], "supported"),
        n("ob_runtime", OB, "Obstacle NOT removed by this campaign: per-event runtime is the binding cost of the evolve step. 17 cells at 160 substeps per 0.1 ms ARC event produced no candidate in 60 min on the 4090 (exit 124); more cells raise it. D delivers one capped, receipted launch; finishing a stage inside a cap is a separate registered problem.",
          "Levers, not chosen here: cell count, substeps per event, event count per stage.", [EVOLVE], "supported"),
    ]


def build_edges():
    return [
        cm.edge("io_graph", "io_gate", "prerequisite", "candidate pool"),
        cm.edge("io_skel", "io_gate", "prerequisite", "candidate archive"),
        cm.edge("datum_fire", "io_gate", "prerequisite", "acceptance"),
        cm.edge("datum_rest", "io_gate", "prerequisite", "acceptance"),
        cm.edge("io_gate", "io_net", "prerequisite", "passing cells"),
        cm.edge("datum_syn", "io_net", "prerequisite", "acceptance"),
        cm.edge("io_net", "goal", "prerequisite", "manifest and launch"),
        cm.edge("ob_archive", "io_skel", "describes", "obstacle addressed by this task"),
        cm.edge("ob_partners", "io_graph", "describes", "obstacle addressed by this task"),
        cm.edge("ob_box", "io_graph", "describes", "obstacle addressed by this task"),
        cm.edge("ob_runtime", "goal", "describes", "obstacle recorded, not removed"),
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
            "id": "c3_campaign", "title": TITLE, "effect": "A connected human E/I core for Example 21",
            "conditions": "17 kept proofread cells; the C3 release's cell table, skeletons, subcompartments and synapse export; deployed donor profiles held fixed; Vast 50616476 as the only executor.",
            "narrative": "A prerequisite tree: green is the goal, blue the intermediate objectives (workstreams A-D), purple the datums that define acceptance by testing to failure, orange the obstacles. Each objective carries its registered prediction and falsifier; the closing agent sets its status and evidence file.",
            "uncertainties": "Y7 predicts that most interneuron candidates fail at donor densities; the per-event runtime of the evolve step is not addressed.",
            "predictions": "A: partners exist for every kept cell. B: 40 of 40 import. C: E 1-4 of 20, I 0-2 of 20, about half the kept L2 block under 3x. D: at least one construction-ready contact.",
            "nodeIds": [x["id"] for x in nodes], "testIds": [], "evidence": [], "status": "unverified", "origin": ORIGIN}],
        "nodes": nodes,
        "project": {"calendar": {"exceptions": [], "hours": 8, "weekdays": [1, 2, 3, 4, 5]}, "date": DATE, "direction": "forward", "resources": []},
        "relationshipReviews": cm.reviews([e for e in edges if e["relation"] in cm.REVIEW_TEXT]),
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
    fresh = OUT.with_name(f"h01-c3-campaign.{uuid.uuid4().hex[:8]}.reasoning.json")
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
        target = cm.HERE / f"h01-c3-campaign.{fmt}"
        cm.request({"action": "export", "document": opened["id"], "path": str(target), "format": fmt})
        print("exported", target, target.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
