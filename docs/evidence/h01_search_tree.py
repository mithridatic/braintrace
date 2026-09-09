"""The H01 programme drawn as one Hartshorne search tree.

Every split the programme made is a branch. Eliminated branches are struck through,
failed-at-cap branches are red, open branches are dashed, and each node names the
decision JSON that justifies it, relative to docs/evidence. The tree is curated by hand
from those decision records (the records have no common schema); the test checks that
every cited file exists so the picture cannot drift from the evidence.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVIDENCE = Path(__file__).resolve().parent

# (id, parent, label, status, reference)
# status: closed = necessary/sufficient established; eliminated = branch excluded;
# fail = search closed at cap with reassessment; untested = launched, no verdict; open = never split.
NODES = [
    ("root", None, "Why is no H01 cell human-qualified?", "question", "h01-population-status.md"),
    ("sim", "root", "Simulator identity\nBrainCell = NEURON at matched mesh and dt", "closed",
     "h01-i-transfer/sp2-close-decision.json"),
    ("dt", "sim", "Residual is time step only\ndt <= 0.000625 ms at 0.27 nA", "closed",
     "h01-i-transfer/sp2-close-decision.json"),
    ("dt019", "sim", "0.19 nA step", "open", "h01-i-transfer/sp2-close-decision.json"),
    ("e", "root", "E cell (L2/3 pyramid, Allen donor)\nlow-drive gain: 4 vs 1 spikes at 200 pA",
     "fail", "h01-e-gain/stage-close-decision.json"),
    ("e_g0", "e", "SK / Ca with axonal Na site\nsufficient for late rate at 310 pA", "closed",
     "h01-e-gain/stage-g0-decision.json"),
    ("e_g1", "e", "Ih density\nleft 250 pA at 8 spikes", "eliminated",
     "h01-e-gain/stage-g-decision.json"),
    ("e_g2", "e", "Leak\nsets rheobase as a step, not a slope", "eliminated",
     "h01-e-gain/stage-g-decision.json"),
    ("e_open", "e", "Slow Na inactivation or Kv7/M kinetics\nabsent from the fit; second donor",
     "open", "h01-e-gain/stage-close-decision.json"),
    ("i", "root", "I cell (PV basket, HL5BN1 donor)\npost-trough burst and accommodation", "fail",
     "h01-i-reserve/stage-1-decision.json"),
    ("i_kv3", "i", "Somatic Kv3 close factor 2.0\nloop and trough within repeatability", "closed",
     "h01-i-energetic/stage-f-decision.json"),
    ("i_soma_na", "i", "Somatic Na availability\nh >= 0.7 at trough + 6 ms", "eliminated",
     "h01-i-reserve/stage-0-decision.json"),
    ("i_axon_na", "i", "Axonal NaTg density x1.5\nmoved cycle 2 by -0.126 ms", "eliminated",
     "h01-i-reserve/stage-1-decision.json"),
    ("i_open", "i", "Slow state after the trough", "open", "h01-i-reserve/stage-1-decision.json"),
    ("circ", "root", "Circuit", "question", "h01-ie-inhibition/decision.json"),
    ("circ_inh", "circ", "Functional inhibition at the measured pair\n0.003 mV at the E soma",
     "untested", "h01-ie-inhibition/decision.json"),
    ("circ_syn", "circ", "Synapse strength\n20 nS assumed vs 3.1 nS human median", "open",
     "h01-ie-synapse-literature.json"),
    ("circ_conn", "circ", "Connectivity\n12 synapses among 104 cells in the whole export",
     "closed", "h01-full-export-join-progress.json"),
    ("pop", "root", "104-cell population", "question", "h01-104-corrected-construction-decision.json"),
    ("pop_build", "pop", "Construction 519 s, 807,588 compartments", "closed",
     "h01-104-corrected-construction-decision.json"),
    ("pop_init", "pop", "Initialization 144 s", "closed",
     "h01-104-corrected-initialization-decision.json"),
    ("pop_run", "pop", "10 ms runtime\n16,000-step program, twice killed", "untested",
     "h01-104-corrected-initialization-decision.json"),
    ("pop_donor", "pop", "Donors for the other 102 cells\nL4 and SST fits not reproduced",
     "eliminated", "h01-donors/stage-allen-l4-decision.json"),
]
STYLE = {"question": ("white", "k", "-"), "closed": ("#d9ead3", "k", "-"),
         "eliminated": ("#eeeeee", "0.5", "-"), "fail": ("#f4cccc", "darkred", "-"),
         "untested": ("#fff2cc", "k", "--"), "open": ("white", "k", "--")}


def children(node_id):
    return [n for n in NODES if n[1] == node_id]


def layout():
    """Leaves get consecutive x slots; parents sit over the mean of their children."""
    pos, counter = {}, [0]

    def place(node, depth):
        kids = children(node[0])
        if not kids:
            pos[node[0]] = (counter[0], -depth-.9*(counter[0] % 2))
            counter[0] += 1
        else:
            xs = [place(k, depth+1) for k in kids]
            pos[node[0]] = (sum(xs)/len(xs), -depth)
        return pos[node[0]][0]

    place(NODES[0], 0)
    return pos


def draw(out):
    pos = layout()
    fig, axis = plt.subplots(figsize=(26, 10))
    by_id = {n[0]: n for n in NODES}
    for node_id, parent, label, status, ref in NODES:
        x, y = pos[node_id]
        if parent:
            px, py = pos[parent]
            axis.plot([px, x], [py-.2, y+.28], color="0.4", lw=1,
                      ls=STYLE[status][2], zorder=1)
        face, edge, _ = STYLE[status]
        text = label if status != "eliminated" else "X  "+label.replace("\n", "\nX  ")
        axis.text(x, y, f"{text}\n[{status}] {ref}", ha="center", va="center", fontsize=6.6,
                  color=edge, bbox=dict(boxstyle="round,pad=.35", fc=face, ec=edge, lw=1), zorder=2)
    axis.set_xlim(-.7, max(x for x, _ in pos.values())+.7)
    axis.set_ylim(min(y for _, y in pos.values())-.6, .5)
    axis.set_axis_off()
    axis.set_title("H01 programme search tree (Hartshorne): green = established, grey struck = eliminated, "
                   "red = failed at cap with reassessment, yellow = launched but untested, dashed = open",
                   fontsize=10)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=140)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return {s: sum(n[3] == s for n in NODES) for s in STYLE}


def main(out_dir=EVIDENCE):
    counts = draw(Path(out_dir)/"h01-search-tree")
    (Path(out_dir)/"h01-search-tree.json").write_text(json.dumps(
        {"nodes": [dict(zip(("id", "parent", "label", "status", "reference"), n)) for n in NODES],
         "status_counts": counts}, indent=2)+"\n")
    return counts


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:2])))
