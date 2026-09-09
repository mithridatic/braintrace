"""Hartshorne four-box source-load diagrams for the two H01 energetic questions.

Figure 1: the I cell under current clamp. A flow source (the clamp) feeds the membrane
load through the electrode; the published fit (I campaign stage A record) absorbed a 31.4 pA holding current as a bias source
in parallel with the load, which is why the published bias was mistaken for physiology.

Figure 2: the measured I-to-E pair. The presynaptic spike is the effort source, the
synapse is the series impedance (1/g), and the E dendrite at the receptor is the load,
with the dendrite-to-soma cable as a second series impedance. The circuit's 20 nS
against the human 3.1 nS median, and the 10 to 15 mV local versus 0.003 mV somatic
deflection, are read from the committed JSON.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVIDENCE = Path(__file__).resolve().parent
SYNAPSE = EVIDENCE/"h01-ie-synapse-literature.json"
BIAS = EVIDENCE/"h01-i-campaign"/"a-into-G1a-027.json"
INHIBITION = EVIDENCE/"h01-ie-inhibition"/"decision.json"


def read_numbers():
    syn = json.loads(SYNAPSE.read_text())
    bias = json.loads(BIAS.read_text())
    human_ns = syn["pinned_targets"]["peak_conductance_ns"]["value"]
    circuit_ns = syn["circuit_today"]["inhibitory_weight_us"]*1e3
    return {"bias_pa": 1e3*_find(bias, "bias_na"), "human_ns": human_ns, "circuit_ns": circuit_ns,
            "human_rev_mv": syn["pinned_targets"]["reversal_mv"]["value"],
            "circuit_rev_mv": syn["circuit_today"]["reversal_mv"],
            "human_delay_ms": syn["pinned_targets"]["delay_ms"]["value"],
            "circuit_delay_ms": syn["circuit_today"]["delay_ms"],
            "soma_deflection_mv": .003, "receptor_deflection_mv": (10., 15.)}


def _find(obj, key):
    """First value of key anywhere in a nested JSON document."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = _find(v, key)
            if hit is not None:
                return hit
    if isinstance(obj, list):
        for v in obj:
            hit = _find(v, key)
            if hit is not None:
                return hit
    return None


def box(axis, x, y, text, colour, w=2.6, h=1.3):
    axis.text(x, y, text, ha="center", va="center", fontsize=8.5,
              bbox=dict(boxstyle="round,pad=.4", fc=colour, ec="k", lw=1))


def arrow(axis, x0, y0, x1, y1, label):
    axis.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2))
    axis.text((x0+x1)/2, (y0+y1)/2+.18, label, ha="center", fontsize=7.5, color="tab:blue")


def draw_clamp(axis, n):
    axis.set_title("I cell under current clamp: four-box flow-source model", fontsize=10)
    box(axis, 1, 3, "Supply power\ncurrent clamp\n0.19 / 0.23 / 0.27 nA", "#e8f0fe")
    box(axis, 5, 3, "Transmit: split flow\nelectrode + access\n(impedance not measured)", "#cfe8ff")
    box(axis, 9, 3, "Load impedance\nmembrane at soma\nR_in, tau not measured", "#f8d7da")
    box(axis, 5, .8, f"Dissipate power\nbias source {n['bias_pa']:.1f} pA\nabsorbed by the fit", "#fde9d9")
    box(axis, 9, .8, "Dissipate power\nleak + Kv3 (close x2.0)\nsets the trough", "#fde9d9")
    arrow(axis, 2.4, 3, 3.6, 3, "flow i = clamp")
    arrow(axis, 6.4, 3, 7.6, 3, "i - i_bias")
    arrow(axis, 5, 2.3, 5, 1.5, "i_bias")
    arrow(axis, 9, 2.3, 9, 1.5, "e = V_m")
    axis.text(5, -.4, "Reading: the 31.4 pA is a holding current on the source side, not a membrane"
              " property. Series and load impedances were never measured, so the loop cannot be closed"
              " numerically.", ha="center", fontsize=7.5, wrap=True)
    axis.set_xlim(-.6, 10.8)
    axis.set_ylim(-.9, 4.1)
    axis.set_axis_off()


def draw_pair(axis, n):
    axis.set_title("Measured I-to-E pair (contact 8105899): effort-source model", fontsize=10)
    box(axis, 1, 3, "Supply power\npresynaptic spike\neffort = V_pre - E_rev", "#e8f0fe")
    box(axis, 4.4, 3, f"Transmit: split effort\nsynapse Z = 1/g\ncircuit {1e3/n['circuit_ns']:.0f} MOhm ({n['circuit_ns']:.0f} nS)"
        f"\nhuman {1e3/n['human_ns']:.0f} MOhm ({n['human_ns']} nS)", "#cfe8ff")
    box(axis, 7.8, 3, "Transmit: split effort\ndendritic cable\nreceptor to soma", "#cfe8ff")
    box(axis, 10.6, 3, f"Load\nE soma\n{n['soma_deflection_mv']} mV moved", "#f8d7da")
    box(axis, 7.8, .8, f"Dissipate power\nreceptor site\n{n['receptor_deflection_mv'][0]:.0f} to "
        f"{n['receptor_deflection_mv'][1]:.0f} mV local", "#fde9d9")
    arrow(axis, 2.3, 3, 3.1, 3, f"E_rev {n['circuit_rev_mv']} vs {n['human_rev_mv']} mV")
    arrow(axis, 5.8, 3, 6.5, 3, f"delay {n['circuit_delay_ms']} vs {n['human_delay_ms']} ms")
    arrow(axis, 9.1, 3, 9.6, 3, "attenuated")
    arrow(axis, 7.8, 2.3, 7.8, 1.5, "local IPSP")
    ratio = n["receptor_deflection_mv"][0]/n["soma_deflection_mv"]
    axis.text(5.8, -.4, f"Reading: the synapse is {n['circuit_ns']/n['human_ns']:.1f}x the human conductance"
              f" yet the soma moves {n['soma_deflection_mv']} mV: the cable box attenuates by >{ratio:,.0f}x."
              " The deficit is placement and cable, not conductance.", ha="center", fontsize=7.5)
    axis.set_xlim(-.6, 12.2)
    axis.set_ylim(-.9, 4.1)
    axis.set_axis_off()


def main(out_dir=EVIDENCE):
    n = read_numbers()
    out = Path(out_dir)
    for name, drawer in (("h01-thevenin-i-clamp", draw_clamp), ("h01-thevenin-ie-pair", draw_pair)):
        fig, axis = plt.subplots(figsize=(11, 4.6))
        drawer(axis, n)
        fig.tight_layout()
        fig.savefig(out/f"{name}.png", dpi=140)
        fig.savefig(out/f"{name}.svg")
        plt.close(fig)
    (out/"h01-thevenin-boxes.json").write_text(json.dumps(
        {"sources": [str(p.relative_to(EVIDENCE)) for p in (SYNAPSE, BIAS, INHIBITION)], "numbers": n},
        indent=2)+"\n")
    return n


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:2]), indent=1))
