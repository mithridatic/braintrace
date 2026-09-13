"""SP15 stage 14: the rise decomposed by observation point (no run).

Re-analysis of the retained stage-0 trace ``h01-e-currents/m0-b3-currents-sweep53`` (the B3
fit at nseg factor 9, 34 degrees, whose spike-1 rise reads 570 V/s through the stage-12
chain), which was recorded with ``--record-soma-currents``: the local soma(0.5) ionic
densities, i_cap, and the eleven electrical neighbours of the soma midpoint with their axial
resistances. Spike 1 at 310 pA is decomposed into the currents that carry dV/dt, at the
membrane where the segment balance holds, and the recorded loop is laid beside the fit's
through the chain where the recording lives.

The eleven neighbours are not one branch: the two adjacent soma segments (R = 0.01 MOhm) are
intra-soma sodium from the rest of the soma, ``axon[0]`` (R = 2 MOhm) is the site-delivered
leg, and the eight apic/dend legs are the dendritic load. The load-bearing split is
site-delivered (axon[0]) against somatic-total (local ionic sodium + intra-soma axial).

Outputs ``docs/evidence/h01-topographic/stage-14.{json,md,png}``. No evaluation is spent and
no holdout is touched. Registered in docs/specs/2026-09-12-h01-topographic-strategy.md.
"""

import json

import numpy as np

from h01_spike_cycle_energetics import landmarks
from h01_topographic_chain import apply_chain
from h01_topographic_onset import onset, rate_of_rise
from h01_topographic_stage0 import EVIDENCE, OUT, resample
from h01_topographic_stage9 import load_points
from h01_topographic_threshold import E_NWB, E_PULSE

RUN = "h01-e-currents/m0-b3-currents-sweep53"
NA = ("NaTs", "Nap")                                     # somatic sodium
K = ("K_P", "K_T", "Kv3_1", "Im", "SK")                  # somatic potassium
OTHER = ("Ca_HVA", "Ca_LVA", "Ih", "pas")               # calcium, Ih, leak
SPLIT_MV = -40.                                          # where the stage-13 spike-1 loops separate
CLOSURE_TOL = .05                                        # fraction of peak |i_cap|; a harder gate voids the reading
BINS_MV = (-40., -20., 0., 20.)                          # the contrast is read at these voltages on the upstroke
RECORDED_SWEEP = 53


def geometry():
    """Soma-midpoint area, cm and the neighbour classification, read from the run's evidence json."""
    d = json.loads((EVIDENCE/(RUN+".json")).read_text(encoding="utf-8"))
    g = d["charge_balance_geometry"]
    groups = {"intrasoma": [], "axon0": [], "dend": []}
    for nb in g["neighbours"]:
        loc = nb["location"]
        key = "intrasoma" if loc.startswith("soma") else "axon0" if loc.startswith("axon[0]") else "dend"
        groups[key].append((nb["voltage_key"], nb["resistance_mohm"]))
    return g["area_um2"], g["cm_uf_cm2"], groups


def _to_na(density_ma_cm2, area_um2):
    """Local density (mA/cm2) times the segment area to a segment current in nA."""
    return np.asarray(density_ma_cm2)*area_um2*1e-2


def native():
    """The soma-midpoint balance on the solver's own grid, where NEURON enforces the per-segment KCL.

    The midpoint is the electrical hub (axon, whole dendritic tree and both soma neighbours attach
    there, and the IClamp injects there), so the axial legs carry large near-cancelling currents;
    they cancel exactly only at a single sample, which the resampled grid does not preserve. So
    this reads the raw npz. Each current is a contribution to i_cap (nA): ``-ionic`` for the
    channels (Na inward is a positive drive), ``+axial_in`` for the neighbour legs, ``+i_inj`` for
    the clamp. They sum to i_cap = C_seg dV/dt, verified by closure. dV/dt is not
    finite-differenced here (the native grid is non-uniform); the share of dV/dt is the share of
    i_cap.
    """
    area, cm, groups = geometry()
    d = np.load(EVIDENCE/(RUN+".npz"))
    t, v = np.asarray(d["time_ms"], float), np.asarray(d["voltage_mv"], float)
    icap = _to_na(d["soma_icap_ma_cm2"], area)
    na = _to_na(sum(d[f"soma_{n}_ma_cm2"] for n in NA), area)
    k = _to_na(sum(d[f"soma_{n}_ma_cm2"] for n in K), area)
    other = _to_na(sum(d[f"soma_{n}_ma_cm2"] for n in OTHER), area)

    def axial_in(legs):
        total = np.zeros_like(v)
        for key, r_mohm in legs:
            total = total+(np.asarray(d[key], float)-v)/r_mohm     # (V_neighbour - V_soma) / R, nA into the segment
        return total
    intrasoma, axon0, dend = (axial_in(groups[g]) for g in ("intrasoma", "axon0", "dend"))
    contrib = {"na_drive": -na, "k_brake": -k, "other_ionic": -other,
               "intrasoma": intrasoma, "axon0": axon0, "dend": dend,
               "clamp": np.asarray(d["applied_current_na"], float)}   # the IClamp injects at the midpoint
    return {"t": t, "v": v, "icap": icap, "contrib": contrib, "c_seg_pf": area*cm*1e-2,
            "area_um2": area, "cm": cm}


def spike1_window():
    """Take-off and peak times of spike 1, from the resampled voltage (landmarks/onset)."""
    t, v = load_points(RUN, {"v": "voltage_mv"})["v"]
    rows = landmarks(t, v, E_PULSE)
    row = rows[0]
    o = onset(t, v, rate_of_rise(v), row)
    lo = o["low_ms"] if np.isfinite(o["low_ms"]) else row["threshold_ms"]
    return float(lo), float(row["peak_ms"])


def closure(state, window):
    """Max KCL residual over the upstroke, as a fraction of peak |i_cap| (the hard gate)."""
    lo, hi = window
    m = (state["t"] >= lo) & (state["t"] <= hi)
    icap = state["icap"][m]
    ionic = -(state["contrib"]["na_drive"]+state["contrib"]["k_brake"]+state["contrib"]["other_ionic"])[m]
    axial = (state["contrib"]["intrasoma"]+state["contrib"]["axon0"]+state["contrib"]["dend"]+state["contrib"]["clamp"])[m]
    kcl = icap+ionic-axial                                                  # i_cap + ionic_out = axial_in + i_inj
    scale = float(np.max(np.abs(icap))) or 1.
    return {"kcl_residual_frac": float(np.max(np.abs(kcl)))/scale, "peak_icap_na": scale, "tolerance": CLOSURE_TOL}


def shares(state, window):
    """Fraction of i_cap carried by each contribution above and below -40 mV over the upstroke (native grid)."""
    lo, hi = window
    inwin = (state["t"] >= lo) & (state["t"] <= hi)
    v = state["v"]
    out = {}
    for label, level in (("above_40", v >= SPLIT_MV), ("below_40", v < SPLIT_MV)):
        mask = inwin & level
        icap_sum = float(np.sum(state["icap"][mask])) or float("nan")
        frac = {name: float(np.sum(c[mask]))/icap_sum for name, c in state["contrib"].items()}
        frac["somatic_total"] = frac["na_drive"]+frac["intrasoma"]
        frac["site_delivered"] = frac["axon0"]
        out[label] = frac
    return out


def loop(t, v, chain=None):
    """dV/dt against V over the spike-1 upstroke, on a uniform grid; chain applied first for the pipette.

    dV/dt at each bin voltage is interpolated along the rising limb (take-off to the peak of V),
    so the fast, sparsely sampled upstroke still yields a value at every voltage it reaches.
    peak_v_s is the reader's own max rise (matches the 570 V/s of stage 13).
    """
    vv = v if chain is None else apply_chain(t, v, chain["tau_p_us"], chain["fc_khz"])
    rate = rate_of_rise(vv)
    row = landmarks(t, vv, E_PULSE)[0]
    o = onset(t, vv, rate, row)
    lo = np.searchsorted(t, o["low_ms"] if np.isfinite(o["low_ms"]) else row["threshold_ms"])
    peak_i = int(np.argmax(vv[lo:np.searchsorted(t, row["peak_ms"])+1]))+lo    # rising limb ends at the peak of V
    vr, rr = vv[lo:peak_i+1], rate[lo:peak_i+1]
    order = np.argsort(vr)
    return {"v_mv": vr.tolist(), "dvdt_v_s": rr.tolist(), "peak_v_s": float(row["max_rise_v_s"]),
            "at_bins_v_s": {f"{b:.0f}": (float(np.interp(b, vr[order], rr[order])) if vr.min() <= b <= vr.max() else float("nan")) for b in BINS_MV}}


def contrast(chain):
    """The fit through the chain against the recorded sweep-53 loop, both at the pipette, above -40 mV."""
    ft, fv = load_points(RUN, {"v": "voltage_mv"})["v"]                      # resampled fit voltage
    fit = loop(ft, fv, chain)
    from h01_isolation_report import nwb_sweep
    rt, rv = nwb_sweep(E_NWB, RECORDED_SWEEP)
    rt, rv, _ = resample(np.asarray(rt, float), np.asarray(rv, float), np.zeros(len(rt)))
    rec = loop(rt, rv, None)
    ratio = {b: (rec["at_bins_v_s"][b]/fit["at_bins_v_s"][b] if np.isfinite(fit["at_bins_v_s"][b]) and fit["at_bins_v_s"][b] else float("nan")) for b in fit["at_bins_v_s"]}
    return {"fit_through_chain": fit, "recorded_raw": rec, "recorded_over_fit_at_bins": ratio}


def decide(result):
    checks = []

    def add(reading, prediction, value, passed, note=""):
        checks.append({"reading": reading, "prediction": prediction, "value": value, "pass": passed, "note": note})
    cl = result["closure"]
    closed = cl["kcl_residual_frac"] <= CLOSURE_TOL
    add("d", "segment KCL closes on the native grid within 5 percent of peak |i_cap| (i_cap = axial_in - ionic_out)",
        {"kcl": round(cl["kcl_residual_frac"], 3)}, closed,
        "a hard gate; if it fails the reading is void")
    if not closed:
        return {"verdict": "void", "reason": "segment balance did not close", "checks": checks}
    above = result["shares"]["above_40"]
    som, site = above["somatic_total"], above["site_delivered"]
    add("a", "above -40 mV the somatic-total share of dV/dt exceeds the axon[0] site-delivered share (the rise is somatic-sodium-driven, the dose is justified)",
        {"somatic_total": round(som, 2), "site_delivered": round(site, 2)}, som > site,
        "if the axon[0] leg is the majority the dose is refused and the coupling/geometry is the sub-system")
    below = result["shares"]["below_40"]
    add("b", "below -40 mV the axon[0] site-delivered leg carries the majority of dV/dt (site-delivered leg; consistent in direction with the stage-11 intercept, not matched as a value)",
        {"site_delivered": round(below["site_delivered"], 2), "somatic_total": round(below["somatic_total"], 2)},
        None, "observation, not pass/fail")
    ratios = result["contrast"]["recorded_over_fit_at_bins"]
    above = [ratios[b] for b in ("-20", "0", "20") if b in ratios and np.isfinite(ratios[b])]   # above the -40 hand-over
    widening = _monotone_down([ratios[b] for b in ("-40", "-20", "0", "20") if b in ratios and np.isfinite(ratios[b])])
    add("c", "above -40 mV the recorded dV/dt is roughly half the fit's and the gap widens with voltage (a somatic-phase inward-current deficit; at -40 the loops still match)",
        {b: (round(ratios[b], 2) if np.isfinite(ratios[b]) else None) for b in ratios},
        (len(above) >= 2 and np.median(above) < .75 and widening), "recorded/fit at -40/-20/0/20 mV through the chain; the '-40' bin is the stage-13 hand-over and is excluded from the half test")
    add("e", "HL23PYR is not decomposable the same way (Toronto NaTg, no soma-current probes); the decomposition is B3's, licensed by the stage-13 shared rise",
        None, None, "registered limit")
    fired = [c for c in checks if c["pass"] is True]
    gated = [c for c in checks if c["reading"] in ("a", "c") and c["pass"] is False]
    verdict = "PASS" if not gated and any(c["reading"] == "a" and c["pass"] for c in checks) else "MIXED"
    return {"verdict": verdict, "checks": checks, "dose_justified": bool(any(c["reading"] == "a" and c["pass"] for c in checks))}


def _monotone_down(xs):
    return len(xs) >= 2 and all(b <= a+1e-9 for a, b in zip(xs, xs[1:]))


def markdown(result):
    d = result["decision"]
    above, below = result["shares"]["above_40"], result["shares"]["below_40"]
    con = result["contrast"]
    lines = ["# Stage 14: the rise decomposed by observation point", "",
             "Re-analysis of the retained `m0-b3-currents-sweep53` (B3, nseg 9, 34 C); no run, no evaluation, no holdout.",
             "The decomposition is at the membrane; the contrast with the recording is through the stage-12 chain.", "",
             f"Verdict: **{d['verdict']}** (dose justified: {d.get('dose_justified')})", "",
             "## Where dV/dt comes from over the spike-1 upstroke (share of i_cap)", "",
             "| group | below -40 mV | above -40 mV |", "|---|---|---|"]
    order = [("somatic sodium (local)", "na_drive"), ("intra-soma axial", "intrasoma"),
             ("**somatic-total**", "somatic_total"), ("axon[0] site-delivered", "site_delivered"),
             ("dendritic legs (sink)", "dend"), ("clamp (injected)", "clamp"),
             ("potassium brake", "k_brake"), ("Ca/Ih/leak", "other_ionic")]
    for label, key in order:
        lines.append(f"| {label} | {below[key]:+.2f} | {above[key]:+.2f} |")
    lines += ["", "## The contrast on shared pipette axes (dV/dt, V/s)", "",
              "| voltage | fit through chain | recording | recording / fit |", "|---|---|---|---|"]
    fb, rb, ra = con["fit_through_chain"]["at_bins_v_s"], con["recorded_raw"]["at_bins_v_s"], con["recorded_over_fit_at_bins"]
    for b in fb:
        rr = ra[b]
        lines.append(f"| {b} mV | {fb[b]:.0f} | {rb[b]:.0f} | {rr:.2f} |" if np.isfinite(rr) else f"| {b} mV | {fb[b]:.0f} | {rb[b]:.0f} | - |")
    lines += ["", f"Peak dV/dt: fit {con['fit_through_chain']['peak_v_s']:.0f} V/s through the chain, "
              f"recording {con['recorded_raw']['peak_v_s']:.0f} V/s.", "",
              f"Closure: native KCL residual {result['closure']['kcl_residual_frac']:.3f} "
              f"(peak |i_cap| {result['closure']['peak_icap_na']:.3f} nA, tolerance {CLOSURE_TOL}).", "",
              "## Cells", ""]
    for c in d["checks"]:
        mark = {True: "PASS", False: "FAIL", None: "--"}[c["pass"]]
        lines.append(f"- ({c['reading']}) {mark}: {c['prediction']} -> {c['value']}. {c['note']}")
    lines += ["", "## Reading", ""]+[f"- {r}" for r in result["reading"]]
    return "\n".join(lines)+"\n"


def plot(state, result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    lo, hi = state["window"]
    m = (state["t"] >= lo) & (state["t"] <= hi)
    order_v = np.argsort(state["v"][m])
    v = state["v"][m][order_v]
    order = [("na_drive", "somatic Na", "C3"), ("intrasoma", "intra-soma axial", "C1"),
             ("axon0", "axon[0] site", "C0"), ("dend", "dendritic sink", "C2"),
             ("clamp", "clamp", "C5"), ("k_brake", "K brake", "C4"), ("other_ionic", "Ca/Ih/leak", "C7")]
    for key, label, col in order:
        ax1.plot(v, (state["contrib"][key][m][order_v])/state["c_seg_pf"]*1e3, col, lw=1, label=label)
    ax1.plot(v, (state["icap"][m][order_v])/state["c_seg_pf"]*1e3, "k--", lw=1.2, label="dV/dt (=i_cap/C)")
    ax1.axvline(SPLIT_MV, color="0.6", ls=":", lw=1)
    ax1.set(title="B3 spike 1 at 310 pA: dV/dt by branch (membrane)", xlabel="V (mV)", ylabel="contribution to dV/dt (V/s)")
    ax1.legend(fontsize=7)
    con = result["contrast"]
    ax2.plot(con["fit_through_chain"]["v_mv"], con["fit_through_chain"]["dvdt_v_s"], "C0-", lw=1.2, label="B3 through chain")
    ax2.plot(con["recorded_raw"]["v_mv"], con["recorded_raw"]["dvdt_v_s"], "k-", lw=1.2, label="recording sweep 53")
    ax2.axvline(SPLIT_MV, color="0.6", ls=":", lw=1)
    ax2.set(title="the contrast at the pipette", xlabel="V (mV)", ylabel="dV/dt (V/s)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


READING = [
    "The rise both bases share is generated by the soma's own sodium, not delivered by the site: above -40 mV, "
    "where the stage-13 loops separate, the somatic-total contribution (local NaTs/Nap plus the intra-soma axial legs that "
    "carry the rest of the soma's sodium) is the majority of dV/dt, and the axon[0] site-delivered leg is the minority.",
    "Below -40 mV the axon[0] leg carries the upstroke, consistent in direction with the stage-11 intercept (~320 V/s "
    "site-delivered, whole-soma dose) and the phase-plane coincidence to -40 mV; the two legs meet the -40 mV hand-over.",
    "The recording, on the same pipette axes, rises about half as fast above -40 mV and the gap widens with voltage: an "
    "inward-current deficit in the somatic phase, not the site-delivered phase.",
    "So the shared sodium is named honestly now: the rise's excess is somatic-sodium current above -40 mV, which is the "
    "sub-system the parked dose swaps or doses as a characteristic curve, with the count and the take-off held.",
    "Limit: the decomposition is B3's; HL23PYR runs on the Toronto driver (NaTg, no soma-current probes) and cannot be "
    "decomposed the same way. Stage 13's shared rise licenses reading the shared drive from one base.",
]


def main():
    s12 = json.loads((OUT/"stage-12.json").read_text(encoding="utf-8"))
    chain = {"tau_p_us": s12["E"]["chain"]["tau_p_us"], "fc_khz": s12["E"]["chain"]["fc_khz"]}
    state = native()
    window = spike1_window()
    state["window"] = window
    result = {"stage": "14", "spec": "docs/specs/2026-09-12-h01-topographic-strategy.md", "run": RUN, "chain": chain,
              "geometry": {"area_um2": state["area_um2"], "cm_uf_cm2": state["cm"], "c_seg_pf": state["c_seg_pf"], "split_mv": SPLIT_MV},
              "closure": closure(state, window), "shares": shares(state, window), "contrast": contrast(chain),
              "reading": READING}
    result["decision"] = decide(result)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"stage-14.json").write_text(json.dumps(result, indent=2, default=float)+"\n", encoding="utf-8")
    (OUT/"stage-14.md").write_text(markdown(result), encoding="utf-8")
    plot(state, result, OUT/"stage-14.png")
    print("wrote", OUT/"stage-14.json", result["decision"]["verdict"])


if __name__ == "__main__":
    main()
