"""SP15 stage 12: the recording's measurement chain, measured from its own step edges and
applied to the model traces (retained traces only, no run).

Every recorded quantity is read at the top of a patch pipette through the amplifier's bridge,
its capacitance neutralisation and the acquisition filter; every model quantity is read at
the membrane. The chain is measured from the subthreshold short-square onset edges, where the
membrane is linear and the input is known: a current step I0 gives the membrane ramp k·t, the
bridge term (R_s·LP[u] − R_b·u)·I0 with the pipette pole tau_p = R_s·C_p, then a 4-pole Bessel
at f_c and a delay t0. The fitted (tau_p, f_c) is applied to the model traces and every
campaign quantity re-read with the campaign's own readers.

Outputs ``docs/evidence/h01-topographic/stage-12.{json,md,png}``.
"""

import json

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import bessel, lfilter, welch

from h01_isolation_report import human_trace, nwb_sweep
from h01_spike_cycle_energetics import landmarks
from h01_topographic_coupling import slide
from h01_topographic_onset import onset, rate_of_rise
from h01_topographic_stage0 import CELLS, OUT, ROOT, resample
from h01_topographic_stage9 import E_POINTS, I_PULSE, load_points, point_rows
from h01_topographic_threshold import E_NWB, E_PULSE, I_NWB, approach_rate

CHAIN = {
    "E": {"nwb": E_NWB, "edges": (15, 16, 17, 18, 22, 23, 24, 25, 26), "onset_ms": 1020., "pulse": E_PULSE, "human_sweep": 53,
          "fit": "h01-e-coupling-b/s10-d2-density", "control": "h01-e-cable-load/c10-area-310-n9-sweep53", "recorded_peak_mv": None},
    "I": {"nwb": I_NWB, "edges": (7, 8, 9, 11, 12, 13), "onset_ms": 1020., "pulse": I_PULSE, "human_input": "0.19 nA",
          "fit": "h01-i-takeoff/s9-finalist-probes-019-probes"},
}
RAMPS = {"ramp01": 1.1, "ramp1": 11., "ramp10": 110.}
RISE_SIGMA_E = 1.5                                # stage-0 E repeat sigma of the spike-1 rise, V/s (within the recording chain)
FINE_MS = .005
EDGE_WINDOW_MS = (-.3, .5)
TAU_GRID_US = (0., 1., 2., 5., 10., 20., 30., 50., 75., 100.)
NOISE_SWEEPS = {"E": range(32, 48), "I": range(20, 40)}   # long-square sweeps whose first second (100 to 1000 ms) is unstimulated
NOISE_WINDOW_MS = (100., 1000.)
NOISE_REF_KHZ = (1., 2.)
NOISE_SLOPE_KHZ = (10., 20.)
FOUR_POLE_DB_PER_OCTAVE = -24.
FC_FLOOR_KHZ = 20.                                # the acquisition corner read from a noise floor that is flat to Nyquist
FC_CONSERVATIVE_KHZ = 10.                         # an assumed corner below any the noise floor allows; the chain is applied here as its largest plausible effect
LIMITS = {"tau_profile_ratio": 1.2, "take_off_mv": .1, "approach_fraction": .05, "lead_ms": .02, "e_span_mv": 1., "i_span_mv": 8., "rise_sigma_v_s": 3*RISE_SIGMA_E}


def sweep_step(path, sweep):
    """Command amplitude (nA) and bridge (MOhm) of one NWB sweep."""
    with h5py.File(path, "r") as nwb:
        g = nwb[f"acquisition/timeseries/Sweep_{sweep}"]
        return float(g["aibs_stimulus_amplitude_pa"][()])*1e-3, float(g["bridge_balance"][()])*1e-6


def edge(path, sweep, onset_ms, window=EDGE_WINDOW_MS):
    """One onset edge: time relative to the nominal step (ms), voltage relative to the pre-step mean (mV), I0 (nA), R_b (MOhm)."""
    t, v = nwb_sweep(path, sweep)
    t = np.asarray(t, float)-onset_ms; v = np.asarray(v, float)
    m = (t >= window[0]) & (t <= window[1])
    base = v[(t >= window[0]) & (t < -.05)].mean()
    i0, rb = sweep_step(path, sweep)
    return {"sweep": sweep, "t_ms": t[m], "v_mv": v[m]-base, "i0_na": i0, "rb_mohm": rb}


def single_pole(x, dt_ms, tau_ms):
    """Exact-discretisation first-order low-pass; identity when tau is zero."""
    if tau_ms <= 0:
        return x
    a = 1.-np.exp(-dt_ms/tau_ms)
    return lfilter([a], [1., -(1.-a)], x)


def bessel4(x, dt_ms, fc_khz):
    b, a = bessel(4, fc_khz, fs=1./dt_ms)
    return lfilter(b, a, x)


def chain_response(t_ms, i0_na, rb_mohm, slope_mv_ms, tau_p_us, fc_khz, t0_ms, delta_r_mohm, tau_cmd_us=0.):
    """Recorded voltage predicted at the data times for a current step at t = 0 (fine-grid simulation, then delay and sampling).

    The command reaches the pipette through its own pole tau_cmd (the step edge measures the round trip); the membrane
    ramp is the integral of the delivered current; the pipette pole tau_p acts on the node; the bridge subtracts R_b times
    the delivered current; the acquisition filter is a 4-pole Bessel at f_c.
    """
    tf = np.arange(t_ms[0]-.5, t_ms[-1]+.5, FINE_MS)
    u = single_pole((tf >= 0.).astype(float), FINE_MS, tau_cmd_us*1e-3)
    rs = rb_mohm+delta_r_mohm
    ramp = slope_mv_ms*np.cumsum(u)*FINE_MS
    node = single_pole(ramp+rs*i0_na*u, FINE_MS, tau_p_us*1e-3)-rb_mohm*i0_na*u
    out = bessel4(node, FINE_MS, fc_khz)
    return np.interp(t_ms-t0_ms, tf, out)


def noise_corner(path, sweeps=None, cell="E"):
    """The acquisition corner bounded from the unstimulated baseline's noise spectrum.

    A 4-pole corner at f_c falls 24 dB per octave above it; the slope of the spectrum between 10 and 20 kHz therefore
    bounds the corner from below. The chain is applied at FC_FLOOR_KHZ when the slope is flatter than half of a
    4-pole roll-off there (a lower bound on the corner, so an upper bound on the chain's effect).
    """
    segs = []
    for s in (sweeps if sweeps is not None else NOISE_SWEEPS[cell]):
        try:
            t, v = nwb_sweep(path, s)
        except KeyError:
            continue
        t = np.asarray(t, float); v = np.asarray(v, float)
        m = (t > NOISE_WINDOW_MS[0]) & (t < NOISE_WINDOW_MS[1])
        segs.append(v[m]-v[m].mean())
    x = np.concatenate(segs)
    f, p = welch(x, fs=50000., nperseg=4096)
    ref = np.median(p[(f > NOISE_REF_KHZ[0]*1e3) & (f < NOISE_REF_KHZ[1]*1e3)])
    db = {fk: float(10*np.log10(p[np.argmin(np.abs(f-fk*1e3))]/ref)) for fk in (1, 2, 5, 10, 15, 20, 24)}
    slope = db[int(NOISE_SLOPE_KHZ[1])]-db[int(NOISE_SLOPE_KHZ[0])]
    flat = slope > FOUR_POLE_DB_PER_OCTAVE/2
    return {"n_sweeps": len(segs), "rms_mv": float(x.std()), "db_vs_1_2_khz": db, "slope_10_to_20_khz_db_per_octave": float(slope),
            "flat_to_nyquist": bool(flat), "fc_khz": FC_FLOOR_KHZ if flat else float(NOISE_SLOPE_KHZ[0])}


def fit_chain(edges, fc_khz, tau_p_us=None):
    """Joint fit of tau_p (unless held), the command pole, t0 and R_s - R_b over every edge at a held acquisition corner,
    with a free membrane slope per edge. The transient's area (R_s·I0·tau_p) is invariant to the command pole, so tau_p
    is identified even though the edge only measures the round trip."""
    n = len(edges)
    held = tau_p_us is not None
    def unpack(p):
        tau = tau_p_us if held else p[0]
        rest = p[1:] if not held else p
        return tau, rest[0], rest[1], rest[2], rest[3:]
    def residual(p):
        tau, tcmd, t0, dr, slopes = unpack(p)
        return np.concatenate([chain_response(e["t_ms"], e["i0_na"], e["rb_mohm"], s, tau, fc_khz, t0, dr, tcmd)-e["v_mv"] for e, s in zip(edges, slopes)])
    slopes0 = [max(np.polyfit(e["t_ms"][e["t_ms"] > .15], e["v_mv"][e["t_ms"] > .15], 1)[0], .1) for e in edges]
    p0 = ([5.] if not held else [])+[50., .05, 0.]+slopes0
    lo = ([0.] if not held else [])+[0., -.2, -3.]+[0.]*n
    hi = ([300.] if not held else [])+[500., .3, 3.]+[60.]*n
    sol = least_squares(residual, p0, bounds=(lo, hi))
    tau, tcmd, t0, dr, slopes = unpack(sol.x)
    rms = float(np.sqrt(np.mean(sol.fun**2)))
    return {"tau_p_us": float(tau), "fc_khz": float(fc_khz), "tau_cmd_us": float(tcmd), "t0_ms": float(t0), "delta_r_mohm": float(dr),
            "slopes_mv_ms": [float(s) for s in slopes], "rms_mv": rms}


def profile_tau(edges, fc_khz, grid=TAU_GRID_US, ratio=LIMITS["tau_profile_ratio"]):
    """Residual against a held tau_p; the bound is the largest tau_p within the ratio of the minimum."""
    rows = [{"tau_p_us": tau, **{k: v for k, v in fit_chain(edges, fc_khz, tau).items() if k in ("tau_cmd_us", "rms_mv")}} for tau in grid]
    best = min(r["rms_mv"] for r in rows)
    within = [r["tau_p_us"] for r in rows if r["rms_mv"] <= ratio*best]
    return {"rows": rows, "min_rms_mv": best, "tau_p_bound_us": max(within), "tau_p_best_us": min(rows, key=lambda r: r["rms_mv"])["tau_p_us"]}


def apply_chain(t_ms, v_mv, tau_p_us, fc_khz):
    """The model trace as the recording's chain would report it (fine grid, pole, Bessel, back to the data grid)."""
    tf = np.arange(t_ms[0], t_ms[-1], FINE_MS)
    vf = np.interp(tf, t_ms, v_mv)
    out = bessel4(single_pole(vf, FINE_MS, tau_p_us*1e-3), FINE_MS, fc_khz)
    return np.interp(t_ms, tf, out)


def read_spike(t, v, pulse):
    """Count and spike-1 rise, peak, fall, span, rapidness, take-off, approach of one soma trace."""
    marks = landmarks(t, v, pulse)
    if not marks:
        return {"count": 0}
    r = marks[0]; o = onset(t, v, rate_of_rise(v), r)
    return {"count": len(marks), "rise_v_s": r["max_rise_v_s"], "peak_mv": r["peak_mv"], "fall_v_s": r["max_fall_v_s"], "span_mv": o["span_mv"],
            "rapidness_per_ms": o["rapidness_per_ms"], "take_off_mv": o["low_mv"], "approach_mv_per_ms": approach_rate(t, v, r["threshold_ms"], pulse[0])}


def soma_trace(base):
    traces = load_points(base, {"soma": "voltage_mv"})
    return None if traces is None else traces["soma"]


def step_through(base, pulse, chain):
    """One run's soma trace read raw and through the chain."""
    tr = soma_trace(base)
    if tr is None:
        return {"status": "not run"}
    t, v = tr
    return {"status": "read", "raw": read_spike(t, v, pulse), "chain": read_spike(t, apply_chain(t, v, chain["tau_p_us"], chain["fc_khz"]), pulse)}


def peak_bound(base, pulse, fc_khz, recorded_peak_mv, grid_us=np.arange(0., 301., 5.)):
    """The tau_p at which the filtered spike-1 peak first falls to the recorded peak, and the rise there."""
    t, v = soma_trace(base)
    last = None
    for tau in grid_us:
        row = read_spike(t, apply_chain(t, v, tau, fc_khz), pulse)
        if row["count"] == 0:
            break
        last = {"tau_p_us": float(tau), **row}
        if row["peak_mv"] <= recorded_peak_mv:
            return last
    return {**last, "note": "peak never reaches the recorded value on the grid"} if last else {"note": "no spike"}


def arm_through(base, chain):
    """The x2 density-held ramps and step at every recorded point, raw and through the chain: soma slide, lead, approach, count."""
    out = {}
    for label in ("raw", "chain"):
        ramps = {}
        for name, rate in RAMPS.items():
            traces = load_points(f"{base}-{name}", E_POINTS)
            if traces is None:
                ramps[name] = {"rate_pa_per_ms": rate, "status": "not run"}; continue
            if label == "chain":
                traces = {k: (t, apply_chain(t, v, chain["tau_p_us"], chain["fc_khz"])) for k, (t, v) in traces.items()}
            rows = point_rows(traces, E_PULSE, E_PULSE[0])
            ramps[name] = {"rate_pa_per_ms": rate, "status": "read" if rows else "no spike", "spike_1": rows[0] if rows else None}
        read = [r for r in ramps.values() if r["status"] == "read"]
        out[label] = {"soma_slide": slide(ramps, "soma"), "approach": [r["spike_1"]["approach_mv_per_ms"] for r in read],
                      "axon1_lead_ms": [r["spike_1"]["points"]["axon1"]["lead_ms"] for r in read if "axon1" in r["spike_1"]["points"]],
                      "soma_span_mv": [r["spike_1"]["points"]["soma"]["span_mv"] for r in read]}
    return out


def human_step(cell):
    c = CHAIN[cell]
    if cell == "E":
        t, v = nwb_sweep(c["nwb"], c["human_sweep"])
    else:
        t, v = human_trace(ROOT/CELLS["I"]["human_dir"]/CELLS["I"]["inputs"][c["human_input"]][0])
    t, v, _ = resample(np.asarray(t, float), np.asarray(v, float), np.zeros(len(t)))
    return read_spike(t, v, c["pulse"])


def decide(result):
    checks = []
    def add(cell, reading, prediction, value, passed, note=""):
        checks.append({"cell": cell, "reading": reading, "prediction": prediction, "value": None if value is None else float(value), "pass": None if passed is None else bool(passed), "note": note})
    for cell in ("E", "I"):
        ch = result[cell]["chain"]; pr = result[cell]["profile"]; nz = result[cell]["noise"]
        add(cell, "a", "the noise floor bounds the acquisition corner (slope 10 to 20 kHz flatter than half a 4-pole roll-off)", nz["slope_10_to_20_khz_db_per_octave"], nz["flat_to_nyquist"],
            f"noise-floor corner {nz['fc_khz']:.0f} kHz, chain applied at the conservative {FC_CONSERVATIVE_KHZ:.0f} kHz; baseline rms {nz['rms_mv']:.3f} mV over {nz['n_sweeps']} sweeps")
        constrained = pr["tau_p_bound_us"]*1e-3 < 2./(2*np.pi*ch["fc_khz"])
        add(cell, "a", "the edges bound tau_p to within a factor of two of the acquisition corner", pr["tau_p_bound_us"], constrained,
            f"best tau_p {pr['tau_p_best_us']:.0f} us (joint fit {ch['tau_p_us']:.1f}), command pole {ch['tau_cmd_us']:.0f} us, rms {ch['rms_mv']:.3f} mV, R_s - R_b {ch['delta_r_mohm']:+.2f} MOhm")
    e = result["E"]; fit, hum = e["fit_step"], e["human"]
    if fit["status"] == "read":
        gap = fit["raw"]["rise_v_s"]-hum["rise_v_s"]
        frac = (fit["raw"]["rise_v_s"]-fit["chain"]["rise_v_s"])/gap
        pb = e["peak_bound"]
        frac_pb = (fit["raw"]["rise_v_s"]-pb["rise_v_s"])/gap if "rise_v_s" in pb else None
        add("E", "b", "the chain accounts for less than half of the rise gap under the peak bound", frac_pb, frac_pb is not None and frac_pb < .5,
            f"measured chain accounts for {100*frac:.0f} percent at {FC_CONSERVATIVE_KHZ:.0f} kHz (rise {fit['chain']['rise_v_s']:.0f}) and {100*(fit['raw']['rise_v_s']-fit['chain_floor']['rise_v_s'])/gap:.0f} percent at the noise floor (rise {fit['chain_floor']['rise_v_s']:.0f}) against the recorded {hum['rise_v_s']:.0f}; peak bound at tau_p {pb.get('tau_p_us', float('nan')):.0f} us gives rise {pb.get('rise_v_s', float('nan')):.0f}")
        add("E", "b", "the residual rise gap exceeds 3 sigma of the within-chain repeat (4.5 V/s)", fit["chain"]["rise_v_s"]-hum["rise_v_s"], fit["chain"]["rise_v_s"]-hum["rise_v_s"] > LIMITS["rise_sigma_v_s"])
        add("E", "d", "E fit onset span through the chain within 1 mV of the recording's", fit["chain"]["span_mv"]-hum["span_mv"], abs(fit["chain"]["span_mv"]-hum["span_mv"]) < LIMITS["e_span_mv"],
            f"raw {fit['raw']['span_mv']:.2f}, chain {fit['chain']['span_mv']:.2f}, recorded {hum['span_mv']:.2f}; rapidness {fit['raw']['rapidness_per_ms']:.0f} / {fit['chain']['rapidness_per_ms']:.0f} / {hum['rapidness_per_ms']:.0f} per ms")
    arm = e["arm"]
    if arm["raw"]["soma_slide"] and arm["chain"]["soma_slide"]:
        d_take = max(abs(a-b) for a, b in zip(arm["raw"]["soma_slide"]["take_off_mv"], arm["chain"]["soma_slide"]["take_off_mv"]))
        d_app = max(abs(a-b)/a for a, b in zip(arm["raw"]["approach"], arm["chain"]["approach"]))
        d_lead = max(abs(a-b) for a, b in zip(arm["raw"]["axon1_lead_ms"], arm["chain"]["axon1_lead_ms"]))
        add("E", "c", "slow set held: take-off moves less than 0.1 mV", d_take, d_take < LIMITS["take_off_mv"])
        add("E", "c", "slow set held: approach within 5 percent", d_app, d_app < LIMITS["approach_fraction"])
        add("E", "c", "slow set held: axon lead within 0.02 ms", d_lead, d_lead < LIMITS["lead_ms"])
        add("E", "c", "slow set held: 310 pA count unchanged", fit["chain"]["count"]-fit["raw"]["count"], fit["chain"]["count"] == fit["raw"]["count"])
    i = result["I"]
    if i["fit_step"]["status"] == "read":
        add("I", "d", "I fit onset span through the chain stays above 8 mV (stage-8 reading survives its chain)", i["fit_step"]["chain"]["span_mv"], i["fit_step"]["chain"]["span_mv"] > LIMITS["i_span_mv"],
            f"raw {i['fit_step']['raw']['span_mv']:.1f}, recorded {i['human']['span_mv']:.1f} mV; rapidness {i['fit_step']['raw']['rapidness_per_ms']:.0f} / {i['fit_step']['chain']['rapidness_per_ms']:.0f} / {i['human']['rapidness_per_ms']:.0f} per ms")
    verdict = "no reading" if any(c["pass"] is None for c in checks) else ("PASS" if all(c["pass"] for c in checks) else "FAIL")
    return {"checks": checks, "verdict": verdict}


def markdown(result):
    lines = ["# SP15 stage 12: the recording's measurement chain, measured and applied", ""]
    for cell in ("E", "I"):
        ch, pr, nz = result[cell]["chain"], result[cell]["profile"], result[cell]["noise"]
        lines += [f"## {cell} chain", "",
                  f"Chain applied: pipette pole from the edges, acquisition corner at an assumed conservative {FC_CONSERVATIVE_KHZ:.0f} kHz (the noise floor bounds the corner only from below; its {nz['fc_khz']:.0f} kHz reading beside it).",
                  f"Noise floor ({nz['n_sweeps']} unstimulated seconds, rms {nz['rms_mv']:.3f} mV), dB against 1 to 2 kHz: "
                  + ", ".join(f"{k} kHz {v:+.1f}" for k, v in nz["db_vs_1_2_khz"].items())
                  + f"; slope 10 to 20 kHz {nz['slope_10_to_20_khz_db_per_octave']:+.1f} dB per octave (a 4-pole corner below 10 kHz would give -24); corner applied {nz['fc_khz']:.0f} kHz.",
                  f"Onset edges of sweeps {', '.join(str(s) for s in CHAIN[cell]['edges'])}, joint fit at {FC_CONSERVATIVE_KHZ:.0f} kHz: tau_p {ch['tau_p_us']:.1f} us, command pole {ch['tau_cmd_us']:.0f} us, t0 {ch['t0_ms']*1e3:+.0f} us, R_s - R_b {ch['delta_r_mohm']:+.2f} MOhm, rms {ch['rms_mv']:.3f} mV.",
                  f"Profile: best tau_p {pr['tau_p_best_us']:.0f} us, bound (rms within 20 percent of the minimum) {pr['tau_p_bound_us']:.0f} us.", "",
                  "| tau_p us | command pole us | rms mV |", "|---|---|---|"] + [f"| {r['tau_p_us']:.0f} | {r['tau_cmd_us']:.0f} | {r['rms_mv']:.3f} |" for r in pr["rows"]] + [""]
    e = result["E"]
    lines += ["## E spike 1 at 310 pA, raw and through the chain", "", "| trace | count | rise V/s | peak mV | fall V/s | span mV | rapidness /ms | take-off mV | approach mV/ms |", "|---|---|---|---|---|---|---|---|---|"]
    def row(label, r):
        if r.get("count", 0) == 0:
            return f"| {label} | 0 | | | | | | | |"
        return f"| {label} | {r['count']} | {r['rise_v_s']:.0f} | {r['peak_mv']:.1f} | {r['fall_v_s']:.0f} | {r['span_mv']:.2f} | {r['rapidness_per_ms']:.0f} | {r['take_off_mv']:.2f} | {r['approach_mv_per_ms']:.2f} |"
    for label, key in (("x2 density held", "fit_step"), ("B3 control 1 um", "control_step")):
        st = e[key]
        if st["status"] == "read":
            lines += [row(f"{label}, raw", st["raw"]), row(f"{label}, chain at {FC_CONSERVATIVE_KHZ:.0f} kHz", st["chain"]), row(f"{label}, chain at the noise floor {e['noise']['fc_khz']:.0f} kHz", st["chain_floor"])]
    pb = e["peak_bound"]
    if "rise_v_s" in pb:
        lines.append(row(f"x2 density held at the peak bound (tau_p {pb['tau_p_us']:.0f} us)", pb))
    lines += [row("recording sweep 53", e["human"]), ""]
    arm = e["arm"]
    lines += ["## E slow set on the x2 density-held ramps, raw against chain", ""]
    for label in ("raw", "chain"):
        a = arm[label]
        if a["soma_slide"]:
            lines.append(f"- {label}: soma take-off {', '.join(f'{x:.2f}' for x in a['soma_slide']['take_off_mv'])} mV (slide {a['soma_slide']['slide_mv']:+.2f}); approach {', '.join(f'{x:.2f}' for x in a['approach'])}; axon lead {', '.join(f'{x:.2f}' for x in a['axon1_lead_ms'])} ms; span {', '.join(f'{x:.1f}' for x in a['soma_span_mv'])}")
    i = result["I"]
    lines += ["", "## I onset through the I chain", "", "| trace | count | rise V/s | peak mV | fall V/s | span mV | rapidness /ms | take-off mV | approach mV/ms |", "|---|---|---|---|---|---|---|---|---|"]
    if i["fit_step"]["status"] == "read":
        lines += [row("I fit, raw", i["fit_step"]["raw"]), row("I fit, chain", i["fit_step"]["chain"])]
    lines += [row("I recording 0.19 nA", i["human"]), "", "## Decision", ""]
    for c in result["decision"]["checks"]:
        status = "no reading" if c["pass"] is None else ("pass" if c["pass"] else "FAIL")
        value = "n/a" if c["value"] is None else f"{c['value']:+.3g}"
        lines.append(f"- {c['cell']} ({c['reading']}) {status}: {c['prediction']} ({value}{'; ' + c['note'] if c['note'] else ''})")
    lines += ["", f"Verdict: {result['decision']['verdict']}", "", "## Reading", ""] + [f"- {r}" for r in result["reading"]] + [""]
    return "\n".join(lines)


def plot(result, edges, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, cell in enumerate(("E", "I")):
        ax = axes[0][col]; ch = result[cell]["chain"]
        for e, s in zip(edges[cell], ch["slopes_mv_ms"]):
            ax.plot(e["t_ms"]*1e3, e["v_mv"], ".", color="0.6", ms=3)
            ax.plot(e["t_ms"]*1e3, chain_response(e["t_ms"], e["i0_na"], e["rb_mohm"], s, ch["tau_p_us"], ch["fc_khz"], ch["t0_ms"], ch["delta_r_mohm"], ch["tau_cmd_us"]), "-", color="C3", lw=.8)
        ax.set(title=f"{cell}: onset edges (dots), fitted round trip (lines); tau_p {ch['tau_p_us']:.0f} us, command {ch['tau_cmd_us']:.0f} us, f_c {ch['fc_khz']:.0f} kHz", xlabel="us after the nominal step", ylabel="mV", xlim=(-100, 300), ylim=(-.6, 2.5))
    c = CHAIN["E"]; t, v = soma_trace(f"{c['fit']}-sweep53"); ch = result["E"]["chain"]
    th, vh = nwb_sweep(c["nwb"], c["human_sweep"]); th, vh, _ = resample(np.asarray(th, float), np.asarray(vh, float), np.zeros(len(th)))
    ax = axes[1][0]
    for label, (tt, vv), style in (("recording", (th, vh), "k-"), ("fit raw", (t, v), "C0-"), ("fit through chain", (t, apply_chain(t, v, ch["tau_p_us"], ch["fc_khz"])), "C3-"),
                                   (f"fit at the peak bound", (t, apply_chain(t, v, result["E"]["peak_bound"].get("tau_p_us", 0.), ch["fc_khz"])), "C1--")):
        m = landmarks(tt, vv, c["pulse"])
        if not m:
            continue
        w = (tt >= m[0]["threshold_ms"]-1.) & (tt <= m[0]["threshold_ms"]+2.)
        ax.plot(vv[w], rate_of_rise(vv)[w], style, lw=1, label=label)
    ax.set(title="E spike 1 at 310 pA, phase plane", xlabel="mV", ylabel="V/s"); ax.legend(fontsize=8)
    ci = CHAIN["I"]; tr = soma_trace(ci["fit"]); chi = result["I"]["chain"]
    ax = axes[1][1]
    if tr is not None:
        t, v = tr
        thi, vhi = human_trace(ROOT/CELLS["I"]["human_dir"]/CELLS["I"]["inputs"][ci["human_input"]][0]); thi, vhi, _ = resample(np.asarray(thi, float), np.asarray(vhi, float), np.zeros(len(thi)))
        for label, (tt, vv), style in (("recording", (thi, vhi), "k-"), ("fit raw", (t, v), "C0-"), ("fit through chain", (t, apply_chain(t, v, chi["tau_p_us"], chi["fc_khz"])), "C3-")):
            m = landmarks(tt, vv, ci["pulse"])
            if not m:
                continue
            w = (tt >= m[0]["threshold_ms"]-1.) & (tt <= m[0]["threshold_ms"]+2.)
            ax.plot(vv[w], rate_of_rise(vv)[w], style, lw=1, label=label)
    ax.set(title="I spike 1 at 0.19 nA, phase plane", xlabel="mV", ylabel="V/s"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


READING = [
    "The chain is measured where it can be and bounded where it cannot. The onset edges show a jump of at most two samples and no "
    "transient larger than 0.4 mV at 1.2 nA, where a live 5 pF pipette pole would put 9 mV: the capacitance neutralisation is live "
    "in practice whatever the metadata field holds, and the pipette pole reads 5 us (E) and 2 us (I) at best, profile flat to 30 us "
    "because the edge measures the round trip and the command's own pole (E 150 us, I 44 us) absorbs it. The unstimulated noise "
    "floor is flat to Nyquist in both files (E -0.7 dB per octave between 10 and 20 kHz, I +0.1, against -24 for a 4-pole corner "
    "below 10 kHz), which reads as a corner at or above 20 kHz if the floor is pipette noise and says nothing if it is the "
    "digitiser's; the chain is therefore applied at an assumed 10 kHz corner as the conservative case, with the 20 kHz "
    "reading beside it.",
    "Through the chain the E fit's spike-1 rise moves from 655 to 610 V/s at 10 kHz and 638 at 20 kHz (the control 639 to 570 and "
    "615): 5 to 15 percent of the gap to the recorded 348, 17 to 45 V/s, which is 11 to 30 sigma of the within-chain repeat and "
    "changes no decision. The residual gap, 260 to 290 V/s, is real. The bound independent of the edges, the pole at which the "
    "filtered peak meets the recorded 35.3 mV, would cover about half of the gap; the registered 'less than half' fired by the letter "
    "at that bound, not at the measurement, which is an order of magnitude smaller. A third leg: the recording repolarises faster "
    "than the fit (-104 against -91 and -95 V/s) while it rises slower; a chain that halved the rise would have halved the fall.",
    "The slow set holds through the chain: the take-off shifts by a constant 0.13 mV at 10 kHz (the registered 0.1 fired by the "
    "letter; the slide across the ramps, the quantity stage 10 decided on, holds to 0.01 mV), approach within 1 percent, axon lead "
    "within 0.02 ms, count unchanged; stages 7 to 11 stand. The I fit's rise, equal to the recording's when read raw (592 against "
    "597), reads 530 through the 10 kHz chain: that match was a coincidence of observation points, and the I fit rises about a tenth "
    "too slowly when both are read through the chain. The E onset span does not hold: it moves by more than 1 mV and non-monotonically across the "
    "ramps under a chain that moves the rise by a few percent; the 10 to 100 V/s span is a fragile reading at a 0.6 mV/ms approach "
    "and the stage-8 E cell (no discrimination) gains that qualification. The I fit's span survives its chain (13.5 against the "
    "recorded 1.9 mV), so the stage-8 I reading stands.",
    "First execution, recorded: with the acquisition corner free the registered edge fit returned 2.65 kHz (I 3.3), which applied "
    "to the fit gave a rise of 384 V/s against the recorded 348 and would have read the whole gap as the chain. The noise floor "
    "refutes that corner; the low value was the command's own pole, which a step edge cannot separate from the output filter. The "
    "registration is amended: the pipette pole from the transient's area, the corner from the noise floor where it is pipette "
    "noise and from an assumed conservative 10 kHz otherwise.",
    "Rule, independent of the outcome and now in the causal model: a fast model quantity is comparable to a recorded one only "
    "through the recording's own chain (here a 3 to 7 percent correction to every rise), and a repeat sigma bounds repeatability, "
    "not bias between the two chains.",
]


def main():
    result = {"stage": "12", "spec": "docs/specs/2026-09-12-h01-topographic-strategy.md", "limits": LIMITS, "reading": READING}
    edges = {}
    for cell, c in CHAIN.items():
        edges[cell] = [edge(c["nwb"], s, c["onset_ms"]) for s in c["edges"]]
        noise = noise_corner(c["nwb"], cell=cell)
        chain = fit_chain(edges[cell], FC_CONSERVATIVE_KHZ)
        result[cell] = {"noise": noise, "chain": chain, "chain_floor": fit_chain(edges[cell], noise["fc_khz"]), "profile": profile_tau(edges[cell], FC_CONSERVATIVE_KHZ), "human": human_step(cell),
                        "edges": [{"sweep": e["sweep"], "i0_na": e["i0_na"], "rb_mohm": e["rb_mohm"]} for e in edges[cell]]}
    e = result["E"]; c = CHAIN["E"]
    e["fit_step"] = step_through(f"{c['fit']}-sweep53", c["pulse"], e["chain"])
    e["fit_step"]["chain_floor"] = step_through(f"{c['fit']}-sweep53", c["pulse"], e["chain_floor"])["chain"]
    e["control_step"] = step_through(c["control"], c["pulse"], e["chain"])
    e["control_step"]["chain_floor"] = step_through(c["control"], c["pulse"], e["chain_floor"])["chain"]
    e["peak_bound"] = peak_bound(f"{c['fit']}-sweep53", c["pulse"], e["chain"]["fc_khz"], e["human"]["peak_mv"])
    e["arm"] = arm_through(c["fit"], e["chain"])
    ci = CHAIN["I"]
    result["I"]["fit_step"] = step_through(ci["fit"], ci["pulse"], result["I"]["chain"])
    result["decision"] = decide(result)
    (OUT/"stage-12.json").write_text(json.dumps(result, indent=2, default=float)+"\n", encoding="utf-8")
    (OUT/"stage-12.md").write_text(markdown(result), encoding="utf-8")
    plot(result, edges, OUT/"stage-12.png")
    print("wrote", OUT/"stage-12.json", result["decision"]["verdict"])


if __name__ == "__main__":
    main()
