"""Human long-square datums per deployed donor for the C3 test-to-failure gate.

For each donor recording in ``.cache/`` the long-square sweep family is listed with its
amplitude, pulse window, -20 mV crossing count inside the pulse and pre-pulse rest, and the
firing-range datums are derived: ``rheobase_pa`` (lowest firing amplitude), ``sweep_step_pa``
(median amplitude step of the family), ``highest_firing_pa`` (highest amplitude that still
fires) and the registered repeat counts at the primary input (spec
2026-09-16-h01-keep-drop, table). NWB voltages are corrected by the -14 mV liquid junction
potential once, as ``h01_keep_drop_rest.py`` does. Read-only analysis; runs on the laptop.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from h01_keep_drop_rest import load_nwb_sweep, rest_window  # noqa: E402
from h01_pv_human_datums import spike_datums  # noqa: E402

JUNCTION_MV = -14.
DETECTION_MV = -20.
REST_MS = 100.

DONORS = {
    "l2-pyramidal-allen-541563728": dict(
        nwb="human-pyramidal-l2/recording.nwb", primary_pa=310., repeat_counts=[10],
        source_cell="Allen 541563728, the deployed donor's own recording"),
    "l4-pyramidal-allen-527952884": dict(
        nwb="human-pyramidal-l4/527952752_ephys.nwb", primary_pa=90., repeat_counts=[12],
        source_cell="Allen 527952884 (ephys 527952752), the deployed donor's own recording"),
    "l5-pv-basket-hl5bn1": dict(
        nwb="human-pv/528687507_ephys.nwb", primary_pa=190., repeat_counts=[12],
        source_cell="Allen 528687520 (ephys 528687507), the human PV recording the campaign's PV "
                    "repeat spread comes from; the deployed HL5BN1 donor's own protocol datums are the "
                    "ModelDB exports listed under donor_exports (0.19/0.23/0.27 nA, all firing)",
        exports=[("human-pv/active-0.npz", .19), ("human-pv/active-1.npz", .23), ("human-pv/active-2.npz", .27)],
        export_pulse_ms=(270., 1270.)),
    "l3-sst-interneuron-hl5mn1": dict(
        nwb="human-sst-l3/571700399_ephys.nwb", primary_pa=100., repeat_counts=[14, 14, 13, 12],
        source_cell="Allen 571700399 (HL5MN1 donor recording)"),
}


def stimulus_window(time_ms, current_pa, baseline_fraction=.05, floor_pa=1.):
    """Amplitude and pulse window of a long-square command.

    The pulse is the longest contiguous run of samples that leave the holding level (the
    Allen protocol's short test pulse at the start of every sweep is shorter than the long
    square, whatever its amplitude).

    Parameters
    ----------
    time_ms, current_pa : array_like
        Command waveform.
    baseline_fraction : float, optional
        Leading fraction of samples that defines the holding level.
    floor_pa : float, optional
        Deviation under which a sample counts as holding.

    Returns
    -------
    dict
        ``amplitude_pa`` (median signed deviation over the pulse), ``on_ms``, ``off_ms`` (the
        run's first sample and one step past its last) and ``baseline_pa``; the window is
        None when the command never leaves its holding level.
    """
    time_ms, current_pa = np.asarray(time_ms, float), np.asarray(current_pa, float)
    lead = max(1, int(len(current_pa)*baseline_fraction))
    baseline = float(np.median(current_pa[:lead]))
    deviation = current_pa-baseline
    away = np.abs(deviation) > floor_pa
    if not away.any():
        return dict(amplitude_pa=0., on_ms=None, off_ms=None, baseline_pa=baseline)
    edges = np.flatnonzero(np.diff(np.r_[0, away.astype(int), 0]))
    starts, stops = edges[::2], edges[1::2]
    longest = int(np.argmax(stops-starts))
    first, last = int(starts[longest]), int(stops[longest])-1
    step = float(np.median(np.diff(time_ms))) if len(time_ms) > 1 else 0.
    return dict(amplitude_pa=float(np.median(deviation[first:last+1])), on_ms=float(time_ms[first]),
                off_ms=float(time_ms[last])+step, baseline_pa=baseline)


def sweep_features(time_ms, voltage_mv, command_pa, amplitude_attr_pa=None):
    """One sweep's amplitude, window, crossing count inside the window and pre-pulse rest."""
    window = stimulus_window(time_ms, command_pa)
    amplitude = window["amplitude_pa"] if amplitude_attr_pa is None else float(amplitude_attr_pa)
    row = dict(amplitude_pa=amplitude, measured_amplitude_pa=window["amplitude_pa"],
               on_ms=window["on_ms"], off_ms=window["off_ms"], count=None, rest_mv=None, rest_sd_mv=None)
    if window["on_ms"] is None:
        return row
    time_ms, voltage_mv = np.asarray(time_ms, float), np.asarray(voltage_mv, float)
    mask = (time_ms >= window["on_ms"]) & (time_ms < window["off_ms"])
    row["count"] = len(spike_datums(time_ms[mask], voltage_mv[mask], DETECTION_MV))
    rest = rest_window(time_ms, voltage_mv, window["on_ms"], REST_MS)
    row.update(rest_mv=rest["mean_mv"], rest_sd_mv=rest["sd_mv"])
    return row


def read_long_square_sweeps(path, shift_mv=JUNCTION_MV):
    """Every long-square sweep of an Allen IVSCC NWB file with its features.

    The command is read from ``stimulus/presentation/Sweep_<n>/data``; samples with a
    magnitude under 1e-6 are amperes (scaled to pA), larger ones already pA. The
    ``aibs_stimulus_amplitude_pa`` attribute, when present, is the recorded amplitude and the
    measured deviation is kept beside it.
    """
    import h5py  # noqa: PLC0415 - optional dependency, only needed for NWB sources

    rows = []
    with h5py.File(path, "r") as handle:
        presentation = handle["stimulus/presentation"]
        for name in sorted(presentation, key=lambda n: int(n.split("_")[-1])):
            group = presentation[name]
            kind = group["aibs_stimulus_name"][()] if "aibs_stimulus_name" in group else b""
            kind = kind.decode() if isinstance(kind, bytes) else str(kind)
            if "long square" not in kind.lower():
                continue
            command = np.asarray(group["data"][()], float)
            if np.nanmax(np.abs(command)) < 1e-6:
                command = command*1e12
            rate = float(group["starting_time"].attrs["rate"])
            attr = float(group["aibs_stimulus_amplitude_pa"][()]) if "aibs_stimulus_amplitude_pa" in group else None
            sweep = int(name.split("_")[-1])
            time_ms, voltage_mv = load_nwb_sweep(path, sweep)
            row = sweep_features(np.arange(len(command))/rate*1e3, voltage_mv+shift_mv, command, attr)
            rows.append(dict(sweep=sweep, stimulus=kind, **row))
    return rows


def derive_datums(rows, primary_pa, repeat_counts):
    """Firing-range datums from a sweep family; None where the family does not reach them.

    ``sweep_step_pa`` is the median upward amplitude step between consecutive positive sweeps
    in recording order, so repeat sweeps and the odd extra level do not shift the family's
    step (the Allen ascending long-square family is contiguous in sweep numbering).
    """
    positive = sorted((r for r in rows if r["count"] is not None and r["amplitude_pa"] > 0.),
                      key=lambda r: r["amplitude_pa"])
    firing = [r for r in positive if r["count"] >= 1]
    levels = sorted({round(r["amplitude_pa"], 3) for r in positive})
    ordered = sorted(positive, key=lambda r: r.get("sweep", 0))
    steps = [b["amplitude_pa"]-a["amplitude_pa"] for a, b in zip(ordered, ordered[1:])
             if b["amplitude_pa"]-a["amplitude_pa"] > .5]   # upward steps between consecutive sweeps
    at_primary = [r for r in positive if abs(r["amplitude_pa"]-primary_pa) < .5]
    rests = [r["rest_mv"] for r in rows if r["rest_mv"] is not None]
    return dict(
        primary_pa=primary_pa, repeat_counts=list(repeat_counts),
        rheobase_pa=firing[0]["amplitude_pa"] if firing else None,
        rheobase_count=firing[0]["count"] if firing else None,
        sweep_step_pa=float(np.median(steps)) if len(steps) else None,
        highest_firing_pa=firing[-1]["amplitude_pa"] if firing else None,
        highest_firing_count=firing[-1]["count"] if firing else None,
        highest_recorded_pa=positive[-1]["amplitude_pa"] if positive else None,
        measured_counts_at_primary=[r["count"] for r in at_primary],
        measured_sweeps_at_primary=[r.get("sweep") for r in at_primary],
        family_amplitudes_pa=levels,
        rest_repeat_mean_mv=float(np.mean(rests)) if rests else None,
        rest_repeat_sd_mv=float(np.std(rests)) if len(rests) >= 2 else None)


def export_counts(cache, exports, pulse_ms):
    """Crossing counts of the deployed donor's own exported traces, in their own convention."""
    rows = []
    for name, current_na in exports:
        data = np.load(Path(cache)/name)
        time_ms, voltage_mv = np.asarray(data["time"], float), np.asarray(data["voltage"], float)
        mask = (time_ms >= pulse_ms[0]) & (time_ms < pulse_ms[1])
        rows.append(dict(source=name, amplitude_pa=current_na*1e3,
                         count=len(spike_datums(time_ms[mask], voltage_mv[mask], DETECTION_MV)),
                         sha256=hashlib.sha256((Path(cache)/name).read_bytes()).hexdigest()))
    return rows


def measure(cache, donors=None):
    """Datums for every donor from the recordings under ``cache``."""
    cache = Path(cache)
    donors = DONORS if donors is None else donors
    result = {}
    for donor, spec in donors.items():
        path = cache/spec["nwb"]
        rows = read_long_square_sweeps(path)
        entry = dict(source=spec["nwb"], sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     source_cell=spec["source_cell"], convention=f"NWB reported {JUNCTION_MV:+g} mV LJP",
                     crossing_threshold_mv=DETECTION_MV, sweeps=rows,
                     **derive_datums(rows, spec["primary_pa"], spec["repeat_counts"]))
        if "exports" in spec:
            entry["donor_exports"] = export_counts(cache, spec["exports"], spec["export_pulse_ms"])
        result[donor] = entry
    return dict(
        definition=("Long-square family per donor: amplitude (aibs_stimulus_amplitude_pa, measured deviation "
                    "beside it), pulse window from the command, -20 mV crossing count inside the window on "
                    "LJP-corrected voltage, rest over the 100 ms before onset. rheobase_pa = lowest firing "
                    "amplitude; sweep_step_pa = median upward step between consecutive positive sweeps in recording "
                    "order; highest_firing_pa = "
                    "highest amplitude with a crossing; repeat_counts = registered counts at the primary "
                    "input (spec 2026-09-16-h01-keep-drop); measured counts at that input are recorded beside "
                    "them and are not the gate's datum. rest_repeat_mean_mv / rest_repeat_sd_mv = mean and sd of the "
                    "pre-pulse rest across the long-square sweeps: the rest datum and its tolerance for the "
                    "test-to-failure gate; donor-rest.json keeps the single-sweep value for the legacy verdict."),
        donors=result)


def render(report):
    """One table row per donor."""
    lines = ["donor | rheobase_pa | step_pa | highest_firing_pa | highest_recorded_pa | primary_pa | "
             "registered repeats | measured at primary | rest_mv (repeat sd)"]
    for donor, row in report["donors"].items():
        lines.append(f"{donor} | {row['rheobase_pa']} | {row['sweep_step_pa']} | {row['highest_firing_pa']} | "
                     f"{row['highest_recorded_pa']} | {row['primary_pa']} | {row['repeat_counts']} | "
                     f"{row['measured_counts_at_primary']} | {row['rest_repeat_mean_mv']:.2f} ({row['rest_repeat_sd_mv']:.2f})")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/h01-c3-keep-drop/human-datums.json"))
    args = parser.parse_args(argv)
    report = measure(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", newline="\n")
    print(render(report))


if __name__ == "__main__":
    main()
