"""Measure each donor recording's pre-pulse rest for the keep/drop rule (spec 2026-09-16-h01-keep-drop).

The rest is the mean output voltage over the 100 ms before the pulse in the recording's own
clock. Allen NWB files store voltages uncorrected for the -14 mV liquid junction potential; the
correction is applied here once, as the donor import specs prescribe. Exported npz traces are
read in the convention they were exported in.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DEFINITION = ("Mean and sd of the recorded voltage over samples with onset-100 <= t < onset ms, in the "
              "recording's own clock; corrected voltages where the source stores the LJP uncorrected.")

DONORS = {
    "l2-pyramidal-allen-541563728": dict(
        source="human-pyramidal-l2/sweep-50.npz", npz=("time_ms", "corrected_voltage_mv"), onset_ms=1020.,
        shift_mv=0., convention="corrected_voltage_mv = reported - 14 mV LJP, as exported"),
    "l4-pyramidal-allen-527952884": dict(
        source="human-pyramidal-l4/527952752_ephys.nwb", sweep=39, onset_ms=1020., shift_mv=-14.,
        convention="NWB reported - 14 mV LJP"),
    "l3-sst-interneuron-hl5mn1": dict(
        source="human-sst-l3/571700399_ephys.nwb", sweep=44, onset_ms=1020., shift_mv=-14.,
        convention="NWB reported - 14 mV LJP"),
    "l5-pv-basket-hl5bn1": dict(
        source="human-pv/active-0.npz", npz=("time", "voltage"), onset_ms=270., shift_mv=0.,
        convention="as exported (already the comparison convention used by the PV campaign)"),
}


def rest_window(time_ms, voltage_mv, onset_ms, width_ms=100.):
    """Mean/sd of the voltage over onset-width <= t < onset; raises when the window is empty."""
    time_ms, voltage_mv = np.asarray(time_ms, float), np.asarray(voltage_mv, float)
    mask = (time_ms >= onset_ms-width_ms) & (time_ms < onset_ms)
    if not mask.any():
        raise ValueError(f"no samples in [{onset_ms-width_ms}, {onset_ms}) ms")
    segment = voltage_mv[mask]
    return dict(mean_mv=float(segment.mean()), sd_mv=float(segment.std()), n=int(mask.sum()),
                window_ms=[float(onset_ms-width_ms), float(onset_ms)])


def load_npz(path, time_key, voltage_key):
    """Exported trace: (time_ms, voltage_mv) arrays under the named keys."""
    archive = np.load(path)
    return np.asarray(archive[time_key], float), np.asarray(archive[voltage_key], float)


def load_nwb_sweep(path, sweep):
    """Allen IVSCC 1.0 NWB sweep: (time_ms, voltage_mv) from acquisition/timeseries/Sweep_<n>.

    The stored samples are volts in both donor files (unit attribute ``Volts``), while the
    ``conversion`` attribute is 0.001 in one file and 1.0 in the other; it is not applied. A
    raw mean below 1 in magnitude is read as volts and scaled to mV.
    """
    import h5py  # noqa: PLC0415 - optional dependency, only needed for NWB sources

    with h5py.File(path, "r") as handle:
        group = handle["acquisition/timeseries"][f"Sweep_{sweep}"]
        rate = float(group["starting_time"].attrs["rate"])
        voltage = np.asarray(group["data"][()], float)
    if abs(np.nanmean(voltage)) < 1.:
        voltage = voltage*1e3
    return np.arange(len(voltage))/rate*1e3, voltage


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(cache):
    cache = Path(cache)
    donors = {}
    for donor, spec in DONORS.items():
        path = cache/spec["source"]
        if "npz" in spec:
            time_ms, voltage_mv = load_npz(path, *spec["npz"])
        else:
            time_ms, voltage_mv = load_nwb_sweep(path, spec["sweep"])
        rest = rest_window(time_ms, voltage_mv+spec["shift_mv"], spec["onset_ms"])
        donors[donor] = dict(rest_mv=rest["mean_mv"], sd_mv=rest["sd_mv"], n=rest["n"], window_ms=rest["window_ms"],
                             source=spec["source"], sha256=sha256(path), convention=spec["convention"])
    return dict(definition=DEFINITION, donors=donors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = measure(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+"\n", newline="\n")
    for donor, row in receipt["donors"].items():
        print(f"{donor}: {row['rest_mv']:.2f} mV (sd {row['sd_mv']:.2f}, n {row['n']})")


if __name__ == "__main__":
    main()
