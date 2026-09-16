"""Copy keep/drop run receipts into the evidence tree with decimated traces.

Spec: docs/specs/2026-09-16-h01-keep-drop.md. For every ``transfer-all-<cell>[-dthalf|-ramp]``
folder that holds a ``terminal.json`` this copies ``launch.json``, ``run.json`` and
``terminal.json`` unchanged and writes ``trace.npz`` with the voltages sampled every
``--every`` steps (0.5 ms at the primary dt) plus the exact spike times (and the decimated
ramp current of a ramp run), so the full
460,000-sample traces (5 MB per cell) stay on the box while the evidence tree keeps a
reproducible summary of each run.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

RECEIPTS = ("launch.json", "run.json", "terminal.json")


def decimate(npz, every):
    """Sub-sampled time/voltage arrays and the exact spike times of one run."""
    every = int(every)
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    time_ms = np.asarray(npz["time_ms"])
    events = np.asarray(npz["events"]).reshape(len(time_ms), -1).any(axis=1)
    out = dict(time_ms=time_ms[::every], voltage=np.asarray(npz["voltage"])[::every],
               output_voltage=np.asarray(npz["output_voltage"])[::every],
               spike_times_ms=time_ms[events], every=np.int64(every))
    if "ramp_current_na" in npz:   # ramp runs (--ramp-na) save the injected current beside the voltages
        out["ramp_current_na"] = np.asarray(npz["ramp_current_na"])[::every]
    return out


def copy_run(src, dst, every):
    """Copy one run folder's receipts; returns False when the run has not terminated."""
    src, dst = Path(src), Path(dst)
    if not (src/"terminal.json").exists():
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for name in RECEIPTS:
        if (src/name).exists():
            shutil.copyfile(src/name, dst/name)
    with np.load(src/"run.npz") as npz:
        np.savez_compressed(dst/"trace.npz", **decimate(npz, every))
    return True


def copy_all(folder, out_dir, every):
    """Copy every terminated transfer-all run; returns the sorted list of copied labels."""
    copied = []
    for src in sorted(Path(folder).glob("transfer-all-*")):
        if src.is_dir() and copy_run(src, Path(out_dir)/src.name, every):
            copied.append(src.name)
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="var/h01-driven")
    parser.add_argument("--out-dir", default="docs/evidence/h01-keep-drop/runs")
    parser.add_argument("--every", type=int, default=100)
    args = parser.parse_args()
    copied = copy_all(args.folder, args.out_dir, args.every)
    Path(args.out_dir, "index.json").write_text(json.dumps(dict(every=args.every, runs=copied), indent=2)+"\n")
    print(len(copied), "runs copied to", args.out_dir)


if __name__ == "__main__":
    main()
