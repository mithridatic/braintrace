"""Multivari small multiples and conjugate-pair views of direct H01 observations."""

from pathlib import Path

import numpy as np


def _panels(count, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(count, 1, figsize=(13, max(4, count*3)), squeeze=False,
                             sharey=True)
    fig.subplots_adjust(left=.08, right=.98, bottom=.08, top=.94, hspace=.55)
    fig.suptitle(title)
    return fig, axes[:, 0]


def multivari(records, output, offsets=(0., 2., 6.)):
    """Plot individual phase observations grouped by record and selected cycle.

    Parameters
    ----------
    records : list of dict
        Named records containing direct observation windows.
    output : pathlib.Path
        PNG output path. All panels use one voltage scale.
    offsets : tuple of float
        Fixed recovery offsets in ms, displayed as separate observations.

    Returns
    -------
    None
        Writes the multivari figure without averaging observations.
    """
    import matplotlib.pyplot as plt
    fig, axes = _panels(len(records), "Matryoshka / multivari: phase within window, cycle within input/system")
    for ax, item in zip(axes, records):
        ticks, labels = [], []
        windows = [w for w in item["observation"]["windows"] if w["kind"] in ("recovery", "tail", "no_spike_pulse")]
        for index, window in enumerate(windows):
            samples = [s for s in window["phase_samples"] if s["offset_ms"] in offsets]
            xx = index*4+np.arange(len(samples))
            yy = [s["voltage_mv"] if s["status"] == "available" else np.nan for s in samples]
            ax.plot(xx, yy, "o-", lw=1, ms=4)
            ticks.append(index*4+1)
            labels.append(f"{window['kind']}\ncycle {window['cycle'] or '-'}")
            if window["status"] != "available":
                ax.text(index*4+1, .1, "unavailable", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
            ax.axvline(index*4+3.3, color="0.85", lw=.6)
        ax.set(title=item["label"], ylabel="Membrane voltage (mV)", xticks=ticks, xticklabels=labels)
        ax.grid(axis="y", alpha=.2)
    axes[-1].set_xlabel(f"Offsets {offsets} ms: from trough for recovery/tail, pulse onset for no-spike; missing phases remain gaps")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def conjugates(records, arrays, output):
    """Plot local voltage/current and voltage/charge paths per selected window.

    Parameters
    ----------
    records : list of dict
        Named observations with boundary metadata.
    arrays : mapping
        Retained selected samples, prefixed by each record's array key.
    output : pathlib.Path
        Destination for the paired-variable figure.

    Returns
    -------
    None
        Saves two physically distinct views with shared axes within each column.
    """
    import matplotlib.pyplot as plt
    models = [r for r in records if r["observation"]["current_evidence"] in ("available", "partial")]
    if not models:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(.5, .5, "Conjugate current/charge observations unavailable", ha="center")
        ax.set_axis_off()
    else:
        fig, axes = plt.subplots(len(models), 2, figsize=(13, 4*len(models)), squeeze=False,
                                 sharex="col", sharey="col", layout="constrained")
        for row, item in enumerate(models):
            record = item["observation"]
            for index, w in enumerate(record["windows"]):
                if w["status"] != "available":
                    continue
                p = f"{item['array_key']}_{w['array_prefix']}"
                voltage = arrays[f"{p}_voltage_mv"]
                label = f"{w['kind']} {w['cycle'] or '-'}"
                # Storage is the net local charging current, not one guessed channel.
                if f"{p}_current_storage_na" in arrays:
                    current = arrays[f"{p}_current_storage_na"]
                    color = f"C{index % 10}"
                    axes[row, 0].plot(voltage, current, label=label, lw=1, color=color)
                    axes[row, 0].plot(voltage[0], current[0], "o", ms=3, color=color)
                    axes[row, 1].plot(voltage, arrays[f"{p}_charge_storage_pc"], label=label, lw=1, color=color)
            for ax in axes[row]:
                ax.grid(alpha=.2)
                ax.set_xlabel("Membrane voltage (mV)")
                if ax.lines:
                    ax.legend(fontsize=7)
            axes[row, 0].set(title=item["label"], ylabel="Local charge-storage current (nA)")
            axes[row, 1].set(title="Cumulative charge from each window's start", ylabel="Charge (pC)")
        fig.suptitle("Conjugate pairs: V-I and V-Q | dots mark starts | no whole-cell energy claim")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def trajectories(records, arrays, output):
    """Plot selected original trajectories with explicit pulse and window clocks.

    Parameters
    ----------
    records : list of dict
        Observation records.
    arrays : mapping
        Original selected samples.
    output : pathlib.Path
        Output image path.

    Returns
    -------
    None
        Writes the observed trajectories on a shared voltage scale.
    """
    import matplotlib.pyplot as plt
    fig, axes = _panels(len(records), "Direct response: selected original samples, original pulse clock")
    for ax, item in zip(axes, records):
        obs = item["observation"]
        for w in obs["windows"]:
            if w["status"] != "available":
                continue
            p = f"{item['array_key']}_{w['array_prefix']}"
            ax.plot(arrays[f"{p}_time_ms"]-obs["pulse_ms"][0], arrays[f"{p}_voltage_mv"],
                    label=f"{w['kind']} {w['cycle'] or '-'}", lw=1)
        ax.set(title=item["label"], ylabel="Voltage (mV)")
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("Time since pulse onset (ms); gaps are unselected intervals")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def render_bundle(record, arrays, stem):
    """Render the three required views for a saved direct-observation bundle.

    Parameters
    ----------
    record : dict
        Bundle with a records list.
    arrays : mapping
        Stored selected samples.
    stem : path-like
        Output prefix.

    Returns
    -------
    list of str
        Written PNG paths.
    """
    outputs = []
    for name, plot in (("multivari", multivari), ("conjugates", conjugates), ("trajectories", trajectories)):
        output = Path(f"{stem}-{name}.png")
        if name == "multivari":
            plot(record["records"], output)
        else:
            plot(record["records"], arrays, output)
        outputs.append(str(output))
    slow = Path(f"{stem}-multivari-slow.png")
    multivari(record["records"], slow, (20., 60., 200.))
    outputs.append(str(slow))
    return outputs
