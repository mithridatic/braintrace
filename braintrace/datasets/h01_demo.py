"""Run a passive H01 cable model with ``python -m braintrace.datasets.h01_demo``."""

import argparse
import json
from pathlib import Path

import braincell
import brainunit as u
import numpy as np

from .h01 import H01Archive, fetch_h01


def make_passive_cell(imported, *, current_na=0.001, duration_ms=1.0, cv_policy=None,
                      stimulus_location=None, recording_location=None):
    """Construct a configurable demonstration cell from an H01 component.

    Parameters
    ----------
    imported : H01Component
        Geometry returned by ``H01Archive.load``.
    current_na : float, optional
        Current injected at the SWC root in nanoamperes; not necessarily soma.
    duration_ms : float, optional
        Clamp duration in milliseconds.
    cv_policy : braincell.CVPolicy or None, optional
        Caller-selected spatial discretization.
    stimulus_location, recording_location : LocsetExpr or None, optional
        Anatomical locations for current injection and voltage recording.
        Each defaults to the source root when omitted.

    Returns
    -------
    braincell.Cell
        Uninitialized cell with a voltage probe named ``voltage``.

    Notes
    -----
    Uniform passive parameters are demonstration assumptions, not a fit to
    this human neuron. The imported anatomy itself carries no ion channels.
    """
    from braincell.filter import AllRegion, RootLocation
    from braincell.mech import Channel, StateProbe

    if not np.isfinite(current_na) or not np.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("Current must be finite and duration must be positive and finite.")
    cell = braincell.Cell(imported.morphology, cv_policy=cv_policy, V_init=-65 * u.mV)
    cell.paint(AllRegion(), braincell.CableProperty(
        resting_potential=-65 * u.mV,
        membrane_capacitance=1.0 * u.uF / u.cm**2,
        axial_resistivity=100.0 * u.ohm * u.cm,
    ))
    cell.paint(AllRegion(), Channel("IL", g_max=.1 * u.mS / u.cm**2, E=-65 * u.mV))
    cell.place(recording_location if recording_location is not None else RootLocation(0.),
               StateProbe(field="v", name="voltage"))
    cell.place(stimulus_location if stimulus_location is not None else RootLocation(0.), braincell.CurrentClamp(
        delay=0 * u.ms, durations=duration_ms * u.ms, amplitudes=current_na * u.nA,
    ))
    return cell


def main(argv=None):
    """Run an explicit download or offline H01 import and simulation.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments. ``None`` reads the process arguments.

    Returns
    -------
    dict
        Import and simulation evidence, also emitted as JSON.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Existing official SWC ZIP (offline).")
    source.add_argument("--download", type=Path, metavar="CACHE_DIR", help="Download/cache official SWCs.")
    parser.add_argument("--neuron", default="810151953")
    parser.add_argument("--component", type=int, default=0)
    parser.add_argument("--duration-ms", type=float, default=1.0)
    parser.add_argument("--dt-ms", type=float, default=.025)
    parser.add_argument("--current-na", type=float, default=.001)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence file.")
    args = parser.parse_args(argv)
    if not np.isfinite(args.dt_ms) or args.dt_ms <= 0:
        parser.error("--dt-ms must be positive and finite")
    archive = H01Archive(args.archive) if args.archive else fetch_h01(args.download)
    imported = archive.load(args.neuron, component=args.component)
    cell = make_passive_cell(imported, current_na=args.current_na, duration_ms=args.duration_ms)
    # Cell.run uses brainstate.transform.for_loop for the entire time sequence.
    result = cell.run(dt=args.dt_ms * u.ms, duration=args.duration_ms * u.ms)
    voltage = np.asarray(result.traces["voltage"].to_decimal(u.mV))
    if not np.isfinite(voltage).all():
        raise RuntimeError("H01 demonstration produced non-finite voltages.")
    report = {
        "provenance": imported.provenance,
        "braincell_version": braincell.__version__,
        "cell_components": archive.components(args.neuron),
        "source_points": len(imported.source_rows), "compartments": len(cell.cvs),
        "import_diagnostics": imported.report.format(),
        "model": "uniform passive cable; no human electrophysiology fit",
        "duration_ms": args.duration_ms, "dt_ms": args.dt_ms, "current_na": args.current_na,
        "steps": len(voltage), "voltage_min_mv": float(voltage.min()),
        "voltage_max_mv": float(voltage.max()), "voltage_final_mv": float(voltage[-1]),
        "finite": True,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return report


if __name__ == "__main__":
    main()
