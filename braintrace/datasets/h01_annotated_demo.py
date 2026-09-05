"""Stimulate an H01 synapse position and record at an annotated soma."""

import argparse
from collections import Counter
import json
from pathlib import Path

import brainunit as u
import brainstate
import numpy as np
from braincell.mech import Channel

from .h01 import H01Archive
from .h01_annotations import H01Annotations
from .h01_demo import make_passive_cell


def make_annotated_cell(imported, annotations, *, max_distance_um, current_na=.001, duration_ms=1.):
    """Construct a source-guided passive demonstration with explicit assumptions.

    Parameters
    ----------
    imported : H01Component
        Connected component containing labelled soma samples.
    annotations : H01Annotations
        Offline cell metadata and synapse table.
    max_distance_um : float
        Explicit synapse projection tolerance.
    current_na, duration_ms : float, optional
        Demonstration input amplitude and duration; not measured physiology.

    Returns
    -------
    cell : braincell.Cell
        Cell with input at the closest accepted postsynaptic endpoint, voltage
        recording at soma, and doubled leak on strictly annotated dendrite cable.
    evidence : dict
        Metadata, source provenance, projection results and modeling assumptions.
    """
    anatomy = imported.anatomy()
    soma = anatomy.soma_location()
    metadata = annotations.metadata(imported.neuron_id)
    rows = annotations.synapses(imported.neuron_id, role="post")
    projections = anatomy.project_synapses(rows, max_distance_um=max_distance_um)
    accepted = [i for i, p in enumerate(projections) if p.status == "projected"]
    if not accepted:
        raise ValueError("No postsynaptic endpoints project within the supplied tolerance.")
    index = min(accepted, key=lambda i: (projections[i].distance_um, rows[i].source_row))
    selected = projections[index]
    cell = make_passive_cell(imported, current_na=current_na, duration_ms=duration_ms,
                             stimulus_location=selected.location, recording_location=soma)
    dendrite = anatomy.region("dendrite")
    if dendrite.intervals:
        cell.paint(dendrite, Channel("IL", g_max=.2 * u.mS / u.cm**2, E=-65 * u.mV))
    evidence = {
        "anatomy": anatomy.provenance,
        "metadata": {"tags": metadata.tags, "measurements": dict(metadata.measurements),
                     "descriptions": dict(metadata.descriptions), "provenance": dict(metadata.provenance)},
        "synapses": {"provenance": annotations.synapse_provenance, "post_csv_rows": len(rows),
                     "projection_counts": dict(Counter(p.status for p in projections)),
                     "max_distance_um": max_distance_um, "stimulated_source_row": rows[index].source_row,
                     "stimulated_position_um": rows[index].position_um,
                     "projection_distance_um": selected.distance_um},
        "soma_location": soma.evaluate(imported.morphology).points,
        "dendrite_intervals": len(dendrite.intervals),
        "model": "passive demonstration; current clamp at a synapse coordinate, not a fitted receptor model",
        "parameters": {"current_na": current_na, "duration_ms": duration_ms,
                       "background_leak_ms_cm2": .1, "strict_dendrite_leak_ms_cm2": .2,
                       "region_policy": "strict", "rest_mv": -65, "capacitance_uf_cm2": 1,
                       "axial_resistivity_ohm_cm": 100},
    }
    return cell, evidence


def main(argv=None):
    """Run the annotated demonstration from installed BrainTrace.

    Parameters
    ----------
    argv : list of str or None, optional
        CLI arguments; defaults to process arguments.

    Returns
    -------
    dict
        Source and compiled-run evidence, printed as JSON.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neuron", default="810151953")
    parser.add_argument("--component", type=int, default=0)
    parser.add_argument("--max-distance-um", type=float, required=True)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    with brainstate.environ.context(precision=args.precision):
        imported = H01Archive(args.archive).load(args.neuron, component=args.component)
        cell, evidence = make_annotated_cell(imported, H01Annotations(args.annotations),
                                             max_distance_um=args.max_distance_um)
        run = cell.run(dt=.025 * u.ms, duration=1 * u.ms)
        voltage = np.asarray(run.traces["voltage"].to_decimal(u.mV))
    if not np.isfinite(voltage).all():
        raise RuntimeError("Annotated H01 demo produced non-finite voltages.")
    evidence["simulation"] = {"steps": len(voltage), "dt_ms": .025, "compartments": len(cell.cvs),
                              "precision_bits": voltage.dtype.itemsize * 8,
                              "finite": True, "voltage_min_mv": float(voltage.min()),
                              "voltage_max_mv": float(voltage.max()), "voltage_final_mv": float(voltage[-1])}
    text = json.dumps(evidence, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return evidence


if __name__ == "__main__":
    main()
